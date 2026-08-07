import json

import h5py
import numpy as np
from click.testing import CliRunner

from xrd_app.config import ProjectConfig, default_config
from xrd_app.core import io, xrf_selection
from xrd_app.xrf_cli import main
from xrd_app.xrf_project import XRFProject


def _selection():
    return {
        "attrs": {"scan": 24, "n_total_frames": 3, "channels": [0, 1]},
        "source_files": ["raw/a.h5", "raw/b.h5"],
        "frames": {
            "global_frame_index": [0, 1, 2],
            "source_file_index": [0, 0, 1],
            "source_frame_index": [0, 1, 0],
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        },
        "materials": {
            "Br": {
                "intensity": [10.0, 20.0, 30.0],
                "keep": [False, True, True],
                "attrs": {
                    "minimum_counts": 20.0,
                    "energy_range_kev": [11.7, 12.1],
                },
            },
            "Pb": {
                "intensity": [4.0, 5.0, 6.0],
                "keep": [False, False, True],
                "attrs": {"minimum_counts": 6.0},
            },
        },
        "spectrum": {
            "energy_kev": [1.0, 2.0, 3.0],
            "summed_counts": [100.0, 200.0, 300.0],
        },
    }


def test_selection_round_trip_and_stable_hash(tmp_path):
    output = xrf_selection.save(tmp_path / "selection.h5", _selection())
    loaded = xrf_selection.load(output)

    assert loaded["attrs"]["format"] == xrf_selection.FORMAT
    assert loaded["attrs"]["channels"] == [0, 1]
    assert loaded["materials"]["Br"]["attrs"]["energy_range_kev"] == [11.7, 12.1]
    np.testing.assert_array_equal(loaded["materials"]["Br"]["keep"], [False, True, True])
    assert loaded["attrs"]["selection_hash"] == xrf_selection.validate(_selection())["attrs"]["selection_hash"]
    assert xrf_selection.summary(loaded)["materials"]["Br"]["retained_frames"] == 2


def test_xrf_addon_can_be_created_inside_existing_xrd_project(tmp_path):
    config = ProjectConfig(tmp_path, data=default_config("existing", tmp_path, 7))
    config.create_tree()
    config.save()
    original = config.config_path.read_text()

    project = XRFProject.load(tmp_path).create_addon()

    assert project.root == tmp_path
    assert project.config_path == tmp_path / "XRF" / "xrf_config.yaml"
    assert project.config_path.exists()
    assert config.config_path.read_text() == original
    assert project.data["active_scan"] == "Scan_0007"


def test_xrf_cli_init_adds_to_existing_xrd_project(tmp_path):
    config = ProjectConfig(tmp_path, data=default_config("existing", tmp_path))
    config.create_tree()
    config.save()

    result = CliRunner().invoke(main, [
        "init", "--name", "ignored add-on label", "--root", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert "Created XRF add-on" in result.output
    assert (tmp_path / "XRF" / "xrf_config.yaml").exists()


def test_position_offset_is_copied_to_canonical_project_path(tmp_path):
    config = ProjectConfig(tmp_path, data=default_config("existing", tmp_path))
    config.create_tree()
    config.save()
    project = XRFProject.load(tmp_path).create_addon()
    source = tmp_path / "incoming-offset.json"
    source.write_text(json.dumps({"theta": [1.0], "y_offset": [2.0]}))

    destination = project.set_position_offset(source)

    assert destination == tmp_path / "XRF" / "Metadata" / "position_offset.json"
    assert json.loads(destination.read_text()) == {"theta": [1.0], "y_offset": [2.0]}
    assert project.data["data_sources"]["position_offset"] == "Metadata/position_offset.json"
    assert not destination.is_symlink()


def test_threshold_recalculation_uses_full_cached_intensities():
    updated = xrf_selection.apply_threshold(_selection(), "Br", 25)

    np.testing.assert_array_equal(updated["materials"]["Br"]["keep"], [False, False, True])
    assert updated["materials"]["Br"]["attrs"]["minimum_counts"] == 25


def test_quadratic_pixel_energy_round_trip():
    calibration = {
        "quadratic_kev": 5.263744e-7,
        "linear_kev": 8.41967e-3,
        "offset_kev": 1.136032,
    }
    pixels = np.asarray([0.0, 1056.0, 1191.0])

    energy = xrf_selection.pixel_to_kev(pixels, calibration)
    recovered = xrf_selection.kev_to_pixel(energy, calibration)

    np.testing.assert_allclose(recovered, pixels, atol=1e-8)


def test_xrf_cli_loads_canonical_selection(tmp_path):
    project = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(main, ["init", "--name", "test", "--root", str(project)]).exit_code == 0
    source = xrf_selection.save(tmp_path / "external.h5", _selection())

    result = runner.invoke(main, [
        "load-data", "--root", str(project), "--source", str(source),
    ])

    assert result.exit_code == 0, result.output
    loaded = xrf_selection.load(project / "XRF" / "Processed" / "Scan_0024_xrf_selection.h5")
    assert sorted(loaded["materials"]) == ["Br", "Pb"]


def test_xrf_cli_registers_raw_me7_directory(tmp_path):
    project = tmp_path / "project"
    me7 = tmp_path / "Scan_0024" / "ME7"
    me7.mkdir(parents=True)
    with h5py.File(me7 / "scan_0024_00001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=np.zeros((1, 7, 4096), dtype=np.uint32))
    runner = CliRunner()
    assert runner.invoke(main, ["init", "--name", "test", "--root", str(project)]).exit_code == 0

    result = runner.invoke(main, [
        "load-data", "--root", str(project), "--source", str(me7), "--scan", "24",
    ])

    assert result.exit_code == 0, result.output
    registered = XRFProject.load(project)
    assert registered.data["scans"]["Scan_0024"]["me7_dir"] == str(me7.resolve())


def test_xrf_cli_loads_scan_set(tmp_path):
    project = tmp_path / "project"
    scans = tmp_path / "scans"
    for scan in (24, 25):
        me7 = scans / f"Scan_{scan:04d}" / "ME7"
        me7.mkdir(parents=True)
        with h5py.File(me7 / f"scan_{scan:04d}_00001.h5", "w") as handle:
            handle.create_dataset(
                "entry/data/data", data=np.zeros((2, 7, 4096), dtype=np.uint32)
            )
    (scans / "Scan_9999").mkdir()
    runner = CliRunner()
    assert runner.invoke(main, ["init", "--name", "test", "--root", str(project)]).exit_code == 0

    result = runner.invoke(main, [
        "load-data", "--root", str(project), "--source", str(scans),
    ])

    assert result.exit_code == 0, result.output
    registered = XRFProject.load(project)
    assert sorted(registered.data["scans"]) == ["Scan_0024", "Scan_0025"]
    assert "Registered Scan_0024: 1 ME7 files, 2 points" in result.output
    assert "Registered Scan_0025: 1 ME7 files, 2 points" in result.output


def test_xrf_project_discovers_local_processed_selection(tmp_path):
    project = tmp_path / "project"
    runner = CliRunner()
    assert runner.invoke(main, ["init", "--name", "test", "--root", str(project)]).exit_code == 0
    path = project / "XRF" / "Processed" / "Scan_0024_xrf_selection.h5"
    xrf_selection.save(path, _selection())

    addon = XRFProject.load(project)
    discovered = addon.discover_processed()

    assert discovered == ["Scan_0024"]
    assert addon.data["scans"]["Scan_0024"]["selection"]["path"] == str(path.resolve())


def test_sum_selected_xrd_reads_only_retained_frames(tmp_path):
    xrd = tmp_path / "scan_0024_00001.h5"
    frames = np.stack([
        np.full((2, 2), value, dtype=np.uint16) for value in (1, 10, 100)
    ])
    with h5py.File(xrd, "w") as handle:
        handle.create_dataset("entry/data/data", data=frames)
    selection = _selection()
    selection["source_files"] = [str(xrd)]
    selection["frames"]["source_file_index"] = [0, 0, 0]
    selection["frames"]["source_frame_index"] = [0, 1, 2]
    selection["materials"]["Br"]["keep"] = [False, True, True]
    progress = []

    image, count = xrf_selection.sum_selected_xrd(
        selection, "Br", progress=lambda done, total: progress.append((done, total))
    )

    np.testing.assert_array_equal(image, np.full((2, 2), 110.0))
    assert count == 2
    assert progress[-1] == (2, 2)


def test_xrf_frame_cut_filters_arbitrary_saved_grid(tmp_path):
    grid_path = tmp_path / "grid_mapping_7x7.h5"
    grid = {
        "bin_size": 7,
        "xrd_files": ["/raw/scan_0024_00001.h5", "/raw/scan_0024_00002.h5"],
        "frame_map": [[0, 0], [0, 1], [1, 0]],
        "bins": {"0_0": [0, 1], "0_1": [2]},
        "n_bins": 2,
    }
    io.save_grid_mapping(grid_path, grid)
    np.savez_compressed(
        tmp_path / "xrf_frame_cut.npz",
        source_file=np.asarray(["scan_0024_00001.h5", "scan_0024_00002.h5"]),
        source_frame_index=np.asarray([1, 0]),
        material=np.asarray("Br"),
    )

    cut = io.load_grid_mapping(grid_path)

    assert cut["bins"] == {"0_0": [1], "0_1": [2]}
    assert cut["xrf_material_cut"] == "Br"


def test_xrf_cli_processes_registered_raw_me7_without_grid(tmp_path):
    raw_root = tmp_path / "raw"
    project = raw_root / "project"
    scan_dir = raw_root / "Raw" / "Scan_0024"
    me7 = scan_dir / "ME7"
    xrd = scan_dir / "XRD"
    me7.mkdir(parents=True)
    xrd.mkdir()
    positions = raw_root / "processed" / "SOCKETSERVER" / "Scan_0024_position.h5"
    positions.parent.mkdir(parents=True)
    with h5py.File(positions, "w") as handle:
        group = handle.create_group("entry/data/Position")
        group.create_dataset("X_Position", data=np.arange(5, dtype=float))
        group.create_dataset("Y_Position", data=np.arange(10, 15, dtype=float))
    with h5py.File(raw_root / "Scan_0024.h5", "w") as handle:
        handle.create_dataset(
            "entry/instrument/bluesky/streams/baseline/sample_theta/value", data=[2.0]
        )
    offset = raw_root / "position_offset.json"
    offset.write_text(json.dumps({"theta": [0, 2], "y_offset": [0, -25]}))
    for number, (me7_count, xrd_count) in enumerate(((2, 3), (2, 2)), start=1):
        spectra = np.zeros((me7_count, 7, 4096), dtype=np.uint32)
        spectra[:, 0, 1192] = np.arange(1, me7_count + 1)
        filename = f"scan_0024_{number:05d}.h5"
        with h5py.File(me7 / filename, "w") as handle:
            handle.create_dataset("entry/data/data", data=spectra)
        with h5py.File(xrd / filename, "w") as handle:
            handle.create_dataset(
                "entry/data/data", data=np.zeros((xrd_count, 2, 2), dtype=np.uint16)
            )
    runner = CliRunner()
    assert runner.invoke(main, [
        "init", "--name", "test", "--scan-number", "24", "--root", str(project),
    ]).exit_code == 0
    addon = XRFProject.load(project)
    addon.set_position_offset(offset)
    assert runner.invoke(main, [
        "load-data", "--root", str(project), "--source", str(me7), "--scan", "24",
    ]).exit_code == 0

    result = runner.invoke(main, [
        "process-raw", "--root", str(project), "--scan", "24",
    ])

    assert result.exit_code == 0, result.output
    assert "PROGRESS 2/2 files" in result.output
    selection_path = project / "XRF" / "Processed" / "Scan_0024_xrf_selection.h5"
    selection = xrf_selection.load(selection_path)
    assert selection["frames"]["global_frame_index"].size == 0
    assert selection["attrs"]["linked_dataset"] is False
    assert selection["spectrum"]["summed_counts"][1192] == 6

    definitions = project / "definitions.json"
    definitions.write_text(json.dumps({
        "Br": {"display_name": "Br", "pixel_range": [1192, 1193],
               "energy_range_kev": [11.0, 12.0], "minimum_counts": 2}
    }))
    linked = runner.invoke(main, [
        "link-dataset", "--root", str(project), "--scan", "24",
        "--definitions", str(definitions),
    ])

    assert linked.exit_code == 0, linked.output
    selection = xrf_selection.load(selection_path)
    np.testing.assert_array_equal(selection["frames"]["global_frame_index"], [0, 1, 3, 4])
    np.testing.assert_array_equal(selection["frames"]["x"], [0, 1, 3, 4])
    np.testing.assert_array_equal(selection["frames"]["y"], [-15, -14, -12, -11])
    assert selection["attrs"]["n_total_frames"] == 5
    assert selection["attrs"]["linked_dataset"] is True
    assert selection["materials"]["Br"]["attrs"]["minimum_counts"] == 2
    np.testing.assert_array_equal(selection["materials"]["Br"]["keep"], [False, True, False, True])
    assert selection["attrs"]["channels"] == list(range(6))
    assert selection["attrs"]["deadtime_correction"] is False
