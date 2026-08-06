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


def _write_legacy(tmp_path):
    linker = tmp_path / "Scan_0024_xrf_xrd_links.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(linker, "w") as handle:
        handle.attrs["scan"] = 24
        group = handle.create_group("links")
        group.create_dataset("Material", data=np.asarray(["Br", "Br", "Pb"], dtype=object),
                             dtype=string_dtype)
        group.create_dataset("Global Frame Index", data=[1, 2, 2])
        group.create_dataset("XRF Intensity", data=[20.0, 30.0, 6.0])
        group.create_dataset("X", data=[2.0, 3.0, 3.0])
        group.create_dataset("Y", data=[5.0, 6.0, 6.0])
        group.create_dataset("XRD File Link", data=np.asarray(["a.h5", "b.h5", "b.h5"], dtype=object),
                             dtype=string_dtype)
        group.create_dataset("XRD Frame Index", data=[1, 0, 0])
    masks = tmp_path / "Scan_0024_xrf_threshold_masks.npz"
    np.savez_compressed(
        masks,
        names=np.asarray(["Br", "Pb"]),
        keep_masks=np.asarray([[False, True, True], [False, False, True]]),
        global_frame_indices=np.asarray([0, 1, 2]),
        minimum_counts=np.asarray([20.0, 6.0]),
    )
    roi = tmp_path / "Scan_0024_xrf_rois.json"
    roi.write_text(json.dumps({
        "channels": [0, 1],
        "deadtime_correction": False,
        "rois": {"Br": {"energy_range_kev": [11.7, 12.1]}},
    }))
    return linker, masks, roi


def test_import_legacy_linker_preserves_masks_and_overlap(tmp_path):
    linker, masks, roi = _write_legacy(tmp_path)
    intensities = tmp_path / "Scan_0024_xrf_roi_intensities.npz"
    np.savez_compressed(
        intensities,
        materials=np.asarray(["Br", "Pb"]),
        intensities=np.asarray([[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]]),
    )

    selection = xrf_selection.import_legacy_linker(
        linker, mask_path=masks, roi_config_path=roi,
        intensity_path=intensities, scan=24,
    )

    np.testing.assert_array_equal(selection["materials"]["Br"]["keep"], [False, True, True])
    np.testing.assert_array_equal(selection["materials"]["Br"]["intensity"], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(selection["materials"]["Pb"]["keep"], [False, False, True])
    assert selection["materials"]["Br"]["intensity"][2] == 30.0
    assert selection["materials"]["Pb"]["intensity"][2] == 6.0
    assert selection["attrs"]["unresolved_frame_identities"] == 1


def test_xrf_cli_init_import_and_status(tmp_path):
    runner = CliRunner()
    project = tmp_path / "xrf project"
    linker, masks, roi = _write_legacy(tmp_path)

    initialized = runner.invoke(main, ["init", "--name", "test", "--root", str(project)])
    imported = runner.invoke(main, [
        "import-legacy", "--root", str(project), "--scan", "24",
        "--linker", str(linker), "--masks", str(masks), "--roi-config", str(roi),
    ])
    status = runner.invoke(main, ["status", "--root", str(project), "--json-output"])

    assert initialized.exit_code == 0, initialized.output
    assert (project / "config.yaml").exists()
    assert (project / "XRF" / "xrf_config.yaml").exists()
    assert (project / "Metadata" / "reflections.json").exists()
    assert imported.exit_code == 0, imported.output
    assert "Br: 2 retained" in imported.output
    assert status.exit_code == 0, status.output
    report = json.loads(status.output)
    assert report["scans"]["Scan_0024"]["valid"] is True
    assert report["scans"]["Scan_0024"]["materials"]["Pb"]["retained_frames"] == 1


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


def test_xrf_cli_processes_registered_raw_me7(tmp_path):
    project = tmp_path / "project"
    me7 = tmp_path / "Scan_0024" / "ME7"
    me7.mkdir(parents=True)
    spectra = np.zeros((2, 7, 4096), dtype=np.uint32)
    spectra[:, 0, 1192] = [10, 20]
    with h5py.File(me7 / "scan_0024_00001.h5", "w") as handle:
        handle.create_dataset("entry/data/data", data=spectra)
    runner = CliRunner()
    assert runner.invoke(main, [
        "init", "--name", "test", "--scan-number", "24", "--root", str(project),
    ]).exit_code == 0
    grid = {
        "scan": "Scan_0024", "bin_size": 1, "positions_real": True,
        "n_rows": 1, "n_cols": 2, "n_bin_rows": 1, "n_bin_cols": 2,
        "n_total_frames": 2, "n_bins": 2, "coordinate_source": "positions_xy",
        "xrd_files": [str(tmp_path / "scan_0024_00001.h5")],
        "frame_map": [[0, 0], [0, 1]], "bins": {"0_0": [0], "0_1": [1]},
    }
    grid_path = project / "Metadata" / "Scan_0024" / "grid_mapping_1x1.h5"
    io.save_grid_mapping(grid_path, grid)
    assert runner.invoke(main, [
        "load-data", "--root", str(project), "--source", str(me7), "--scan", "24",
    ]).exit_code == 0

    result = runner.invoke(main, [
        "process-raw", "--root", str(project), "--scan", "24",
    ])

    assert result.exit_code == 0, result.output
    selection = xrf_selection.load(
        project / "XRF" / "Processed" / "Scan_0024_xrf_selection.h5"
    )
    assert selection["spectrum"]["summed_counts"][1192] == 30
    assert selection["frames"]["global_frame_index"].size == 2
    assert (project / "XRF" / "Cache" / "Scan_0024" / "Scan_0024_xrf_points.npz").exists()
