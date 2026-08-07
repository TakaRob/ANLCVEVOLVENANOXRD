import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from xrd_app.cli import (_same_grid_lattice, bin, grid, main, make_bins, peaks,
                         roi_shapes, run_pipeline)
from xrd_app.config import ProjectConfig
from xrd_app.core import catalogs, io


@pytest.mark.parametrize("command", sorted(main.commands))
def test_every_command_help_succeeds(command):
    result = CliRunner().invoke(main, [command, "--help"])

    assert result.exit_code == 0, result.output


def test_run_cvevolve_uses_default_prompt_and_non_tty_container(monkeypatch, tmp_path):
    config_dir = tmp_path / "external session"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    prompt = config_dir / "prompt.md"
    holdout_prompt = config_dir / "holdout_test_prompt.md"
    data_dir = tmp_path / "external data"
    config.write_text(
        f"workspace:\n  root_dir: {config_dir / 'sessions'}\n"
        f"  data_dir: {data_dir}\n  holdout_data_dir: {data_dir}\n"
    )
    prompt.write_text("task")
    holdout_prompt.write_text("holdout task")
    calls = []
    monkeypatch.setattr("xrd_app.cli.shutil.which", lambda engine: f"/usr/bin/{engine}")
    monkeypatch.setattr("subprocess.call", lambda command: calls.append(command) or 0)

    result = CliRunner().invoke(main, [
        "run-cvevolve", "--config", str(config), "--engine", "podman",
        "--root", str(tmp_path / "project"),
    ])

    assert result.exit_code == 0, result.output
    command = calls[0]
    assert "-i" in command
    assert "-it" not in command
    assert ["--prompt", str(prompt.resolve())] == command[-4:-2]
    assert command[-2:] == ["--holdout-test-prompt", str(holdout_prompt.resolve())]
    assert f"{config_dir.resolve()}:{config_dir.resolve()}" in command
    assert f"{data_dir.resolve()}:{data_dir.resolve()}" in command


def test_help_lists_workflow_and_defaults():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])
    grid_help = runner.invoke(main, ["grid", "--help"])

    assert result.exit_code == 0
    assert "make-bins" in result.output
    assert "run-pipeline" in result.output
    assert grid_help.exit_code == 0
    assert "[default: 3]" in grid_help.output
    defaults = {param.name: param.default for param in grid.params}
    assert defaults["deskew_method"] == "auto"
    peak_params = {param.name: param for param in peaks.params}
    assert peak_params["workers"].default is None
    assert "--workers" in runner.invoke(main, ["peaks", "--help"]).output
    pipeline_params = {param.name: param for param in run_pipeline.params}
    assert pipeline_params["workers"].default is None
    assert "--workers" in runner.invoke(main, ["run-pipeline", "--help"]).output
    roi_params = {param.name: param for param in roi_shapes.params}
    assert roi_params["normalize_frames"].default is False
    roi_help = runner.invoke(main, ["roi-shapes", "--help"]).output
    assert "--normalize-frames" in roi_help
    assert "--sample-crop" in roi_help
    for command in (bin, make_bins):
        params = {param.name: param for param in command.params}
        assert params["normalize_frames"].default is False
        assert "--normalize-frames" in runner.invoke(
            main, [command.name, "--help"]).output


def test_init_requires_name_without_prompting(tmp_path):
    result = CliRunner().invoke(main, ["init", "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "Missing option '--name'" in result.stderr


def test_init_creates_editable_project_defaults(tmp_path):
    result = CliRunner().invoke(
        main, ["init", "--name", "test-project", "--root", str(tmp_path)]
    )
    package = Path(__file__).resolve().parent.parent

    assert result.exit_code == 0, result.output
    assert (tmp_path / "Metadata" / "reflections.json").read_bytes() == (
        package / "assets" / "reflections.json").read_bytes()
    assert (tmp_path / "Metadata" / "tth.tiff").read_bytes() == (
        package / "assets" / "tth.tiff").read_bytes()
    detector = tmp_path / "Algorithms" / "PeakAlgorithms" / "default_detector.py"
    assert detector.read_bytes() == (
        package / "PeakAlgorithms" / "5x5_tophat_band_adaptive_snr.py").read_bytes()
    catalog = json.loads((detector.parent / "catalog.json").read_text())
    assert catalog["detectors"][0]["file"] == detector.name


def test_link_replaces_calibrations_at_canonical_paths(tmp_path):
    runner = CliRunner()
    assert runner.invoke(main, [
        "init", "--name", "test-project", "--root", str(tmp_path),
    ]).exit_code == 0
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    tth = source_dir / "custom-name.tif"
    reflections = source_dir / "custom-name.json"
    tth.write_bytes(b"replacement tiff")
    reflections.write_text('[{"name": "custom", "two_theta": 8, "width": 0.4}]')

    result = runner.invoke(main, [
        "link", "--root", str(tmp_path), "--tth", str(tth),
        "--reflections", str(reflections),
    ])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "Metadata" / "tth.tiff").read_bytes() == tth.read_bytes()
    assert (tmp_path / "Metadata" / "reflections.json").read_bytes() == reflections.read_bytes()
    assert not (tmp_path / "Metadata" / "tth.tiff").is_symlink()
    assert not (tmp_path / "Metadata" / "reflections.json").is_symlink()
    config = ProjectConfig.load(tmp_path)
    assert config.get("data_sources", "tth_map") is None
    assert config.get("data_sources", "reflections") is None


def test_init_refuses_to_overwrite_existing_project(tmp_path):
    runner = CliRunner()

    first = runner.invoke(
        main, ["init", "--name", "test-project", "--root", str(tmp_path)]
    )
    original = (tmp_path / "config.yaml").read_text()
    second = runner.invoke(
        main, ["init", "--name", "replacement", "--root", str(tmp_path)]
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "refusing to overwrite" in second.stderr
    assert (tmp_path / "config.yaml").read_text() == original


def test_scan_detect_usage_errors_go_to_stderr():
    result = CliRunner().invoke(main, ["scan-detect"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Provide --scans-dir" in result.stderr


def test_scan_detect_rejects_multiple_sources():
    result = CliRunner().invoke(
        main, ["scan-detect", "--scans-dir", "scans", "--scan-file", "scan.h5"]
    )

    assert result.exit_code == 2
    assert "Use only one" in result.stderr


def test_lineage_missing_explicit_target_fails(tmp_path):
    result = CliRunner().invoke(
        main,
        ["lineage", "missing.json", "--root", str(tmp_path), "--scan", "203"],
    )

    assert result.exit_code == 1
    assert "Result catalog not found" in result.stderr


def _project_config(tmp_path):
    (tmp_path / "Labels" / "Scan_0203").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(
        "name: test\nscan:\n  number: 203\n  name: Scan_0203\n"
        "paths:\n  raw_dir: Raw\n  binned_dir: Binned\n"
        "  metadata_dir: Metadata\n  labels_dir: Labels\n"
    )


def test_shapes_rejects_wrong_bin_from_peaks(tmp_path):
    _project_config(tmp_path)
    peaks_path = tmp_path / "foreign_peaks.h5"
    catalogs.save_result(peaks_path, {
        "lineage": {"stage": "peaks", "scan": "Scan_0203", "bin_size": 3},
        "peaks_by_bin": {},
    })

    result = CliRunner().invoke(main, [
        "shapes", "--root", str(tmp_path), "--scan", "203", "--bin-size", "5",
        "--from-peaks", str(peaks_path),
    ])

    assert result.exit_code == 1
    assert "bin 3x3 != 5x5" in result.stderr


@pytest.mark.parametrize("fast_args", [[], ["--fast"]])
def test_roi_shapes_missing_h5_uses_fallback_without_building(
        monkeypatch, tmp_path, fast_args):
    _project_config(tmp_path)
    metadata = tmp_path / "Metadata" / "Scan_0203"
    metadata.mkdir(parents=True)
    io.save_grid_mapping(metadata / "grid_mapping_3x3.h5", {
        "bin_size": 3, "n_bin_rows": 1, "n_bin_cols": 1,
        "xrd_files": [], "frame_map": [], "bins": {"0_0": [0]},
    })
    (metadata / "tth.tiff").touch()

    class Source:
        def keys(self):
            return ["0_0"]

        def region(self, key, y0, y1, x0, x1):
            return np.ones((y1 - y0, x1 - x0))

        def close(self):
            pass

    built = []
    monkeypatch.setattr("xrd_app.core.io.build_bins", lambda *a, **k: built.append(True))
    monkeypatch.setattr("xrd_app.core.io.open_bin_source", lambda *a, **k: Source())
    monkeypatch.setattr("xrd_app.core.io.load_tth_map", lambda path: np.ones((4, 4)))
    monkeypatch.setattr("xrd_app.core.processing.estimate_beam_center", lambda image: (2, 2))

    result = CliRunner().invoke(main, [
        "roi-shapes", "--root", str(tmp_path), "--scan", "203", "--bin-size", "3",
        "--roi", "0,0,2,2", "--name", "preview", *fast_args,
        "--preview-output", str(tmp_path / "preview.json"),
    ])

    assert result.exit_code == 0, result.output
    assert built == []


def test_roi_shapes_passes_frame_normalization(monkeypatch, tmp_path):
    _project_config(tmp_path)
    metadata = tmp_path / "Metadata" / "Scan_0203"
    metadata.mkdir(parents=True)
    io.save_grid_mapping(metadata / "grid_mapping_3x3.h5", {
        "bin_size": 3, "n_bin_rows": 1, "n_bin_cols": 1,
        "xrd_files": [], "frame_map": [], "bins": {"0_0": [0, 1]},
    })
    (metadata / "tth.tiff").touch()

    class Source:
        def close(self):
            pass

    observed = {}

    def sample_rois(source, rois, **kwargs):
        observed.update(kwargs)
        return [{
            "roi": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
            "metric": "integrated", "normalize_frames": True,
            "center_bin": "0_0",
            "profile": {"0_0": {"intensity": 1.0, "integrated": 4.0,
                                  "mean": 1.0, "n_pixels": 4, "n_frames": 2}},
        }]

    monkeypatch.setattr("xrd_app.core.io.open_bin_source", lambda *a, **k: Source())
    monkeypatch.setattr("xrd_app.core.io.load_tth_map", lambda path: np.ones((4, 4)))
    monkeypatch.setattr("xrd_app.core.roi_map.sample_rois", sample_rois)
    monkeypatch.setattr("xrd_app.core.processing.estimate_beam_center", lambda image: (2, 2))

    output = tmp_path / "preview.json"
    result = CliRunner().invoke(main, [
        "roi-shapes", "--root", str(tmp_path), "--scan", "203", "--bin-size", "3",
        "--roi", "0,0,2,2", "--name", "preview", "--fast", "--normalize-frames",
        "--preview-output", str(output),
    ])

    assert result.exit_code == 0, result.output
    assert observed["normalize_frames"] is True
    preview = json.loads(output.read_text())
    assert preview["normalize_frames"] is True
    assert preview["intensity_definition"].startswith("mean detector counts")


def test_hd_child_grid_requires_matching_lattice_provenance():
    source = {
        "coordinate_source": "positions_faithful",
        "positions_real": True,
        "xrd_files": ["a.h5"],
        "frame_map": [[0, 0], [0, 1]],
    }
    assert _same_grid_lattice(dict(source), source)
    assert not _same_grid_lattice(
        {**source, "coordinate_source": "positions_xy"}, source)
    assert not _same_grid_lattice(
        {**source, "frame_map": [[0, 1], [0, 0]]}, source)
    assert not _same_grid_lattice({}, source)


def test_run_combined_rejects_non_1x1_before_resolving_artifacts(tmp_path):
    result = CliRunner().invoke(main, [
        "run-combined", "--root", str(tmp_path), "--scan", "203",
        "--bin-size", "3", "--algorithm", "anything",
    ])

    assert result.exit_code == 2
    assert "require --bin-size 1" in result.stderr
