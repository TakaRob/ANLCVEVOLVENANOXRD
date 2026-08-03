import json

import numpy as np
import pytest
from click.testing import CliRunner

from xrd_app.cli import _same_grid_lattice, bin, grid, main, make_bins, peaks, roi_shapes


@pytest.mark.parametrize("command", sorted(main.commands))
def test_every_command_help_succeeds(command):
    result = CliRunner().invoke(main, [command, "--help"])

    assert result.exit_code == 0, result.output


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
    roi_params = {param.name: param for param in roi_shapes.params}
    assert roi_params["normalize_frames"].default is False
    assert "--normalize-frames" in runner.invoke(main, ["roi-shapes", "--help"]).output
    for command in (bin, make_bins):
        params = {param.name: param for param in command.params}
        assert params["normalize_frames"].default is False
        assert "--normalize-frames" in runner.invoke(
            main, [command.name, "--help"]).output


def test_init_requires_name_without_prompting(tmp_path):
    result = CliRunner().invoke(main, ["init", "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "Missing option '--name'" in result.stderr


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
    assert "Result JSON not found" in result.stderr


def _project_config(tmp_path):
    (tmp_path / "Labels" / "Scan_0203").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(
        "name: test\nscan:\n  number: 203\n  name: Scan_0203\n"
        "paths:\n  raw_dir: Raw\n  binned_dir: Binned\n"
        "  metadata_dir: Metadata\n  labels_dir: Labels\n"
    )


def test_shapes_rejects_wrong_bin_from_peaks(tmp_path):
    _project_config(tmp_path)
    peaks_path = tmp_path / "foreign_peaks.json"
    peaks_path.write_text(json.dumps({
        "lineage": {"stage": "peaks", "scan": "Scan_0203", "bin_size": 3},
        "peaks_by_bin": {},
    }))

    result = CliRunner().invoke(main, [
        "shapes", "--root", str(tmp_path), "--scan", "203", "--bin-size", "5",
        "--from-peaks", str(peaks_path),
    ])

    assert result.exit_code == 1
    assert "bin 3x3 != 5x5" in result.stderr


def test_fast_roi_shapes_missing_h5_uses_fallback_without_building(monkeypatch, tmp_path):
    _project_config(tmp_path)
    metadata = tmp_path / "Metadata" / "Scan_0203"
    metadata.mkdir(parents=True)
    (metadata / "grid_mapping_3x3.json").write_text(json.dumps({
        "bin_size": 3, "n_bin_rows": 1, "n_bin_cols": 1,
        "bins": {"0_0": [0]},
    }))
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
        "--roi", "0,0,2,2", "--name", "preview", "--fast",
        "--preview-output", str(tmp_path / "preview.json"),
    ])

    assert result.exit_code == 0, result.output
    assert built == []


def test_roi_shapes_passes_frame_normalization(monkeypatch, tmp_path):
    _project_config(tmp_path)
    metadata = tmp_path / "Metadata" / "Scan_0203"
    metadata.mkdir(parents=True)
    (metadata / "grid_mapping_3x3.json").write_text(json.dumps({
        "bin_size": 3, "n_bin_rows": 1, "n_bin_cols": 1,
        "bins": {"0_0": [0, 1]},
    }))
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
