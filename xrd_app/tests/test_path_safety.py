"""Containment checks for user- and config-derived project paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from xrd_app import cli, workspace
from xrd_app.config import DataManager, ProjectConfig


@pytest.mark.parametrize(
    "scan",
    ["../Scan_0007", "/tmp/Scan_0007", r"..\Scan_0007", ".", ".."],
)
def test_scan_names_reject_paths(scan):
    with pytest.raises(ValueError, match="Invalid scan name"):
        DataManager.scan_name_of(scan)


@pytest.mark.parametrize(
    ("scan", "expected"),
    [(7, "Scan_0007"), ("7", "Scan_0007"), ("Scan_7", "Scan_0007"),
     ("Scan_0007", "Scan_0007")],
)
def test_scan_names_retain_canonical_support(scan, expected):
    assert DataManager.scan_name_of(scan) == expected


@pytest.mark.parametrize("name", ["../project", "/tmp/project", r"..\project", ".", ".."])
def test_workspace_project_names_reject_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid project name"):
        workspace.create_project(name, workspace=tmp_path)


def test_workspace_project_creation_stays_contained_and_refuses_existing(tmp_path):
    root = workspace.create_project("safe project", workspace=tmp_path)
    original = (root / "config.yaml").read_text()

    assert root == tmp_path.resolve() / "safe project"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        workspace.create_project("safe project", workspace=tmp_path)
    assert (root / "config.yaml").read_text() == original


@pytest.mark.parametrize(
    ("name", "tag"),
    [("grain (001)!", "grain_001"), ("../escape", "escape"),
     (r"..\windows\escape", "windows_escape"), ("/tmp/escape", "tmp_escape")],
)
def test_roi_catalog_names_normalize_inside_scan_labels(tmp_path, name, tag):
    dm = DataManager(tmp_path, scan=7)

    output = dm.roi_map_json(name, 3)

    assert output == tmp_path / "Labels" / "Scan_0007" / f"{tag}_roimap_3x3.json"


@pytest.mark.parametrize("constructor", ["peaks_json", "shapes_json", "hd_map_json"])
def test_algorithm_output_names_reject_path_components(tmp_path, constructor):
    dm = DataManager(tmp_path, scan=7)

    with pytest.raises(ValueError, match="Invalid algorithm name"):
        getattr(dm, constructor)("../escape", 3)


def test_variant_output_names_reject_path_components(tmp_path):
    dm = DataManager(tmp_path, scan=7)

    with pytest.raises(ValueError, match="Invalid variant"):
        dm.grid_mapping(bin_size=3, variant=r"..\escape")


def test_roi_save_uses_normalized_name_for_output_and_persistence(tmp_path, monkeypatch):
    def fake_roi_shapes(**kwargs):
        Path(kwargs["preview_output"]).write_text(json.dumps({"features": []}))

    monkeypatch.setattr(cli.roi_shapes, "callback", fake_roi_shapes)
    result = CliRunner().invoke(
        cli.main,
        ["roi-save", "--roi", "1,2,3,4", "--name", "../grain (001)!",
         "--scan", "7", "--root", str(tmp_path)],
    )
    output = tmp_path / "Labels" / "Scan_0007" / "grain_001_roimap_3x3.json"

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert json.loads(output.read_text())["name"] == "grain_001"
    assert not (tmp_path / "grain (001)!_roimap_3x3.json").exists()


@pytest.mark.parametrize("configured", ["../outside", "/tmp/outside", r"..\outside"])
def test_configured_project_directories_reject_escape(tmp_path, configured):
    data = {"paths": {"labels_dir": configured}}
    dm = DataManager(config=ProjectConfig(tmp_path, data=data))

    with pytest.raises(ValueError, match=r"paths\.labels_dir"):
        _ = dm.labels_dir_root


def test_configured_project_directory_component_stays_under_root(tmp_path):
    data = yaml.safe_load("paths:\n  labels_dir: CustomLabels\n")
    dm = DataManager(config=ProjectConfig(tmp_path, data=data))

    assert dm.labels_dir_root == tmp_path.resolve() / "CustomLabels"
