"""Reflection loading and DataManager resolution precedence."""

import json

from xrd_app.config import DataManager
from xrd_app.core import io


def _write_json(path, angle, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {"name": name, "two_theta": angle, "width": 0.4},
    ]))
    return path


def test_load_reflections_accepts_json(tmp_path):
    json_path = _write_json(tmp_path / "reflections.json", 7.25, "json")

    assert io.load_reflections(json_path) == ([7.25], ["json"])


def test_reflections_resolve_scan_then_project_json(tmp_path):
    dm = DataManager(tmp_path, scan=7)
    project = _write_json(tmp_path / "Metadata" / "reflections.json", 7.0, "project")
    scan = _write_json(
        tmp_path / "Metadata" / "Scan_0007" / "reflections.json", 8.0, "scan")

    assert dm.reflections() == scan
    scan.unlink()
    assert dm.reflections() == project


def test_reflections_keep_configured_external_json(tmp_path):
    external = _write_json(tmp_path / "external.json", 10.0, "external")
    dm = DataManager(tmp_path, scan=7)
    dm.config.data = {"data_sources": {"reflections": str(external)}}

    assert dm.reflections() == external


def test_reflections_fall_back_to_bundled_json(tmp_path):
    path = DataManager(tmp_path, scan=7).reflections()

    assert path.name == "reflections.json"
    degs, labels = io.load_reflections(path)
    assert degs and len(degs) == len(labels)
