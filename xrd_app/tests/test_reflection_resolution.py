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


def test_load_reflections_accepts_json_and_legacy_python(tmp_path):
    json_path = _write_json(tmp_path / "reflections.json", 7.25, "json")
    py_path = tmp_path / "reflections.py"
    py_path.write_text("degs = [8.5]\ndeg_labels = ['python']\n")

    assert io.load_reflections(json_path) == ([7.25], ["json"])
    assert io.load_reflections(py_path) == ([8.5], ["python"])


def test_reflections_resolve_scan_then_project_json(tmp_path):
    dm = DataManager(tmp_path, scan=7)
    project = _write_json(tmp_path / "Metadata" / "reflections.json", 7.0, "project")
    scan = _write_json(
        tmp_path / "Metadata" / "Scan_0007" / "reflections.json", 8.0, "scan")

    assert dm.reflections() == scan
    scan.unlink()
    assert dm.reflections() == project


def test_reflections_keep_legacy_project_and_external_sources(tmp_path):
    metadata = tmp_path / "Metadata"
    metadata.mkdir()
    legacy = metadata / "reflections.py"
    legacy.write_text("degs = [9.0]\ndeg_labels = ['legacy']\n")
    dm = DataManager(tmp_path, scan=7)

    assert dm.reflections() == legacy

    external = tmp_path / "external.py"
    external.write_text("degs = [10.0]\ndeg_labels = ['external']\n")
    dm.config.data = {"data_sources": {"reflections": str(external)}}
    assert dm.reflections() == external


def test_reflections_fall_back_to_bundled_json(tmp_path):
    path = DataManager(tmp_path, scan=7).reflections()

    assert path.name == "reflections.json"
    degs, labels = io.load_reflections(path)
    assert degs and len(degs) == len(labels)
