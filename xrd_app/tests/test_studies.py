"""Headless tests for study discovery + registry + result loaders (core/studies.py)."""
from __future__ import annotations

import csv
import json

import numpy as np

from xrd_app.core import studies


def _write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_study(root, name="Study", with_rsm=True):
    d = root / name
    d.mkdir(parents=True)
    _write_csv(d / "features.csv", ["scan", "reflection"], [{"scan": "Scan_0203", "reflection": "(001)"}])
    _write_csv(d / "device_map.csv", ["row", "col"], [{"row": 0, "col": 0}])
    _write_csv(d / "rocking_curves.csv",
               ["track_id", "reflection", "status", "theta_bragg", "fwhm"],
               [{"track_id": 1, "reflection": "(001)", "status": "fit",
                 "theta_bragg": 5.0, "fwhm": 0.3}])
    (d / "tracks.json").write_text(json.dumps({
        "bin_size": 3, "n_tracks": 1,
        "tracks": [{"track_id": 1, "reflection": "(001)", "is_recurrent": True,
                    "members": [{"theta": 4.0, "intensity": 10.0},
                                {"theta": 5.0, "intensity": 20.0}]}]}))
    (d / "prediction_report.md").write_text("# report\nrecall 0.7\n")
    np.savez_compressed(d / "combined_device.npz",
                        n_rows=2, n_cols=2, reflections=np.array(["(001)"]),
                        max_intensity=np.ones((2, 2)))
    if with_rsm:
        np.savez_compressed(
            d / "rsm.npz", volume=np.zeros((4, 4, 4)),
            qx_edges=np.linspace(0, 1, 5), qy_edges=np.linspace(0, 1, 5),
            qz_edges=np.linspace(0, 1, 5))
        (d / "rsm.summary.json").write_text(json.dumps(
            {"scans": ["Scan_0203", "Scan_0204"], "thetas": [20.5, 20.0],
             "grid_shape": [4, 4, 4]}))
    return d


def test_discovers_study_dirs(tmp_path):
    _make_study(tmp_path, "Study")
    _make_study(tmp_path, "Study_1x1", with_rsm=False)
    (tmp_path / "Binned").mkdir()  # a data dir that must NOT be listed
    found = studies.list_studies(tmp_path)
    names = sorted(e["path"] for e in found)
    assert names == ["Study", "Study_1x1"]


def test_artifacts_and_meta(tmp_path):
    _make_study(tmp_path, "Study")
    e = next(x for x in studies.list_studies(tmp_path) if x["path"] == "Study")
    a = e["artifacts"]
    assert a["rsm"] and a["rocking"] and a["combined_device"] and a["tracks"]
    assert e["bin_size"] == 3            # from tracks.json
    assert e["scans"] == ["Scan_0203", "Scan_0204"]  # from rsm.summary.json
    assert "rsm" in studies.describe(e)


def test_registry_overlay_roundtrip(tmp_path):
    _make_study(tmp_path, "Study")
    studies.register_study(tmp_path, tmp_path / "Study", name="My run",
                           notes="hello", created="2026-07-02T00:00:00")
    # persisted to studies.json
    reg = json.loads((tmp_path / "studies.json").read_text())
    assert reg["studies"][0]["name"] == "My run"
    # overlay shows up in discovery
    e = next(x for x in studies.list_studies(tmp_path) if x["path"] == "Study")
    assert e["name"] == "My run" and e["notes"] == "hello"


def test_registry_update_does_not_duplicate(tmp_path):
    _make_study(tmp_path, "Study")
    studies.register_study(tmp_path, tmp_path / "Study", name="A")
    studies.register_study(tmp_path, tmp_path / "Study", notes="added later")
    reg = studies.load_registry(tmp_path)
    assert len(reg["studies"]) == 1
    assert reg["studies"][0]["name"] == "A"           # preserved
    assert reg["studies"][0]["notes"] == "added later"  # merged


def test_result_loaders(tmp_path):
    _make_study(tmp_path, "Study")
    d = tmp_path / "Study"
    rc = studies.load_rocking_curves(d)
    assert rc and rc[0]["track_id"] == 1.0 and rc[0]["status"] == "fit"
    tr = studies.load_tracks(d)
    assert tr and tr[0]["is_recurrent"] is True
    cd = studies.load_combined_device(d)
    assert cd is not None and cd["max_intensity"].shape == (2, 2)
    assert "recall" in studies.load_prediction_report(d)


def test_missing_artifacts_return_empty(tmp_path):
    empty = tmp_path / "Nope"
    empty.mkdir()
    assert not studies.is_study_dir(empty)
    assert studies.load_rocking_curves(empty) == []
    assert studies.load_tracks(empty) == []
    assert studies.load_combined_device(empty) is None
    assert studies.load_prediction_report(empty) is None
