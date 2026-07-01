"""Headless tests for cross-scan aggregation (core/aggregate.py + `xrd-app aggregate`).

Builds a synthetic Labels/ tree (no bins, no detector needed) so the aggregate
read path, column mapping, and CSV/SQLite emit are all exercised offline.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from xrd_app.core import aggregate as agg


def _feature(fid, reflection, tth, chi, prof):
    """A shapes-catalog 'kept' feature with a per-bin intensity_profile."""
    return {
        "feature_id": fid,
        "reflection": reflection,
        "ref_tth": tth,
        "center_bin": "5_7",
        "center_row": 5, "center_col": 7,
        "detector_x": 123.0, "detector_y": 456.0,
        "chi_deg": chi,
        "peak_intensity": max(e["intensity"] for e in prof.values()),
        "mean_snr": 8.5,
        "n_bins": len(prof),
        "rocking_fwhm": 1.2,      # legacy alias for chi_fwhm
        "strain_breadth": 0.05,   # legacy alias for tth_fwhm
        "spatial_extent": ["5_7", "5_8"],
        "reason": "kept",
        "intensity_profile": prof,
    }


def _write_shapes(labels_root: Path, scan: str, bin_size: int, kept):
    d = labels_root / scan
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"gaussian_shapes_{bin_size}x{bin_size}.json"
    p.write_text(json.dumps({"scan": scan, "kept": kept, "filtered": []}))
    return p


def _make_project(tmp_path: Path):
    labels = tmp_path / "Labels"
    prof_a = {
        "5_7": {"intensity": 1000.0, "integrated": 5000.0, "det_x": 123.0,
                "det_y": 456.0, "tth": 7.52, "chi": 30.0},
        "5_8": {"intensity": 800.0, "integrated": 4000.0, "det_x": 124.0,
                "det_y": 457.0, "tth": 7.50, "chi": 31.0},
    }
    prof_b = {
        "9_2": {"intensity": 600.0, "integrated": 3000.0, "det_x": 200.0,
                "det_y": 300.0, "tth": 10.62, "chi": -12.0},
    }
    _write_shapes(labels, "Scan_0203", 3, [_feature(0, "(001)", 7.514, 30.0, prof_a)])
    _write_shapes(labels, "Scan_0214", 3, [
        _feature(0, "(001)", 7.514, 30.5, prof_a),
        _feature(1, "(011)", 10.617, -12.0, prof_b),
    ])
    return labels


def test_aggregate_collects_features_and_devicemap(tmp_path):
    labels = _make_project(tmp_path)
    features, device_map = agg.aggregate(labels, log=lambda *_: None)

    # 1 feature in 203 + 2 in 214 = 3 features.
    assert len(features) == 3
    scans = sorted(r["scan"] for r in features)
    assert scans == ["Scan_0203", "Scan_0214", "Scan_0214"]

    # device_map rows = total bins across all profiles: 2 + 2 + 1 = 5.
    assert len(device_map) == 5

    # Column mapping: legacy rocking_fwhm/strain_breadth land in chi_fwhm/tth_fwhm.
    f0 = features[0]
    assert f0["chi_fwhm"] == 1.2
    assert f0["tth_fwhm"] == 0.05
    assert f0["peak_intensity"] == 1000.0
    assert f0["mean_intensity"] == 900.0          # (1000+800)/2
    assert f0["n_bins"] == 2
    # intensity-weighted measured 2θ COM: (7.52·1000 + 7.50·800)/1800
    assert abs(f0["tth_com"] - 7.51111) < 1e-4

    # device_map rows carry parsed row/col from the bin_key.
    dm0 = next(r for r in device_map if r["bin_key"] == "5_7")
    assert (dm0["row"], dm0["col"]) == (5, 7)
    assert dm0["reflection"] == "(001)"


def test_aggregate_bin_size_filter(tmp_path):
    labels = _make_project(tmp_path)
    # All catalogs are 3x3; filtering to 5x5 yields nothing.
    features, _ = agg.aggregate(labels, bin_size=5, log=lambda *_: None)
    assert features == []
    features3, _ = agg.aggregate(labels, bin_size=3, log=lambda *_: None)
    assert len(features3) == 3


def test_aggregate_scan_filter(tmp_path):
    labels = _make_project(tmp_path)
    features, _ = agg.aggregate(labels, scans=["Scan_0203"], log=lambda *_: None)
    assert {r["scan"] for r in features} == {"Scan_0203"}


def test_write_csv_and_sqlite_roundtrip(tmp_path):
    labels = _make_project(tmp_path)
    features, device_map = agg.aggregate(labels, log=lambda *_: None)

    fcsv = agg.write_csv(features, agg.FEATURE_COLUMNS, tmp_path / "Study" / "features.csv")
    dcsv = agg.write_csv(device_map, agg.DEVICEMAP_COLUMNS, tmp_path / "Study" / "device_map.csv")
    assert fcsv.exists() and dcsv.exists()
    # header + 3 feature rows
    assert len(fcsv.read_text().strip().splitlines()) == 1 + 3

    db = agg.write_sqlite(tmp_path / "Study" / "study.db", features, device_map)
    con = sqlite3.connect(str(db))
    try:
        assert con.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 3
        assert con.execute("SELECT COUNT(*) FROM device_map").fetchone()[0] == 5
        # SQLite columns match the declared order.
        cols = [r[1] for r in con.execute("PRAGMA table_info(features)")]
        assert cols == agg.FEATURE_COLUMNS
    finally:
        con.close()
