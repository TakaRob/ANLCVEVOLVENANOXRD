"""
Aggregate per-scan feature catalogs into a single comparable dataset.

Walks ``results/<scan>/feature_catalog_NxN.json`` across all scans and bin sizes
and produces two flat tables you can open in Excel/pandas or query with SQL:

  features    one row per detected feature — intensity, prevalence (n_bins),
              shape (rocking_fwhm / strain_breadth), orientation (chi_deg).
  device_map  one row per (scan, reflection, spatial bin) — the per-bin
              intensity map data, in long/tidy form.

Outputs a CSV per table plus a combined SQLite ``.db`` with both as tables.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Callable, Optional, Union

# Column order for the per-feature summary table.
FEATURE_COLUMNS = [
    "scan", "bin_size", "feature_id", "reflection", "ref_tth",
    "center_bin", "center_row", "center_col", "detector_x", "detector_y",
    "chi_deg",
    "peak_intensity", "mean_intensity", "sum_integrated", "mean_snr",  # intensity
    "n_bins",                                                          # prevalence
    "rocking_fwhm", "strain_breadth",                                  # shape
    "spatial_extent", "reason",
]

# Column order for the long device-map table (one row per spatial bin).
DEVICEMAP_COLUMNS = [
    "scan", "bin_size", "feature_id", "reflection", "bin_key", "row", "col",
    "det_x", "det_y", "intensity", "integrated", "tth", "chi",
]

_CATALOG_RE = re.compile(r"feature_catalog_(\d+)x(\d+)\.json$")


def iter_catalogs(results_dir: Path, scans=None, bin_size: Optional[int] = None):
    """Yield (scan_name, bin_size, catalog_path) for every catalog found."""
    if not results_dir.exists():
        return
    wanted = set(scans) if scans else None
    for scan_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if wanted and scan_dir.name not in wanted:
            continue
        for cat in sorted(scan_dir.glob("feature_catalog_*x*.json")):
            m = _CATALOG_RE.search(cat.name)
            bs = int(m.group(1)) if m else None
            if bin_size and bs != bin_size:
                continue
            yield scan_dir.name, bs, cat


def _feature_row(scan: str, bin_size: Optional[int], f: dict) -> dict:
    profile = f.get("intensity_profile", {}) or {}
    intensities = [e["intensity"] for e in profile.values()
                   if isinstance(e, dict) and "intensity" in e]
    integrated = [e["integrated"] for e in profile.values()
                  if isinstance(e, dict) and "integrated" in e]
    return {
        "scan": scan,
        "bin_size": bin_size,
        "feature_id": f.get("feature_id"),
        "reflection": f.get("reflection"),
        "ref_tth": f.get("ref_tth"),
        "center_bin": f.get("center_bin"),
        "center_row": f.get("center_row"),
        "center_col": f.get("center_col"),
        "detector_x": f.get("detector_x"),
        "detector_y": f.get("detector_y"),
        "chi_deg": f.get("chi_deg"),
        "peak_intensity": f.get("peak_intensity"),
        "mean_intensity": round(sum(intensities) / len(intensities), 2) if intensities else None,
        "sum_integrated": round(sum(integrated), 1) if integrated else None,
        "mean_snr": round(f["mean_snr"], 2) if "mean_snr" in f else None,
        "n_bins": f.get("n_bins"),
        "rocking_fwhm": f.get("rocking_fwhm"),
        "strain_breadth": f.get("strain_breadth"),
        "spatial_extent": " ".join(f.get("spatial_extent", [])),
        "reason": f.get("reason"),
    }


def _devicemap_rows(scan: str, bin_size: Optional[int], f: dict):
    rows = []
    for bk, e in (f.get("intensity_profile", {}) or {}).items():
        if not isinstance(e, dict):
            continue
        parts = bk.split("_")
        row = int(parts[0]) if len(parts) == 2 and parts[0].isdigit() else None
        col = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else None
        rows.append({
            "scan": scan, "bin_size": bin_size, "feature_id": f.get("feature_id"),
            "reflection": f.get("reflection"), "bin_key": bk, "row": row, "col": col,
            "det_x": e.get("det_x"), "det_y": e.get("det_y"),
            "intensity": e.get("intensity"), "integrated": e.get("integrated"),
            "tth": e.get("tth"), "chi": e.get("chi"),
        })
    return rows


def aggregate(results_dir: Union[str, Path], scans=None, bin_size: Optional[int] = None,
              log: Callable[[str], None] = print):
    """Collect feature + device-map rows from all matching catalogs."""
    results_dir = Path(results_dir)
    features, device_map = [], []
    n_catalogs = 0
    for scan, bs, cat in iter_catalogs(results_dir, scans, bin_size):
        with open(cat) as fh:
            catalog = json.load(fh)
        n_catalogs += 1
        for f in catalog:
            features.append(_feature_row(scan, bs, f))
            device_map.extend(_devicemap_rows(scan, bs, f))
        log(f"  {scan} [{bs}x{bs}]: {len(catalog)} features")
    log(f"Aggregated {len(features)} features from {n_catalogs} catalog(s)")
    return features, device_map


def write_csv(rows, columns, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def write_sqlite(db_path: Union[str, Path], features, device_map) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    try:
        _create_table(con, "features", FEATURE_COLUMNS, features)
        _create_table(con, "device_map", DEVICEMAP_COLUMNS, device_map)
        con.commit()
    finally:
        con.close()
    return db_path


def _create_table(con, name, columns, rows):
    cols_sql = ", ".join(f'"{c}"' for c in columns)
    con.execute(f'CREATE TABLE "{name}" ({cols_sql})')
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f'INSERT INTO "{name}" VALUES ({placeholders})',
        [tuple(r.get(c) for c in columns) for r in rows],
    )
