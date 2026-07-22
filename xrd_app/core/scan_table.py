"""Per-scan summary table — one row per scan, comparable across a project.

The cross-scan companion to :mod:`core.aggregate` (which is one row per
*feature*). For a chosen **bin size** and **catalog type** (a lineage key
``(kind, algo, tag)`` shared across scans — e.g. the gaussian shapes at 3×3, or
a *territorial* mapping) it walks every scan's matching catalog and produces one
row of comparable metrics:

  Features     kept-shape count.
  Area sum     Σ per-shape footprint (overlaps counted).  grid bins | CSV² area.
  Area union   footprint of the set-union of all shapes (overlaps once).
  Coverage %   Area union ÷ the scan total (grid bins, or the outline hull area
               for a territorial mapping — "outlining the outer points").
  Preferred χ  tip χ of the most-populous (area-weighted) azimuthal KDE cluster
               — the same dominant cluster the Orientation Map paints — with a
               ``± range`` (half its angular span).
  Fill %       area-weighted mean *solidity* of the shapes: how solidly each
               shape fills its own convex-hull outline (holes / missing
               low-intensity detections lower it), weighted by size so large,
               well-defined shapes dominate.

Two geometry modes, chosen automatically from the catalog type:

* **grid** (shapes/combined at N×N): areas are bin counts; the scan total is the
  ``n_bin_rows·n_bin_cols`` grid; ``spatial_extent`` keys are ``"row_col"``.
* **territory** (tag contains ``territory``): areas are in **coordinate-CSV
  units** from the ``territories`` polygon block of the territorial grid mapping;
  ``spatial_extent`` keys are ``"<tid>_0"`` territory ids. The scan total is the
  convex-hull area of every territory's polygon vertices.

Pure logic — no PyQt, no click. The χ-cluster math is a port of
:func:`gui.orientation.cluster_features_by_chi` (area-weighted) so this table and
the Orientation Map agree; the solidity/hull helpers are shared with the
Territory Map's per-shape ``bounding area``/``fill %`` display.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from . import catalogs
from .territory import _polygon_area

# One row per scan; the tab / CLI format these.
COLUMNS = [
    "Scan", "Features", "Area sum", "Area union", "Coverage %",
    "Preferred χ", "χ ± range", "Fill %", "Total",
]


# ─────────────────────────────────────────────────────────────────────
# Azimuthal (χ) dominant-cluster — area-weighted, matches the Orientation Map
# ─────────────────────────────────────────────────────────────────────
def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _feature_area(f) -> float:
    """Footprint used as the χ weight and the size weight (kept-shape n_bins)."""
    return float(f.get("n_bins") or len(f.get("intensity_profile") or {}) or 1)


def _kde_wrapped(chi, w, bandwidth, grid):
    """Circular Gaussian KDE over χ (degrees) — wrapped Δχ, area-weighted."""
    kde = np.zeros_like(grid, dtype=float)
    for c, wi in zip(chi, w):
        diff = (grid - c + 180.0) % 360.0 - 180.0
        kde += wi * np.exp(-0.5 * (diff / bandwidth) ** 2)
    return kde


def _chi_clusters(feats, bandwidth):
    """Area-weighted χ clusters split at KDE valleys (port of
    ``orientation.cluster_features_by_chi``). Returns ``[{features, area}]``."""
    from scipy.signal import find_peaks

    items = [(_wrap180([f["chi_deg"]])[0], f)
             for f in feats if f.get("chi_deg") is not None]
    if not items:
        return []
    items.sort(key=lambda t: t[0])
    chis = np.array([t[0] for t in items])
    ws = np.array([_feature_area(f) for _, f in items], dtype=float)
    if len(items) < 3:
        return [{"features": [f for _, f in items], "area": float(ws.sum())}]

    grid = np.linspace(-180, 179, 360)
    kde = _kde_wrapped(chis, ws, bandwidth, grid)
    pad = max(4, int(bandwidth * 2))
    ext = np.concatenate([kde[-pad:], kde, kde[:pad]])
    vidx, _ = find_peaks(-ext, distance=max(4, int(bandwidth * 1.5)),
                         prominence=0.3 * kde.max())
    vidx = vidx - pad
    vidx = vidx[(vidx >= 0) & (vidx < 360)]
    if len(vidx) < 2 or kde.max() == 0:
        return [{"features": [f for _, f in items], "area": float(ws.sum())}]

    vnorm = np.sort((grid[vidx] + 180) % 360)
    nseg = len(vnorm)
    groups = defaultdict(list)
    for c, f in items:
        cn = (c + 180) % 360
        idx = int(np.searchsorted(vnorm, cn, side="right")) % nseg
        groups[idx].append(f)
    return [{"features": g, "area": sum(_feature_area(f) for f in g)}
            for g in groups.values() if g]


def _chi_span(chi_vals) -> float:
    """Angular span of a set of χ (degrees), circular-aware (handles the seam)."""
    if len(chi_vals) < 2:
        return 0.0
    v = _wrap180(chi_vals)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo > 180.0:                       # straddles ±180 — unwrap and retry
        s = np.where(v < 0, v + 360.0, v)
        return float(s.max() - s.min())
    return hi - lo


def dominant_chi(feats, bandwidth=5.0):
    """(peak χ, ± half-range) of the most-populous area-weighted χ cluster.

    ``peak χ`` is the tip of that cluster's Gaussian KDE (its argmax); the range
    is half the cluster's angular span. ``(nan, 0.0)`` when no feature has a χ.
    """
    cls = _chi_clusters(feats, bandwidth)
    if not cls:
        return math.nan, 0.0
    dom = max(cls, key=lambda c: c["area"])
    chi = np.array([_wrap180([f["chi_deg"]])[0] for f in dom["features"]
                    if f.get("chi_deg") is not None])
    if chi.size == 0:
        return math.nan, 0.0
    w = np.array([_feature_area(f) for f in dom["features"]
                  if f.get("chi_deg") is not None], dtype=float)
    grid = np.arange(-180, 180, 0.5)
    peak = float(grid[int(np.argmax(_kde_wrapped(chi, w, bandwidth, grid)))])
    return peak, _chi_span(chi) / 2.0


# ─────────────────────────────────────────────────────────────────────
# Solidity ("fill") — how solidly a shape fills its convex-hull outline
# ─────────────────────────────────────────────────────────────────────
def _parse_rc(key):
    """``"row_col"`` → ``(row, col)`` ints, or None (skips territory ids etc.)."""
    parts = str(key).split("_")
    if len(parts) == 2 and parts[0].lstrip("-").isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def _hull_cell_count(points) -> int:
    """Integer lattice cells inside/on the convex hull of ``points`` (row,col).

    The denominator for grid solidity: how many bins the shape's outline
    encloses. Falls back to the point count for <3 / collinear layouts (no area).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return n
    try:
        from matplotlib.path import Path as MplPath
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return n
    verts = pts[hull.vertices]
    r0, c0 = np.floor(pts.min(axis=0)).astype(int)
    r1, c1 = np.ceil(pts.max(axis=0)).astype(int)
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    grid_pts = np.column_stack([rr.ravel(), cc.ravel()])
    inside = MplPath(verts).contains_points(grid_pts, radius=1e-9)
    # contains_points can drop boundary lattice points; never report fewer cells
    # than the shape actually occupies.
    return max(int(inside.sum()), n)


def grid_solidity(spatial_extent, n_bins) -> float:
    """Fill fraction (0..1) of a grid shape: bins ÷ convex-hull cells."""
    rc = [p for p in (_parse_rc(k) for k in (spatial_extent or [])) if p]
    if len(rc) < 3:
        return 1.0
    cells = _hull_cell_count(rc)
    n = float(n_bins or len(rc))
    return min(n / cells, 1.0) if cells else 1.0


def territory_fill(territories: dict, spatial_extent):
    """(bounding area, fill fraction) for a territorial shape, in CSV units.

    ``bounding area`` = convex-hull area of the shape's territory polygons'
    vertices (the outline "around the outer points"); ``fill`` = the shape's
    summed territory area ÷ that bounding area. Solid shapes → ~1.0; a large
    shape with holes / missing territories fills less. ``(0.0, 1.0)`` when the
    territories carry no polygons.
    """
    keys = list(dict.fromkeys(spatial_extent or []))      # unique, order-stable
    filled = 0.0
    verts = []
    for k in keys:
        t = territories.get(k)
        if not isinstance(t, dict):
            continue
        filled += float(t.get("area") or 0.0)
        verts.extend(t.get("polygon") or [])
    if len(verts) < 3:
        return filled, 1.0
    try:
        from scipy.spatial import ConvexHull
        pts = np.asarray(verts, dtype=float)
        hull = ConvexHull(pts)
        bounding = _polygon_area([[pts[v, 0], pts[v, 1]] for v in hull.vertices])
    except Exception:
        bounding = filled
    if bounding <= 0:
        return filled, 1.0
    return bounding, min(filled / bounding, 1.0)


# ─────────────────────────────────────────────────────────────────────
# Catalog-type discovery (the "type of json" selector)
# ─────────────────────────────────────────────────────────────────────
def _lineage_label(key) -> str:
    """Human label for a lineage key ``(kind, algo, tag)``."""
    kind, algo, tag = key
    bits = [algo or "?", kind or "?"]
    label = " ".join(bits)
    if tag:
        label += f" · {tag}"
    return label


def catalog_types(dm, bin_size: int) -> list:
    """Distinct catalog types available at ``bin_size`` across all project scans.

    Each entry: ``{"key": (kind, algo, tag), "label": str, "territory": bool,
    "scans": int}`` — one option for the "type of json" selector. A type is
    offered if at least one scan has a matching shapes/combined/feature catalog.
    """
    seen: dict = {}
    for scan in dm.discover_scans(selected_only=True):
        rd = dm.results_dir(scan)
        for p in catalogs.feature_sources(rd, bin_size):
            key = catalogs.lineage_key(p)
            info = catalogs.parse_name(p.name) or {}
            entry = seen.setdefault(key, {
                "key": key,
                "label": _lineage_label(key),
                "territory": "territory" in (info.get("tag") or ""),
                "scans": 0,
            })
            entry["scans"] += 1
    # Territory types last; otherwise by label.
    return sorted(seen.values(), key=lambda e: (e["territory"], e["label"]))


def _match_catalog(rd, bin_size, lineage_key):
    """The scan's catalog whose lineage key matches, or None."""
    for p in catalogs.feature_sources(rd, bin_size):
        if catalogs.lineage_key(p) == lineage_key:
            return p
    return None


def catalog_reflections(dm, bin_size: int, lineage_key) -> list:
    """Sorted reflections present across all scans for a bin + catalog type.

    Feeds the reflection-filter selector; empty if the type resolves to nothing.
    """
    refs = set()
    for scan in dm.discover_scans(selected_only=True):
        cat = _match_catalog(dm.results_dir(scan), bin_size, lineage_key)
        if cat is None:
            continue
        kept, _ = catalogs.load_features_any(cat)
        refs.update(f.get("reflection") for f in kept if f.get("reflection"))
    return sorted(refs)


# ─────────────────────────────────────────────────────────────────────
# Geometry per scan
# ─────────────────────────────────────────────────────────────────────
def _grid_total_bins(dm, scan, bin_size) -> Optional[int]:
    gm_path = dm.grid_mapping(bin_size=bin_size, scan=scan)
    try:
        with open(gm_path) as f:
            gm = json.load(f)
        return int(gm["n_bin_rows"]) * int(gm["n_bin_cols"])
    except Exception:
        return None


def _territory_mapping(dm, scan) -> Optional[dict]:
    gm_path = dm.grid_mapping(bin_size=1, variant="territory", scan=scan)
    try:
        with open(gm_path) as f:
            gm = json.load(f)
    except Exception:
        return None
    return gm if gm.get("territories") else None


def _outline_area(territories: dict) -> float:
    """Convex-hull ("outer points") area of every territory polygon, CSV units."""
    verts = []
    for t in territories.values():
        verts.extend(t.get("polygon") or [])
    if len(verts) < 3:
        return sum(float(t.get("area") or 0.0) for t in territories.values())
    try:
        from scipy.spatial import ConvexHull
        pts = np.asarray(verts, dtype=float)
        hull = ConvexHull(pts)
        return _polygon_area([[pts[v, 0], pts[v, 1]] for v in hull.vertices])
    except Exception:
        return sum(float(t.get("area") or 0.0) for t in territories.values())


def _grid_row(scan, feats, total_bins, bandwidth):
    area_sum = sum(_feature_area(f) for f in feats)
    union = set()
    fill_num = fill_den = 0.0
    for f in feats:
        ext = f.get("spatial_extent", []) or []
        union.update(ext)
        w = _feature_area(f)
        fill_num += grid_solidity(ext, f.get("n_bins")) * w
        fill_den += w
    union_area = len(union)
    peak, half = dominant_chi(feats, bandwidth)
    cov = (100.0 * union_area / total_bins) if total_bins else math.nan
    return {
        "Scan": scan,
        "Features": len(feats),
        "Area sum": area_sum,
        "Area union": union_area,
        "Coverage %": cov,
        "Preferred χ": peak,
        "χ ± range": half,
        "Fill %": (100.0 * fill_num / fill_den) if fill_den else math.nan,
        "Total": total_bins,
    }


def _territory_row(scan, feats, territories, bandwidth):
    area_of = {k: float(t.get("area") or 0.0) for k, t in territories.items()}
    area_sum = 0.0
    union = set()
    fill_num = fill_den = 0.0
    for f in feats:
        ext = list(dict.fromkeys(f.get("spatial_extent", []) or []))
        area_sum += sum(area_of.get(k, 0.0) for k in ext)
        union.update(ext)
        _bound, fill = territory_fill(territories, ext)
        w = _feature_area(f)
        fill_num += fill * w
        fill_den += w
    union_area = sum(area_of.get(k, 0.0) for k in union)
    total_area = _outline_area(territories)
    peak, half = dominant_chi(feats, bandwidth)
    cov = (100.0 * union_area / total_area) if total_area else math.nan
    return {
        "Scan": scan,
        "Features": len(feats),
        "Area sum": area_sum,
        "Area union": union_area,
        "Coverage %": cov,
        "Preferred χ": peak,
        "χ ± range": half,
        "Fill %": (100.0 * fill_num / fill_den) if fill_den else math.nan,
        "Total": round(total_area, 1),
    }


# ─────────────────────────────────────────────────────────────────────
# Public: build the whole table
# ─────────────────────────────────────────────────────────────────────
def scan_table_rows(dm, bin_size: int, lineage_key, refs=None, bandwidth: float = 5.0):
    """One summary row per project scan for the given bin + catalog type.

    ``lineage_key`` selects the catalog type ``(kind, algo, tag)`` matched in each
    scan; a territorial type (tag contains ``territory``) switches geometry to
    coordinate-CSV units. ``refs`` optionally filters to a set of reflections
    (e.g. ``{"(001)"}``); None keeps all. Returns ``(rows, meta)`` where meta
    carries the resolved units / territory flag for the header.
    """
    is_territory = bool(lineage_key) and "territory" in (lineage_key[2] or "")
    refset = set(refs) if refs else None
    rows = []
    for scan in dm.discover_scans(selected_only=True):
        rd = dm.results_dir(scan)
        cat = _match_catalog(rd, bin_size, lineage_key) if lineage_key else None
        if cat is None:
            continue
        kept, _ = catalogs.load_features_any(cat)
        if refset is not None:
            kept = [f for f in kept if f.get("reflection") in refset]
        if is_territory:
            terr = _territory_mapping(dm, scan)
            if terr is None:
                continue
            rows.append(_territory_row(scan, kept, terr["territories"], bandwidth))
        else:
            total = _grid_total_bins(dm, scan, bin_size)
            rows.append(_grid_row(scan, kept, total, bandwidth))
    meta = {
        "territory": is_territory,
        "units": "CSV²" if is_territory else "bins",
        "bin_size": bin_size,
        "lineage_key": lineage_key,
    }
    return rows, meta


def format_table(rows, meta) -> str:
    """Render rows as a fixed-width text table (the CLI's stdout view)."""
    units = meta.get("units", "bins")
    headers = ["Scan", "Features", f"Area sum ({units})", f"Area union ({units})",
               "Coverage %", "Preferred χ", "χ ± range", "Fill %", "Total"]
    keys = COLUMNS

    def cell(r, k):
        v = r.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if k in ("Coverage %", "Fill %"):
            return f"{v:.1f}%"
        if k == "Preferred χ":
            return f"{v:.0f}°"
        if k == "χ ± range":
            return f"±{v:.0f}°"
        if k in ("Area sum", "Area union", "Total") and isinstance(v, float):
            return f"{v:,.0f}"
        return str(v)

    table = [headers] + [[cell(r, k) for k in keys] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = []
    for ri, row in enumerate(table):
        lines.append("  ".join(c.rjust(widths[i]) if i else c.ljust(widths[i])
                               for i, c in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
