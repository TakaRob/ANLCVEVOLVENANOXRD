"""Territorial / cell-model partition — a skew-free *reference* binning.

Bins frames by **true stage (X, Y) positions** instead of the reconstructed
serpentine grid, so the partition is immune to the even/odd-row backlash that
skews the N×N grid (see :func:`core.io.assign_grid_from_positions`). Frames are
grouped into irregular **territories** that grow outward over the (X, Y)
Delaunay neighbor graph until each reaches a target frame count — a *clean
partition* (every frame in exactly one territory) with adaptive SNR: denser
regions get tighter territories, sparser regions get more averaging.

The output is a grid-mapping dict compatible with :func:`core.io.build_bins` /
:func:`core.processing.run_peaks` / :func:`core.processing.run_shapes`, plus a
``territories`` block carrying each cell's centroid, area, frame count, polygon
footprint, and physical neighbor list. The territorial shape linker
(``ShapeAlgorithms/territory.py``) links peaks across those *physical* neighbors
instead of the N×N 8-neighborhood; every other pipeline stage is reused
unchanged.

This is the **source of truth** the fast post-processing (skew / backlash
fixing) is optimized against — see ``compare_to_truth.py`` at the repo root.

Bin keys are ``"<tid>_0"`` so the existing ``int(k.split("_")[...])`` parsers
(``io._bin_sort_key``, ``BinImageSource``, the gaussian linker) keep working
without edits; the real adjacency comes from ``territories[*].neighbors``.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from . import io


# ─────────────────────────────────────────────────────────────────────
# Neighbor graph over true (X, Y) frame positions
# ─────────────────────────────────────────────────────────────────────
def _delaunay_adjacency(points: np.ndarray) -> list:
    """Local frame adjacency from a distance-pruned Delaunay triangulation.

    Delaunay fills the convex hull, so an unfiltered graph bridges across concave
    scan boundaries (for example, joining opposite ends of a bowtie raster).
    Keep only edges within three typical frame spacings. Falls back to a kNN
    graph if the triangulation is degenerate (collinear / duplicate positions).
    """
    n = len(points)
    adj = [set() for _ in range(n)]
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        max_edge = 3.0 * _median_step(points)
        for simplex in tri.simplices:
            m = len(simplex)
            for i in range(m):
                for j in range(i + 1, m):
                    a, b = int(simplex[i]), int(simplex[j])
                    if np.linalg.norm(points[a] - points[b]) <= max_edge:
                        adj[a].add(b)
                        adj[b].add(a)
        if any(adj):
            return adj
    except Exception:
        pass
    return _knn_adjacency(points, k=8)


def _knn_adjacency(points: np.ndarray, k: int = 8) -> list:
    """k-nearest-neighbor adjacency — Delaunay fallback for degenerate layouts."""
    from scipy.spatial import cKDTree
    n = len(points)
    adj = [set() for _ in range(n)]
    if n <= 1:
        return adj
    tree = cKDTree(points)
    kk = min(k + 1, n)
    _, idx = tree.query(points, k=kk)
    idx = np.atleast_2d(idx)
    for i in range(n):
        for j in idx[i][1:]:
            j = int(j)
            adj[i].add(j)
            adj[j].add(i)
    return adj


def _median_step(points: np.ndarray) -> float:
    """Median nearest-neighbor spacing — the natural one-frame length scale.

    Used to normalize centroids into grid-like units (so the shape filter's
    distance check behaves as on the N×N grid) and to size fallback footprints.
    """
    from scipy.spatial import cKDTree
    if len(points) < 2:
        return 1.0
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]
    return float(np.median(nn)) if len(nn) else 1.0


# ─────────────────────────────────────────────────────────────────────
# Greedy region-growing partition
# ─────────────────────────────────────────────────────────────────────
def _grow_territories(adj: list, points: np.ndarray, target_size: int):
    """Greedy region-grow over ``adj`` into a clean partition.

    The lowest-index unassigned frame seeds a territory; it absorbs its nearest
    unassigned neighbors (Euclidean distance to the seed) until it reaches
    ``target_size`` frames or its frontier is exhausted. Every frame lands in
    exactly one territory. Seed order is the frame index, so the partition is
    deterministic / reproducible.

    Returns ``(territory_of, territories)`` — a per-frame territory id array and
    a list of member-index lists.
    """
    n = len(adj)
    territory_of = np.full(n, -1, dtype=np.int64)
    territories: list = []
    for seed in range(n):
        if territory_of[seed] != -1:
            continue
        tid = len(territories)
        territory_of[seed] = tid
        members = [seed]
        sx, sy = points[seed]
        frontier: list = []
        for nb in adj[seed]:
            if territory_of[nb] == -1:
                dx, dy = points[nb][0] - sx, points[nb][1] - sy
                heapq.heappush(frontier, (dx * dx + dy * dy, nb))
        while len(members) < target_size and frontier:
            _, f = heapq.heappop(frontier)
            if territory_of[f] != -1:
                continue
            territory_of[f] = tid
            members.append(f)
            for nb in adj[f]:
                if territory_of[nb] == -1:
                    dx, dy = points[nb][0] - sx, points[nb][1] - sy
                    heapq.heappush(frontier, (dx * dx + dy * dy, nb))
        territories.append(members)
    return territory_of, territories


def _territory_neighbors(territory_of: np.ndarray, adj: list, n_terr: int) -> list:
    """Territory adjacency: A ~ B if any member frames are Delaunay-adjacent."""
    nbrs = [set() for _ in range(n_terr)]
    for f, ta in enumerate(territory_of):
        for nb in adj[f]:
            tb = int(territory_of[nb])
            if tb != ta:
                nbrs[int(ta)].add(tb)
                nbrs[tb].add(int(ta))
    return nbrs


def grow_peak_feature(peaks_by_cell: dict, neighbors: dict, seed_cell: str,
                      seed_peak: dict, link_tolerance: float = 5.0,
                      anchor: str = "frontier", allowed_cells=None) -> dict:
    """Grow one feature through adjacent cells using detector-peak continuity.

    ``frontier`` compares each candidate with the accepted peak in the adjoining
    cell, allowing a Bragg peak to drift smoothly across real space. ``seed``
    reproduces the old fixed-center behavior for controlled comparisons.
    Returns ``{cell_key: peak}``, keeping the strongest matching peak per cell.
    """
    if anchor not in {"frontier", "seed"}:
        raise ValueError("anchor must be 'frontier' or 'seed'")

    accepted = {seed_cell: seed_peak}
    queue = [seed_cell]
    tol2 = float(link_tolerance) ** 2
    reflection = seed_peak.get("label")

    while queue:
        cell = queue.pop(0)
        reference = accepted[cell] if anchor == "frontier" else seed_peak
        for neighbor in neighbors.get(cell, []):
            if neighbor in accepted:
                continue
            if allowed_cells is not None and neighbor not in allowed_cells:
                continue
            matches = []
            for peak in peaks_by_cell.get(neighbor, []):
                if peak.get("label") != reflection:
                    continue
                d2 = ((float(peak["x"]) - float(reference["x"])) ** 2 +
                      (float(peak["y"]) - float(reference["y"])) ** 2)
                if d2 <= tol2:
                    matches.append(peak)
            if matches:
                accepted[neighbor] = max(
                    matches, key=lambda p: float(p.get("snr", 0.0)))
                queue.append(neighbor)

    return accepted


# ─────────────────────────────────────────────────────────────────────
# Per-territory geometry (to-scale footprint)
# ─────────────────────────────────────────────────────────────────────
def _polygon_area(poly: list) -> float:
    """Shoelace area of a polygon given as ``[[x, y], ...]``."""
    if len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _hull_or_box(pts: np.ndarray, step: float) -> list:
    """Convex-hull polygon of the member points; a step-box for <3 / collinear."""
    if len(pts) >= 3:
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(pts)
            return [[float(pts[v, 0]), float(pts[v, 1])] for v in h.vertices]
        except Exception:
            pass
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    hw = step / 2.0
    return [[cx - hw, cy - hw], [cx + hw, cy - hw],
            [cx + hw, cy + hw], [cx - hw, cy + hw]]


# ─────────────────────────────────────────────────────────────────────
# Public: build the territorial grid mapping
# ─────────────────────────────────────────────────────────────────────
def build_territory_mapping(
    xrd_dir: Union[str, Path],
    pos_csv: Union[str, Path],
    target_size: int = 9,
    scan_number: int = 203,
    output: Optional[Union[str, Path]] = None,
    log: Callable[[str], None] = print,
    archive: Optional[Union[str, Path]] = None,
) -> dict:
    """Build (and optionally write) the territorial grid-mapping dict.

    Requires a real position CSV with ``X_Position``/``Y_Position`` — the
    skew-free coordinate source. Raises ``FileNotFoundError`` / ``ValueError``
    otherwise rather than silently falling back to the reconstructed grid, so a
    "truth" is never built on the very skew it is meant to measure.
    """
    log(f"Loading scan metadata from {xrd_dir} ...")
    xrd_files, frame_map, n_total = io.load_xrd_metadata(
        xrd_dir, scan_number, archive=archive)
    log(f"  {n_total} frames across {len(xrd_files)} H5 files")

    have_position_file = pos_csv is not None and Path(pos_csv).exists()
    have_archive_positions = bool(
        archive and io.archive_has_real_positions(archive))
    if not have_position_file and not have_archive_positions:
        raise FileNotFoundError(
            f"Territorial binning needs real positions (got {pos_csv!r}, and the "
            "unbinned archive has none).")
    if have_position_file and io.is_recreated_csv(pos_csv):
        raise ValueError(
            f"{pos_csv} is a recreated CSV (synthetic lattice, not true "
            "positions). Territorial binning needs real stage (X, Y) positions.")

    if have_position_file:
        frame_x, frame_y = io.load_positions_xy(pos_csv, n_total)
    else:
        frame_x, frame_y = io.archive_positions(archive)
    if not np.isfinite(frame_y).any():
        raise ValueError(
            f"{pos_csv} has no Y_Position column. Territorial binning needs "
            "true 2-D (X, Y) positions.")

    # Interpolate the occasional dropped reading so every frame gets a position
    # (same treatment the grid path gives partial position traces).
    x = io._interp_nan(frame_x)
    y = io._interp_nan(frame_y)
    points = np.column_stack([x, y]).astype(np.float64)

    step = _median_step(points)
    x_min, y_min = float(x.min()), float(y.min())
    log(f"  Median frame spacing (step): {step:.4g}")

    log("Building (X, Y) neighbor graph (Delaunay) ...")
    adj = _delaunay_adjacency(points)

    log(f"Region-growing territories (target {target_size} frames/cell) ...")
    territory_of, terr_members = _grow_territories(adj, points, target_size)
    n_terr = len(terr_members)
    terr_nbrs = _territory_neighbors(territory_of, adj, n_terr)

    counts = np.array([len(m) for m in terr_members])
    log(f"  {n_terr} territories  ·  frames/cell: "
        f"min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    bins: dict = {}
    territories: dict = {}
    max_r = max_c = 0.0
    for tid, members in enumerate(terr_members):
        key = f"{tid}_0"
        bins[key] = [int(m) for m in members]
        pts = points[members]
        cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        # Normalize centroid into grid-like units (≈1 per frame step) so the
        # shape filter's distance-monotonicity check (which uses node row/col)
        # behaves exactly as on the N×N grid.
        cr, cc = (cy - y_min) / step, (cx - x_min) / step
        max_r, max_c = max(max_r, cr), max(max_c, cc)
        polygon = _hull_or_box(pts, step)
        territories[key] = {
            "centroid_xy": [round(cx, 4), round(cy, 4)],
            "centroid_rc": [round(cr, 3), round(cc, 3)],
            "area": round(_polygon_area(polygon), 4),
            "count": len(members),
            "neighbors": [f"{nb}_0" for nb in sorted(terr_nbrs[tid])],
            "polygon": [[round(px, 4), round(py, 4)] for px, py in polygon],
        }

    n_rows, n_cols = int(max_r) + 1, int(max_c) + 1
    result = {
        "bin_size": 1,                       # nominal; territories override geometry
        "coordinate_source": "territory_xy",
        # Territorial binning always requires a real (X, Y) CSV (enforced above),
        # so this mapping is real-positions by construction — `bin` accepts it.
        "positions_csv": str(pos_csv) if have_position_file else None,
        "positions_archive": str(archive) if have_archive_positions else None,
        "positions_real": True,
        "target_size": target_size,
        "step": step,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_bin_rows": n_rows,
        "n_bin_cols": n_cols,
        "n_total_frames": n_total,
        "n_bins": len(bins),
        "h5_dataset": io.H5_DATASET,
        "xrd_files": xrd_files,
        "bins": bins,
        "frame_map": frame_map,
        "territories": territories,
    }

    if output is not None:
        io.atomic_write_json(output, result, indent=None)
        size_kb = Path(output).stat().st_size / 1024
        log(f"Wrote {output} ({size_kb:.0f} KB)")

    return result


def orient_centroids_to_grid(gm: dict, reference_bins: dict,
                             reference_bin_size: int = 1):
    """Orient territory centroids to a regular grid using shared frame IDs.

    Territorial ``centroid_rc`` coordinates start at physical ``(Y_min, X_min)``,
    while regular scan rows/columns follow the acquisition-oriented grid. Choose
    the transpose and axis reversals that correlate with the regular grid, then
    return ``(cell_to_rc, n_rows, n_cols)``. If the mappings do not share enough
    frames to establish an orientation, the physical coordinates are unchanged.
    """
    territories = gm.get("territories") or {}
    source_bins = gm.get("bins") or {}
    source = {}
    for key, info in territories.items():
        rc = info.get("centroid_rc")
        if rc and len(rc) == 2:
            try:
                source[key] = (float(rc[0]), float(rc[1]))
            except (TypeError, ValueError):
                pass

    frame_target = {}
    bs = max(int(reference_bin_size), 1)
    for key, frames in (reference_bins or {}).items():
        try:
            row, col = (int(v) for v in key.split("_"))
        except (AttributeError, TypeError, ValueError):
            continue
        target = (row * bs + (bs - 1) / 2, col * bs + (bs - 1) / 2)
        for frame in frames:
            frame_target[frame] = target

    src_rows, src_cols, target_rows, target_cols = [], [], [], []
    for key, frames in source_bins.items():
        if key not in source:
            continue
        targets = [frame_target[frame] for frame in frames if frame in frame_target]
        if not targets:
            continue
        src_rows.append(source[key][0])
        src_cols.append(source[key][1])
        target_rows.append(float(np.mean([rc[0] for rc in targets])))
        target_cols.append(float(np.mean([rc[1] for rc in targets])))

    def correlation(a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    direct = (correlation(src_rows, target_rows),
              correlation(src_cols, target_cols))
    swapped = (correlation(src_cols, target_rows),
               correlation(src_rows, target_cols))
    transpose = sum(abs(v) for v in swapped) > sum(abs(v) for v in direct)
    row_corr, col_corr = swapped if transpose else direct

    n_source_rows = int(gm.get("n_bin_rows") or 0)
    n_source_cols = int(gm.get("n_bin_cols") or 0)
    if not n_source_rows and source:
        n_source_rows = int(max(rc[0] for rc in source.values())) + 1
    if not n_source_cols and source:
        n_source_cols = int(max(rc[1] for rc in source.values())) + 1
    n_rows, n_cols = ((n_source_cols, n_source_rows) if transpose
                      else (n_source_rows, n_source_cols))

    remap = {}
    for key, (source_row, source_col) in source.items():
        row, col = ((source_col, source_row) if transpose
                    else (source_row, source_col))
        if row_corr < 0:
            row = n_rows - 1 - row
        if col_corr < 0:
            col = n_cols - 1 - col
        remap[key] = (int(round(row)), int(round(col)))
    return remap, n_rows, n_cols


# ─────────────────────────────────────────────────────────────────────
# Gridless coordinate linking for an existing (1×1) grid mapping
# ─────────────────────────────────────────────────────────────────────
def add_coordinate_neighbors(gm: dict, frame_x, frame_y,
                             log: Callable[[str], None] = print) -> dict:
    """Augment an existing grid mapping for **gridless coordinate linking**.

    The serpentine/backlash skew is a *grid* artefact: linking by (row, col)
    adjacency on a backlash-misregistered grid fragments features. This keeps the
    same cells and peaks but replaces grid adjacency with the physical (Delaunay)
    neighbor graph over each cell's true (X, Y) centroid, so the territory linker
    links by coordinate rather than by grid. The existing bin keys are kept only
    as labels — adjacency is purely physical, so the skewed keys never drive
    linking. Intended for 1×1 (one frame per cell). Mutates and returns ``gm``.

    ``frame_x`` / ``frame_y`` are per-frame true positions
    (``io.load_positions_xy``), indexed by global frame index.
    """
    bins = gm["bins"]
    x = io._interp_nan(np.asarray(frame_x, dtype=np.float64))
    y = io._interp_nan(np.asarray(frame_y, dtype=np.float64))
    keys = list(bins.keys())
    pts = np.array([[float(np.mean(x[bins[k]])), float(np.mean(y[bins[k]]))]
                    for k in keys], dtype=np.float64)
    step = _median_step(pts)
    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    log(f"  Coordinate neighbors: {len(keys)} cells · step {step:.4g}")

    adj = _delaunay_adjacency(pts)
    territories: dict = {}
    for i, k in enumerate(keys):
        cx, cy = pts[i]
        territories[k] = {
            "centroid_xy": [round(float(cx), 4), round(float(cy), 4)],
            "centroid_rc": [round((cy - y_min) / step, 3),
                            round((cx - x_min) / step, 3)],
            "count": len(bins[k]),
            "neighbors": [keys[j] for j in sorted(adj[i])],
        }

    gm["territories"] = territories
    gm["coordinate_source"] = "coord_xy"
    gm["step"] = step
    return gm
