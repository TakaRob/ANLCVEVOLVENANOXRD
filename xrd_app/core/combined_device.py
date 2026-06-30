"""Fuse the per-θ device maps into one spatial *combined device view*.

Each scan's ``device_map`` (one row per scan×reflection×spatial bin, from
:mod:`core.aggregate`) is a per-orientation slice. Fusing them over θ yields a
single (row, col) canvas richer than any one scan:

  * ``max_intensity[r,c]``       — strongest diffraction seen at that spot, any θ
  * ``argmax_theta[r,c]``        — the θ that produced it → a local-orientation map
  * ``n_theta_present[r,c]``     — how many θ that spot lit up at (recurrence)
  * ``layer_intensity[k,r,c]``   — same, split per reflection k (per-band layers)
  * ``layer_argmax_theta[k,r,c]``— orientation map per reflection

Plus track centroids (from :mod:`core.tracking`) for an overlay. This is a pure
data layer — arrays + metadata saved to a ``.npz`` — exactly what a future
"Combined Device View" tab would render. No GUI here.

Pure module — no click, no Qt. numpy imported lazily.
"""

from __future__ import annotations

from typing import Callable, Optional

from . import tracking


def _num(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    f = _num(v)
    return int(f) if f is not None else None


def build_combined(device_map_rows, theta_by_scan: Optional[dict] = None,
                   grid_shape=None, intensity_key: str = "integrated",
                   tracks=None, log: Callable[[str], None] = print) -> dict:
    """Fuse device-map rows over θ into stacked spatial arrays.

    Args:
        device_map_rows: dicts with scan, reflection, row, col, intensity,
                         integrated (the aggregate device_map table).
        intensity_key:   'integrated' (default) or 'intensity' — which column
                         drives the max/argmax canvases.
        grid_shape:      (n_rows, n_cols); inferred from the data when None.
        tracks:          optional track dicts → centroid overlay metadata.

    Returns a dict of numpy arrays + metadata (see module docstring).
    """
    import numpy as np

    theta_by_scan = theta_by_scan or tracking.THETA_BY_SCAN
    rows = []
    reflections = []
    refl_index = {}
    max_r = max_c = -1
    for d in device_map_rows:
        r, c = _int(d.get("row")), _int(d.get("col"))
        if r is None or c is None:
            continue
        scan = d.get("scan")
        theta = theta_by_scan.get(scan)
        if theta is None:
            continue
        val = _num(d.get(intensity_key))
        if val is None:
            val = _num(d.get("intensity")) or 0.0
        refl = d.get("reflection")
        if refl not in refl_index:
            refl_index[refl] = len(reflections)
            reflections.append(refl)
        rows.append((r, c, refl_index[refl], float(theta), val))
        max_r, max_c = max(max_r, r), max(max_c, c)

    if grid_shape is not None:
        nrows, ncols = grid_shape
    else:
        nrows, ncols = max_r + 1, max_c + 1
    nrefl = len(reflections)

    max_intensity = np.zeros((nrows, ncols), float)
    argmax_theta = np.full((nrows, ncols), np.nan, float)
    thetas_seen = [[set() for _ in range(ncols)] for _ in range(nrows)]
    layer_intensity = np.zeros((nrefl, nrows, ncols), float)
    layer_argmax_theta = np.full((nrefl, nrows, ncols), np.nan, float)

    for r, c, k, theta, val in rows:
        if val > max_intensity[r, c]:
            max_intensity[r, c] = val
            argmax_theta[r, c] = theta
        if val > layer_intensity[k, r, c]:
            layer_intensity[k, r, c] = val
            layer_argmax_theta[k, r, c] = theta
        if val > 0:
            thetas_seen[r][c].add(theta)

    n_theta_present = np.array([[len(thetas_seen[r][c]) for c in range(ncols)]
                                for r in range(nrows)], dtype=int)

    track_meta = _track_overlay(tracks) if tracks else {
        "track_id": [], "reflection": [], "centroid_row": [],
        "centroid_col": [], "theta_at_max_I": [], "is_recurrent": []}

    all_thetas = sorted({r[3] for r in rows})
    log(f"Combined device: {nrows}×{ncols} grid, {nrefl} reflection layers, "
        f"{int((n_theta_present > 0).sum())} bins with signal, "
        f"{len(track_meta['track_id'])} track centroids")
    return {
        "bin_size": None,
        "n_rows": nrows, "n_cols": ncols,
        "reflections": reflections,
        "thetas": all_thetas,
        "intensity_key": intensity_key,
        "max_intensity": max_intensity,
        "argmax_theta": argmax_theta,
        "n_theta_present": n_theta_present,
        "layer_intensity": layer_intensity,
        "layer_argmax_theta": layer_argmax_theta,
        "tracks": track_meta,
    }


def _track_overlay(tracks) -> dict:
    cols = {"track_id": [], "reflection": [], "centroid_row": [],
            "centroid_col": [], "theta_at_max_I": [], "is_recurrent": []}
    for t in tracks:
        cols["track_id"].append(t.get("track_id"))
        cols["reflection"].append(t.get("reflection"))
        cols["centroid_row"].append(t.get("centroid_row"))
        cols["centroid_col"].append(t.get("centroid_col"))
        cols["theta_at_max_I"].append(t.get("theta_at_max_I"))
        cols["is_recurrent"].append(bool(t.get("is_recurrent")))
    return cols


def save_npz(path, combined: dict):
    """Write the combined-device arrays + metadata to a compressed ``.npz``."""
    import numpy as np
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tr = combined["tracks"]
    np.savez_compressed(
        path,
        n_rows=combined["n_rows"], n_cols=combined["n_cols"],
        reflections=np.array([str(r) for r in combined["reflections"]]),
        thetas=np.array(combined["thetas"], float),
        intensity_key=combined["intensity_key"],
        max_intensity=combined["max_intensity"],
        argmax_theta=combined["argmax_theta"],
        n_theta_present=combined["n_theta_present"],
        layer_intensity=combined["layer_intensity"],
        layer_argmax_theta=combined["layer_argmax_theta"],
        track_id=np.array(tr["track_id"]),
        track_reflection=np.array([str(r) for r in tr["reflection"]]),
        track_centroid_row=np.array(tr["centroid_row"], float),
        track_centroid_col=np.array(tr["centroid_col"], float),
        track_theta_at_max=np.array([float(x) if x is not None else np.nan
                                     for x in tr["theta_at_max_I"]], float),
        track_is_recurrent=np.array(tr["is_recurrent"], bool),
    )
    return path


def summary(combined: dict) -> dict:
    """Small JSON-friendly summary of the combined canvas (no big arrays)."""
    import numpy as np
    mi = combined["max_intensity"]
    npres = combined["n_theta_present"]
    return {
        "n_rows": combined["n_rows"], "n_cols": combined["n_cols"],
        "reflections": combined["reflections"],
        "thetas": combined["thetas"],
        "intensity_key": combined["intensity_key"],
        "bins_with_signal": int((npres > 0).sum()),
        "bins_multi_theta": int((npres > 1).sum()),
        "max_intensity_overall": float(mi.max()) if mi.size else 0.0,
        "n_tracks": len(combined["tracks"]["track_id"]),
        "n_recurrent_tracks": int(sum(combined["tracks"]["is_recurrent"])),
    }
