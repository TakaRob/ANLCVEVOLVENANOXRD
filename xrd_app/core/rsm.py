"""Fuse per-scan q-maps into one binned 3D reciprocal-space map (RSM).

``xrd-app qspace`` writes, per scan, a ``<scan>_qmap.npz`` holding the per-pixel
scattering vectors ``(qx, qy, qz)`` at that scan's sample θ plus the summed
detector intensity at that θ. This module histograms that intensity into a shared
regular 3D grid in reciprocal space and accumulates it across the θ series — the
reciprocal-space analogue of ``combined_device.py`` (which fuses the same scans
into a real-space spatial canvas).

The result is a 3D volume ``I(qx, qy, qz)``: a reflection shows up as a compact
blob whose radial position encodes lattice spacing (**strain**) and whose
transverse spread encodes **tilt / mosaic**. A per-voxel coverage count is kept
alongside so unsampled voxels are distinguishable from sampled-but-dark ones.

The intensity is summed over the whole illuminated (x,y) map at each θ, so this is
a sample-integrated RSM. For a per-grain reciprocal-space view use the feature
q-coordinates in ``<scan>_features_q.csv`` instead (see QSPACE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class QMap:
    """One scan's per-pixel Q field (+ optional summed intensity) at its θ."""
    scan: str
    theta_deg: Optional[float]
    qx: np.ndarray
    qy: np.ndarray
    qz: np.ndarray
    intensity: Optional[np.ndarray]


def load_qmap(path) -> QMap:
    """Load a ``<scan>_qmap.npz`` written by ``xrd-app qspace``."""
    d = np.load(path, allow_pickle=False)
    inten = d["intensity"] if "intensity" in d.files else None
    scan = str(d["scan"]) if "scan" in d.files else Path(path).stem
    theta = float(d["theta_deg"]) if "theta_deg" in d.files else None
    return QMap(scan, theta, d["qx"], d["qy"], d["qz"], inten)


def common_grid(qmaps: Sequence[QMap], nbins=128, pad: float = 0.0) -> List[np.ndarray]:
    """Shared ``(qx, qy, qz)`` bin edges spanning every map.

    ``nbins`` is an int (same for all axes) or a ``(nx, ny, nz)`` triple; ``pad``
    grows each axis range by that fraction of its span (0 = tight).
    """
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for m in qmaps:
        for i, a in enumerate((m.qx, m.qy, m.qz)):
            lo[i] = min(lo[i], float(np.min(a)))
            hi[i] = max(hi[i], float(np.max(a)))
    span = hi - lo
    lo = lo - pad * span
    hi = hi + pad * span
    if np.isscalar(nbins):
        nbins = (int(nbins),) * 3
    return [np.linspace(lo[i], hi[i], int(nbins[i]) + 1) for i in range(3)]


def accumulate(qmaps: Sequence[QMap], edges: Sequence[np.ndarray],
               min_intensity: float = 0.0, subtract_median: bool = True,
               progress: Optional[Callable[[int, int], None]] = None):
    """Histogram intensity into the 3D grid, summed across maps.

    For each map, per-pixel intensity is (optionally) baseline-subtracted by its
    median, thresholded at ``min_intensity``, and binned by its ``(qx,qy,qz)``.
    Returns ``(volume, counts)`` each of shape ``(nx, ny, nz)`` — the summed
    intensity and the number of contributing pixels per voxel.
    """
    shape = tuple(len(e) - 1 for e in edges)
    volume = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    n = len(qmaps)
    for j, m in enumerate(qmaps):
        if m.intensity is not None:
            w = m.intensity.ravel().astype(np.float64)
            pts = np.stack([m.qx.ravel(), m.qy.ravel(), m.qz.ravel()], axis=1)
            good = np.isfinite(w) & np.all(np.isfinite(pts), axis=1)
            w, pts = w[good], pts[good]
            if subtract_median and w.size:
                w = w - float(np.median(w))
            keep = w > min_intensity
            if np.any(keep):
                pts, w = pts[keep], w[keep]
                volume += np.histogramdd(pts, bins=edges, weights=w)[0]
                counts += np.histogramdd(pts, bins=edges)[0].astype(np.int64)
        if progress is not None:
            progress(j + 1, n)
    return volume, counts


def centers(edges: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [0.5 * (e[:-1] + e[1:]) for e in edges]


def projections(volume: np.ndarray) -> dict:
    """Max-intensity 2D projections along each axis (quick 2D views of the RSM)."""
    return {
        "qx_qy": volume.max(axis=2),  # collapse qz
        "qx_qz": volume.max(axis=1),  # collapse qy
        "qy_qz": volume.max(axis=0),  # collapse qx
    }


def save_npz(path, volume, counts, edges, meta: Optional[dict] = None,
             projs: Optional[dict] = None) -> Path:
    """Write the RSM volume + counts + edges/centers + projections to ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, cz = centers(edges)
    p = projs or projections(volume)
    payload = dict(
        volume=volume, counts=counts,
        qx_edges=edges[0], qy_edges=edges[1], qz_edges=edges[2],
        qx_centers=cx, qy_centers=cy, qz_centers=cz,
        proj_qx_qy=p["qx_qy"], proj_qx_qz=p["qx_qz"], proj_qy_qz=p["qy_qz"],
    )
    for k, v in (meta or {}).items():
        payload[k] = v
    np.savez_compressed(path, **payload)
    return path


def load_rsm(path) -> dict:
    """Load an ``rsm.npz`` written by ``xrd-app rsm`` into a tidy dict.

    Returns projections + axis edges/centers + coverage, so the GUI is
    render-only. ``volume``/``counts`` are included when present.
    """
    d = np.load(path, allow_pickle=False)
    files = set(d.files)
    return {
        "volume": d["volume"] if "volume" in files else None,
        "counts": d["counts"] if "counts" in files else None,
        "edges": (d["qx_edges"], d["qy_edges"], d["qz_edges"]),
        "centers": (d["qx_centers"], d["qy_centers"], d["qz_centers"]),
        "proj": {"qx_qy": d["proj_qx_qy"], "qx_qz": d["proj_qx_qz"],
                 "qy_qz": d["proj_qy_qz"]},
        "scans": [str(s) for s in d["scans"]] if "scans" in files else [],
        "thetas": [float(t) for t in d["thetas"]] if "thetas" in files else [],
    }


def load_feature_cloud(qspace_dir, scans=None, theta_by_scan=None) -> dict:
    """Read every ``<scan>_features_q.csv`` under ``qspace_dir`` into arrays.

    Each detected feature is one point in reciprocal space. θ per point comes
    from ``core.tracking.theta_of`` (features_q.csv has no θ column). Returns a
    dict of parallel arrays/lists: ``qx, qy, qz, q_mag, intensity`` (floats),
    ``reflection, scan`` (str), ``theta`` (float, NaN if unknown), and ``n``.
    """
    import csv as _csv
    from . import tracking

    qdir = Path(qspace_dir)
    want = set(scans) if scans else None
    rows, scan_of = [], []
    for f in sorted(qdir.glob("*_features_q.csv")):
        scan = f.name[:-len("_features_q.csv")]
        if want is not None and scan not in want:
            continue
        with open(f, newline="") as fh:
            for r in _csv.DictReader(fh):
                rows.append(r)
                scan_of.append(r.get("scan") or scan)

    def _f(name):
        out = np.empty(len(rows), dtype=np.float64)
        for i, r in enumerate(rows):
            try:
                out[i] = float(r.get(name, ""))
            except (TypeError, ValueError):
                out[i] = np.nan
        return out

    thetas = np.array(
        [tracking.theta_of(s, theta_by_scan) if tracking.theta_of(s, theta_by_scan)
         is not None else np.nan for s in scan_of], dtype=np.float64)
    return {
        "n": len(rows),
        "qx": _f("qx"), "qy": _f("qy"), "qz": _f("qz"),
        "q_mag": _f("q_mag"), "intensity": _f("sum_integrated"),
        "reflection": [r.get("reflection", "") for r in rows],
        "scan": scan_of,
        "theta": thetas,
    }


def summary(volume, counts, edges, scans, thetas) -> dict:
    nz = int(np.count_nonzero(counts))
    return {
        "n_scans": len(scans),
        "scans": list(scans),
        "thetas": [None if (t is None or (isinstance(t, float) and np.isnan(t)))
                   else float(t) for t in thetas],
        "grid_shape": list(volume.shape),
        "q_ranges": {ax: [float(e[0]), float(e[-1])]
                     for ax, e in zip(("qx", "qy", "qz"), edges)},
        "total_intensity": float(volume.sum()),
        "nonzero_voxels": nz,
        "fill_fraction": nz / float(volume.size) if volume.size else 0.0,
        "peak_voxel_intensity": float(volume.max()) if volume.size else 0.0,
    }
