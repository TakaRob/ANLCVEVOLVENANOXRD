"""
Grand-sum of a scan's binned frames — the artifact behind the Setup → manual
reflections "Compute histogram" button.

The 2θ histogram in that dialog is always re-derived on the fly from a single
persisted image: the sum of every bin (equivalently, every raw frame) for the
scan. Summing is the expensive part; deriving the radial profile from the sum is
instant. This module owns that sum so the GUI button, the ``reflection-sum`` CLI
command, and the post-binning hook all produce the identical ``reflection_sum.npz``.

All bin sizes give the same grand sum, so we read from the fastest available
source: a prebuilt ``xrd_NxN_bins.h5`` (fewest datasets) when one exists, else
raw 1×1 frames. The saved file matches what ``reflection_popup`` reads back:
``image`` (float32), ``is_raw`` (bool), ``max_bins`` (int, 0 = no limit).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import io

SUM_FILENAME = "reflection_sum.npz"


def sum_dir(dm, scan=None) -> Path:
    """Directory the sum is persisted in (per-scan metadata, mirrors the popup)."""
    return Path(dm.metadata_scan_dir(scan)) if scan else Path(dm.metadata_dir)


def sum_path(dm, scan=None) -> Path:
    """Path of the persisted ``reflection_sum.npz`` for a scan."""
    return sum_dir(dm, scan) / SUM_FILENAME


def source_bin(dm, scan=None) -> int:
    """Bin size to sum from — the largest prebuilt NxN h5, else 1 (raw).

    Mirrors ``reflection_popup._source_bin``: all sizes yield the same grand
    sum, so prefer the coarsest built bins (fewest datasets → fastest read).
    """
    sizes = set()
    try:
        bdir = dm.binned_dir(scan)
        if bdir.is_dir():
            for p in bdir.glob("xrd_*x*_bins.h5"):
                m = re.match(r"xrd_(\d+)x(\d+)_bins", p.name)
                if m and int(m.group(1)) != 1:
                    sizes.add(int(m.group(1)))
    except Exception:
        pass
    return max(sizes) if sizes else 1


def save(dm, scan, image: np.ndarray, is_raw: bool, max_bins: int = 0) -> Path:
    """Persist a summed image atomically in the format the popup reads back."""
    path = sum_path(dm, scan)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez_compressed(
        str(tmp),
        image=image.astype(np.float32),
        is_raw=np.array(bool(is_raw)),
        max_bins=np.array(int(max_bins or 0)))
    os.replace(str(tmp), str(path))
    return path


def compute_and_save(dm, scan=None, max_bins: Optional[int] = None,
                     overwrite: bool = True,
                     progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Sum all of a scan's bins and persist ``reflection_sum.npz``.

    Returns a status dict: ``{scan, path, shape, is_raw, bin_size, skipped}``.
    With ``overwrite=False`` an existing sum is left in place (``skipped=True``).
    ``max_bins`` caps how many bins are summed (0/None = all); it is recorded in
    the file so the GUI shows the same cap.
    """
    scan_name = dm._scan(scan) if hasattr(dm, "_scan") else scan
    path = sum_path(dm, scan)
    if not overwrite and path.exists():
        return {"scan": scan_name, "path": path, "shape": None,
                "is_raw": None, "bin_size": None, "skipped": True}

    bin_size = source_bin(dm, scan)
    src = io.open_bin_source(dm, bin_size, scan)
    try:
        image = src.sum_all(max_bins=max_bins or None, progress=progress)
        is_raw = src.is_raw
    finally:
        src.close()

    save(dm, scan, image, is_raw, max_bins=int(max_bins or 0))
    return {"scan": scan_name, "path": path, "shape": tuple(image.shape),
            "is_raw": bool(is_raw), "bin_size": bin_size, "skipped": False}
