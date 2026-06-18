"""Build CVEvolve dev/holdout splits from a labeled source.

The "Pick Development Set" step selects bins from one of:
  - **verified** manual labels (bins marked ``__reviewed__``), or
  - an **algorithm's** peak or shape output (which sets the evolved kind).

A single holdout percentage splits the bins (seeded, reproducible) into a dev set
(``test_data/``) and a holdout set (``holdout_data/``), each written in the format
CVEvolve already expects: ``bin_annotations.json`` + ``empty_bins.json`` +
``grid_mapping.json`` (copied).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Optional, Tuple

Annotations = dict  # {bin_key: {reflection: [[row, col], ...]}}


# ----- source → annotations ------------------------------------------
def bins_from_verified(bin_annotations_path) -> Tuple[Annotations, list]:
    """From a labeling ``bin_annotations_*.json`` keep only reviewed bins.

    Returns (annotations, empty_bins) where empty_bins are reviewed-but-empty.
    """
    with open(bin_annotations_path) as f:
        all_ann = json.load(f)
    ann, empty = {}, []
    for bk, ref_dict in all_ann.items():
        if not isinstance(ref_dict, dict) or not ref_dict.get("__reviewed__", False):
            continue
        peaks = {r: pts for r, pts in ref_dict.items() if r != "__reviewed__" and pts}
        if peaks:
            ann[bk] = peaks
        else:
            empty.append(bk)
    return ann, empty


def bins_from_peaks(peaks_json) -> Tuple[Annotations, list]:
    """Convert a ``*_peaks.json`` (peaks_by_bin) to bin_annotations format."""
    if not isinstance(peaks_json, dict):
        with open(peaks_json) as f:
            peaks_json = json.load(f)
    ann = {}
    for bk, peaks in peaks_json.get("peaks_by_bin", {}).items():
        by_ref = {}
        for p in peaks:
            r, c = int(round(p["y"])), int(round(p["x"]))
            by_ref.setdefault(p.get("label", "unknown"), []).append([r, c])
        if by_ref:
            ann[bk] = by_ref
    return ann, []


def bins_from_shapes(shapes_json) -> Tuple[Annotations, list]:
    """Convert a ``*_shapes.json`` (kept features) to bin_annotations format.

    Each kept feature contributes its per-bin detector positions (from the
    feature's intensity_profile) under the feature's reflection.
    """
    if not isinstance(shapes_json, dict):
        with open(shapes_json) as f:
            shapes_json = json.load(f)
    ann = {}
    for feat in shapes_json.get("kept", []):
        ref = feat.get("reflection", "unknown")
        for bk, entry in feat.get("intensity_profile", {}).items():
            if isinstance(entry, dict) and "det_x" in entry and "det_y" in entry:
                r, c = int(entry["det_y"]), int(entry["det_x"])
            else:
                r, c = int(feat.get("detector_y", 0)), int(feat.get("detector_x", 0))
            ann.setdefault(bk, {}).setdefault(ref, []).append([r, c])
    return ann, []


# ----- split + write -------------------------------------------------
def _write_set(dest: Path, ann: Annotations, empty: list, grid_mapping: Optional[Path]):
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "bin_annotations.json", "w") as f:
        json.dump(ann, f, indent=2)
    with open(dest / "empty_bins.json", "w") as f:
        json.dump(empty, f, indent=2)
    if grid_mapping and Path(grid_mapping).exists():
        shutil.copy2(grid_mapping, dest / "grid_mapping.json")


def build_split(annotations: Annotations, empty_bins: list, *, holdout_pct: float,
                seed: int, dest_dev, dest_holdout,
                grid_mapping=None) -> dict:
    """Split labeled bins into dev + holdout sets (seeded) and write both.

    ``holdout_pct`` is the fraction (0-100) of *labeled* bins reserved for holdout.
    Empty bins are split in the same proportion. Returns counts.
    """
    dest_dev, dest_holdout = Path(dest_dev), Path(dest_holdout)
    rng = random.Random(seed)

    keys = sorted(annotations.keys())
    rng.shuffle(keys)
    n_hold = int(round(len(keys) * holdout_pct / 100.0))
    hold_keys = set(keys[:n_hold])

    empty = sorted(empty_bins)
    rng.shuffle(empty)
    n_ehold = int(round(len(empty) * holdout_pct / 100.0))
    hold_empty = set(empty[:n_ehold])

    dev_ann = {k: v for k, v in annotations.items() if k not in hold_keys}
    hold_ann = {k: v for k, v in annotations.items() if k in hold_keys}
    dev_empty = [k for k in empty if k not in hold_empty]
    hold_empty_l = [k for k in empty if k in hold_empty]

    _write_set(dest_dev, dev_ann, dev_empty, grid_mapping)
    _write_set(dest_holdout, hold_ann, hold_empty_l, grid_mapping)

    n_pts = sum(len(pts) for refs in annotations.values() for pts in refs.values())
    return {
        "total_bins": len(annotations),
        "total_points": n_pts,
        "dev_bins": len(dev_ann),
        "holdout_bins": len(hold_ann),
        "dev_empty": len(dev_empty),
        "holdout_empty": len(hold_empty_l),
        "seed": seed,
        "holdout_pct": holdout_pct,
    }
