"""Reflection sets: JSON data + a generated ``reflections.py`` loader.

The GUI edits reflection data as JSON (``[{name, two_theta, width}, ...]``) and
never hand-edits Python. ``reflections.py`` is *generated* from that JSON and is
what the pipeline imports (it exposes ``degs``, ``deg_labels``, and ``widths``).
Reflections can be per-scan (different scans, different angles), so both files
live under ``Metadata/<scan>/`` (or project ``Metadata/`` as the default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

DEFAULT_WIDTH = 0.4  # ± degrees; the manual `width` drives the detection band

# Label shared by every tile of a whole-detector reflection set. A set whose
# entries all carry this one label OR-merges (in build_tth_band_masks) into a
# single band that spans the detector — the "no known reflections" workflow,
# expressed as an ordinary reflection set with no special-case code anywhere.
WHOLE_FRAME_LABEL = "(no reflections)"

# Perovskite default reflection set used to seed a new project's reflections.json
# (and the bundled assets/reflections.py fallback). The first 8 are the labeled
# Bragg peaks / phase markers; the last 3 had no label historically, so they are
# named by their angle (easy to rename in the manual editor).
DEFAULT_REFLECTIONS = [
    {"name": "PbI2",  "two_theta": 6.81319,  "width": DEFAULT_WIDTH},
    {"name": "(001)", "two_theta": 7.51422,  "width": DEFAULT_WIDTH},
    {"name": "(011)", "two_theta": 10.61748, "width": DEFAULT_WIDTH},
    {"name": "(111)", "two_theta": 13.00831, "width": DEFAULT_WIDTH},
    {"name": "(002)", "two_theta": 15.01266, "width": DEFAULT_WIDTH},
    {"name": "ITO",   "two_theta": 16.07224, "width": DEFAULT_WIDTH},
    {"name": "(012)", "two_theta": 16.79944, "width": DEFAULT_WIDTH},
    {"name": "(112)", "two_theta": 18.42549, "width": DEFAULT_WIDTH},
    {"name": "21.30", "two_theta": 21.29655, "width": DEFAULT_WIDTH},
    {"name": "22.60", "two_theta": 22.59817, "width": DEFAULT_WIDTH},
    {"name": "26.16", "two_theta": 26.16205, "width": DEFAULT_WIDTH},
]


def default_reflections() -> List[dict]:
    """A fresh copy of the default perovskite reflection set."""
    return [dict(r) for r in DEFAULT_REFLECTIONS]


def whole_frame_reflections(
    tth_map=None,
    *,
    label: str = WHOLE_FRAME_LABEL,
    spacing: float = 0.3,
    margin: float = 1.0,
    tth_min: float = 0.0,
    tth_max: float = 40.0,
) -> List[dict]:
    """Tile the 2θ range with entries that all share ``label``.

    Every entry uses the same ``label`` so ``build_tth_band_masks`` OR-merges them
    into one band covering the whole detector — an ordinary reflection set that
    lets the band-restricted detector search everything (for datasets with no
    known Bragg reflections). ``spacing`` 0.3° < the detector tolerance (0.4°) so
    the merged band is contiguous with no gaps.

    When a ``tth_map`` is given the range is clamped to its observed span (padded
    by ``margin``) to keep the tile count small; otherwise the fixed ``tth_min``…
    ``tth_max`` default is used (tiles off the detector just contribute empty
    masks, so it stays correct across recalibration).
    """
    lo, hi = float(tth_min), float(tth_max)
    if tth_map is not None:
        import numpy as np

        finite = np.asarray(tth_map, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo = max(0.0, float(finite.min()) - margin)
            hi = float(finite.max()) + margin
    step = float(spacing)
    if step <= 0:
        raise ValueError("spacing must be > 0")
    reflections = []
    deg = lo
    # inclusive of hi (fp-safe) so the top of the range is covered
    while deg <= hi + step / 2:
        reflections.append({"name": label, "two_theta": round(deg, 5), "width": DEFAULT_WIDTH})
        deg += step
    return reflections


def read_json(path) -> List[dict]:
    """Read a reflections.json (list of {name, two_theta, width})."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("reflections", [])


def write_json(reflections: List[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norm = [
        {
            "name": r["name"],
            "two_theta": round(float(r["two_theta"]), 5),
            "width": round(float(r.get("width", DEFAULT_WIDTH)), 4),
        }
        for r in reflections
    ]
    with open(path, "w") as f:
        json.dump(norm, f, indent=2)
    return path


def generate_py(reflections: List[dict], path) -> Path:
    """Write a ``reflections.py`` exposing degs / deg_labels / widths.

    Matches the loader contract used by ``core.io.load_reflections`` (which reads
    ``mod.degs`` and ``mod.deg_labels``); ``widths`` is additional so the detector
    band can follow each reflection's drawn width.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    degs = [round(float(r["two_theta"]), 5) for r in reflections]
    labels = [r["name"] for r in reflections]
    widths = [round(float(r.get("width", DEFAULT_WIDTH)), 4) for r in reflections]
    content = (
        "# Auto-generated by xrd-app from reflections.json — do not edit by hand.\n"
        f"degs = {degs}\n"
        f"deg_labels = {labels}\n"
        f"widths = {widths}\n"
    )
    path.write_text(content)
    return path


def save(reflections: List[dict], json_path, py_path=None) -> Path:
    """Write reflections.json and the sibling reflections.py loader."""
    jp = write_json(reflections, json_path)
    generate_py(reflections, py_path or jp.with_suffix(".py"))
    return jp
