"""Canonical JSON reflection sets used per scan or at project level."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

DEFAULT_WIDTH = 0.4  # ± degrees; the manual `width` drives the detection band

# This reserved label tells detectors that the single reflection has unlimited
# width. Detectors recognize the label and use an all-true mask regardless of
# the calibrated 2theta range.
WHOLE_FRAME_LABEL = "(no reflections)"
WHOLE_FRAME_WIDTH = 180.0

# Perovskite default reflection set used to seed a new project's reflections.json.
# The first 8 are labeled Bragg peaks / phase markers; the last 3 are named by
# their angle and can be renamed in the manual editor.
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


def whole_frame_reflections(label: str = WHOLE_FRAME_LABEL) -> List[dict]:
    """Return one reserved reflection with unlimited detector width."""
    return [{"name": label, "two_theta": 0.0, "width": WHOLE_FRAME_WIDTH}]


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


def save(reflections: List[dict], json_path) -> Path:
    """Write canonical reflection JSON."""
    return write_json(reflections, json_path)
