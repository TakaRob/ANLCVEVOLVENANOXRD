"""ROI candidates detected on a fully summed detector image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io


def detect(image, algorithm_path, **params) -> list[dict]:
    """Run a summed-image ROI detector and normalize its rectangle output."""
    module = io.load_module(algorithm_path)
    if not hasattr(module, "detect_rois"):
        raise TypeError(f"ROI algorithm must define detect_rois(image, **params): {algorithm_path}")
    h, w = np.asarray(image).shape
    results = []
    for candidate in module.detect_rois(np.asarray(image, dtype=float), **params) or []:
        if isinstance(candidate, dict):
            roi = candidate.get("roi") or [candidate.get(k) for k in ("x0", "y0", "x1", "y1")]
            score = candidate.get("score")
        else:
            roi, score = candidate, None
        try:
            x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
        except (TypeError, ValueError):
            continue
        x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
        y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        entry = {"roi": (x0, y0, x1, y1)}
        if score is not None:
            entry["score"] = float(score)
        results.append(entry)
    return results


def default_algorithm() -> Path:
    return Path(__file__).resolve().parent.parent / "ROIAlgorithms" / "baseline.py"


def discover_algorithms(dm) -> list[dict]:
    """Bundled baseline plus evolved algorithms from the ROI CVEvolve session."""
    import json
    library = Path(__file__).resolve().parent.parent / "ROIAlgorithms"
    algorithms = [
        {"name": "Wieghold conservative (P=0.992, F1=0.797)",
         "file": str(library / "wieghold_peak_conservative.py"),
         "source": "manual F1 training", "default_sensitivity": 0.65},
        {"name": "Wieghold balanced (P=0.946, F1=0.834)",
         "file": str(library / "wieghold_peak_balanced.py"),
         "source": "manual F1 training", "default_sensitivity": 0.50},
        {"name": "Wieghold very conservative (P=0.991, F1=0.780)",
         "file": str(library / "wieghold_peak_very_conservative.py"),
         "source": "manual F1 training", "default_sensitivity": 0.80},
        {"name": "Baseline summed-image detector",
         "file": str(default_algorithm()), "source": "bundled", "default_sensitivity": 4.0},
    ]
    data_dir = dm.cvevolve_dir / "roi_summed_detection" / "test_data"
    manifest = data_dir / "top_algorithms.json"
    if not manifest.exists():
        return algorithms
    try:
        entries = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return algorithms
    for entry in entries if isinstance(entries, list) else []:
        path = data_dir / entry.get("file", "")
        if not path.is_file():
            continue
        score = entry.get("holdout_f2", entry.get("holdout_f1"))
        suffix = f" F2={score:.3f}" if isinstance(score, (int, float)) else ""
        algorithms.append({"name": f"{entry.get('name', path.stem)} ({entry.get('source', 'CVEvolve')}{suffix})",
                           "file": str(path), "source": entry.get("source", "CVEvolve"),
                           "default_sensitivity": entry.get("sensitivity", 0.5)})
    return algorithms
