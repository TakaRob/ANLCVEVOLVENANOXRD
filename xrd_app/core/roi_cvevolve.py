"""Build a CVEvolve session for ROI detection on fully summed images."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from . import reflection_sum, roi_catalog


def collect_examples(dm) -> list[dict]:
    """Collect scans with a sum image and at least one saved manual ROI."""
    examples = []
    for scan_dir in sorted(dm.labels_dir_root.glob("Scan_*")):
        scan = scan_dir.name
        catalogs = []
        for bin_size in (1, 2, 3, 4, 5):
            catalogs.extend(roi_catalog.discover(scan_dir, bin_size))
        features = []
        for path in catalogs:
            features.extend(roi_catalog.load(path).get("features", []))
        sum_path = reflection_sum.sum_path(dm, scan)
        if features and sum_path.exists():
            examples.append({"scan": scan, "sum_path": sum_path,
                             "rois": [f["manual_roi"] for f in features
                                      if f.get("manual_roi")]})
    return examples


def create_session(dm, dest=None, holdout_pct=20.0, seed=42) -> dict:
    """Create a self-contained ROI detector CVEvolve session from manual labels."""
    if not np.isfinite(holdout_pct) or not 0 <= holdout_pct < 100:
        raise ValueError("holdout_pct must be between 0 (inclusive) and 100 (exclusive)")
    examples = collect_examples(dm)
    if not examples:
        raise ValueError(
            "No training examples found. Save manual ROI > Shape features first; "
            "expected Labels/<scan>/*_roimap_NxN.h5 plus Metadata/<scan>/reflection_sum.npz.")
    dest = Path(dest or (dm.cvevolve_dir / "roi_summed_detection")).resolve()
    data = dest / "test_data"
    holdout = dest / "holdout_data"
    data.mkdir(parents=True, exist_ok=True)
    holdout.mkdir(parents=True, exist_ok=True)
    generated = ("Scan_*_reflection_sum.npz", "Scan_*_rois.json", "evaluate.py")
    for directory in (data, holdout):
        for pattern in generated:
            for path in directory.glob(pattern):
                if path.is_file() or path.is_symlink():
                    path.unlink()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(examples))
    n_holdout = min(len(examples) - 1,
                    max(1, int(round(len(examples) * holdout_pct / 100.0)))) \
        if len(examples) > 1 and holdout_pct > 0 else 0
    holdout_idx = set(order[:n_holdout].tolist())
    splits = {"test_data": [], "holdout_data": []}
    for index, example in enumerate(examples):
        target = holdout if index in holdout_idx else data
        scan = example["scan"]
        shutil.copy2(example["sum_path"], target / f"{scan}_reflection_sum.npz")
        label_path = target / f"{scan}_rois.json"
        label_path.write_text(json.dumps({"scan": scan, "rois": example["rois"]}, indent=2))
        splits[target.name].append(scan)

    baseline = Path(__file__).resolve().parent.parent / "ROIAlgorithms" / "baseline.py"
    shutil.copy2(baseline, data / "baseline.py")
    evaluate = _evaluate_script()
    (data / "evaluate.py").write_text(evaluate)
    if n_holdout:
        (holdout / "evaluate.py").write_text(evaluate)

    config = {
        "name": "roi_summed_detection",
        "num_workers_generate": 2,
        "num_workers_tune": 2,
        "model": {"model_name": "claudeopus46", "api_key_env_var": "ARGO_API_KEY",
                  "api_base": "https://apps.inside.anl.gov/argoapi/v1", "max_retries": 15},
        "workspace": {"root_dir": str(dest / "sessions"), "data_dir": str(data),
                      "holdout_data_dir": str(holdout),
                      "require_dangerous_command_approval": True},
        "metric": {"name_hint": "mean F2 score for summed-image ROI detection",
                   "direction_hint": "maximize", "target_value": None,
                   "description_hint": "Match detected ROI centers to manually labeled ROI centers within 25 detector pixels. Primary metric is mean F2 (beta=2), prioritizing recall while penalizing excessive proposals."},
        "branching": {"warmup_rounds": 3, "tune_every": 3, "evolve_every": 2,
                      "min_scored_for_tune": 1, "min_scored_for_evolve": 2},
        "stopping": {"max_rounds": 25, "patience_rounds": 12, "min_improvement": 0.0},
    }
    with open(dest / "config.yaml", "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    prompt = _prompt()
    (dest / "prompt.md").write_text(prompt)
    (dest / "holdout_test_prompt.md").write_text(prompt)
    return {"dest": dest, "examples": len(examples), "splits": splits}


def _prompt():
    return """# Fully summed detector ROI detection

Evolve `baseline.py` to detect manually labeled feature ROIs on a scan's fully
summed detector image. The candidate must define:

```python
def detect_rois(image, sensitivity=4.0, min_distance=12, max_rois=200):
    return [{"roi": (x0, y0, x1, y1), "score": float}, ...]
```

Use only the summed image; no reflection labels or 2-theta restriction are
required. Favor recall of manual ROIs, but suppress duplicate and diffuse
background candidates. `evaluate.py --candidate candidate.py` reports mean F2
(primary), precision, recall, and F1. Final evaluation must use all scans.
"""


def _evaluate_script():
    return '''#!/usr/bin/env python3
import argparse, importlib.util, json
from pathlib import Path
import numpy as np


def load_module(path):
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def score(pred, truth, tolerance=25.0):
    p = [((r[0]+r[2])/2, (r[1]+r[3])/2) for r in pred]
    t = [((r["x0"]+r["x1"])/2, (r["y0"]+r["y1"])/2) for r in truth]
    used=set(); tp=0
    for tx,ty in t:
        choices=[(i,(px-tx)**2+(py-ty)**2) for i,(px,py) in enumerate(p) if i not in used]
        if choices:
            i,d=min(choices,key=lambda z:z[1])
            if d <= tolerance*tolerance: used.add(i); tp += 1
    precision=tp/max(len(p),1); recall=tp/max(len(t),1)
    f1=2*precision*recall/max(precision+recall,1e-12)
    f2=5*precision*recall/max(4*precision+recall,1e-12)
    return precision,recall,f1,f2


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--candidate",required=True); args=ap.parse_args()
    root=Path(__file__).parent; module=load_module(args.candidate); rows=[]
    for labels in sorted(root.glob("Scan_*_rois.json")):
        data=json.loads(labels.read_text()); scan=data["scan"]
        image=np.load(root/f"{scan}_reflection_sum.npz")["image"].astype(float)
        raw=module.detect_rois(image); pred=[list(x.get("roi",x)) for x in raw]
        rows.append(score(pred,data["rois"]))
    mean=np.mean(rows,axis=0) if rows else np.zeros(4)
    print(f"precision={mean[0]:.6f} recall={mean[1]:.6f} f1={mean[2]:.6f} f2={mean[3]:.6f}")
    print(f"METRIC {mean[3]:.6f}")
if __name__ == "__main__": main()
'''
