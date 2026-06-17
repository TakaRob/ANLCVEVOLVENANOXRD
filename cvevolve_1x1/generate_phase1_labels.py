"""
Generate Phase 1 evaluation labels from the 3x3 feature catalog.

Converts the 214 validated features into per-center-bin label files
for 1x1 CVEvolve evaluation. Each feature's 3x3 center_bin maps to
a 1x1 center frame at (row*3+1, col*3+1).

Splits features into test (70%) and holdout (30%) sets, keeping
features sharing the same center_bin in the same split.

Usage:
    python generate_phase1_labels.py
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEATURE_CATALOG = BASE / "results" / "scan203" / "feature_catalog_3x3.json"
CVEVOLVE_DIR = BASE / "cvevolve_1x1"

BIN_SIZE_3X3 = 3
TEST_FRACTION = 0.7
RANDOM_SEED = 42


def map_3x3_center_to_1x1(center_bin_3x3):
    r3, c3 = int(center_bin_3x3.split("_")[0]), int(center_bin_3x3.split("_")[1])
    r1 = r3 * BIN_SIZE_3X3 + BIN_SIZE_3X3 // 2
    c1 = c3 * BIN_SIZE_3X3 + BIN_SIZE_3X3 // 2
    return f"{r1}_{c1}"


def main():
    with open(FEATURE_CATALOG) as f:
        features = json.load(f)
    print(f"Loaded {len(features)} features from feature catalog")

    # Group features by their 1x1 center bin
    by_center = defaultdict(list)
    for feat in features:
        center_1x1 = map_3x3_center_to_1x1(feat["center_bin"])
        by_center[center_1x1].append(feat)

    print(f"Features map to {len(by_center)} unique 1x1 center bins")

    # Build per-bin annotations: {bin_key: {reflection: [[x,y], ...]}}
    all_annotations = {}
    for center_1x1, feats in by_center.items():
        ann = {}
        for feat in feats:
            ref = feat["reflection"]
            x, y = feat["detector_x"], feat["detector_y"]
            ann.setdefault(ref, []).append([x, y])
        all_annotations[center_1x1] = ann

    # Split into test/holdout by center bin
    center_keys = sorted(all_annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(center_keys)
    n_test = int(len(center_keys) * TEST_FRACTION)
    test_keys = set(center_keys[:n_test])
    holdout_keys = set(center_keys[n_test:])

    test_ann = {k: v for k, v in all_annotations.items() if k in test_keys}
    holdout_ann = {k: v for k, v in all_annotations.items() if k in holdout_keys}

    test_features = sum(
        sum(len(pts) for pts in v.values()) for v in test_ann.values()
    )
    holdout_features = sum(
        sum(len(pts) for pts in v.values()) for v in holdout_ann.values()
    )

    print(f"\nSplit:")
    print(f"  Test:    {len(test_ann)} bins, {test_features} feature points")
    print(f"  Holdout: {len(holdout_ann)} bins, {holdout_features} feature points")

    # Write test data
    test_dir = CVEVOLVE_DIR / "test_data"
    test_labels_dir = test_dir / "labels"
    if test_labels_dir.exists():
        shutil.rmtree(test_labels_dir)
    test_labels_dir.mkdir(parents=True)

    with open(test_dir / "bin_annotations.json", "w") as f:
        json.dump(test_ann, f, indent=2)

    for bk, ann in test_ann.items():
        with open(test_labels_dir / f"{bk}.json", "w") as f:
            json.dump(ann, f, indent=2)

    with open(test_dir / "empty_bins.json", "w") as f:
        json.dump([], f)

    # Write holdout data
    holdout_dir = CVEVOLVE_DIR / "holdout_data"
    holdout_labels_dir = holdout_dir / "labels"
    if holdout_labels_dir.exists():
        shutil.rmtree(holdout_labels_dir)
    holdout_labels_dir.mkdir(parents=True)

    with open(holdout_dir / "bin_annotations.json", "w") as f:
        json.dump(holdout_ann, f, indent=2)

    for bk, ann in holdout_ann.items():
        with open(holdout_labels_dir / f"{bk}.json", "w") as f:
            json.dump(ann, f, indent=2)

    with open(holdout_dir / "empty_bins.json", "w") as f:
        json.dump([], f)

    # Also write the full feature catalog as a reference file in test_data
    with open(test_dir / "feature_catalog.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"\nWrote labels to:")
    print(f"  {test_dir / 'bin_annotations.json'}")
    print(f"  {test_labels_dir}/ ({len(test_ann)} files)")
    print(f"  {holdout_dir / 'bin_annotations.json'}")
    print(f"  {holdout_labels_dir}/ ({len(holdout_ann)} files)")

    # Summary stats
    print(f"\nFeature distribution by reflection:")
    ref_counts = defaultdict(int)
    for feat in features:
        ref_counts[feat["reflection"]] += 1
    for ref, count in sorted(ref_counts.items(), key=lambda x: -x[1]):
        print(f"  {ref:>12}: {count}")


if __name__ == "__main__":
    main()
