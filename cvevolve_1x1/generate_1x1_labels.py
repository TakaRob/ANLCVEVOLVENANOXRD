"""
Generate 1x1 evaluation labels from 3x3 bin annotations.

Each 3x3 bin covers a 3x3 block of 1x1 frames. A peak annotated in a
3x3 bin (at a given detector x,y) must exist in at least some of the
constituent 1x1 frames. We assign the annotation to the CENTER 1x1 frame
of each 3x3 block, since the peak is most likely strongest there.

This produces:
  - Per-bin label JSON files in test_data/labels/ and holdout_data/labels/
  - bin_annotations.json and empty_bins.json for each split

Usage:
    python generate_1x1_labels.py
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ANN_3X3_HOLDOUT = BASE / "cvevolve_3x3" / "holdout_data" / "bin_annotations.json"
ANN_3X3_TEST = BASE / "cvevolve_3x3" / "test_data" / "annotations_summed.json"
GRID_1X1 = None  # will be generated

BIN_SIZE_3X3 = 3


def map_3x3_to_1x1_center(bin_key_3x3):
    """Map a 3x3 bin key to the center 1x1 frame."""
    r3, c3 = int(bin_key_3x3.split("_")[0]), int(bin_key_3x3.split("_")[1])
    r1 = r3 * BIN_SIZE_3X3 + BIN_SIZE_3X3 // 2
    c1 = c3 * BIN_SIZE_3X3 + BIN_SIZE_3X3 // 2
    return f"{r1}_{c1}"


def map_3x3_to_all_1x1(bin_key_3x3):
    """Map a 3x3 bin key to ALL constituent 1x1 frames."""
    r3, c3 = int(bin_key_3x3.split("_")[0]), int(bin_key_3x3.split("_")[1])
    frames = []
    for dr in range(BIN_SIZE_3X3):
        for dc in range(BIN_SIZE_3X3):
            r1 = r3 * BIN_SIZE_3X3 + dr
            c1 = c3 * BIN_SIZE_3X3 + dc
            frames.append(f"{r1}_{c1}")
    return frames


def generate_labels(ann_3x3_path, output_dir, label_name):
    """Generate 1x1 labels from 3x3 annotations."""
    with open(ann_3x3_path) as f:
        ann_3x3 = json.load(f)

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    bin_annotations_1x1 = {}
    all_1x1_bins_with_labels = set()

    for bin_key_3x3, reflections in ann_3x3.items():
        if bin_key_3x3.startswith("__"):
            continue
        if not isinstance(reflections, dict):
            continue

        center_1x1 = map_3x3_to_1x1_center(bin_key_3x3)

        label_data = {}
        for ref_name, points in reflections.items():
            if ref_name.startswith("__"):
                continue
            if isinstance(points, list) and len(points) > 0:
                label_data[ref_name] = points

        if label_data:
            bin_annotations_1x1[center_1x1] = label_data
            all_1x1_bins_with_labels.add(center_1x1)

            label_file = labels_dir / f"{center_1x1}.json"
            with open(label_file, "w") as f:
                json.dump(label_data, f, indent=2)

    ann_path = output_dir / "bin_annotations.json"
    with open(ann_path, "w") as f:
        json.dump(bin_annotations_1x1, f, indent=2)

    print(f"  {label_name}: {len(bin_annotations_1x1)} bins with annotations")
    total_pts = sum(
        sum(len(pts) for pts in v.values())
        for v in bin_annotations_1x1.values()
    )
    print(f"  Total annotation points: {total_pts}")

    empty_path = output_dir / "empty_bins.json"
    with open(empty_path, "w") as f:
        json.dump([], f)

    return bin_annotations_1x1


def main():
    print("Generating 1x1 evaluation labels from 3x3 annotations")
    print("=" * 60)

    holdout_dir = BASE / "cvevolve_1x1" / "holdout_data"
    test_dir = BASE / "cvevolve_1x1" / "test_data"

    print("\nHoldout data:")
    generate_labels(ANN_3X3_HOLDOUT, holdout_dir, "holdout")

    if ANN_3X3_TEST.exists():
        ann_test = json.load(open(ANN_3X3_TEST))
        has_bins = any(
            "_" in k for k in ann_test.keys()
            if not k.startswith("__") and isinstance(ann_test[k], list)
        )
        if not has_bins:
            test_ann_path = BASE / "cvevolve_3x3" / "test_data" / "annotations_summed.json"
            holdout_ann = BASE / "cvevolve_3x3" / "holdout_data" / "bin_annotations.json"
            print("\nTest data (using holdout annotations as test split):")
            generate_labels(holdout_ann, test_dir, "test")
        else:
            print("\nTest data:")
            generate_labels(ANN_3X3_TEST, test_dir, "test")
    else:
        print("\nNo test annotations found, copying holdout as test")
        generate_labels(ANN_3X3_HOLDOUT, test_dir, "test")

    print("\nDone!")


if __name__ == "__main__":
    main()
