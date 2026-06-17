Evaluate the submitted algorithm against the holdout validation set of single-exposure (1×1) XRD frames.

## Purpose

This is **Phase 1** evaluation: the algorithm must recover known Bragg peak features from the validated feature catalog. The ground truth comes from 214 features validated on 3×3 data, mapped to their corresponding 1×1 center frames.

## Holdout data directory contents

- **`bin_annotations.json`**: Ground truth annotations for bins that contain peaks. Format:
  ```json
  {
    "1_1": {
      "(001)": [[842, 972], [855, 557]],
      "(111)": [[388, 1017], [374, 752], [467, 324]],
      ...
    },
    ...
  }
  ```
  Keys are bin identifiers (`"row_col"` in the 1×1 grid), values are dicts mapping reflection names to lists of `[x, y]` pixel coordinates.

- **`empty_bins.json`**: A list of bin keys confirmed to contain **no peaks**. A correct algorithm should return zero detections for these bins.

- **`grid_mapping.json`**: Spatial grid metadata — `n_bin_rows` (156), `n_bin_cols` (221).

- **`tth.tiff`**: The per-pixel 2-theta map (same as test data).

- **`reflections.py`**: Reflection 2-theta values and labels (same as test data).

- **`noise_reduction_algorithms.py`**: Noise reduction library (same as test data).

- **`labels/`**: Directory of per-bin JSON label files (e.g., `1_1.json`), each containing the ground truth annotations for that center bin.

## Loading frame images

All frames are pre-computed in a single HDF5 file at **`/home/takaji/xrd_1x1_bins.h5`**. Load any frame with:

```python
import h5py
import numpy as np

def load_frame(bin_key, bins_h5_path="/home/takaji/xrd_1x1_bins.h5"):
    with h5py.File(bins_h5_path, "r") as f:
        return f[bin_key][:].astype(np.float64)
```

## Evaluation procedure

1. For each bin in `bin_annotations.json`, plus the bins in `empty_bins.json`:
   a. Run the submitted algorithm with that bin as the `--center-bin`.
   b. The algorithm should detect peaks in the center frame AND neighboring frames within a spatial radius, link them across frames, apply Voigt profile shape verification, and output only the spatially-validated peaks for the center bin.
   c. Compare detections against ground truth annotations for that center bin.

2. **Matching criterion**: A detected point within **40 pixels** (Euclidean distance) of a ground truth point counts as a true positive. Each ground truth point can match at most one detection, and vice versa.

3. **Per-bin F1 score**:
   - If a bin has no ground truth peaks and the algorithm correctly detects nothing: F1 = 1.0
   - If a bin has no ground truth peaks but the algorithm detects one or more: F1 = 0.0
   - Otherwise: F1 = 2 * precision * recall / (precision + recall)

4. **Aggregate metric**: The primary metric is the **mean F1 score** across all validation bins, weighted equally.

5. Print the aggregate F1 score as the metric value. Also print per-bin scores for debugging.

## Running the evaluation

The algorithm script should follow the IO format from the task prompt:
```bash
python <algorithm>.py \
    --center-bin <key> \
    --bins-h5 /home/takaji/xrd_1x1_bins.h5 \
    --two-theta <holdout_dir>/tth.tiff \
    --reflections <holdout_dir>/reflections.py \
    --grid-mapping <holdout_dir>/grid_mapping.json \
    --output <output_csv>
```

Loop over all holdout bins, collect per-bin F1 scores, and compute the mean.

**Subprocess timeout**: Set the subprocess timeout to at least **300 seconds** (5 minutes) per bin. Each evaluation processes ~121 neighboring frames (spatial_radius=5 → 11×11 grid), so individual bins may take 1-3 minutes.

## Important notes

- The holdout bins span a variety of signal levels: some have many strong peaks, some have a single faint peak, and some have no peaks at all.
- Do not train or tune hyperparameters on this holdout set. It is for final evaluation only.
- **Recall is critical.** These are known, validated features. Missing a known feature is worse than a false positive.
