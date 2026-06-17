Evaluate the submitted algorithm against the holdout validation set of spatially-binned XRD images.

## Holdout data directory contents

- **`bin_annotations.json`**: Ground truth annotations for bins that contain peaks. Format:
  ```json
  {
    "0_0": {
      "(001)": [[842, 972], [855, 557]],
      "(111)": [[388, 1017], [374, 752], [467, 324]],
      ...
    },
    ...
  }
  ```
  Keys are bin identifiers (`"row_col"`), values are dicts mapping reflection names to lists of `[x, y]` pixel coordinates.

- **`empty_bins.json`**: A list of bin keys that were reviewed and confirmed to contain **no peaks**. A correct algorithm should return zero detections for these bins.

- **`tth.tiff`**: The per-pixel 2-theta map (same as test data).

- **`reflections.py`**: Reflection 2-theta values and labels (same as test data).

- **`noise_reduction_algorithms.py`**: Noise reduction library (same as test data).

## Loading bin images

All bin images are pre-computed in a single HDF5 file at **`/home/takaji/xrd_4x4_bins.h5`**. Load any bin with:

```python
import h5py
import numpy as np

def load_bin_image(bin_key, bins_h5_path="/home/takaji/xrd_4x4_bins.h5"):
    with h5py.File(bins_h5_path, "r") as f:
        return f[bin_key][:].astype(np.float64)
```

## Evaluation procedure

1. For each bin in `bin_annotations.json`, plus the bins in `empty_bins.json`:
   a. Load the bin image from `/home/takaji/xrd_4x4_bins.h5`.
   b. Run the submitted algorithm to produce detections.
   c. Compare detections against ground truth annotations for that bin.

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
    --bin-key <key> \
    --bins-h5 /home/takaji/xrd_4x4_bins.h5 \
    --two-theta <holdout_dir>/tth.tiff \
    --reflections <holdout_dir>/reflections.py \
    --output <output_csv>
```

Loop over all holdout bins, collect per-bin F1 scores, and compute the mean.

## Important notes

- The holdout bins span a variety of signal levels: some have many strong peaks, some have a single faint peak, and some have no peaks at all.
- Do not train or tune hyperparameters on this holdout set. It is for final evaluation only.
- Note: holdout annotations start empty and will be populated via the labeling GUI.
