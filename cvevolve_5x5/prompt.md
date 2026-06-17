Find a robust and accurate algorithm that can detect Bragg peaks (bright spots) in spatially-binned X-ray crystal diffraction (XRD) images.

## 1. Problem statement

An X-ray diffraction experiment scans a sample with a focused X-ray beam across a 2D spatial grid. At each grid position, a 2D detector records a diffraction pattern. When the sample contains crystalline material, Bragg peaks appear as bright spots at positions determined by the crystal lattice spacing (characterized by 2-theta angle) and orientation.

**Spatial binning.** Rather than analyzing each single-exposure frame independently (which is noisy), nearby frames are grouped into spatial bins and summed. A **5×5 binning** groups 5 rows × 5 columns of adjacent grid positions, summing all ~25 frames in each bin into one image. This reduces noise while preserving spatial variation across the sample. There are **1188 bins** total (32 bin-rows × 45 bin-cols), each producing a 1062×1028 pixel image.

Your task is to design an algorithm that, given one of these binned XRD images, a map indicating the 2-theta value at every pixel, and a list of lattice planes (reflections) to look for, identifies and locates the Bragg peaks. The algorithm must work reliably across all bins — some bins have many bright peaks, some have faint peaks barely above background, and some have no peaks at all.

## 2. Key physics

Different lattice planes deflect X-rays by different 2-theta angles. On the detector, pixels at the same 2-theta form an arc (since 2-theta corresponds to radial distance from the beam center). When a crystal grain satisfies the Bragg condition, it produces a bright spot somewhere along the arc for that plane.

The dominant background structure is **radial**: intensity varies smoothly with 2-theta (radial distance from beam center) but is roughly uniform along arcs (azimuthally). This makes radial profile subtraction an effective noise reduction strategy — fit or smooth the median intensity vs. 2-theta curve, map it back to 2D, and subtract.

Peaks vary widely in brightness. Some are strong and obvious; others are faint and require careful background subtraction to distinguish from noise. The algorithm should handle both extremes.

## 3. Data

### Pre-built bin images

All 1188 bin images are pre-computed and stored in a single HDF5 file:

**`/home/takaji/xrd_5x5_bins.h5`**

Each dataset is keyed by bin identifier (e.g., `"0_0"`, `"3_7"`) and contains a float32 array of shape (1062, 1028). Bad pixels have already been clamped (negatives → 0, values > 1e9 → 0).

```python
import h5py
import numpy as np

def load_bin_image(bin_key, bins_h5_path="/home/takaji/xrd_5x5_bins.h5"):
    with h5py.File(bins_h5_path, "r") as f:
        return f[bin_key][:].astype(np.float64)
```

To list all available bins:
```python
with h5py.File("/home/takaji/xrd_5x5_bins.h5", "r") as f:
    bin_keys = list(f.keys())  # 1188 keys like "0_0", "0_1", ...
```

### Files in the data directory

- **`tth.tiff`**: Per-pixel 2-theta map (1062×1028 float). Same geometry for all bins.
- **`reflections.py`**: The 2-theta values (`degs`) and names (`deg_labels`) of the valid reflections in the sample.
- **`baseline.py`**: A baseline algorithm that demonstrates data loading, noise reduction, peak detection, and evaluation. **Use this as your starting reference and try to outperform it.** It applies radial Gaussian background subtraction followed by global percentile thresholding.
- **`noise_reduction_algorithms.py`**: A library of four radial background models (Gaussian, Split Gaussian, Skewed Gaussian, Fourier low-pass) that fit the median-intensity-vs-2-theta profile and subtract it. Imported by the baseline. You may use, modify, or replace these.
- **`annotations_summed.json`**: Ground truth annotations for a **fully-summed image** (all 25,170 frames summed into one). This is provided purely as context — it shows what peaks look like when signal-to-noise is very high. Your algorithm must work on individual per-bin images, not this summed image.

### Important notes about the images

- The detector shape is 1062 rows × 1028 columns.
- Setting visualization percentile range to 10–99 is recommended. Use `np.log1p()` for better dynamic range when viewing.
- Edge pixels (first/last 2 rows and columns) often have artifacts and should be ignored during detection.

## 4. Evaluation

Your algorithm must take as input a bin key (e.g., `"3_7"`), the 2-theta map, and the list of reflections. It should output detected peak locations as `(x, y)` pixel coordinates with reflection labels.

**Matching criterion**: A detected point within **40 pixels** of a ground truth point counts as a match.

**Metric**: F1 score computed across all detected and ground truth points. The goal is to maximize recall (find all true peaks) while minimizing false positives.

**Multi-bin evaluation**: The primary evaluation runs across many bins and averages the F1 score. Some bins have no peaks — the algorithm should correctly return no detections for those.

When evaluating the baseline, use default settings (`--noise-algorithm gaussian --percentile 97.0`).

## 5. IO format

Your submitted algorithm script should accept:
- `--bin-key`: The bin identifier (e.g., `"3_7"`)
- `--bins-h5`: Path to the pre-built bin images HDF5 (default: `/home/takaji/xrd_5x5_bins.h5`)
- `--two-theta`: Path to `tth.tiff`
- `--reflections`: Path to `reflections.py`
- `--output`: Output CSV file path

The output CSV should have three columns: `reflection`, `x`, `y`. Each row is a detected peak.

If `--labels` is provided (path to a JSON file with ground truth for that bin), compute and print the F1 score.

## 6. Tips

- The **radial background** is the dominant noise source. Subtracting it well is crucial. The provided noise reduction models work, but there may be better approaches (adaptive thresholding, wavelet decomposition, morphological operations, etc.).
- Consider **adaptive thresholding** rather than a fixed global percentile — different bins have different signal levels.
- **Local contrast** measures (e.g., comparing a pixel to its local neighborhood) can help distinguish faint peaks from background fluctuations.
- The 2-theta map tells you where to expect peaks of each reflection. Consider using this as a prior.
- Some bins contain strong **Debye-Scherrer ring** segments (continuous arcs of intensity from polycrystalline material) — these are NOT Bragg peaks and should not be detected. They tend to be diffuse, while Bragg peaks are compact.
- Loading is fast: `h5py.File('/home/takaji/xrd_5x5_bins.h5')['3_7'][:]` takes milliseconds. No need to cache.
