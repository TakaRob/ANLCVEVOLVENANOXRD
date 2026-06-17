Find an algorithm that reliably detects **214 known Bragg peak features** in single-exposure (1×1, un-binned) X-ray diffraction frames, using spatial profile verification with a Voigt distribution.

## 1. Background — what we already know

We have two proven algorithms that work on binned data. Your job is to combine them, adapt them for noisy 1×1 single-exposure frames, and improve from there.

**Algorithm 1 — Peak detection** (`tophat_band_adaptive_snr.py` in the data directory):
The best per-frame Bragg peak detector, evolved on 5×5 binned data. Pipeline:
1. `radial_median_subtract()` — robust radial background removal (most important step)
2. `fast_tophat(size=15)` — white top-hat morphological filter for compact features
3. `build_tth_band_masks(tth_tolerance=0.4)` — restrict search to narrow 2-theta bands
4. `detect_in_band(snr_threshold=4.0, min_pixels=3, max_pixels=150, min_compactness=0.12)` — per-band MAD thresholding + connected component analysis
5. Cross-band duplicate suppression

**Algorithm 2 — Spatial linking + shape verification** (the approach from `spatial_feature_analysis.py`, adapted in `baseline.py`):
After detecting peaks in individual frames, link them across the spatial grid and verify shape:
1. **Union-Find linking**: Each detection is a node `(bin_key, peak_index, row, col, x, y)`. Detections in adjacent frames within `link_tolerance=5` pixels are unioned. Connected components = features.
2. **Profile shape check**: For each feature, measure intensity vs. distance from center. On 3×3 data this used a Gaussian monotonicity check. For 1×1 data, use a **Voigt profile** fit (pseudo-Voigt: `V(r) = amp * [η*L(r,γ) + (1-η)*G(r,σ)]`).

These two algorithms already produced the 214 validated features when run on 3×3 data. Your starting point is to combine them for 1×1 data, then creatively improve.

## 2. Your starting architecture

**Start by combining Algorithm 1 + Algorithm 2 directly.** Read both files in the data directory. Then adapt:

### Stage 1: Use `tophat_band_adaptive_snr.py` as-is, but lower thresholds

The 5×5 detector used `snr_threshold=4.0` because binned data has high SNR. For noisy 1×1 frames, **lower it to 2.0-3.0**. Also relax `min_pixels` to 2 and `tth_tolerance` to 0.5. This is just a candidate generator — be aggressive, let Stage 3 do the filtering.

Write this as a standalone function: `detect_peaks_in_frame(image, tth_map, reflections, params) -> list[dict]`

### Stage 2: Use Union-Find linking from `baseline.py`

Same approach — link detections at matching (x,y) positions across neighboring frames. `link_tolerance=5` pixels worked on 3×3 and should work on 1×1.

Write this as a standalone function: `link_peaks_across_frames(all_detections, grid_info) -> list[Feature]`

### Stage 3: Upgrade Gaussian check to Voigt profile fitting

The original used `check_gaussian_profile()` which just checked monotonic intensity decrease. Replace with proper Voigt fitting:
- Fit intensity vs. spatial distance using pseudo-Voigt
- R² threshold ~0.3-0.5
- Reject features with < 3 bins (can't fit a profile)
- Reject flat profiles (CV < 0.05)

Write this as a standalone function: `verify_voigt_profiles(features, params) -> list[Feature]`

## 3. KEEP STAGES ISOLATED

**Do NOT merge stages into a monolithic function.** Each stage must be a separate function with a clear interface:

- **Stage 1 output** → list of candidate peaks per frame: `[(x, y, reflection, snr, ...), ...]`
- **Stage 2 output** → list of linked features: `[(feature_id, [(bin_key, x, y, intensity), ...]), ...]`
- **Stage 3 output** → filtered list of spatially-validated features

This separation lets each stage be tuned independently. When you improve Stage 3's Voigt fitting, you don't touch Stage 1. When you lower Stage 1's SNR threshold, Stage 3 still works unchanged.

## 4. Then improve creatively

Once the baseline combination works, look for improvements. Ideas to explore:

- **Stack-then-detect**: Sum a spatial neighborhood of frames before detection (gives ~5-25× SNR boost), then verify profiles on individual frames. A previous worker found this gave F1=0.591 and eliminated Union-Find entirely. If you try this, still keep detection and verification as separate functions.
- **Multi-scale top-hat**: Different kernel sizes catch peaks of different sizes.
- **Weighted Voigt fitting**: Weight bins by their detection confidence.
- **Adaptive link tolerance**: Vary by reflection or spatial region.
- **Intensity-aware filtering**: Use the feature catalog to understand typical peak intensities per reflection.

## 5. The 214 known features

The file `feature_catalog.json` in the data directory contains all 214 features. Each has:

```json
{
  "feature_id": 1,
  "reflection": "(112)",
  "detector_x": 109, "detector_y": 49,
  "peak_intensity": 49.0, "mean_snr": 24.93,
  "n_bins": 4,
  "spatial_extent": ["0_0", "0_1", "0_2", "0_3"],
  "center_bin": "0_0", "center_row": 0, "center_col": 0,
  "intensity_profile": { ... }
}
```

Key fields:
- **`detector_x`, `detector_y`**: Where the peak appears on the detector (pixel coordinates).
- **`center_bin`**: The 3×3 spatial bin where the peak is brightest. In 1×1 coordinates: `(center_row*3+1, center_col*3+1)`.
- **`n_bins`**: How many 3×3 bins the feature spans (2-730). In 1×1, features span ~9× more frames.
- **`intensity_profile`**: Per-bin intensity — this is the shape your Voigt verification must match.

The features span 7 reflections: (002)=63, (001)=42, ITO=41, (012)=33, (111)=23, (112)=7, (011)=5.

Use the catalog to understand what features look like, but **do not hardcode positions** — your algorithm must work generically.

## 6. Data

### Frame images

All ~25,170 single-exposure frames are in: **`/home/takaji/xrd_1x1_bins.h5`**

Keyed by grid position (e.g., `"1_1"`, `"30_50"`), each dataset is float32 shape (1062, 1028).

```python
import h5py
def load_frame(bin_key, path="/home/takaji/xrd_1x1_bins.h5"):
    with h5py.File(path, "r") as f:
        return f[bin_key][:].astype(np.float64)
```

### Files in the data directory

- **`tophat_band_adaptive_snr.py`**: **USE THIS** — best per-frame detector from 5×5 optimization. Read it, understand the functions, adapt thresholds for 1×1.
- **`baseline.py`**: **USE THIS** — 3-stage baseline combining detection + linking + Voigt verification. Read it for the spatial linking and Voigt fitting code.
- **`tth.tiff`**: Per-pixel 2-theta map (1062×1028). Same for all frames.
- **`reflections.py`**: 2-theta values and names of the reflections.
- **`grid_mapping.json`**: Grid metadata — 156 rows × 221 cols.
- **`feature_catalog.json`**: The 214 known features (full detail).
- **`noise_reduction_algorithms.py`**: Radial background models.

### Important notes

- Detector shape: 1062 × 1028 pixels.
- Edge pixels (first/last 2-3 rows/cols) have artifacts.
- 1×1 frames are noisy (single exposure, ~5-25× less signal than 5×5 bins).

## 7. Evaluation

Your algorithm receives a **center bin** key (e.g., `"4_4"`). It must:
1. Detect peaks in the center frame and neighboring frames
2. Link detections across frames
3. Apply Voigt profile verification
4. Output spatially-validated peaks for the **center bin only**

**Matching**: Detection within **40 pixels** of a known feature position = match.

**Metric**: Per-bin F1 score averaged across evaluation bins.

**Priority**: **Recall is critical.** These are known, validated features. Missing a known feature is worse than a false positive.

**Subprocess timeout**: Set to at least **300 seconds** (5 minutes) per bin.

## 8. IO format

```bash
python algorithm.py \
    --center-bin 4_4 \
    --bins-h5 /home/takaji/xrd_1x1_bins.h5 \
    --two-theta tth.tiff \
    --reflections reflections.py \
    --grid-mapping grid_mapping.json \
    --output detections.csv \
    --spatial-radius 5
```

Output CSV: `reflection`, `x`, `y`. If `--labels` is provided, compute and print F1.

## 9. Tips

- **Read both reference algorithms first** — `tophat_band_adaptive_snr.py` and `baseline.py`. Understand every function before writing new code.
- **Stage 1 matters least.** It's a candidate generator. Use SNR 2.0-3.0. Don't optimize it.
- **Stage 3 (Voigt verification) is where to spend effort.** This is the precision filter.
- **Spatial radius of 4-6** is good. Too small = can't build profiles. Too large = slow.
- **Voigt R² threshold**: 0.3-0.5 is a good range. Too lenient (<0.2) keeps noise.
- **Link tolerance**: 3-7 pixels for matching across frames.
- Many bins with known features have only 1-2 peaks. Some have up to 7.
