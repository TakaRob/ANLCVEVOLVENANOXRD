Find a robust and accurate algorithm that detects Bragg peaks in single-exposure (1×1, un-binned) X-ray diffraction frames AND validates them by checking that their spatial intensity profile across neighboring frames follows a Voigt distribution.

## 1. Problem statement

An X-ray diffraction experiment scans a sample with a focused X-ray beam across a 2D spatial grid. At each grid position, a 2D detector records a diffraction pattern. When the sample contains crystalline material, Bragg peaks appear as bright spots at positions determined by the crystal lattice spacing (characterized by 2-theta angle) and orientation.

**This task uses 1×1 data — single exposures with NO spatial binning.** This means each "bin" is a single detector frame, and frames are very noisy compared to binned data. There are **~34,000 frames** arranged on a 156-row × 221-column spatial grid (raster scan).

### Why spatial verification matters

A focused X-ray beam illuminates a ~1 μm spot. As the beam rasters across a crystal grain, the Bragg peak intensity rises, reaches a maximum at the grain center, and falls off — producing a characteristic spatial intensity profile. Real peaks appear in **clusters of neighboring frames**. Noise spikes appear in isolated single frames.

The spatial profile is a **Voigt distribution** (not a pure Gaussian). This is because:
- **Gaussian component (σ)**: arises from strain distribution in the crystal and the beam profile
- **Lorentzian component (γ)**: arises from finite crystal size (Scherrer broadening) and mosaicity

The Voigt profile is the convolution of Gaussian and Lorentzian. A good pseudo-Voigt approximation is:
```
V(r) = amplitude * [η * L(r, γ) + (1 - η) * G(r, σ)]
```
where η depends on the relative widths of the Gaussian and Lorentzian components.

### Full pipeline — multi-stage with intentional over-detection

Your algorithm must implement THREE stages as separate logical steps. The key design philosophy is: **Stage 1 should be intentionally aggressive/sensitive** (low thresholds, high recall), because Stage 3 will filter out false positives. It is much better to detect too many candidates and filter them spatially than to miss faint real peaks with a strict per-frame threshold.

**Stage 1: Per-frame peak detection (aggressive candidate generator).** Given a single-frame image, detect ALL candidate Bragg peak positions. **This stage is NOT the final arbiter — it is just a candidate generator.** Use low SNR thresholds (2.0-3.0), permissive size/compactness filters, wider 2-theta band tolerances, and cast a very wide net. False positives at this stage are completely acceptable and expected — the spatial verification in Stage 3 is what enforces precision. Missing a real peak here means it is permanently lost from the pipeline. Therefore: **err heavily on the side of over-detection.** Do not spend optimization effort on making the per-frame detector precise. Spend your effort on making Stage 3's shape verification as accurate as possible.

**Stage 2: Spatial linking.** Link detections at the same detector (x, y) position across neighboring spatial frames. Two detections in adjacent frames (within `link_tolerance` pixels at the same detector position) are the same physical peak observed from neighboring beam positions. Use Union-Find or equivalent clustering.

**Stage 3: Voigt profile shape verification (selective).** For each linked feature, examine how the peak intensity varies across the spatial grid. Fit a Voigt profile to intensity vs. spatial distance from the feature center. Keep features whose spatial intensity profile matches a Voigt shape. Reject isolated detections and features with flat or random spatial profiles. This stage is where precision is enforced — it should be the strict filter that the pipeline relies on for accuracy.

The stages can be separate functions, separate scripts, or a single integrated pipeline. What matters is that the detection is sensitive and the spatial verification is rigorous.

## 2. Prior art — what worked on binned data

### Best peak detection algorithm (from 5×5 binned CVEvolve optimization)

The following pipeline achieved the best F1 scores on 5×5 and 3×3 binned data. It is included as `tophat_band_adaptive_snr.py` for reference. Key ideas:

1. **Radial median background subtraction**: Compute median intensity in narrow 2-theta annular bins. Smooth the median profile. Subtract to remove the radially-symmetric background. This is the single most important step — the dominant noise source is radial.

2. **White top-hat morphological filter**: `image - opening(image)` using min/max filters with a square kernel (size ~13-15 pixels). Extracts compact bright features while suppressing broad/diffuse intensity.

3. **2-theta band restriction**: For each reflection, only search within a narrow band around its known 2-theta value (±0.4°). This dramatically reduces false positives by constraining where peaks can appear.

4. **Per-band adaptive thresholding**: Within each 2-theta band, compute the median and MAD (median absolute deviation). Set threshold at `median + SNR_threshold × 1.4826 × MAD`. This adapts to the local noise level in each band.

5. **Connected component analysis**: Label connected pixels above threshold. Filter by size (min_pixels=3, max_pixels=150), compactness (aspect_ratio × fill_ratio > 0.12). Compute intensity-weighted centroid.

6. **Cross-band duplicate suppression**: Sort all peaks by SNR, keep highest-SNR peak when two are within `dup_distance` pixels.

**Important for 1×1**: Since single frames have ~5-25× less signal than 5×5 bins, you will likely need to use a lower SNR threshold (e.g., 2.5-3.5 instead of 4.0) to catch faint peaks. The spatial verification in Stage 3 compensates by filtering false positives.

### Spatial feature analysis (what worked for shape verification on 3×3 data)

The spatial feature analysis used Union-Find to link peaks across neighboring bins:
- Each detection is a node: `(bin_key, peak_index, row, col, x, y, peak_dict)`
- Two detections in adjacent bins (8-connected neighbors) with detector positions within `link_tolerance` pixels are unioned
- Connected components form "features" — the same physical Bragg peak observed across multiple spatial positions

For filtering on 3×3 data, a Gaussian profile check was used (checking if intensity decreases monotonically with distance from center). **For 1×1 data, use a Voigt profile** instead, as the physical peak shape includes both Gaussian and Lorentzian components.

## 3. Data

### Pre-built frame images

All ~34,000 single-exposure frames are stored in a single HDF5 file:

**`/home/takaji/xrd_1x1_bins.h5`**

Each dataset is keyed by grid position (e.g., `"0_0"`, `"30_50"`) and contains a float32 array of shape (1062, 1028). Bad pixels have been clamped (negatives → 0, values > 1e9 → 0).

```python
import h5py
import numpy as np

def load_frame(bin_key, bins_h5_path="/home/takaji/xrd_1x1_bins.h5"):
    with h5py.File(bins_h5_path, "r") as f:
        return f[bin_key][:].astype(np.float64)
```

### Files in the data directory

- **`tth.tiff`**: Per-pixel 2-theta map (1062×1028 float). Same geometry for all frames.
- **`reflections.py`**: The 2-theta values (`degs`) and names (`deg_labels`) of the valid reflections.
- **`grid_mapping.json`**: Spatial grid metadata — `n_bin_rows`, `n_bin_cols`, and the mapping from bin keys to frame indices.
- **`baseline.py`**: A baseline algorithm implementing all 3 stages. **Use this as your starting reference and try to outperform it.** It applies per-frame detection, Union-Find spatial linking, and Voigt profile fitting.
- **`noise_reduction_algorithms.py`**: Library of radial background models. You may use, modify, or replace these.
- **`tophat_band_adaptive_snr.py`**: The best per-bin detection algorithm from the 5×5 optimization. Use as reference for Stage 1.
- **`annotations_summed.json`**: Ground truth for a fully-summed image (context only).

### Important notes

- Detector shape: 1062 rows × 1028 columns.
- For visualization: percentile range 10–99, use `np.log1p()` for dynamic range.
- Edge pixels (first/last 2-3 rows and columns) often have artifacts.
- Loading is fast: `h5py.File(path)[key][:]` takes milliseconds.

## 4. Evaluation

Your algorithm receives a **center bin** key (e.g., `"30_50"`). It must:
1. Detect peaks in the center frame AND neighboring frames within a spatial radius
2. Link detections across frames
3. Apply Voigt profile shape verification
4. Output only the spatially-validated peaks that appear in the **center bin**

**Matching criterion**: A detected point within **40 pixels** of a ground truth point counts as a match.

**Metric — PRIMARY GOAL is RECALL of the known 3×3 features.** The score is the **mean per-center-bin F2 score** (recall-weighted, β=2: recall counts ~4× precision), averaged across evaluation bins.

> **Why F2, not F1.** The ground truth is derived from **3×3 annotations mapped to the center 1×1 frame**, so it *undercounts* the peaks that are actually visible at full 1×1 sensitivity. Many of your "extra" detections are real peaks the 3×3 labels simply never recorded. Penalizing them as hard false positives (plain F1) would push you to suppress exactly the sensitivity we want. So: **find every known 3×3 feature first.** Extra detections cost only a little. Do not sacrifice recall of the 3×3 features to chase precision — sensitivity can always be tightened later. (Spraying thousands of detections still tanks precision enough to hurt F2, so over-detection must be *reasonable*, not unbounded.)

Use the provided **`evaluate.py`** harness — it reports mean F2 (primary) plus F1/precision/recall for context.

**Why center-bin evaluation**: The ground truth annotations exist for specific bins. Your algorithm processes a neighborhood of frames around each evaluation bin, but only reports validated peaks for the center bin.

### Speed — iterate fast, score honestly

Full-resolution `spatial_radius=5` processes ~121 frames per bin. The provided `evaluate.py` and `baseline.py` expose **development-mode speed knobs** so you can iterate quickly, then confirm the real score at full settings:

- **`--subset N`** — evaluate a seeded, representative sample of N bins (e.g. 20–30) instead of all ~138.
- **`--spatial-radius R`** — use 2–3 while iterating (fewer frames per bin); use **5 for the final score**.
- **`--downsample F`** — block-average each frame F×F before detection (F=2 → ~4× fewer pixels, ~4× faster filters). Output coordinates are rescaled to full resolution automatically. Use **1 for the final score**.
- **`--workers W`** — evaluate independent bins in parallel.

**Fast dev loop** (seconds):
```
python data/evaluate.py --candidate <your_script.py> --subset 25 \
    --spatial-radius 2 --downsample 2 --workers 8
```
**Final / reportable score** (matches holdout scoring):
```
python data/evaluate.py --candidate <your_script.py> --spatial-radius 5 --downsample 1
```
(`data/evaluate.py` is provided in the workspace; its data-file paths default
correctly regardless of the directory you run it from.)
Dev mode is an *approximation* used to rank candidates cheaply; always confirm a promising algorithm at full settings before trusting its score. **Performance note:** keep connected-component analysis vectorized — never call `np.where(cc == comp_id)` inside a per-component loop (it rescans the whole image per component). Gather labeled pixels once and group them by a single sort, as `baseline.py`'s `detect_in_band` now does.

## 5. IO format

Your submitted algorithm script should accept:
- `--center-bin`: The center bin key (e.g., `"30_50"`)
- `--bins-h5`: Path to the HDF5 file (default: `/home/takaji/xrd_1x1_bins.h5`)
- `--two-theta`: Path to `tth.tiff`
- `--reflections`: Path to `reflections.py`
- `--grid-mapping`: Path to `grid_mapping.json`
- `--output`: Output CSV file path
- `--spatial-radius`: How many bins in each direction to include (default: 5)

The output CSV should have three columns: `reflection`, `x`, `y`. Each row is a spatially-validated peak in the center bin.

If `--labels` is provided (path to a JSON file with ground truth), compute and print the F1 score.

## 6. Tips

- **The per-frame detector (Stage 1) matters LEAST.** Don't over-optimize it. It's a candidate generator — it just needs high recall. The included baseline already uses lenient thresholds (SNR 3.0, min_pixels 2, tth_tolerance 0.5). You can go even lower (SNR 2.0-2.5). The spatial Voigt verification (Stage 3) is where the real precision comes from. Spend your optimization effort there.
- **Lower the per-frame SNR threshold** compared to binned data. 1×1 frames are noisy — use ~2.0-3.0 instead of 4.0. The spatial verification will remove false positives.
- **The spatial radius matters.** Too small (1-2) and you can't build meaningful profiles. Too large (>8) and you waste time on distant frames that don't contribute. 4-6 is a good range.
- **Voigt fitting needs ≥3 bins** to be meaningful. For 2-bin features, consider a simpler test (e.g., intensity ratio).
- **The Voigt R² threshold** is a key parameter. Too strict (>0.8) and you lose real peaks with noisy profiles. Too lenient (<0.2) and you keep noise. 0.3-0.5 is a reasonable range.
- **Consider intensity weighting** when computing the Voigt fit — bins where the peak is barely detected should matter less.
- **Some peaks appear in Debye-Scherrer ring segments** (continuous arcs from polycrystalline material). These are NOT Bragg peaks. They tend to produce flat or very broad spatial profiles rather than peaked Voigt shapes, so shape verification should filter them.
- **Speed matters.** With ~34,000 frames, even per-frame detection needs to be efficient. Pre-compute the 2-theta binning data once and reuse it. Vectorize where possible.
- The `link_tolerance` for matching peaks across frames should be small (3-7 pixels) since the detector geometry is fixed — a peak from the same grain should appear at nearly the same (x, y) in adjacent frames.
- Consider **multi-scale top-hat** or **wavelet-based** approaches for detecting peaks at different sizes in noisy single-exposure data.
