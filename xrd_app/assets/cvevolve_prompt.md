Find a robust and accurate algorithm that detects Bragg peaks in single-exposure (1×1, un-binned) X-ray diffraction frames.

## 1. Problem statement

An X-ray diffraction experiment scans a sample with a focused X-ray beam across a 2D spatial grid. At each grid position, a 2D detector records a diffraction pattern. When the sample contains crystalline material, Bragg peaks appear as bright spots at positions determined by the crystal lattice spacing (characterized by 2-theta angle) and orientation.

**This task uses 1×1 data — single exposures with NO spatial binning.** This means each "bin" is a single detector frame, and frames are very noisy compared to binned data. There are **~34,000 frames** arranged on a 156-row × 221-column spatial grid (raster scan).

### Why spatial persistence matters

A focused X-ray beam illuminates a ~1 μm spot. As the beam rasters across a crystal grain, the same Bragg peak appears at nearly the same detector position in **clusters of neighboring frames**. Noise spikes, by contrast, appear in **isolated single frames**. You can exploit this: a candidate that persists at the same (x, y) across adjacent frames is far more likely to be a real peak than one seen only once.

### Pipeline — aggressive detection, optional spatial confirmation

**Stage 1: Per-frame peak detection (the core task).** Given a single-frame image, detect candidate Bragg peak positions within the known 2-theta bands. Because the ground truth undercounts the peaks visible at full 1×1 sensitivity (see Evaluation), **favor recall**: use low SNR thresholds (2.0-3.0), permissive size/compactness filters, and reasonably wide 2-theta band tolerances. Missing a real peak here means it is lost from the pipeline, so err on the side of over-detection — but keep it *reasonable*, not unbounded (spraying thousands of detections still tanks precision enough to hurt the score). **Spend the bulk of your optimization effort on this stage.**

**Stage 2 (optional): Spatial linking for confirmation.** Link detections at the same detector (x, y) position across neighboring spatial frames — two detections in adjacent frames within `link_tolerance` pixels are the same physical peak observed from neighboring beam positions. Use Union-Find or equivalent clustering. Detections that persist across adjacent frames are strong; isolated single-frame detections can be down-weighted or dropped as noise. Keep this **light** — it is a consistency check to trim obvious noise, not a strict filter, and it must not sacrifice recall of the known features.

The stages can be separate functions, separate scripts, or a single integrated pipeline. What matters is that per-frame detection is sensitive and any spatial confirmation is gentle.

## 2. Prior art — best peak detection on binned data

### Best peak detection algorithm (from 5×5 binned CVEvolve optimization)

The following pipeline achieved the best F1 scores on 5×5 and 3×3 binned data. It is included as `tophat_band_adaptive_snr.py` for reference. Key ideas:

1. **Radial median background subtraction**: Compute median intensity in narrow 2-theta annular bins. Smooth the median profile. Subtract to remove the radially-symmetric background. This is the single most important step — the dominant noise source is radial.

2. **White top-hat morphological filter**: `image - opening(image)` using min/max filters with a square kernel (size ~13-15 pixels). Extracts compact bright features while suppressing broad/diffuse intensity.

3. **2-theta band restriction**: For each reflection, only search within a narrow band around its known 2-theta value (±0.4°). This dramatically reduces false positives by constraining where peaks can appear.

4. **Per-band adaptive thresholding**: Within each 2-theta band, compute the median and MAD (median absolute deviation). Set threshold at `median + SNR_threshold × 1.4826 × MAD`. This adapts to the local noise level in each band.

5. **Connected component analysis**: Label connected pixels above threshold. Filter by size (min_pixels=3, max_pixels=150), compactness (aspect_ratio × fill_ratio > 0.12). Compute intensity-weighted centroid.

6. **Cross-band duplicate suppression**: Sort all peaks by SNR, keep highest-SNR peak when two are within `dup_distance` pixels.

**Important for 1×1**: Since single frames have ~5-25× less signal than 5×5 bins, you will likely need to use a lower SNR threshold (e.g., 2.5-3.5 instead of 4.0) to catch faint peaks. Favor recall — the ground truth undercounts, so extra detections are only lightly penalized.

### Spatial linking (what worked for cross-bin analysis on 3×3 data)

Union-Find was used to link peaks across neighboring bins:
- Each detection is a node: `(bin_key, peak_index, row, col, x, y, peak_dict)`
- Two detections in adjacent bins (8-connected neighbors) with detector positions within `link_tolerance` pixels are unioned
- Connected components are the same physical Bragg peak observed across multiple spatial positions

For 1×1 data, use the same linking idea purely as a light spatial-persistence confirmation: peaks that recur across neighbors are kept confidently; single-frame-only detections are the ones to treat with suspicion.

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
- **`baseline.py`**: A baseline algorithm implementing per-frame detection and optional spatial linking. **Use this as your starting reference and try to outperform it.**
- **`noise_reduction_algorithms.py`**: Library of radial background models. You may use, modify, or replace these.
- **`tophat_band_adaptive_snr.py`**: The best per-bin detection algorithm from the 5×5 optimization. Use as reference for Stage 1.
- **`annotations_summed.json`**: Ground truth for a fully-summed image (context only).

### Important notes

- Detector dimensions: 1062 rows × 1028 columns.
- For visualization: percentile range 10–99, use `np.log1p()` for dynamic range.
- Edge pixels (first/last 2-3 rows and columns) often have artifacts.
- Loading is fast: `h5py.File(path)[key][:]` takes milliseconds.

## 4. Evaluation

Your algorithm receives a **center bin** key (e.g., `"30_50"`). It must:
1. Detect peaks in the center frame (optionally also in neighboring frames within a spatial radius)
2. Optionally link detections across frames as a spatial-persistence confirmation
3. Output the peaks that appear in the **center bin**

**Matching criterion**: A detected point within **40 pixels** of a ground truth point counts as a match.

**Metric — PRIMARY GOAL is RECALL of the known 3×3 features.** The score is the **mean per-center-bin F2 score** (recall-weighted, β=2: recall counts ~4× precision), averaged across evaluation bins.

> **Why F2, not F1.** The ground truth is derived from **3×3 annotations mapped to the center 1×1 frame**, so it *undercounts* the peaks that are actually visible at full 1×1 sensitivity. Many of your "extra" detections are real peaks the 3×3 labels simply never recorded. Penalizing them as hard false positives (plain F1) would push you to suppress exactly the sensitivity we want. So: **find every known 3×3 feature first.** Extra detections cost only a little. Do not sacrifice recall of the 3×3 features to chase precision — sensitivity can always be tightened later. (Spraying thousands of detections still tanks precision enough to hurt F2, so over-detection must be *reasonable*, not unbounded.)

Use the provided **`evaluate.py`** harness — it reports mean F2 (primary) plus F1/precision/recall for context.

**Why center-bin evaluation**: The ground truth annotations exist for specific bins. Your algorithm may process a neighborhood of frames around each evaluation bin, but only reports peaks for the center bin.

### Speed — iterate fast, score honestly

If you process a neighborhood, full-resolution `spatial_radius=5` covers ~121 frames per bin. The provided `evaluate.py` and `baseline.py` expose **development-mode speed knobs** so you can iterate quickly, then confirm the real score at full settings:

- **`--subset N`** — evaluate a seeded, representative sample of N bins (e.g. 20–30) instead of all ~138.
- **`--spatial-radius R`** — use 0–2 while iterating (fewer frames per bin); use **5 for the final score** if your algorithm uses neighbors.
- **`--downsample F`** — block-average each frame F×F before detection (F=2 → ~4× fewer pixels, ~4× faster filters). Output coordinates are rescaled to full resolution automatically. Use **1 for the final score**.
- **`--workers W`** — evaluate independent bins in parallel.

**Fast dev loop** (seconds):
```
python data/evaluate.py --candidate <your_script.py> --subset 25 \
    --spatial-radius 1 --downsample 2 --workers 8
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
- `--spatial-radius`: How many bins in each direction to include (default: 5; use 0 for center-only detection)

The output CSV should have three columns: `reflection`, `x`, `y`. Each row is a detected peak in the center bin.

If `--labels` is provided (path to a JSON file with ground truth), compute and print the F1 score.

## 6. Tips

- **Lower the per-frame SNR threshold** compared to binned data. 1×1 frames are noisy — use ~2.0-3.0 instead of 4.0. Favor recall; the undercounting ground truth means extra detections are cheap.
- **The 2-theta band restriction is your strongest precision lever.** Only search within a narrow band around each known reflection's 2-theta value — most spurious detections fall outside these bands.
- **Radial background subtraction first.** The dominant noise is radially symmetric; removing it before thresholding matters more than any other single step.
- **If you use spatial linking, keep `link_tolerance` small** (3-7 pixels): the detector geometry is fixed, so a peak from the same grain appears at nearly the same (x, y) in adjacent frames.
- **Some peaks appear in Debye-Scherrer ring segments** (continuous arcs from polycrystalline material). These are NOT single-crystal Bragg peaks. Because a ring is broad and continuous in the azimuthal direction while a Bragg spot is sharp and localized, an azimuthal (I-vs-χ) discriminator separates them well (see the tip below).
- **Speed matters.** With ~34,000 frames, even per-frame detection needs to be efficient. Pre-compute the 2-theta binning data once and reuse it. Vectorize where possible.
- Consider **multi-scale top-hat** or **wavelet-based** approaches for detecting peaks at different sizes in noisy single-exposure data.
- **Consider an azimuthal (I-vs-χ) 1D profile within each 2-theta band as a candidate generator and ring discriminator.** Since the reflection 2-theta values are already known, the remaining unknown is *where along each ring* a grain's Bragg spot sits — i.e. the azimuthal angle χ. Within a 2-theta band, bin intensity by χ to get a 1D I-vs-χ profile, then find peaks in 1D (cheap and robust) and do precise 2D centroiding only in the azimuthal neighborhood of each 1D peak. This also separates spots from rings *at the candidate stage*: a single-crystal Bragg **spot** is a sharp peak in χ, while a polycrystalline **Debye-Scherrer ring** is flat/broad in χ. Derive χ per pixel from the beam center: estimate the beam center from the 2-theta map (see `estimate_beam_center` in `analysis/spatial_feature_analysis.py`) and compute `chi = atan2(y - cy, x - cx)`. **Important:** a single 1×1 frame is often too noisy for a reliable azimuthal profile — compute I-vs-χ on the **spatial stack** (summed/linked neighboring frames) when you can. This is optional — treat it as one Stage-1 strategy to try and benchmark against multi-scale top-hat, not a required step.
