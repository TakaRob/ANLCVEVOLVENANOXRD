# Nano-XRD Data Processing — Technical Report

**Project:** 2026-1 Luo · perovskite / halide thin films · ISN 26-ID-C, APS (Argonne)
**Author:** Takaji Robson
**Pipeline order:** Binning → CVEvolve detector optimization → Shape / feature finding → Deskewing
**Scope:** data- and image-driven record of each processing step, with the result tables that are already on disk filled in. Commentary is kept minimal.

> 🔴 **TODO (red):** items flagged with `🔴` are gaps to fill before this is final. In the Word version these should be set in **red** so they're obvious. Most are images/figures to paste, or a number to confirm at full settings.

---

## 0. Dataset & instrument (one-line reference)

| Item | Value |
|---|---|
| Beamline | ISN 26-ID-C, APS |
| Energy | 15 keV |
| Pixel size | 75 µm |
| Detector crop / frame | 1062 × 1028 (full); 256 × 256 (ROI crop) |
| Reflections (2θ bands) | PbI₂, (001), (011), (111), (002), ITO, (012), (112) |
| Scan (binning + CVEvolve + shapes + deskew) | **Scan_0203** (all steps — confirmed from grid mappings) |
| 1×1 grid | 156 rows × 221 cols = **34,476 cells**, **25,170 occupied** (151 raw HDF5 files) |
| 5×5 grid | 32 rows × 45 cols = 1,440 cells, **1,188 occupied** |

*Scan reconciliation (resolved):* the draft "151 × 51", the binning report's "25,170", and the cost summary's "34,476" are all the **same scan, Scan_0203** — 34,476 is the full bin grid, 25,170 the occupied bins, 151 the raw file/row count. There is no second scan.

---

## 1. Binning

**Hypothesis (from draft):** Preliminary analysis of nano-XRD data is easier when frames are binned, so there should be a "sweet spot" bin size where peaks are both easy to identify *and* enough of them rise above noise.

### 1.1 Counts vs bin size — Scan_0203

Detector `5x5_tophat_band_adaptive_snr`, snr=4, gaussian shapes, link_tol=5.
Authoritative kept-feature counts from `gaussian_shapes_NxN.json["kept"]`.

| Bin size | Occupied bins | Raw peaks | Kept features | Spatial resolution |
|---|---|---|---|---|
| 1×1 | 25,170 | 186,063 | 1,825 | full |
| 2×2 | 7,231 | 16,915 | 310 | — |
| 3×3 | 3,253 | 9,124 | 213 | 52 × 74 |
| 4×4 | 1,859 | 6,125 | 199 | — |
| 5×5 | 1,188 | 3,889 | 144 | — |

**Result:** total peaks **and** features *decrease* with bin size — bin count falls ~1/N² faster than per-bin yield rises. 

> 🔴 **Figure 1.1:** bar/line plot of raw peaks & kept features vs bin size (data above).

### 1.2 SNR / intensity at real detections — Scan_0203

| Quantity | 1×1 | Binned |
|---|---|---|
| Median peak SNR | 5.3 (barely above threshold) | ~17–24 |
| Strong-peak p90 SNR | 27 | 357 |
| Median feature intensity | 6 | 64 |

**Result:** SNR/intensity *at real detections* rises sharply with binning, even though absolute counts fall. (A random-bin signal/noise sample looks flat because most bins are background and a sub-bin-sized grain doesn't fill the bin: signal grows < N², noise ~ N.)

> 🔴 **Figure 1.2:** SNR distribution (or median ± p90) vs bin size.
> 🔴 **Figure 1.3:** side-by-side detector image of the same hotspot at 1×1 / 3×3 / 5×5 (visual SNR gain).

### 1.3 File size & speed — Scan_0203

Binned HDF5 file sizes on disk (the pre-built `xrd_NxN_bins.h5`):

| Bin size | Occupied bins | Binned HDF5 size |
|---|---|---|
| 1×1 | 25,170 | **10 GB** |
| 3×3 | 3,253 | **7.8 GB** |
| 4×4 | 1,859 | **4.8 GB** |
| 5×5 | 1,188 | **971 MB** |

*(2×2 not pre-built. File size does not fall as fast as bin count because the per-bin detector image is full 1062×1028 regardless of bin size — fewer-but-same-size images.)*

**Speed:** building a binned HDF5 is the bottleneck (~15 min/size, frame-summing over OneDrive); detection itself is seconds–minutes. SNR gain scales as √(frames/bin): 1×1 → 1.0×, 3×3 → 3.0×, 5×5 → 5.0×.

### 1.4 Binning conclusion

**Sweet spot ≈ 3×3:** SNR plateau reached, fragmentation gone (see §4), spatial resolution still 52 × 74. Smaller bins keep resolution but lose SNR and over-segment; larger bins merge distinct grains.

---

## 2. CVEvolve detector optimization

CVEvolve iteratively generates and tunes detector algorithms, scoring each against ground-truth annotations. Model: **Claude Opus 4.6 via Argo API**.

### 2.1 Metric choice

- **Primary metric = mean F2** (recall-weighted, β≈2), **not F1**. (5×5 study reported F1; the metric was switched to F2 for the 1×1 work — see below.)
- **Why F2:** ground truth is derived from 3×3 annotations mapped to the center frame, so it *undercounts* peaks visible at full 1×1 sensitivity. F1 punishes genuine extra detections as false positives; F2 weights recall ~4× precision so we recall all known features and tighten precision later. Tolerance = 40 px.

### 2.2 Two CVEvolve runs (both Scan_0203)

| | **5×5 run** (`hotspot_5x5_binned`) | **1×1 run** (`hotspot_1x1_f2`) |
|---|---|---|
| Metric | F1 (per-bin, 40 px tol) | mean F2 (recall-weighted) |
| Dev / evaluation bins | (annotated dev set) | **138 bins** |
| **Holdout bins** | **116** (113 labeled + 3 empty) | **60 bins** |
| Rounds completed | 11 (rounds 0–10) | 13 (rounds 0–12) |
| Candidates evaluated | 33 | 25 |
| Baseline score | — | F2 = 0.181 (F1 0.139, P 0.111, R 0.243) |
| **Best holdout score** | **F1 = 0.7905** (round 1, `tophat_band_adaptive_snr`) | **F2 = 0.463** (round 10) |
| Best dev score | F1 = 0.892 (round 9) | F2 = 0.470 (round 11) |

> Draft note "0.8 F1, 100/2500, after 2 loops" maps to: holdout **F1 ≈ 0.79**, best candidate found in **round 1** (the first generate loop). "100/2500" 🔴 confirm — likely a peaks-found count on one bin.

### 2.3 Runtimes (from session logs)

| | 5×5 run | 1×1 run |
|---|---|---|
| Wall-clock start | 2026-06-08 22:24 UTC | 2026-06-17 22:28 UTC |
| **Total elapsed** | **44.7 h** | **21.5 h** |
| Round 0 (baseline) | **18.0 h** (I/O outlier — loading raw H5 over `/mnt/c` before prebuild) | 1.2 h |
| Rounds 1+ (search) | 26.7 h | 20.3 h |
| Mean generate round | ~110 min (3 workers) | ~100 min (2 workers) |
| Mean tune round | ~180 min | ~95 min |

### 2.4 F1 report — best detectors (5×5 holdout)

From `cvevolve_5x5/test_data/top_algorithms.json` (ranked by holdout F1):

| Rank | Detector | Holdout F1 |
|---|---|---|
| 1 | `tophat_band_adaptive_snr` | **0.7905** |
| 2 | `tophat_dual_snr_detector` | 0.7778 |
| 3 | `tuned_multiscale_tophat_tight_bands` | 0.7706 |
| 4 | `tuned_multiscale_tophat_dual_snr` | 0.7691 |
| 5 | `log_tophat_hybrid_detector` | 0.7680 |

Winning detector compute: ~31.5 MFLOP/image (1062×1028), ~0.3–0.5 s/bin; full 5×5 dataset (1,188 bins) ≈ 37 GFLOP, ~10 min sequential.

**Search cost (5×5):** ~900–1500 model API calls, estimated **$50–150** (Opus 4.6 via Argo), 0 failures.

> 🔴 **Figure 2.1:** F-score vs round (search curve) — from `sessions/.../history/search_history.sqlite`.
> 🔴 **Figure 2.2:** winning-detector output overlaid on a representative image (detections vs ground truth).

### 2.5 Why CVEvolve was abandoned at 1×1

- Even with band masking/cropping, ROI segmentation, and a data subset, the 1×1 search was **slow** (21.5 h) and the gain was **modest** — holdout F2 rose 0.181 → 0.463, but absolute scores stayed low.
- **Overfitting risk (the deciding factor):** only **138 dev + 60 holdout = 198 labeled bins** against **25,170 occupied bins** (≈0.8% labeled). Tuning a 1×1 detector hard against so few labels would overfit.
- **Decision:** keep the recall-first F2 baseline for 1×1; do real detector optimization at the binned (5×5) scale, where SNR is high and the labeled fraction is far larger.

---

## 3. Shape / feature finding

A **peak** is a single-bin detection (may be noise). A **feature/shape** is a peak cluster linked across neighboring bins (Union-Find) that passes the gaussian-profile filter — the physically trustworthy Bragg reflection. Linking tolerance = 5 px detector agreement.

### 3.1 Peaks → kept features (Scan_0203)

| Bin size | Raw peaks | Kept features | Rejected (filtered) |
|---|---|---|---|
| 1×1 | 186,063 | 1,825 | 🔴 (= raw clusters − kept) |
| 3×3 | 9,124 | 213 | 🔴 |
| 5×5 | 3,889 | 144 | 🔴 |

> 🔴 **Figure 3.1:** example kept feature vs rejected cluster (intensity profile + footprint).
> 🔴 **Figure 3.2:** feature-viewer screenshot of features painted on the scan grid (device map).

### 3.2 Recall check — were missed peaks captured in features?

Open question from the draft: *"We have peaks that weren't found in detection — are they captured in the linked features?"*

- **Method (manual):** scroll images, mark un-found peaks, run a script to check those coordinates in neighboring frames / linked features. Quantify the fraction recovered.
- 🔴 **TODO:** run this check and report the recovered fraction. (Noted as token-expensive for an agent to do by eye; do as a scripted coordinate lookup.)

### 3.3 Morphology metrics available per feature

Each feature carries: `spatial_extent` / `n_bins` (footprint), `intensity_profile`, `chi_deg` (azimuthal angle), `chi_fwhm` (azimuthal breadth — *not* a rocking curve), `tth_fwhm` (radial Δ2θ breadth — *not* calibrated strain). See `TERMINOLOGY.md` §3.

> 🔴 **Figure 3.3:** device-map of χ angle and/or Δ2θ breadth across the sample.

---

## 4. Deskewing (1×1 horizontal-slicing fix)

### 4.1 The problem

At 1×1, features over-segment into **horizontal slices**: 53% of multi-bin features span exactly one scan row (tol=5). Cause: serpentine raster row-registration offset + single-frame centroid jitter exceed the 5 px link tolerance across rows, so only *within-row* links survive. The error is correlated **within** a row (fast X pass) but flips sign **between** rows (direction reversal) → slicing is horizontal.

> 🔴 **Figure 4.1:** skew map / before-after linking on a 1×1 feature (from `scan_203_skew_map.py`).

### 4.2 Remedies tried (Scan_0203)

| Method | What it does | 1×1 single-row fraction | Kept features 1×1 | Binned effect |
|---|---|---|---|---|
| (none) | baseline gaussian linking | 53% | 1,825 | — |
| Position-based (true X,Y) | re-link by true stage position | **33%** | 1,570 | n/a (diagnostic) |
| `gaussian_deskew.py` (shape variant, no CSV) | estimates per-row backlash, adaptive cross-row window | 44% | 2,018 | 3×3 neutral (213→214) |
| `deskew_peaks.py` (CSV pre-filter, true positions) | re-grids peaks onto faithful lattice | **34%** | — | **worse** (2×2 310→422, 3×3 213→328) |

### 4.3 Why deskew hurts binned data

Binning already averages out the serpentine backlash, so re-gridding an already-clean grid only **injects rounding noise**. Deskew is therefore a **1×1-specific** remedy — the scripts warn for bin size ≠ 1.

### 4.4 Diagnostic split of 1×1 fragmentation

- **38%** of 1×1 detections have a same-grain partner one row away but column-shifted outside the 8-window → **deskewable**.
- **45%** have **no** detection in the adjacent row (SNR dropout) → only binning / a lower threshold recovers these.
- For Scan_0203 the systematic skew is small (median ~2 columns), so most 1×1 fragmentation is SNR dropout, not skew. **Binning is the real fix; deskew helps more on larger-backlash scans.**

> 🔴 **Figure 4.2:** before/after device map at 1×1 showing merged slices after deskew.

---

## 5. Summary

| Step | Key result |
|---|---|
| Binning | Counts fall with bin size, but SNR at real peaks rises (median 5.3→~20). Sweet spot ≈ **3×3**. |
| CVEvolve | 5×5 → `tophat_band_adaptive_snr`, **holdout F1 0.791** (44.7 h, ~$50–150). 1×1 → F2 0.181→0.463 but abandoned: 21.5 h and only ~0.8% of bins labeled (overfit risk). |
| Shapes | 1×1: 186k peaks → 1,825 features; 3×3: 9.1k → 213; 5×5: 3.9k → 144. |
| Deskewing | Fixes 1×1 horizontal slicing (53%→33–44%); **1×1-only**, harmful on binned data. |

> 🔴 **Open items:** (1) recovered-fraction of missed peaks (§3.2); (2) confirm what "100/2500" means in the draft; (3) per-bin-size dev-set count for the 5×5 run; (4) paste all figures.

---

*Sources on disk: `binning_benefits_report.py`, `cvevolve_5x5/COST_SUMMARY.md`, `cvevolve_1x1/` configs, `deskew_peaks.py`, `scan_203_skew_map.py`, `TERMINOLOGY.md`.*
