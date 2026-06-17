# CVEvolve 5x5 Binned Bragg Peak Detection — Cost & Scaling Summary

## 1. CVEvolve Search Cost

### Session Overview
- **Total elapsed**: 40.3 hours (rounds 0–9 completed, round 10 running)
- **Total candidates evaluated**: 28 (1 baseline + 27 generated/tuned)
- **Failures**: 0
- **Best holdout F1**: 0.791 (tophat_band_adaptive_snr, round 1)
- **Model**: Claude Opus 4.6 via Argo API

### Per-Round Timing

| Round | Action   | Workers | Duration (min) | Notes |
|-------|----------|---------|----------------|-------|
| 0     | baseline | 1       | 1077           | Slow — H5 loading before prebuild |
| 1     | generate | 3       | 80             | Best holdout result found here |
| 2     | generate | 3       | 133            | |
| 3     | generate | 3       | 116            | End of warmup |
| 4     | tune     | 3       | 140            | |
| 5     | tune     | 3       | 193            | |
| 6     | tune     | 3       | 123            | |
| 7     | tune     | 3       | 146            | |
| 8     | tune     | 3       | 162            | |
| 9     | tune     | 3       | 250            | |

- **Average generate round**: ~110 min (3 workers)
- **Average tune round**: ~165 min (3 workers)
- **Round 0 outlier**: 18 hours — agents were loading bins from raw H5 files over the WSL `/mnt/c/` mount before the pre-built HDF5 existed

### Estimated API Cost

Each worker round involves ~20–50 model calls (code generation, debugging, evaluation). With 3 workers × 10 rounds:
- **~900–1500 model API calls** total
- At Claude Opus 4.6 pricing (~$15/M input, $75/M output tokens), estimated **$50–150** total depending on context length

## 2. Algorithm Computational Cost (Winning Detector)

### Per-Bin FLOP Breakdown

The winning algorithm (`tophat_band_adaptive_snr`) processes each 1062×1028 image through 6 stages:

| Stage | Operation | MFLOP |
|-------|-----------|-------|
| 1 | Radial median background subtraction | 14.0 |
| 2 | Fast white top-hat (min/max filters, size=15) | 4.4 |
| 3 | 2-theta band masking (8 reflections) | 8.7 |
| 4 | Connected component labeling | 1.1 |
| 5 | Per-band MAD thresholding | 3.3 |
| 6 | Component analysis + dedup | <0.1 |
| **Total** | | **31.5 MFLOP** |

**Wall-clock per bin**: ~0.3–0.5 seconds on a modern CPU (dominated by median computation and I/O, not raw FLOPs).

### Full Dataset (5×5 binning, 1188 bins)

| Metric | Value |
|--------|-------|
| Total FLOPs | 37.4 GFLOP |
| Total I/O (read) | 5.2 GB (float32 from HDF5) |
| Estimated wall time | ~10 min (sequential) |

## 3. Scaling by Bin Size

All bin sizes produce the same detector image dimensions (1062×1028). What changes:
- **Number of bins**: smaller bins = more images to process
- **Frames per bin**: larger bins = more frames summed = higher SNR
- **SNR gain**: scales as √(frames_per_bin) relative to single-frame

| Bin Size | Bins | Frames/Bin | SNR Gain | Total GFLOP | I/O (GB) | Est. Time |
|----------|------|------------|----------|-------------|----------|-----------|
| 1×1 | 34,476 | 1 | 1.0× | 1,085 | 150.6 | ~5h |
| 2×2 | 8,658 | 4 | 2.0× | 273 | 37.8 | ~1.2h |
| 3×3 | 3,848 | 9 | 3.0× | 121 | 16.8 | ~32m |
| **5×5** | **1,188** | **25** | **5.0×** | **37** | **5.2** | **~10m** |
| 7×7 | 736 | 49 | 7.0× | 23 | 3.2 | ~6m |
| 10×10 | 368 | 100 | 10.0× | 12 | 1.6 | ~3m |
| 15×15 | 165 | 225 | 15.0× | 5 | 0.7 | ~1m |
| 20×20 | 96 | 400 | 20.0× | 3 | 0.4 | ~48s |

### Key Trade-offs

- **Smaller bins** (1×1, 2×2): Higher spatial resolution but very low SNR. Faint peaks may be undetectable. Algorithm thresholds would need retuning. Compute cost scales linearly with bin count.
- **5×5 (current)**: Good balance — 5× SNR improvement, ~1000 manageable images, captures spatial variation across the sample.
- **Larger bins** (10×10, 20×20): High SNR, fast, but spatial information is lost. Multiple crystal grains in different spatial regions get merged.
- **Full sum** (1 bin): Maximum SNR but zero spatial information. This is the existing summed-image approach.

### Pre-Build Time by Bin Size

The pre-build step (summing H5 frames into the HDF5 file) is dominated by raw H5 I/O:

| Bin Size | Bins | Pre-build File Size | Pre-build Time (est.) |
|----------|------|--------------------|-----------------------|
| 1×1 | 34,476 | ~140 GB (uncompressed) | Not practical |
| 3×3 | 3,848 | ~16 GB | ~2h |
| **5×5** | **1,188** | **~1 GB (compressed)** | **~45 min** |
| 10×10 | 368 | ~1.5 GB | ~45 min (same raw reads) |

Note: Pre-build time is dominated by reading all 25,170 raw frames regardless of bin size. The only saving at larger bin sizes is fewer output datasets.

### If Re-Running CVEvolve at a Different Bin Size

1. **Algorithm transfer**: The winning algorithm's architecture (radial subtract → top-hat → band mask → MAD threshold) is bin-size-agnostic. Only the SNR thresholds need retuning.
2. **CVEvolve re-run**: A focused tune-only run starting from the 5×5 winner (5–10 rounds) should suffice. Estimated ~5–10 hours, ~$20–50 API cost.
3. **Validation labels**: New ground truth annotations would be needed at the new bin size.
