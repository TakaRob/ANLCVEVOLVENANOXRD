# CVEvolve Progress Tracker

**Project:** 2026-1 Luo — Perovskite Bragg Peak Detection (nano-XRD, ISN 26-ID-C)
**Metric:** F1 score (maximize), 40-pixel matching tolerance, averaged across validation bins
**Model:** Claude Opus 4.6 via Argo API (`claudeopus46`)

---

## 5x5 Binned (1,188 bins)

**Session:** `hotspot_5x5_binned`
**Status:** Running (round 11 in progress)
**Config:** 3 workers generate / 3 workers tune, 3 warmup rounds, max 20 rounds, patience 10
**Started:** 2026-06-08 22:24 UTC | **Last completed round:** 2026-06-10 19:04 UTC

### Run Summary

| Metric | Value |
|--------|-------|
| Elapsed time (through R10) | 44.7 hours |
| Rounds completed | 11 (0-10) |
| Total candidates evaluated | 33 |
| Candidates holdout-tested | 31 |
| Failures | 0 |
| Best holdout F1 | **0.790** (R1, tophat_band_adaptive_snr) |
| Estimated input tokens | ~25.5M |
| Estimated output tokens | ~500K |
| Estimated total tokens | ~26M |
| Estimated API cost (no caching) | ~$420 |
| Estimated API cost (with caching) | ~$214 |
| Worker sessions | ~31 (1 prep + 1 baseline + 9×3 generate/tune + 31 holdout) |

### Per-Round Timing

| Round | Action | Workers | Duration | Cumulative | Best F1 This Round |
|-------|--------|---------|----------|------------|-------------------|
| 0 | baseline | 1 | 18.0 h | 18.0 h | 0.022 |
| 1 | generate | 3 | 1.3 h | 19.3 h | **0.790** |
| 2 | generate | 3 | 2.2 h | 21.5 h | 0.768 |
| 3 | generate | 3 | 1.9 h | 23.5 h | 0.760 |
| 4 | tune | 3 | 2.3 h | 25.8 h | 0.771 |
| 5 | tune | 3 | 3.2 h | 29.0 h | 0.750 |
| 6 | tune | 3 | 2.1 h | 31.1 h | 0.735 |
| 7 | tune | 3 | 2.4 h | 33.5 h | 0.769 |
| 8 | tune | 3 | 2.7 h | 36.2 h | 0.750 |
| 9 | tune | 3 | 4.2 h | 40.4 h | 0.726 |
| 10 | tune | 3 | 4.3 h | 44.7 h | 0.681 |
| 11 | tune | 3 | _running_ | — | — |

> Round 0 outlier: 18 h due to agents loading from raw H5 files before the pre-built HDF5 existed.

### Top Algorithms (Holdout F1)

| Rank | Algorithm | Round | Action | Holdout F1 |
|------|-----------|-------|--------|-----------|
| 1 | tophat_band_adaptive_snr | 1 | generate | **0.790** |
| 2 | tophat_dual_snr_detector | 1 | generate | 0.778 |
| 3 | tuned_multiscale_tophat_tight_bands | 4 | tune | 0.771 |
| 4 | tuned_multiscale_tophat_dual_snr | 7 | tune | 0.769 |
| 5 | log_tophat_hybrid_detector | 2 | generate | 0.768 |
| 6 | prominence_peak_detector | 3 | generate | 0.760 |
| 7 | azimuthal_anomaly_detector | 2 | generate | 0.753 |
| 8 | dog_iterative_prominence_v3_az_rescue | 8 | tune | 0.750 |
| 9 | dog_iterative_prominence_v2 | 5 | tune | 0.750 |
| 10 | triple_channel_voting_detector | 2 | generate | 0.745 |

### Token & Cost Breakdown

| Component | Input Tokens | Output Tokens | Cost (uncached) | Cost (cached) |
|-----------|-------------|--------------|-----------------|---------------|
| Prep (orchestrator) | 4,725K | 2K | $71 | ~$20 |
| R0 baseline (1 worker) | 184K | 0.4K | $3 | ~$1 |
| R1-R3 generate (9 workers) | 3,474K | 95K | $59 | ~$25 |
| R4-R10 tune (21 workers) | 15,145K | 276K | $248 | ~$130 |
| Holdout tests (31 agents) | 1,924K | 131K | $39 | ~$18 |
| **Total** | **25,453K** | **504K** | **$420** | **~$214** |

Pricing: Opus 4.6 — $15/M input, $75/M output. Cached: $1.875/M read, $18.75/M write (~70% hit rate assumed). Input tokens dominate because each API call in the agentic loop resends the full conversation history. The prep phase alone accounts for ~19% of input tokens due to 29 model calls with a growing context window.

### Winner Architecture

**tophat_band_adaptive_snr** — Radial median background subtraction, fast white top-hat (size=15), 2-theta band masking (8 reflections), connected component labeling, per-band MAD thresholding, component analysis + dedup. ~31.5 MFLOP per image, ~0.3-0.5 s wall-clock per bin.

---

## 4x4 Binned

**Status:** Not started
**Sessions:** —
**Config:** —

| Metric | Value |
|--------|-------|
| Elapsed time | — |
| Rounds completed | — |
| Total candidates | — |
| Best holdout F1 | — |
| Estimated API cost | — |

### Top Algorithms (Holdout F1)

| Rank | Algorithm | Round | Action | Holdout F1 |
|------|-----------|-------|--------|-----------|
| | | | | |

---

## 3x3 Binned (3,848 bins)

**Status:** Not started
**Sessions:** —
**Config:** —

| Metric | Value |
|--------|-------|
| Elapsed time | — |
| Rounds completed | — |
| Total candidates | — |
| Best holdout F1 | — |
| Estimated API cost | — |

### Top Algorithms (Holdout F1)

| Rank | Algorithm | Round | Action | Holdout F1 |
|------|-----------|-------|--------|-----------|
| | | | | |

---

## 2x2 Binned

**Status:** Not started

| Metric | Value |
|--------|-------|
| Elapsed time | — |
| Rounds completed | — |
| Total candidates | — |
| Best holdout F1 | — |
| Estimated API cost | — |

### Top Algorithms (Holdout F1)

| Rank | Algorithm | Round | Action | Holdout F1 |
|------|-----------|-------|--------|-----------|
| | | | | |

---

## 1x1 (Per-pixel)

**Status:** Not started

| Metric | Value |
|--------|-------|
| Elapsed time | — |
| Rounds completed | — |
| Total candidates | — |
| Best holdout F1 | — |
| Estimated API cost | — |

### Top Algorithms (Holdout F1)

| Rank | Algorithm | Round | Action | Holdout F1 |
|------|-----------|-------|--------|-----------|
| | | | | |

---

## Cross-Scale Summary

| Bin Size | Bins | Best F1 | Best Algorithm | Rounds | Wall Time | Est. Cost |
|----------|------|---------|----------------|--------|-----------|-----------|
| **5x5** | 1,188 | **0.790** | tophat_band_adaptive_snr | 11+ | 44.7 h+ | $214-420 |
| 4x4 | — | — | — | — | — | — |
| 3x3 | 3,848 | — | — | — | — | — |
| 2x2 | — | — | — | — | — | — |
| 1x1 | 34,476 | — | — | — | — | — |

---

*Last updated: 2026-06-10*
