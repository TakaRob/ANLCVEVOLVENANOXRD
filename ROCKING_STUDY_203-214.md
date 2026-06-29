# Rocking-Series Study — `5%_DI_Yes_GB` (Scans 203–214)

> nano-XRD orientation/strain study at ISN 26-ID-C (APS). Same sample, same spot,
> swept through sample θ. Goal: detect Bragg features per scan, **predict** which
> shapes should recur across the θ sweep, **track** them, and **verify** the
> prediction — ending in a *combined* device view that fuses all orientations into
> one spatial map. GUI work is deferred; this plan builds the engine + data layers.

---

## 1. What we're studying

- **Sample:** `5%_DI_Yes_GB`, single spot, `samy = −1.61`.
- **Scans:** 203–214 (logbook id 10270). A θ rocking series — the sample is
  rotated through θ while the nano-beam rasters an (x,y) map at each θ.
- **θ table (the orientation variable):**

  | Scan | θ (°) | Scan | θ (°) |
  |------|-------|------|-------|
  | 203 | 20.5 | 209 | 4.5 |
  | 204 | 20.0 | 210 | 4.0 |
  | 205 | 19.5 | 211 | 3.5 |
  | 206 | 6.0 ⚠*incomplete* | 212 | 3.0 |
  | 207 | 5.5  | 213 | 0.0 |
  | 208 | 5.0  | 214 | 20.5 *(repeat of 203 — reproducibility check)* |

  Note the sampling is **clustered**, not uniform: a dense cluster near θ≈3–6° and
  a few points at 19.5–20.5° and 0°. Rocking-curve fits are only meaningful where
  θ is densely sampled (≈3–6°). 203/214 are a duplicate orientation — use them to
  estimate detection repeatability (an empirical noise floor for the comparison).

- **Geometry:** 15 keV, 6.16 m detector distance, 75 µm pixel, area detector
  `(1062×1028)`. Reflections are fixed 2θ bands (PbI₂, (001), (011), (111), (002),
  ITO, (012), (112), + 3 unlabeled) — see `core/reflections.py`.

- **Physics of the sweep:** a crystalline grain diffracts a given reflection only
  when its orientation satisfies the Bragg condition for the incident beam. As θ
  steps, different grains "light up." For a fixed (x,y) grain, diffracted
  intensity vs θ traces a **rocking curve** peaked at θ_Bragg, whose FWHM is the
  mosaic / rocking spread. χ (azimuth on the detector) should vary *smoothly* with
  θ for a single grain. These are the signals the prediction + tracking exploit.

---

## 2. Scientific goals & hypotheses

- **H1 (recurrence):** a real grain reappears in adjacent-θ scans (within its
  rocking width), at the same de-skewed (X,Y), same reflection band, with χ moving
  smoothly. Isolated single-θ detections are likely noise.
- **H2 (predictability):** from the reflection set + geometry + the first strong
  detections, we can *forecast* the per-θ set of expected shapes (reflection,
  position, χ, approximate intensity) — and the actuals should match it.
- **H3 (rocking curve):** tracked features' intensity(θ) fits a peaked curve;
  θ_Bragg and FWHM are physically sane and reproducible (203 vs 214).
- **Deliverable hypothesis:** fusing all θ into one **combined device view**
  yields a spatial orientation/strain map richer than any single scan.

---

## 3. Data & compute reality — READ FIRST

This dominates scheduling. Measured this session:

- **Transport ceiling:** the link to `\\micdata\data1` (mounted `Z:` → WSL
  `/mnt/z`) tops out at **~23 MB/s** — confirmed Windows-native single-stream
  (22.8) *and* `robocopy /MT:16` (23.2). Not a per-connection cap; it's the WAN
  link. msize tuning (64KB→256KB) and read parallelism only reached ~13 MB/s in
  WSL and **cannot break the ~23 MB/s ceiling**.
- **Per-scan raw size:** `Scan_NNNN/XRD/` ≈ **283 GB** (251 row-files × 278
  frames × 4.37 MB/frame). Frames are chunked `(1,1062,1028)` → one chunk = one
  whole frame, so **ROI cropping does not reduce transfer**.
- **Binning reads the full raw once.** Producing bins (any size) pulls the whole
  283 GB across the WAN: **~3.5 h/scan at 23 MB/s** (≈6 h on plain 9p). For 12
  scans that is **~42 h of reading minimum** on this machine.

### The 1×1 caveat (requested, but flag it)
The bin `.h5` stores **one detector image per bin**. Bin size sets the bin count:

| Bin | # bins (251×278) | binned `.h5` (gzip, est.) | ×12 scans |
|-----|------------------|---------------------------|-----------|
| **1×1** | ~69,561 | ~50–60 GB/scan | **~0.6–0.7 TB** |
| 3×3 | ~7,800 | ~3–7 GB/scan | ~40–80 GB |
| 5×5 | ~2,860 | ~1.5–3 GB/scan | ~20–30 GB |

**1×1 is the largest output and over-segments** (prior finding: 1×1 horizontal
slicing vs serpentine offset). It is kept here **as requested** for full spatial
resolution, but recommend **also producing 3×3** as the working layer for
tracking/prediction (fast, robust) and reserving 1×1 for final high-res maps.

### Decisions that change the schedule by ~10×
1. **Where to bin.** If any **APS-side compute** can read micdata over LAN, bin
   there and download only the small binned `.h5` (≥3×3 → <1 h total). Pulling
   ~0.7 TB (1×1) of binned output, or 2.8 TB of raw, to a laptop is the thing to
   avoid. **This is the single biggest lever.**
2. **Storage target.** Write `Binned/` to **WSL-native ext4 (`/`, 914 GB free)**
   or a non-OneDrive NTFS folder — **never** under `…/OneDrive/…` (it would sync
   every dataset write). 1×1 needs ~0.7 TB free; ext4 fits, C: (340 GB) does not.
3. **Completeness varies.** First check: **Scan_0203 = 151 XRD files** (complete
   reference Scan_0187 = 251). Scans must be validated and partial ones either
   skipped or handled (`batch` already skips scans with no frames; positions can
   be rebuilt with `recreate-positions`).

---

## 4. The existing engine (what we build on)

CLI-is-the-engine (see `xrd_app/CLAUDE.md`). Relevant commands already present:

| Command | Role |
|---------|------|
| `scan-detect` | discover `Scan_*/` (honors `XRD/` subdir) |
| `link` | record tth / reflections / positions (`--position-root` for a folder) |
| `grid` | build de-skewed grid mapping from positions |
| `bin` / `make-bins` | grid-mapping → binned HDF5 (one scan) |
| `peaks` | **Phase 1**: detector over every bin → per-bin peaks |
| `shapes` | **Phase 2**: link peaks across bins (Union-Find) → shapes |
| `run-pipeline` | peaks → shapes for one scan |
| **`batch`** | **grid→bin→peaks→shapes over many scans** (`--scans "203,204"`); auto-skips incomplete scans |
| `recreate-positions` | rebuild a missing `scan_NNNN_position.csv` from frames |

**Shape fields available for tracking/prediction** (`core/aggregate.py`):
`reflection`, `ref_tth`, `chi_deg` (orientation), `chi_fwhm`/`rocking_fwhm`
(mosaic), `tth_fwhm`/`strain_breadth` (strain), `n_bins` (spatial prevalence),
`center_row/col`, `detector_x/y`, and `intensity_profile` (per-bin map → device
map). `aggregate.py` already emits two tidy tables — **`features`** (one row/feature)
and **`device_map`** (one row per scan×reflection×bin) — to CSV + SQLite.

**Gap:** `core/aggregate.py` exists but is **not wired to a CLI command** yet.

---

## 5. What's new (modules to build)

Keep logic in `core/` (pure), expose via `cli.py`, GUI later. New pieces:

- **A. `xrd-app aggregate`** — thin CLI over existing `core/aggregate.py`
  (`--scans`, `--bin-size`, `--out`). Produces `features.csv`, `device_map.csv`,
  `study.db`. *Foundation for everything cross-scan.*
- **B. `core/tracking.py`** — link shapes **across θ** (the rocking analogue of the
  spatial Union-Find): match by de-skewed (X,Y) + reflection band + χ-continuity
  → **tracks** (track_id, per-θ membership, position drift, χ(θ)). Requires
  cross-scan **registration** (align grids; correct stage drift between θ scans).
  Output: `tracks.csv` / `tracks.json`.
- **C. `core/rocking.py`** — per-track **rocking curve**: intensity(θ) (and
  integrated intensity) → fit θ_Bragg, FWHM (mosaicity), amplitude; flag tracks
  too sparsely sampled in θ to fit. Output: `rocking_curves.csv`.
- **D. `core/prediction.py`** — the **predictor**: from reflections + geometry +
  early/strong tracks, forecast the per-θ expected shape catalog (reflection,
  position, χ, approx intensity). Then **compare** to observed: precision/recall
  of predicted-vs-detected, rocking-fit residuals, χ(θ)-smoothness. Output:
  `prediction_report.{json,md}` with the headline "do they appear?" metrics.
- **E. `core/combined_device.py`** — fuse `device_map` across θ into **one spatial
  canvas**: per (X,Y) bin, summarize across θ (e.g. max intensity, argmax-θ =
  local orientation, per-reflection layers, or track-id labeling). Pure data layer
  returning arrays + a saved `combined_device.npz`/`.json`. **No GUI yet** — this
  is exactly the data a future "combined device view" tab will render.

Each new `core/` module gets a thin `cli.py` command and is unit-testable headless.

---

## 6. Execution plan (phases + exact commands)

Run from the project root (`TakaTest/MountTest` or a dedicated study project).
Quote paths (spaces). Write `Binned/` to a non-OneDrive location.

### Phase 0 — Confirm scope (cheap, do first)
- **DONE (2026-06-29):** completeness scan of 203–214. **11 usable** (203, 204,
  205, 207–214; all XRD=151 / SOCKETSERVER=195). **Scan_0206 (θ=6.0) is
  incomplete (XRD=3) — excluded.** Native grid ≈ 151 rows for this sample.
- Confirm SOCKETSERVER positions exist per scan; rebuild if missing
  (`recreate-positions`). Record per-scan grid dims (rows×cols) — they may differ.
- Decide bin size(s): **1×1 (requested) and/or 3×3 (recommended working layer).**
- **Decide where to bin** (APS vs local) — gates the whole timeline (§3).

### Phase 1 — Bins (the big read; ~3.5 h/scan over WAN)
```bash
# discovery + shared inputs (once)
xrd-app scan-detect --root . --scans-dir /mnt/z/isn/2026-1/2026-1-Luo/Raw
xrd-app link --root . --tth <tth.tiff> --position-root /mnt/z/.../<positions>

# 1×1 bins for the series (large + slow — overnight/batched)
xrd-app batch --root . --scans "203,204,205,206,207,208,209,210,211,212,213,214" \
              --bin-size 1 --skip-existing
# (batch runs grid→bin→peaks→shapes; auto-skips incomplete scans)
```
*If splitting binning from analysis, use `make-bins --bin-size 1 --scan Scan_0203`
per scan, then Phase 2.*

### Phase 2 — Peaks + shapes per scan
Covered by `batch` above. To (re)run analysis only on existing bins:
```bash
xrd-app run-pipeline --root . --scan Scan_0203 --bin-size 1 --snr 4.0
```
Physics check (per CLAUDE.md): detected peaks must fall in the expected 2θ bands.

### Phase 3 — Aggregate (build A)
```bash
xrd-app aggregate --root . --scans "203,...,214" --bin-size 1 --out Study/
# → Study/features.csv, Study/device_map.csv, Study/study.db
```

### Phase 4 — Track across θ (build B)
```bash
xrd-app track --root . --scans "203,...,214" --bin-size 1 \
              --match-tol <px/µm> --out Study/tracks.json
```

### Phase 5 — Predict & compare (builds C, D)
```bash
xrd-app rocking  --root . --tracks Study/tracks.json --out Study/rocking_curves.csv
xrd-app predict  --root . --tracks Study/tracks.json --reflections <refl> \
                 --out Study/prediction_report.md
```
Headline outputs: predicted-vs-observed precision/recall, rocking θ_Bragg & FWHM,
χ(θ) smoothness, and 203-vs-214 repeatability as the noise floor.

### Phase 6 — Combined device-view data layer (build E)
```bash
xrd-app combined-device --root . --device-map Study/device_map.csv \
                        --tracks Study/tracks.json --out Study/combined_device.npz
# arrays: per-(X,Y) max-intensity, argmax-θ (orientation), per-reflection layers
```

### Phase 7 — (LATER, on request) GUI
Wire a "Combined Device View" tab over `combined_device.*` + a tracks overlay.
**Not in this plan** — explicitly deferred.

---

## 7. Outputs / deliverables
- `Binned/Scan_02NN/xrd_1x1_bins.h5` (+ optional 3×3) on non-OneDrive disk.
- `Labels/Scan_02NN/…` per-scan peaks + shapes catalogs.
- `Study/features.csv`, `device_map.csv`, `study.db`.
- `Study/tracks.json`, `rocking_curves.csv`.
- `Study/prediction_report.md` — **the answer to "do the predicted shapes appear?"**
- `Study/combined_device.npz` — data behind the future combined device view.

---

## 8. The `/goal` prompt (paste this to kick it off)

> **Goal:** Execute the rocking-series study in `ROCKING_STUDY_203-214.md` for
> sample `5%_DI_Yes_GB`, scans 203–214, at **1×1 bins** (also build 3×3 as the
> fast working layer). Steps: (0) confirm per-scan completeness & positions and
> tell me the usable scan list and the binning location decision before the long
> read; (1) bin all usable scans; (2) run peak + shape finding per scan; (3) add
> and run `xrd-app aggregate`; (4) build `core/tracking.py` + `xrd-app track` to
> link shapes across θ into grain tracks; (5) build `core/prediction.py` +
> `core/rocking.py` to predict which shapes should appear across the θ sweep, fit
> rocking curves, and produce `Study/prediction_report.md` comparing predicted vs
> observed (use 203 vs 214 as the repeatability floor); (6) build
> `core/combined_device.py` + `xrd-app combined-device` to fuse all θ into one
> spatial device-view dataset. Keep logic in `core/`, expose each step as a CLI
> command, unit-test headless, and **do not build the GUI** — stop after the
> combined-device data layer and show me the report. Pause for my OK before the
> ~42 h binning read and before each new module.

---

## 9. Open questions / decisions for you
1. **Bin: 1×1 only, or 1×1 + 3×3?** (Recommend both; 3×3 drives tracking, 1×1 for
   final maps.)
2. **Bin at APS or on this laptop?** (APS ≈ <1 h downloads vs ~42 h here.)
3. **Storage target** for `Binned/` (ext4 `/` recommended; needs ~0.7 TB for 1×1).
4. **Cross-θ registration:** is the spot stable enough to align on de-skewed
   (X,Y), or do we need feature-based registration for stage drift?
5. **Match tolerance** for calling two shapes "the same grain" across θ (px/µm).
6. Include the clustered-θ caveat in fits, or request a uniform re-scan later?
