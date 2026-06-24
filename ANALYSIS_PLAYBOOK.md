# Analysis Playbook — driving xrd-app from chat

**Audience: Claude, acting as an analysis copilot for a beamline scientist who
does not want to write code.** The researcher asks questions in plain language
("how many features in scan 203?", "how spread out is the orientation?"); you
answer by running `xrd-app` CLI commands and reading the result files it writes,
then summarizing. This file maps the questions they ask to the commands/fields
that answer them. Pair it with `CLAUDE.md` (science context) and
`xrd_app/CLAUDE.md` (app architecture).

## Operating mode

- **You run the tools; the researcher reads conclusions.** Run CLI commands,
  read the JSON/CSV outputs, compute the number, and report it in their
  vocabulary (peaks, shapes, reflections, orientation, intensity). Don't make
  them look at raw JSON unless they ask.
- **Never launch the GUI to answer a data question** (`xrd-app gui` blocks and
  needs a display). The GUI and the CLI read the *same* files — everything the
  GUI shows is answerable from the CLI + result files.
- **Read-only first.** `status`, `lineage`, `detectors`, and reading files in
  `Labels/` are cheap and safe — do them freely. The *processing* commands
  (`grid`, `bin`, `peaks`, `shapes`, `batch`, `run-combined`) are **heavy**
  (seconds–minutes per bin, minutes–hours per scan). Confirm with the researcher
  before kicking one off, and say roughly how long it'll take.
- **Start a session by orienting yourself**, not by asking the user to explain
  their project:
  ```bash
  xrd-app status                 # project, scan, frame shape, which inputs resolve ✓/✗
  xrd-app status --bin-size 1    # repeat per bin size of interest (1,3,4,5)
  xrd-app lineage                # every result JSON in Labels/<scan>/ + its provenance
  ls Labels/*/                   # the actual result files on disk
  ```

## The pipeline & what each step writes

```
raw HDF5 frames
  └─ grid   → Metadata/<scan>/grid_mapping_<N>x<N>.json   (frame → spatial bin)
  └─ bin    → Binned/<scan>/xrd_<N>x<N>_bins.h5           (summed CCD per bin)
  └─ peaks  → Labels/<scan>/<algo>_peaks_<N>x<N>.json     (Phase 1: per-bin detections)
  └─ shapes → Labels/<scan>/<algo>_shapes_<N>x<N>.json    (Phase 2: linked, validated)
              + Labels/<scan>/feature_catalog_<N>x<N>.json (kept shapes, viewer format)
              + Labels/<scan>/kept_peaks_<N>x<N>.csv        (the shapes, as a table)
              + Labels/<scan>/filtered_peaks_<N>x<N>.csv     (rejected clusters)
```

- A **peak** is a single-bin detection (may be noise). A **shape** is a peak
  cluster that links across neighboring bins and passes the gaussian-profile
  filter — the *physically trustworthy* feature. **Default to shapes when the
  researcher says "feature"** unless they explicitly want raw peaks.
- `run-pipeline` = `peaks` then `shapes` for one scan. `batch` = full
  `grid→bin→peaks→shapes` over many scans. `run-combined` = a 1×1 per-frame
  algorithm that does detect+link+verify in one pass (its output has **null
  intensities** — see caveats).
- `N` is the spatial bin size (1, 3, 4, 5). Always know which bin size a question
  is about; results are per-bin-size.

## Where the answers live (result-file schemas)

### `<algo>_peaks_<N>x<N>.json` (Phase 1)
Top level: `n_peaks`, `n_bins_with_peaks`, `detector`, `snr`, `bin_size`,
`peaks_by_bin: {"<row>_<col>": [peak, ...]}`. Each **peak**:
| field | meaning |
|---|---|
| `x`, `y` | detector pixel column/row of the peak |
| `snr` | signal-to-noise of the detection |
| `label` | which reflection band it fell in (PbI2, (001), …) |
| `cleaned_intensity` | peak height after background subtraction |
| `integrated_intensity` | integrated counts (if present) |

### `<algo>_shapes_<N>x<N>.json` (Phase 2) — the main analysis object
Top level: `n_kept` (= number of shapes), `n_filtered`, `link_tolerance`,
`bin_size`, `kept: [feature, ...]`, `filtered: [...]`. Each **feature/shape**:
| field | meaning — answers… |
|---|---|
| `feature_id` | stable id within the set |
| `reflection` | assigned reflection (PbI2/(001)/(011)/(111)/(002)/ITO/(012)/(112)) — **"how many of each phase?"** |
| `n_bins` | how many spatial bins the feature spans — **"how large / how big is it?"** |
| `spatial_extent` | the list of bin keys it covers (its footprint on the sample) |
| `center_bin`,`center_row`,`center_col` | where on the sample grid it sits — **"where is it?"** |
| `detector_x`,`detector_y` | mean detector position |
| `peak_intensity` | brightest intensity in the feature — **"how strong / brightest?"** |
| `mean_snr` | mean SNR across its bins — confidence |
| `chi_deg` | azimuthal angle on the detector — **"what orientation?"** |
| `chi_fwhm` | spread of χ across the feature's bins — **"orientation spread / mosaicity / rocking width"** (needs ≥3 bins) |
| `tth_fwhm` | FWHM of Δ2θ across bins — radial/"strain-like" breadth. **NOT a calibrated strain** (see caveats) |
| `ref_tth` | the catalog 2θ for that reflection |
| `intensity_profile` | `{bin_key: {intensity, integrated, det_x, det_y, tth, chi}}` — the per-bin **spatial intensity map** of this one feature |
| `reason` | why it was kept (or filtered) |

> Legacy field names: `chi_fwhm` was once `rocking_fwhm`; `tth_fwhm` was once
> `strain_breadth`. `core/aggregate.py` accepts both — prefer the new names.

### `feature_catalog_<N>x<N>.json`
The `kept` shapes only, in the viewer/device-map format (same fields). This is
what the Device and Orientation maps are built from. `core/aggregate.py` turns
these into two tabular views you can compute yourself when needed:
- **feature rows** — one row per shape (the per-feature fields above, plus
  `mean_intensity`, `sum_integrated`).
- **device-map rows** — one row per (feature, bin): `row, col, intensity,
  integrated, tth, chi` — the data behind a spatial heatmap across the sample.

### `Raw/scans.json` & `config.yaml`
Scan inventory: per scan `n_files`, `n_frames`, `shape` (detector dims) —
**"how big is the scan / how many frames / detector size?"** `config.yaml` holds
the active scan, paths, and `detector.shape`.

## Question → recipe

Run the command, then read the field. Prefer `jq` or a small `python3 -c`/pandas
read; **summarize, don't dump** (these files can be large). Always quote paths
(the project path has spaces).

| Researcher asks… | How to answer |
|---|---|
| "How many features/shapes in scan 203?" | `n_kept` in `<algo>_shapes_3x3.json` (or count `kept`). |
| "How many raw peaks before filtering?" | `n_peaks` / `n_bins_with_peaks` in `<algo>_peaks_*.json`. |
| "Breakdown by reflection / phase?" | group `kept[].reflection` and count. |
| "How large are the features?" | distribution of `n_bins` (and `spatial_extent` length); report min/median/max. |
| "Which is the strongest / brightest?" | max `peak_intensity` over `kept`; report its `reflection`, `center_row/col`. |
| "What's the orientation, and how spread out?" | `chi_deg` per feature; spread = `chi_fwhm` (per feature) or the stdev of `chi_deg` across features. |
| "Any strain / radial broadening?" | `tth_fwhm` distribution — **caveat it's uncalibrated breadth, not strain**. |
| "Map intensity across the sample." | build device-map rows (per-bin `intensity`) from `feature_catalog`; describe hot regions by `row,col`. |
| "Compare two algorithms / two scans." | read both result JSONs, diff `n_kept`, per-reflection counts, intensity stats. |
| "Which detector is best?" | `xrd-app detectors --kind peak\|shape\|combined` — holdout F1/F2 table (**F2 is the primary metric**). |
| "How big is the scan / how many frames?" | `xrd-app status` or `Raw/scans.json` (`n_frames`, `n_files`, `shape`). |
| "What's been run already / where did this file come from?" | `xrd-app lineage` (provenance: detector, snr, link tolerance, peak source). |
| "Did detection use the right settings?" | `snr`, `link_tolerance`, `detector` in the result JSON top level + `lineage`. |

Example reads (adapt names from `ls Labels/<scan>/`):
```bash
# counts + per-reflection breakdown from a shapes file
jq '{shapes:.n_kept, filtered:.n_filtered,
     by_reflection:(.kept|group_by(.reflection)|map({(.[0].reflection):length})|add)}' \
  "Labels/Scan_0203/gaussian_shapes_3x3.json"

# size + intensity + orientation summary with pandas
python3 -c '
import json,statistics as st
d=json.load(open("Labels/Scan_0203/gaussian_shapes_3x3.json"))["kept"]
print("shapes:",len(d))
print("n_bins  min/med/max:",min(f["n_bins"] for f in d),
      st.median(f["n_bins"] for f in d),max(f["n_bins"] for f in d))
chi=[f["chi_deg"] for f in d if f.get("chi_deg") is not None]
print("chi_deg mean/stdev:",round(st.mean(chi),1),round(st.pstdev(chi),1))
'
```

## Interpretation guardrails (state these when relevant)

- **Peaks vs shapes:** a big `n_peaks` with small `n_kept` is normal — most raw
  peaks are noise; shapes are the validated features. Lead with shapes.
- **`tth_fwhm` is not strain.** It's the radial breadth (Δ2θ FWHM) across a
  feature's bins; calling it strain requires calibration this pipeline doesn't
  do. Report it as "radial breadth" and flag the caveat.
- **`chi_deg` / `chi_fwhm` depend on an *estimated* beam center** (fit from the
  tth map), so treat absolute χ as approximate; relative spread is more robust.
  `chi_fwhm` needs ≥3 bins, so single-bin features won't have it.
- **Combined (1×1) outputs have null `peak_intensity`/`mean_snr` and empty
  `intensity_profile`** — intensity/strain/orientation questions can't be
  answered from a combined run; you need a `peaks`→`shapes` run for those.
- **Fast `scan-detect` frame counts are estimates** (`~` prefix); use
  `scan-detect --deep` for exact counts before quoting a precise number.
- **Physics sanity check:** detected peaks should sit in the expected 2θ band for
  their reflection. If counts look implausible or a reflection is wildly
  over-represented, suspect SNR threshold or a calibration/tth issue before
  trusting a metric.

## Clarifying questions worth asking the researcher

Ask before computing when the answer depends on it (otherwise pick a sensible
default and say which you used):
- **Which scan and which bin size?** (default to the config scan; 3×3 is the
  common working size — confirm.)
- **Which algorithm's results?** (multiple `*_shapes_*.json` may exist — list
  them with `lineage` and ask, or use the most recent.)
- **Shapes (validated) or raw peaks?** Default: shapes.
- **All features or a specific reflection / region of the sample?**
- For a heavy run: **OK to spend the compute?** (give the ETA first).

## Lowering friction (optional setup)

If the researcher will do many analysis sessions, propose allow-listing the
read-only operations in `.claude/settings.local.json` so you aren't prompted each
time: `xrd-app status`, `xrd-app lineage`, `xrd-app detectors`, `jq`, and
read-only `python3 -c` over `Labels/`. Keep the heavy processing commands
*prompted* on purpose, so a long run is always a deliberate choice.
