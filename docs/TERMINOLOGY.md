# XRD-App Terminology — Single Source of Truth

This document is the canonical vocabulary for the xrd-app. Code, UI labels, JSON
fields, docstrings, and docs should all use these terms and only these terms. It
exists because the same concept is currently named several different ways
(*peak* / *point* / *feature* / *shape*), and because the device map uses
**rocking** for a quantity that is not a rocking curve and is not measured from
rocking data.

When you add or rename anything user-facing or in the data schema, make it match
this file. If a term here is wrong, fix it *here first*, then propagate.

---

## 1. The core object chain

Detection produces objects at three stages. They are **not** synonyms — each is a
distinct stage with a distinct name. Do not use "peak" for all of them.

| Canonical term | What it is | Stage | Lives in |
|---|---|---|---|
| **peak** | A single raw detection inside **one** spatial bin: a coordinate + intensity/SNR. The direct output of a detector algorithm. | Phase 1 (detection) | `*_peaks.json` → `peaks_by_bin: {bin_key: [peak, ...]}` |
| **member** | One peak after it has been linked into a cluster. Internally a tuple `(bin_key, peak_index, row, col, x, y, peak_dict)`. | Phase 2 (linking) | in-memory only (`processing.py`) |
| **feature** | A linked cluster of members across adjacent bins that passed the Gaussian-profile filter. **This is the physical Bragg reflection / spot.** Fully characterized (center, extent, intensity profile, metrics). | Phase 3 (characterization) | `feature_catalog_NxN.json` |

**shape** is the **stage name** for the verified feature — the same object as
*feature*, named after the Phase-2 "shape finding" step that produces it (linking
+ gaussian-profile filter). It is the canonical name for that stage and its
algorithm-kind across the *public surface*: the `ShapeAlgorithms/` directory, the
`xrd-app shapes` / `run-shapes` CLI commands, `--source shapes`, `kind="shape"`,
the on-disk `*_shapes_NxN.json` files, `config.shapes_json()`, and
`bins_from_shapes()`. Read "a shape" as "a verified feature." Use **feature** when
talking about the data record/object; use **shape** when talking about the
stage, its algorithm-kind, the CLI, or its output files. They are not competing
terms — they name the same thing from two angles.

Rules:

- **peak** = raw, per-bin, pre-link. Never call a characterized object a "peak".
- **feature** = the final object the user inspects, exports, and maps. The device
  map, viewer, and aggregate CSV all operate on **features**, not peaks.
- **shape** = the stage/kind name for that same verified feature (see above).
- **member** is an implementation detail of linking — keep it out of the UI.

### Banned / discouraged words

| Word | Status | Use instead |
|---|---|---|
| **point** | Reserve **only** for manual annotation in the labeling tool (a user-placed ground-truth mark) and for genuine geometry (a click location, "entry point"). Never use "point" for a *detection*. | *peak* (raw) or *feature*/*shape* (verified) |
| **spot** | Banned as a noun for a detection. | *peak* or *feature*/*shape* |
| **blob** | Allowed **only** in the name of the detection *technique* (e.g. "LoG blob detection" — the standard CV term). Never as a noun for an output object. | (technique name OK) / else *peak* |
| **shape** | **Canonical** — but only as the Phase-2 stage/kind/object name (see above), and `.shape` for array dimensions. Do **not** use bare "shape" to mean a feature's *morphology*. | for morphology → *morphology* (see §3) |
| **reflection** | Keep — but it means the **hkl class/label** a feature belongs to (e.g. its `reflection` field), *not* an individual detected object. | — |

---

## 2. Coordinates and frames

A feature lives in two reference frames. Pick the right pair and **always use the
full names** in schemas, function signatures, and exports. Abbreviations
(`det_x`, `row`) are tolerated only as short-lived loop variables.

| Frame | Canonical fields | Meaning |
|---|---|---|
| **Detector frame** | `detector_x`, `detector_y` | Pixel position on the detector array. |
| **Scan / bin-grid frame** | `center_row`, `center_col` | The bin a feature is centered in. A `bin_key` is the string `"row_col"`. |

Fix the existing inconsistency: `aggregate.py` uses `det_x`/`det_y` and
`row`/`col` in the device-map columns while `processing.py` emits
`detector_x`/`detector_y` and `center_row`/`center_col`. **Standardize on the
full names everywhere.**

Note the scan-grid axes are **spatial scan positions**, not angles. In the device
map they are correctly labeled `Col (scan x)` and `Row (scan y)` — keep that.

---

## 3. Morphology and the "rocking" misnomer

A feature's morphology is described by:

- **`spatial_extent`** — the list of `bin_key`s the feature occupies (its footprint).
- **`n_bins`** — how many bins that is (the count).
- **`intensity_profile`** — per-bin `{intensity, integrated, det_x, det_y, tth, chi}`.
- two FWHM-style spread metrics derived from the profile (below).

There is no "morphology" object in code; *morphology = extent + profile*. Use the
word "morphology" in prose; use the concrete field names in code.

### 3.1 The problem: `rocking_fwhm` is not a rocking curve

`feature["rocking_fwhm"]` (`processing.py:309-315`) is the intensity-weighted
**FWHM of the azimuthal angle χ** across the feature's bins:

```python
var = Σ wₙ (χ − μ)²          # weighted variance of χ over bins
rocking_fwhm = 2.3548 · √var  # 2.3548 = 2√(2 ln2)  → FWHM
```

This is **azimuthal (χ) spread along the Debye ring**. It is **not**:

- a rocking curve (an angular scan of the crystal through the Bragg condition), and
- not derived from any rocking data — **the app ingests no rocking scans at all.**

Calling it "rocking" implies a measurement this app never makes. Same issue for
the matching device-map metric label, descriptions, and tab help text.

> Sibling metric `strain_breadth` is the analogous weighted FWHM of `Δ2θ`
> (`tth − ref_tth`) across bins — the *radial* spread. That name is fine; it is
> only the *azimuthal* one that is mislabeled "rocking".

### 3.2 The rename

| Where | Now | Canonical |
|---|---|---|
| Feature field (`processing.py:315`, `aggregate.py`) | `rocking_fwhm` | **`chi_fwhm`** |
| Device-map metric key (`device_map.py` `METRICS`, `PER_FEATURE_METRICS`) | `rocking` | **`chi_breadth`** |
| UI dropdown label (`device_map.py:68`) | `Rocking width` | **`Azimuthal breadth (χ FWHM)`** |
| Colorbar z-label (`device_map.py:76`) | `FWHM χ (°)` | `FWHM χ (°)` — already correct, keep |
| Metric description (`device_map.py:87`) | `Rocking-curve FWHM — mosaic spread / plane curvature` | **`χ-breadth — FWHM of azimuthal angle across the feature's bins (no rocking data involved)`** |
| 2D title (`device_map.py:96`) | `Rocking Width — crystal plane curvature / mosaic spread per feature` | **`Azimuthal Breadth — χ FWHM per feature`** |
| Tab help (`tabs/device.py:15`, `tabs/shape_verify.py:16`) | "...rocking width...", "...rocking-curve FWHM..." | "...azimuthal breadth (χ FWHM, computed from the χ distribution across bins)..." |
| Aggregate column doc (`aggregate.py:8`) | `shape (rocking_fwhm / strain_breadth)` | `morphology (chi_fwhm / strain_breadth)` |

**Why `chi_breadth`/`chi_fwhm` and not "mosaicity":** true mosaicity is measured
with a rocking scan; this is only the azimuthal spread of detected bins. Naming it
after χ keeps the claim honest and matches the already-correct colorbar.

> **Migration note:** renaming the JSON field breaks existing
> `feature_catalog_*.json` files and the device-map reader. Either regenerate
> catalogs, or have the loader accept `chi_fwhm` and fall back to `rocking_fwhm`
> for one release.

---

## 4. Per-bin vs per-feature metrics (device map)

A device-map gotcha worth stating once: some metrics are **per-bin** (a real value
in each bin) and some are **per-feature** (one value painted into every bin of the
feature). Keep this distinction visible in tooltips so the map isn't misread.

| Metric (canonical key) | Granularity | Source |
|---|---|---|
| `intensity` | per-bin | `intensity_profile[bin].integrated` |
| `strain` (Δ2θ vs ref) | per-bin | `tth − ref_tth` per bin |
| `chi` (χ angle) | per-feature | `feat.chi_deg` |
| `chi_breadth` (χ FWHM) | per-feature | `feat.chi_fwhm` |
| `strain_bw` (Δ2θ FWHM) | per-feature | `feat.strain_breadth` |

---

## 5. Quick glossary

- **peak** — raw single-bin detection. Pre-linking.
- **member** — a peak inside a linked cluster (internal tuple).
- **feature** — the final linked, filtered Bragg reflection/spot. The unit of analysis.
- **shape** — the same verified feature, named after the Phase-2 "shape finding" stage; the canonical name for that stage, its algorithm-kind, CLI (`shapes`), and `*_shapes.json` files.
- **reflection** — the hkl label/class a feature belongs to (a grouping, not an object).
- **point** — a manual ground-truth annotation, or genuine geometry (click location, "entry point"). Never a detection.
- **bin / bin_key** — a spatial scan cell; key is `"row_col"`.
- **detector_x / detector_y** — pixel position on the detector.
- **center_row / center_col** — the feature's central bin in the scan grid.
- **spatial_extent / n_bins** — feature footprint (bins) and its count.
- **intensity_profile** — per-bin intensity + geometry for a feature.
- **chi_fwhm** (was `rocking_fwhm`) — FWHM of χ across the feature's bins (azimuthal breadth).
- **strain_breadth** — FWHM of Δ2θ across the feature's bins (radial breadth).
- **chi_deg** — the feature's azimuthal angle on the Debye ring.

---

## 6. Adoption checklist

Done:

- [x] `processing.py`: emit `chi_fwhm` (keep computing it from χ).
- [x] `aggregate.py`: column + doc → `chi_fwhm`; device-map columns → `detector_x/y`.
- [x] `device_map.py`: metric key `rocking`→`chi_breadth` across `METRICS`, `METRIC_ZLABELS`, `METRIC_DESCRIPTIONS`, `METRIC_2D_TITLES`, `PER_FEATURE_METRICS`; legacy `rocking_fwhm` read fallback.
- [x] `tabs/device.py`, `tabs/shape_verify.py`: fixed help text (`rocking`→azimuthal breadth; "shapes"→"features").
- [x] catalog/aggregate readers: accept `chi_fwhm` with `rocking_fwhm` fallback.
- [x] `peak`/`point` reclassification: `cli.py` combined output "points"→"features"; `viewer.py` region overlay labels "peaks"→"features" (they carry `feature_id`/`reason`); `labeling.py` comment "Detection point"→"Peak detection".

Deliberately **not** changed (load-bearing identifiers / correct usage):

- **`shape` stage identifier** — blessed as canonical (see §1): `ShapeAlgorithms/`, `xrd-app shapes`, `kind="shape"`, `*_shapes.json`, `config.shapes_json()`, `bins_from_shapes()` all stay.
- **`*_peaks.json` / `filtered_peaks_*.csv`** — on-disk filename conventions; renaming breaks existing project data. Left as-is.
- **`.shape`** (numpy/detector/grid dimensions) — unrelated to morphology.
- **"LoG blob detection"** — the standard CV name for the *technique*, not an output-object noun (§1).
