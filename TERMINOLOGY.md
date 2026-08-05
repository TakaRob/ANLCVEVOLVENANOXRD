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
| **peak** | A single raw detection inside **one** spatial bin: a coordinate + intensity/SNR. The direct output of a detector algorithm. | Phase 1 (detection) | `*_peaks_NxN.h5` → `peaks_by_bin: {bin_key: [peak, ...]}` |
| **member** | One peak after it has been linked into a cluster. Internally a tuple `(bin_key, peak_index, row, col, x, y, peak_dict)`. | Phase 2 (linking) | in-memory only (`processing.py`) |
| **feature** | A linked cluster of members across adjacent bins that passed the Gaussian-profile filter. **This is the physical Bragg reflection / spot.** Fully characterized (center, extent, intensity profile, metrics). | Phase 3 (characterization) | `*_shapes_NxN.h5` under `kept`, or `*_combined_NxN.h5` under `features` |

**shape** is the **stage name** for the verified feature — the same object as
*feature*, named after the Phase-2 "shape finding" step that produces it (linking
+ gaussian-profile filter). It is the canonical name for that stage and its
algorithm-kind across the *public surface*: the `ShapeAlgorithms/` directory, the
`xrd-app shapes` / `run-shapes` CLI commands, `--source shapes`, `kind="shape"`,
the on-disk `*_shapes_NxN.h5` files, `config.shapes_json()`, and
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

### Coordinate source (de-skew at the source)

The per-frame `(row, col)` is assigned **once**, in `core/io.py::generate_grid_mapping`,
and every downstream stage (binning, peak bin-keys, cross-bin linking, device maps,
aggregate) inherits it. The grid mapping HDF5 records which lattice it is on via
`coordinate_source`:

| `coordinate_source` | How `(row, col)` is assigned | When |
|---|---|---|
| `positions_xy` | Both axes are snapped from measured `(X,Y)` without clipping to a commanded rectangle. | Default at 1×1 and for irregular scans that still need a rectangular grid. |
| `positions_faithful` | Exact file-index rows plus measured fast-axis columns on an approximately square display lattice. | Default at N≥2 for clean file-per-row scans. |
| `positions_faithful_native` | Exact file-index rows plus measured fast-axis columns at native frame density. | Detection/recall work where native sampling matters more than square display pixels. |
| `file_per_row` | File index and within-file rank (the **commanded** fast-axis position), serpentine-aware. | Selected with `--deskew-method commanded`. |
| `territory_xy` | Irregular cells and adjacency from true `(X,Y)` positions. | Built with `territory-grid` when a rectangular lattice is inappropriate. |

> **Why commanded and coordinate methods both exist.** On these scans the even/odd serpentine rows' encoder Y diverges by up to ±33 columns at nominally identical commanded positions because of stage **backlash**. Treating that divergence as a rigid row shift can throw adjacent parts of one feature tens of columns apart. `commanded` preserves within-row rank; `positions_xy` uses measured coordinates as a complete lattice; `faithful*` preserve exact row membership while controlling fast-axis placement. Territory cells are the physical-neighbor option for genuinely irregular layouts.

**Missing position CSV.** The SOCKETSERVER-derived `scan_NNNN_position.csv` (µm; *not* TETRAMM, which is too coarse) is sometimes absent. `xrd-app grid` then **recreates** it from the file-per-row layout (`xrd-app recreate-positions`; tagged with a `# xrd-app coordinate_source=file_per_row` marker so loaders don't mistake it for a real export) and builds a `file_per_row` grid — so downstream never zero-pads positions. Absolute µm scale of a recreated CSV is nominal (`--step-x/--step-y`); the lattice is exact.

**Orientation is preserved** vs. the legacy serpentine grid: here `row ↔ X`
(`corr(row, X) = −1.0`) and `col ↔ Y`, anchored by a reference serpentine pass so
device maps render upright unchanged. The de-skew only fixes the *column*
registration that stage backlash skewed (`corr(col, Y)` 0.585 → 1.0); it does not
transpose or flip the map. A legacy mapping with no `coordinate_source` key is
treated as `serpentine`. To regenerate a scan on the new coordinates: re-run
`xrd-app make-bins` / `grid` / `peaks` / `shapes`.

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
> (`tth − ref_tth`) across bins — the *radial* spread. It had the same overclaim
> problem as "rocking" (it is not a calibrated strain) and was renamed the same
> way: field `strain_breadth` → **`tth_fwhm`**, with honest UI labels. See §3.3.

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

The canonical HDF5 shapes catalog stores this field as `chi_fwhm`.

### 3.3 The "strain" misnomer (radial metric)

The radial metric is `Δ2θ = tth − ref_tth`: the **deviation of the measured 2θ
from the reference Bragg angle**, per bin. The per-feature version
(`strain_breadth`) is its intensity-weighted FWHM across the feature's bins.

This is **not** calibrated lattice strain. True strain needs the conversion
`ε = −½·Δ(2θ)·cot θ_B` *and* the assumption that the 2θ shift is purely a
d-spacing change (not tilt, displacement, or detector geometry). The app applies
no such conversion — it only reports Δ2θ — so "Lattice strain", "d-spacing
deviation", and "lattice parameter gradient" overclaim a measurement this app
does not make. This is the radial twin of the "rocking" misnomer in §3.1.

### 3.2.1 The rename

| Where | Now | Canonical |
|---|---|---|
| Feature field (`ShapeAlgorithms/gaussian.py`, `aggregate.py`) | `strain_breadth` | **`tth_fwhm`** |
| Device-map metric key — per-bin (`device_map.py`) | `strain` | **`tth_dev`** |
| Device-map metric key — per-feature (`device_map.py`) | `strain_bw` | **`tth_breadth`** |
| Device-map label — per-bin (`METRICS`) | `Lattice strain` | **`2θ deviation (Δ2θ)`** |
| Device-map label — per-feature (`METRICS`) | `Strain breadth` | **`Radial breadth (Δ2θ FWHM)`** |
| Per-bin description (`device_map.py`) | `Δ2θ — distance from the reference Bragg angle` | **`Δ2θ — deviation of the measured 2θ from the reference Bragg angle per bin (not a calibrated strain)`** |
| Per-feature description (`device_map.py`) | `Spread of Δ2θ across the feature (strain gradient)` | **`Radial breadth — FWHM of Δ2θ across the feature's bins (not a calibrated strain gradient)`** |
| Per-bin 2D title (`device_map.py`) | `Lattice Strain — d-spacing deviation (Δ2θ from reference)` | **`2θ Deviation — Δ2θ from the reference Bragg angle per bin`** |
| Per-feature 2D title (`device_map.py`) | `Strain Breadth — lattice parameter gradient across feature` | **`Radial Breadth — Δ2θ FWHM per feature`** |
| Colorbar z-labels (`device_map.py`) | `Δ2θ (°)` / `FWHM Δ2θ (°)` | unchanged — already honest, keep |
| Tab help (`tabs/device.py`, `tabs/shape_verify.py`) | "...lattice strain...", "...strain breadth..." | "...2θ deviation (Δ2θ)...", "...radial breadth (Δ2θ FWHM)..." |
| Module docstring (`device_map.py:5`) | "...lattice strain ... mosaicity / domain structure" | "...2θ deviation (Δ2θ) ... azimuthal / radial breadth" |

**Why `tth_fwhm`/`tth_dev` and not "strain":** the app reports only Δ2θ, never
the cot θ_B conversion to true strain. Naming after 2θ keeps the claim honest and
parallels the χ rename (`chi_fwhm`/`chi_breadth`). FWHM is shift-invariant, so the
FWHM of Δ2θ equals the FWHM of 2θ across a feature's bins (one reflection → one
`ref_tth`), making `tth_fwhm` exact.

The canonical HDF5 shapes catalog stores this field as `tth_fwhm`. The
per-bin/per-feature metric keys (`tth_dev` / `tth_breadth`) are not persisted.

---

## 4. Per-bin vs per-feature metrics (device map)

A device-map gotcha worth stating once: some metrics are **per-bin** (a real value
in each bin) and some are **per-feature** (one value painted into every bin of the
feature). Keep this distinction visible in tooltips so the map isn't misread.

| Metric (key) | UI label | Granularity | Source |
|---|---|---|---|
| `intensity` | Intensity | per-bin | `intensity_profile[bin].integrated` |
| `tth_dev` | 2θ deviation (Δ2θ) | per-bin | `tth − ref_tth` per bin |
| `chi` | χ angle | per-feature | `feat.chi_deg` |
| `chi_breadth` | Azimuthal breadth (χ FWHM) | per-feature | `feat.chi_fwhm` |
| `tth_breadth` | Radial breadth (Δ2θ FWHM) | per-feature | `feat.tth_fwhm` (legacy `strain_breadth`) |

---

## 5. Quick glossary

- **peak** — raw single-bin detection. Pre-linking.
- **member** — a peak inside a linked cluster (internal tuple).
- **feature** — the final linked, filtered Bragg reflection/spot. The unit of analysis.
- **shape** — the same verified feature, named after the Phase-2 "shape finding" stage; the canonical name for that stage, its algorithm-kind, CLI (`shapes`), and `*_shapes.h5` files.
- **reflection** — the hkl label/class a feature belongs to (a grouping, not an object).
- **point** — a manual ground-truth annotation, or genuine geometry (click location, "entry point"). Never a detection.
- **bin / bin_key** — a spatial scan cell; key is `"row_col"`.
- **detector_x / detector_y** — pixel position on the detector.
- **center_row / center_col** — the feature's central bin in the scan grid.
- **spatial_extent / n_bins** — feature footprint (bins) and its count.
- **intensity_profile** — per-bin intensity + geometry for a feature.
- **chi_fwhm** (was `rocking_fwhm`) — FWHM of χ across the feature's bins. UI label **"Azimuthal breadth (χ FWHM)"**.
- **tth_fwhm** (was `strain_breadth`) — FWHM of Δ2θ across the feature's bins. UI label **"Radial breadth (Δ2θ FWHM)"** — *not* a calibrated strain (§3.3).
- **tth_dev** (metric key; was `strain`) — per-bin `tth − ref_tth`. UI label **"2θ deviation (Δ2θ)"** — deviation from the reference Bragg angle, *not* calibrated lattice strain (§3.3).
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
- [x] `strain` misnomer (§3.3): field `strain_breadth`→`tth_fwhm` (`ShapeAlgorithms/gaussian.py` emit; `aggregate.py` column); device-map metric keys `strain`→`tth_dev`, `strain_bw`→`tth_breadth` across `METRICS`/`METRIC_ZLABELS`/`METRIC_DESCRIPTIONS`/`METRIC_2D_TITLES`/`PER_FEATURE_METRICS` + comparisons; labels "Lattice strain"→"2θ deviation (Δ2θ)", "Strain breadth"→"Radial breadth (Δ2θ FWHM)"; module docstring; `tabs/device.py` + `tabs/shape_verify.py` help. Legacy `strain_breadth` read-fallback in `device_map.build_device_grids` and `aggregate._feature_row`.

Deliberately **not** changed (load-bearing identifiers / correct usage):

- **`shape` stage identifier** — blessed as canonical (see §1): `ShapeAlgorithms/`, `xrd-app shapes`, `kind="shape"`, `*_shapes.h5`, `config.shapes_json()`, `bins_from_shapes()` all stay.
- **`*_peaks.h5` / `filtered_peaks_*.csv`** — canonical on-disk filename conventions.
- **`.shape`** (numpy/detector/grid dimensions) — unrelated to morphology.
- **"LoG blob detection"** — the standard CV name for the *technique*, not an output-object noun (§1).

---

## 7. Persistence contract

Numerical peaks, shapes, combined, ROI, and HD catalogs are HDF5. A feature is
not a separate catalog kind: it is a record under shapes `kept` or combined
`features`. Grid mappings are `grid_mapping_NxN[_variant].h5`, and cross-scan
tracks are stored in `Study/tracks.h5`.

The older code-occurrence inventory is omitted because it described a removed
catalog format rather than the canonical vocabulary.
