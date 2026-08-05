# PATHWAYS.md — how xrd-app moves data from raw frames to the device views

This is the data-flow map for the **non-GUI** half of `xrd_app/`: what every
stage reads, computes, and writes, the exact structure of every HDF5 / JSON /
NPZ artifact, the `core/` functions that produce them, and the `xrd-app`
command that drives each one. It complements — and defers to — `TERMINOLOGY.md`
for vocabulary and `CLAUDE.md` for architecture. Nothing here describes the GUI;
every capability below is reachable from the command line.

> **The one architectural rule:** the CLI is the engine, the GUI is a face over
> it. Every artifact in this document is produced by a `core/` function called
> from a `cli.py` command. The GUI tabs call the same `core/` functions.

---

## 0. Vocabulary (the object chain)

From `TERMINOLOGY.md` — these are *not* synonyms:

| Term | What it is | Stage | Lives in |
|---|---|---|---|
| **peak** | one raw detection inside **one** spatial bin (coordinate + intensity/SNR) | Phase 1 (detection) | `*_peaks_NxN.h5` → `peaks_by_bin` |
| **member** | a peak after it has been linked into a cluster (in-memory tuple) | Phase 2 (linking) | `processing.py` only |
| **feature** | a linked cluster of members across adjacent bins that passed the profile filter — **the physical Bragg spot** | Phase 3 (characterization) | shapes `kept` or combined `features` in HDF5 |
| **shape** | the *stage name* for a verified feature (Phase-2 "shape finding") — same object as a feature | — | `ShapeAlgorithms/`, `xrd-app shapes`, `*_shapes_*.h5` |

Read "a shape" as "a verified feature." Use **feature** for the record, **shape**
for the stage / CLI / files.

---

## 1. Project tree & path resolution

Everything resolves through `config.ProjectConfig` and `config.DataManager`
(`config.py`) — no path is ever hardcoded. A project is a directory with a
`config.yaml` and this tree (`config.PROJECT_DIRS`):

```
<project>/
  config.yaml            ProjectConfig — data_sources, scans, bins, paths
  Raw/       scans.json (registry) + linked calibration
  Binned/    <scan>/xrd_NxN_bins[_<variant>].h5     ← pre-summed detector images per bin
  Metadata/  reflections.json, tth.tiff, gui_state.json, xrf_elements.json
             <scan>/grid_mapping_NxN[_<variant>].h5, reflections.json, reflection_sum.npz, *_xrf*.npz
  Labels/    <scan>/*_peaks_*.h5, *_shapes_*.h5, *_hdmap_*.h5,
                    *_combined_*.h5, catalog_lineage.json, *.csv
  Figures/   PNG exports
  CVEvolve/  optimizer sessions + optional Hutch databases
  Study/     multi-scan rocking-study outputs (aggregate → track → rocking → predict → combined-device, RSM, qspace/)
```

### `ProjectConfig` (`config.py`)
- `ProjectConfig.load(root=".")` — walks **up** from `root` (git-style) to the
  nearest `config.yaml`, loads YAML.
- `.get(*keys, default=None)` — nested read, e.g. `cfg.get('data_sources','tth_map')`.
- `.create_tree()` — mkdir the six `PROJECT_DIRS`.
- `default_config(name, root, scan_number=None)` — the fresh config dict:
  `name`, `scan:{number,name}`, `detector:{shape}`, `scans:{}` (mirrors
  `Raw/scans.json`), `paths:{raw_dir,binned_dir,...}`, `data_sources:{raw_root,
  position_root, raw_scan_dir, position_csv, tth_map, reflections, grid_mapping,
   detector_script}`, `bins:{}` (bin_size→h5 path).


Path precedence for any input: **explicit override → `data_sources` entry in
`config.yaml` → conventional default → (for tth/reflections) bundled package
asset.**

### `DataManager` (`config.py`) — the path/naming authority
One `DataManager` can drive **many** scans (`_scan_override`), which is how the
rocking study touches 203–214 from one project root.

Scan identity / discovery:
- `scan_name_of(x)` / `scan_number_of(x)` — normalize to `"Scan_NNNN"` / int.
- `discover_scans(usable_only=False, selected_only=False)` — prefers
  `Raw/scans.json`, else scans `Binned/`+`Labels/` for `Scan_\d+`.
- `scans_registry()` / `write_scans_registry()` — read/write `Raw/scans.json`
  (`{scan_name:{dir,n_frames,shape}}`).

Per-scan directories: `labels_dir(scan)`→`Labels/<scan>`,
`binned_dir(scan)`→`Binned/<scan>`, `metadata_scan_dir(scan)`→`Metadata/<scan>`.

**File-naming builders (this is where the `_<variant>` suffixes come from).**
Every builder appends `tag = f"_{variant}" if variant else ""`:

| method | produces |
|---|---|
| `grid_mapping(bin_size, scan, variant)` | `Metadata/<scan>/grid_mapping_NxN[_variant].h5` |
| `binned_h5(bin_size, scan, variant)` | `Binned/<scan>/xrd_NxN_bins[_variant].h5` |
| `peaks_path(algo, bin_size, scan, variant)` | `Labels/<scan>/<algo>_peaks_NxN[_variant].h5` |
| `shapes_path(algo, bin_size, scan, variant)` | `Labels/<scan>/<algo>_shapes_NxN[_variant].h5` |
| `hd_map_path(algo, bin_size, scan, variant)` | `Labels/<scan>/<algo>_hdmap_NxN[_variant].h5` |
| `xrf_product(scan)` | `Metadata/<scan>/<scan>_xrf.npz` |

Current grid methods are `positions_xy`, `faithful`, `faithful_native`, and
`commanded`; `territory` identifies irregular physical cells. Variant tags flow
uniformly through grid mapping → binned HDF5 → peaks → shapes so an experiment
stays self-consistent across stages.

Resolved inputs: `raw_scan_dir()`, `xrd_frames_dir()` (`…/XRD`),
`socketserver_dir()` (`…/SOCKETSERVER`, real positions), `me7_dir()` (XRF),
`position_csv()` (h5 wins over csv), `tth_map()`, `reflections()` /
`reflections_json()` (per-scan → project → bundled asset).

Algorithm libraries (bundled with the package, **not** per-project), each keyed
by its own `catalog.json`: `detectors_dir()` (`PeakAlgorithms/`), `shapes_dir()`
(`ShapeAlgorithms/`), `combined_dir()` (`CombinedAlgorithms/`). See §9.

---

## 2. The pipeline at a glance

```
raw HDF5 frames  ──scan-detect──▶  Raw/scans.json
       │
 (SOCKETSERVER)  ──create-positions──▶  positions CSV (real X,Y per frame)
       │
       ├── grid ─────────────▶  Metadata/<scan>/grid_mapping_NxN.h5   (frame → bin)
       │        territory-grid ▶  grid_mapping_1x1_territory.h5         (frame → territory)
       │
       ├── bin ──────────────▶  Binned/<scan>/xrd_NxN_bins.h5           (one summed image per bin)
       │        reflection-sum ▶ Metadata/<scan>/reflection_sum.npz      (grand sum, for 2θ histogram)
       │
       ├── peaks  (Phase 1) ─▶  Labels/<scan>/<algo>_peaks_NxN.h5     (peaks_by_bin)
       │
       └── shapes (Phase 2/3)▶  Labels/<scan>/<algo>_shapes_NxN.h5    (kept / filtered features)
                    │
                     │    features are shapes `kept` (or combined `features`)
                     │

       ┌────────────┴───────── downstream views ─────────────────────────┐
       │                        │                          │              │
   DEVICE VIEW            HD DEVICE VIEW            TERRITORIAL VIEW   ROCKING STUDY
   combined / aggregate   hd-device-map            territory-build    aggregate→track→
   device_map.csv         *_hdmap_NxN.h5         territory_shapes   rocking→predict→
   *_combined_NxN.h5                             + scan-table       combined-device (Study/)
```

Canonical order (per scan): `init → scan-detect → link → grid → bin → peaks →
shapes`. `xrd-app run-pipeline` runs peaks+shapes back to back; `xrd-app
make-bins` runs grid+bin; `xrd-app batch` runs grid→bin→peaks→shapes over many
scans; `xrd-app territory-build` runs the whole skew-free branch.

Provenance travels **inside** each result HDF5 as a `lineage` block
(`core/lineage.py`) and is read without loading numerical payloads. The per-scan
`catalog_lineage.json` sidecar remains for non-numerical/manual records. See §8.

---

## 3. Raw data & positions

### Raw detector frames
- Location resolved by `DataManager.xrd_frames_dir(scan)` (`…/XRD` if present).
- One HDF5 file per **scan row** (`scan_NNNN_00001.h5`, `…_00002.h5`, …); each
  holds many frames at dataset `entry/data/data` (`io.H5_DATASET`), one 2-D
  detector image per spatial position. Detector is `1062 × 1028`
  (`io.DETECTOR_SHAPE`), raw dtype often uint16/int32 — do **not** upcast.
- `io.scan_h5_files(xrd_dir, scan_number)` — sorted, padding/case-tolerant list
  (trailing `_` guards `203` vs `2030`).
- `io.load_xrd_metadata(xrd_dir, scan_number)` → `(xrd_files, frame_map, n_total)`
  where `frame_map[global_index] = [file_idx, frame_idx]`. This is the master
  index every bin key ultimately dereferences through.
- `io.has_raw_frames(...)` — cheap completeness probe (no HDF5 open).
- `io.detect_frame_shape(scan_dir)` / `io.scan_info(...)` / `io.discover_scans(...)`
  / `io.validate_scan(...)` back **`xrd-app scan-detect`**, which writes
  `Raw/scans.json`.

### Positions — `xrd-app create-positions` (`core/positions.py`)
Real per-frame stage `(X,Y)` are what deskew the grid. Three sources:
1. **SOCKETSERVER interferometry stream** (`method=averaging|basic`):
   `positions.build_positions_csv(socket_dir, output, scan_number, method, theta_deg, reduction)`
   reduces the 24-column interferometer table (constants `_C_TRIGGER=2`,
   `_C_X=(7,19,20)`, `_C_Y=(3,4,5)`, `_C_Z=6`) to one `(X,Y)` µm per trigger via
   `compute_positions(...)` (`averaging`: `x=-sqrt(x_avg²+z²)` + 3-encoder Y;
   `basic`: single encoders + θ cosine correction). Writes a marker-free
   `Trigger,X_Position,Y_Position` CSV (treated as **real** positions).
2. **Lozano pre-reduced position h5** (`--from-h5`):
   `positions.build_positions_csv_from_h5(h5_path, output)` reads
   `entry/data/Position/{X_Position,Y_Position}` (`io.H5_POSITION_GROUP`).
   `positions.find_position_h5(...)` verifies the group is present so a raw XRD
   h5 isn't mistaken for a position file.
3. **No stream** → no CSV is fabricated; the grid is reconstructed directly from
   the one-file-per-row layout (see §4).

A synthetic file-per-row CSV is tagged with
`io.RECREATED_CSV_MARKER = "# xrd-app coordinate_source=file_per_row"`;
`io.is_recreated_csv(path)` detects it so downstream code knows the positions
are reconstructed, not measured.

### Geometry — `xrd-app convert-poni` (`core/geometry.py`)
- `geometry.poni_to_tth(poni_path, shape)` → 2θ-per-pixel map (degrees) from a
  pyFAI `.poni` (pyFAI is the optional `.[poni]` extra; import is lazy).
- `geometry.save_tth_tiff(tth, output)` / `convert_poni_file(...)` write
  `tth.tiff`. `io.load_tth_map(path)` loads it. This `tth` map is the pixel↔2θ
  key used everywhere; **χ** (azimuth) is computed on the fly as
  `atan2(y-beam_y, x-beam_x)` in the shape characterizer, not stored in a map.

---

## 4. Grid mapping — frame → spatial bin  (`xrd-app grid`)

`io.generate_grid_mapping(xrd_dir, pos_csv, bin_size, scan_number, output, ...)`
is the writer. It (a) builds a fine per-frame grid from positions/layout, then
(b) groups `bin_size × bin_size` fine cells into bins
(`io.build_bin_mapping`), and writes HDF5. Loaded mappings expose this dictionary:

**`grid_mapping_NxN[_variant].h5` logical structure:**
```jsonc
{
  "bin_size":         3,
  "coordinate_source":"file_per_row",         // how the grid was derived (below)
  "positions_csv":    "…/positions.csv",       // or null
  "positions_real":   true,                    // true iff a real (X,Y) CSV was used; `bin` requires this
  "n_rows": 151, "n_cols": 167,                // fine grid
  "n_bin_rows": 51, "n_bin_cols": 56,          // binned grid
  "n_total_frames": 25170,
  "n_bins": 2841,
  "h5_dataset": "entry/data/data",
  "xrd_files": ["…/scan_0203_00001.h5", …],    // indexed by file_idx
  "bins":      { "0_0": [166,165,164,167,…], … },  // "binrow_bincol" → GLOBAL frame indices
  "frame_map": [[0,0],[0,1], …]                // global_index → [file_idx, frame_idx]
}
```

The two-level indirection is the heart of the app: **bin key → global frame
index (`bins`) → `[file, frame]` (`frame_map`) → `xrd_files[file]`.** Everything
that needs raw pixels for a bin walks this chain.

**Current `coordinate_source` values** (set inside `generate_grid_mapping`, and
mapped to a grid method by `io.deskew_method_for_source`):
- `positions_xy` — snap both measured axes without clipping; default at 1×1 and
  suitable for irregular scans that still need a rectangular grid.
- `positions_faithful` / `positions_faithful_native` — exact file-index rows and
  measured fast-axis columns, using an approximately square display lattice or
  native frame density; selected by `faithful` / `faithful_native`.
- `file_per_row` — file-index rows and commanded within-file rank; selected by
  `commanded` when backlash makes encoder columns misleading.
- `territory_xy` — irregular true-position cells with physical adjacency, built
  by `territory-grid` rather than `grid`.

Supporting builders in `io.py`: `assign_grid_from_positions`,
`assign_grid_coordinate_faithful`, `build_scan_grid` (serpentine turn-counting),
`build_regular_grid`, `is_file_per_row`, `build_grid_from_frame_map`,
`build_bin_mapping`, `subbin_keys(bin_key, bin_size)` (inverse: a binned key →
its 1×1 sub-cells — used by the HD map).

`io.load_grid_mapping(path_or_dict)` loads (or passes through) a mapping.

---

## 5. Binning — one summed image per bin  (`xrd-app bin` / `make-bins`)

`io.build_bins(grid_mapping, output, bin_size, compression="zstd")` sums each
bin's raw frames into a single detector image and writes an HDF5 (atomic
`.tmp`→`os.replace`, LRU of 32 open source handles).

**`xrd_NxN_bins[_variant].h5` layout:**
- Root attrs: `bin_size`, `n_bin_rows`, `n_bin_cols`, `n_bins`,
  `detector_shape = [1062, 1028]`.
- **One dataset per non-empty bin**, keyed `"row_col"` (e.g. `"3_7"`), dtype
  **float32**, shape = detector frame. Each is the **sum** of that bin's raw
  frames, clamped `<0→0`, `>1e9→0`.

Reading bin images (built h5 *or* raw frames, transparently):
- `io.open_bin_source(dm, bin_size, scan, variant=…)` →
  `_H5Source` if the binned h5 exists, else `_RawSource` (bins raw frames on
  demand). Both expose `is_raw`, `keys()`, `image(key)`, `region(key,y0,y1,x0,x1)`
  (reads only the requested slice), `sum_all(...)`.
- `variant="territory"` selects `xrd_1x1_bins_territory.h5`, whose keys are
  `"<tid>_0"` (§7).

---

## 6. Reflections & the grand-sum image

### Reflections (`core/reflections.py`)
The reflection bands define where peaks are expected. They are stored and loaded
from `reflections.json`; `core/reflections.py` is the source module implementing
that contract and the bundled defaults.

**`reflections.json`** — a plain list:
```json
[{"name":"(111)","two_theta":13.00831,"width":0.4}, …]
```
`DEFAULT_REFLECTIONS` (perovskite/halide + substrate), all `width=0.4°`:

| name | 2θ° | | name | 2θ° |
|---|---|---|---|---|
| PbI2 | 6.81319 | | ITO | 16.07224 |
| (001) | 7.51422 | | (012) | 16.79944 |
| (011) | 10.61748 | | (112) | 18.42549 |
| (111) | 13.00831 | | 21.30 | 21.29655 |
| (002) | 15.01266 | | 22.60 / 26.16 | 22.59817 / 26.16205 |

- `reflections.read_json` / `write_json` / `save` manage the file;
  `io.load_reflections(path)` returns `(degs, deg_labels)`.
- **Resolution order** (in `DataManager`): per-scan selection →
  `data_sources.reflections` → `Metadata/<scan>/reflections.json` → project
  `Metadata/reflections.json` → bundled JSON asset. Re-tuned per scan (offset/scale drift).

### Grand-sum image — `xrd-app reflection-sum` (`core/reflection_sum.py`)
`reflection_sum.compute_and_save(dm, scan)` sums **every** bin (= every frame)
into one image, written to `Metadata/<scan>/reflection_sum.npz`:
- `image` — float32 summed detector image, `is_raw` — bool, `max_bins` — int cap.

It picks the fastest source via `source_bin(...)` (largest prebuilt h5 — all bin
sizes give the same grand sum). The radial 2θ histogram for calibrating
reflections is derived from this instantly (`io.radial_profile(image, tth_map)`).

---

## 7. Peaks → shapes (the detection core, `core/processing.py`)

### Phase 1 — peaks  (`xrd-app peaks`)
`processing.run_peaks(bins_h5, tth_path, detector_path, reflections_path,
bin_size, snr_threshold=4.0, n_workers=…)` runs a detector over every bin
(parallel across cores; each worker opens its own HDF5 handle).

Per bin, `processing.detect_peaks_with_intensity(...)` drives the **detector
contract** (a swappable `PeakAlgorithms/*.py` module loaded by
`io.load_module`): `radial_median_subtract` (2θ background) → `fast_tophat` →
`build_tth_band_masks` (one mask per reflection, ±0.4°) → `detect_in_band` per
band. Peaks are sorted by SNR, de-duplicated within 15 px, capped at
`DEFAULT_MAX_PEAKS_PER_BIN = 25`.

**Peak record** (one entry in `peaks_by_bin["r_c"]`):
```jsonc
{"x":389,"y":1017,"label":"(111)","npix":79,"compactness":0.40,
 "snr":768.9,"peak_val":2208.0,"integrated_intensity":11447.5,
 "cleaned_intensity":2204.7}   // cleaned_intensity = max in ±3px of the bg-subtracted image
```

**`<algo>_peaks_NxN.h5`:**
```jsonc
{ "bin_size":3, "detector":"5x5_tophat_band_adaptive_snr", "snr":4.0,
  "n_peaks":7966, "n_bins_with_peaks":2747,
  "peaks_by_bin": { "0_0":[peak, …], … },
  "scan":"Scan_0203", "algorithm":"…", "lineage":{…} }   // scan/algorithm/lineage added by the CLI
```

### Phase 2/3 — shapes  (`xrd-app shapes`)
`processing.run_shapes(peaks, tth_path, grid_mapping, reflections_path,
bin_size, link_tolerance=5, shape_path=…)` loads the shape module
(`ShapeAlgorithms/gaussian.py` by default) and does:

- **Phase 2 — Union-Find linking** (`gaussian.link_peaks`): each peak is a node
  `(bin_key, peak_index, row, col, x, y, peak_dict)`. Peaks in the 8-neighbor
  bins are `union`ed when their detector positions are within `link_tolerance`
  px. Each connected component = one raw feature. *(Territorial mappings instead
  link across `territories[*].neighbors` via `ShapeAlgorithms/territory.py` — see §10.)*
- **Phase 3 — characterization + filter** (`gaussian.characterize_features` +
  `check_gaussian_profile`). A cluster is **filtered** (dropped) when: <2 members
  (`"isolated"`), non-positive peak intensity, flat profile (CV<0.05), or
  non-monotonic distance-intensity trend (<40%). Otherwise **kept**.

**Shape / feature record** (`kept[]` / `filtered[]`):
```jsonc
{ "reflection":"(111)", "detector_x":464, "detector_y":325,      // mean detector position
  "peak_intensity":57.7, "mean_snr":9.9, "n_bins":21,
  "spatial_extent":["0_2","0_3",…],                              // sorted set of member bin keys
  "center_bin":"0_5", "center_row":0, "center_col":5,            // brightest member
  "intensity_profile":{ "0_2":{"intensity":19.5,"integrated":75.4,
                               "det_x":468,"det_y":324,"tth":13.03,"chi":-153.75}, … },
  "reason":"Gaussian-like: 100% monotonic, 2 bins",
  "ref_tth":13.00831,                                            // reference 2θ of the band
  "chi_fwhm":…,        // intensity-wtd FWHM of χ across bins (mosaicity-ish), ≥3 bins
  "tth_fwhm":…,        // intensity-wtd FWHM of Δ2θ across bins (was "strain_breadth")
  "chi_deg":-153.9,    // azimuth at the mean position
  "feature_id":1 }     // 1-based, assigned to kept only, after sorting
```

**`<algo>_shapes_NxN.h5`:**
```jsonc
{ "bin_size":3, "link_tolerance":5, "n_kept":185, "n_filtered":844,
  "kept":[feature, …], "filtered":[feature, …],
  "scan":"Scan_0203", "shape_algo":"gaussian", "peak_source":"5x5_tophat_band_adaptive_snr",
  "lineage":{…, "peak_source":{…}} }
```

`processing.write_peak_table(...)` also emits `kept_peaks_NxN.csv` /
`filtered_peaks_NxN.csv` tables.

### Combined per-frame  (`xrd-app run-combined`)
`processing.run_combined(...)` runs a `CombinedAlgorithms/*.py` detector that does
peak+shape in **one per-frame pass** (no separate binning), writing
`Labels/<scan>/<algo>_combined_NxN.h5`:
```jsonc
{ "algorithm":"…", "bin_size":1, "n_features":N,
  "by_bin": { "0_0":{"(111)":[[374,752]], "(012)":[[191,194]]}, … },   // {bin:{reflection:[[x,y]]}}
  "features":[ {"feature_id":1,"reflection":"(111)","detector_x":374,"detector_y":752,
                "center_bin":"0_0","center_row":0,"center_col":0,"n_bins":1,
                "peak_intensity":null,"mean_snr":null,"intensity_profile":{}}, … ],
  "scan":"…", "lineage":{…} }
```
These are point-features (no per-bin intensity) — this file **is** the raw
"Device View" data layer (§11).

---

## 8. Catalogs & lineage (`core/catalogs.py`, `core/lineage.py`)

### Catalog kinds (all flat in `Labels/<scan>/`)
`catalogs.parse_name(name)` → `{algo, kind, bin, tag}`. The bin size is parsed
from the segment **after** the kind keyword so an algo name like
`5x5_tophat_band_adaptive_snr` isn't mistaken for a bin.

| kind | filename | payload |
|---|---|---|
| peaks | `<algo>_peaks_NxN[_tag].h5` | `peaks_by_bin` + `lineage` |
| shapes | `<algo>_shapes_NxN[_tag].h5` | `kept`/`filtered` + `lineage` |
| combined | `<algo>_combined_NxN[_tag].h5` | `features` + `lineage` |

There is no separate feature-catalog format. Features live under shapes `kept`
or combined `features`.

Key functions: `list_catalogs(dir, kind, bin_size)`, `available_bins(dir)`,
`feature_sources(dir, bin_size)` (shapes+combined),
`default_feature_source(dir, bin_size)`, `load_features_any(path)` →
`(kept, filtered)` **regardless of on-disk kind** (this is the universal reader
every downstream view uses), `append_features(path, feats)` (assigns
`feature_id`, writes back in the same on-disk shape), `match_across_bin(...)`
(carry a selection across a bin switch via the bin-independent
`lineage_key=(kind,algo,tag)`), `best_grid_mapping(...)`.

### Lineage — provenance
- **In-file** (`core/lineage.py`): every peaks/shapes/combined HDF5 carries a
  `lineage` dict. `peak_lineage` → `{stage:"peaks", scan, bin_size, created,
  app_version, peak_algorithm, detector_file, snr}`; `shape_lineage` adds
  `shape_algorithm`, `link_tolerance`, `peak_source_file`, and nests the upstream
  `peak_source` lineage; `combined_lineage` for the per-frame path.
  `from_peaks_data(...)` recovers upstream lineage from a loaded peaks file.
- **Sidecar** (`core/catalogs.py`): `catalog_lineage.json` (`MANIFEST_NAME`) maps
  `filename → lineage dict` for plain-list files. `record_catalog(dir, filename,
  lineage)` appends on every write; `read_lineage(path)` resolves **in-file →
  manifest → None**; `backfill_feature_lineage(...)` fills missing manifest
  entries. `xrd-app lineage <target>` renders it (`lineage.format_text`).

---

## 9. Detector / shape / noise algorithm libraries

Bundled with the package (not per-project); each dir has a `catalog.json`.

- Discovery/loading: `io.load_module(path)` dynamically imports a `.py` and puts
  `PeakAlgorithms/` and `CombinedAlgorithms/` on `sys.path` for sibling/base
  imports. `DataManager` reads each `catalog.json`: `list_detectors(bin_size)`, `best_detector(bin_size)`
  (highest `holdout_f1`, excludes per-frame), `detector_script(override,bin_size)`,
  and the `list_/resolve_/best_` equivalents for shapes and combined.
  `xrd-app detectors [--kind peak|shape|combined]` lists them.
- Noise models (`core/algorithms.py`): `gaussian`, `split_gaussian`,
  `skewed_gaussian`, `fourier_lowpass` background models + `reduce_noise(...)` →
  `(cleaned, background, fit_info)`.
- Freezing a tuned detector — `xrd-app save-algorithm` (`core/save_algorithm.py`):
  `save_algorithm(base, sensitivity, bin_size, noise_reduction, …, kind, source)`
  writes a runnable `PeakAlgorithms/<stem>.py` (or `<stem>/detector.py` for
  automated/CVEvolve saves) that bakes in an SNR sensitivity + noise reduction,
  and `register_in_catalog(...)` inserts the entry `{name, bin_size, file, role,
  kind, holdout_f1, holdout_f2, source, base, sensitivity, noise_*, …}`.

Holdout scoring (`build-holdout`, `core/holdout.py`) and CVEvolve
(`cvevolve-init`/`run-cvevolve`, `core/cvevolve_setup.py`) sit on top of this
library; CVEvolve optimizes **mean F2** (recall-weighted), not F1. Completed
sessions export `reports/best_candidate.py`; `register-cvevolve` validates that
module's production detector API, copies it into the project-owned
`Algorithms/PeakAlgorithms/<name>/detector.py`, and updates `catalog.json`. The
GUI runs this registration automatically after CVEvolve exits successfully.

---

## 10. TERRITORIAL VIEW — skew-free reference binning (`core/territory.py`)

The N×N serpentine grid is skewed by even/odd-row backlash. The **territorial**
model bins frames by their true stage `(X,Y)` instead, so the partition is
immune to that skew. It is the "source of truth" the fast deskew post-processing
is optimized against.

- **What a territory is:** a cluster of ~`target_size` frames (default 9) grown
  outward over the `(X,Y)` **Delaunay** neighbor graph until it hits the frame
  count — a clean partition (every frame in exactly one territory) with adaptive
  SNR (denser regions → tighter cells). Adjacency is *physical*, not the N×N
  8-neighborhood.
- **`tid` vs frame index:** `tid` is the territory id (`0…n_terr-1`, seed order);
  a frame index is a global frame. Bin keys are **`"<tid>_0"`** (always `_0`
  suffix) so existing `int(k.split("_"))` parsers keep working — but one
  territory key holds **many** frame indices. Real adjacency lives in
  `territories[*].neighbors`. *(See memory note: `tid != frame index`; a View
  1×1 bridge must map frame→tid through the grid `bins[]`.)*

`xrd-app territory-grid` → `territory.build_territory_mapping(xrd_dir, pos_csv,
target_size, scan_number)` → `Metadata/<scan>/grid_mapping_1x1_territory.h5`:
```jsonc
{ "bin_size":1, "coordinate_source":"territory_xy", "positions_real":true,
  "target_size":9, "step":…, "n_rows":179,"n_cols":188,"n_bin_rows":179,"n_bin_cols":188,
  "n_total_frames":25170, "n_bins":2803, "h5_dataset":"…", "xrd_files":[…],
  "bins":      { "0_0":[0,1,2,3,4,5,6,333,331], … },   // "<tid>_0" → member frame indices
  "frame_map": [[0,0], …],
  "territories":{ "0_0":{ "centroid_xy":[446.17,771.44], "centroid_rc":[177.9,187.1],
                          "area":0.0256, "count":9,
                          "neighbors":["1_0","2_0",…], "polygon":[[px,py],…] }, … } }
```

`xrd-app territory-build` chains the whole branch:
`territory-grid → bin --variant territory → peaks --variant territory → shapes
--algorithm territory`. The `shapes` step links across `territories[*].neighbors`
and writes **`territory_shapes_1x1_territory.h5`** (same `kept`/`filtered`
schema as §7, but `center_row`/`center_col` are real fractional positions and
`spatial_extent` holds `"<tid>_0"` keys).

**Coordinate (1×1) variant:** `territory.add_coordinate_neighbors(gm, frame_x,
frame_y)` keeps **one frame per cell** but replaces grid adjacency with Delaunay
adjacency (`coordinate_source="coord_xy"`), so linking is by true position, not
by skewed grid keys. The `shapes` command writes these as
`…_shapes_1x1_coord.h5` / `territory_shapes_1x1_coord.h5`.

**Per-scan territorial summary — `xrd-app scan-table`** (`core/scan_table.py`):
one row per scan (or per reflection) for a chosen bin/catalog: feature count,
footprint sum/union, coverage %, **preferred χ** (area-weighted azimuthal KDE
dominant cluster ± spread), and shape **fill %** (solidity). Two geometry modes:
`grid` (units = bins, `Total = n_bin_rows·n_bin_cols`) and `territory` (units =
CSV², areas from the `territories` polygons, `Total` = outline convex-hull area).
Columns: `Scan, Reflection, Features, Area sum, Area union, Coverage %,
Preferred χ, χ ± range, Fill %, Total`. Writes `Study/scan_summary.csv`.

---

## 11. DEVICE VIEW — features laid on the spatial grid

Two related artifacts:

1. **Per-scan combined map** — `xrd-app run-combined` → `<algo>_combined_NxN.h5`
   (§7). `by_bin` gives `{reflection:[[x,y]]}` per spatial bin — the direct
   device layer for one scan.
2. **Aggregated tidy tables** — `xrd-app aggregate` (`core/aggregate.py`) walks
   every scan's canonical feature source (`catalogs.default_feature_source` +
   `load_features_any`) and flattens to two CSVs + a SQLite DB:
   - `Study/features.csv` (`aggregate.FEATURE_COLUMNS`): one row per feature —
     `scan, bin_size, feature_id, reflection, ref_tth, center_bin, center_row,
     center_col, detector_x, detector_y, chi_deg, tth_com, peak_intensity,
     mean_intensity, sum_integrated, mean_snr, n_bins, chi_fwhm, tth_fwhm,
     spatial_extent, reason`.
   - `Study/device_map.csv` (`aggregate.DEVICEMAP_COLUMNS`): one row per
     scan×reflection×**bin** — `scan, bin_size, feature_id, reflection, bin_key,
     row, col, detector_x, detector_y, intensity, integrated, tth, chi`
     (per-bin values from each feature's `intensity_profile[bin_key]`; `row/col`
     parse only when both parts are digits, so territory `"<tid>_0"` keys give
     `row=0`).
   - `Study/study.db` — `features` + `device_map` tables (`write_sqlite`).

---

## 12. HD DEVICE VIEW — 1×1 intensity beneath a binned feature (`core/hd_map.py`)

The binned device view shows one value per N×N bin. The HD map drops to true 1×1
resolution *inside* each already-found feature: for every N×N bin in a feature's
footprint it expands to the `bin_size²` raw 1×1 cells (`io.subbin_keys`) and
reads that single raw frame, taking max/sum in a small window around the
feature's **detector peak**.

`xrd-app hd-device-map` →
`hd_map.sample_hd_intensity(features, source, bin_size, win=4, cell_xy=…)` over a
source N×N shapes or combined catalog, sampling a 1×1 `io.BinImageSource`. It writes
`Labels/<scan>/<algo>_hdmap_NxN.h5`:
```jsonc
{ "kind":"hd_map", "scan":"Scan_0203", "bin_size":3, "win":4,
  "source_catalog":"gaussian_shapes_3x3.h5",
  "n_bin_rows_1x1":…, "n_bin_cols_1x1":…, "positions_real":true,
  "features":[ { "feature_id":1, "reflection":"(111)", "chi_deg":-153.9, "ref_tth":13.00831,
                 "detector_x":464, "detector_y":325,
                 "hd_profile":{ "0_6":{"intensity":6.0,"integrated":62.0,
                                       "x":445.91,"y":760.30}, … } }, … ] }
```
`hd_profile` is the 1×1 analog of a feature's `intensity_profile`, keyed by 1×1
cell; `x,y` are real stage positions (present only when a real position CSV is
available). The CLI auto-builds the 1×1 grid mapping if missing
(`_ensure_1x1_grid_mapping`) so cell→frame→(x,y) resolves — *if a device map was
built before that grid existed, its lattice can mismatch; rebuild.*
`hd_map.build_cell_xy(gm, positions_csv)` gives `{cell:(x,y)}`;
`hd_map.scan_trajectory(...)` gives the acquisition path.

---

## 13. ROCKING STUDY — features across the θ sweep (`Study/`)

Scans 203–214 are the same sample at different sample tilts θ
(`tracking.THETA_BY_SCAN`). The study links features across θ and fits rocking
curves. Full driver: `xrd-app run-study` (register + run the chain); individual
steps below. `core/studies.py` discovers/registers `Study/` dirs
(`studies.json`, `list-studies` / `register-study`).

### a) tracks — `xrd-app track` (`core/tracking.py`)
`tracking.build_tracks(features, match_tol=2.0, min_theta=2)` groups features
across θ (same reflection + spatial proximity within `match_tol` bins, greedy
strongest-first) — the θ-axis analog of spatial Union-Find. → `Study/tracks.h5`:
```jsonc
{ "bin_size":"3","match_tol":"2.0","min_theta":"2","n_tracks":705,
  "tracks":[ { "track_id":0,"reflection":"(001)","ref_tth":7.51,
               "centroid_row":37.0,"centroid_col":24.0,"pos_drift":0.0,
               "n_theta":1,"n_members":1,"theta_min":5.5,"theta_max":5.5,
               "theta_at_max_I":5.5,"chi_mean":-130.6,"chi_span":0.0,"chi_max_step":0.0,
               "max_intensity":2177955.9,"is_recurrent":false,
               "members":[ {"scan":"Scan_0207","theta":5.5,"center_row":37.0,"center_col":24.0,
                            "chi_deg":-130.6,"tth_com":7.57,"peak_intensity":5086.0,
                            "sum_integrated":2177955.9,"intensity":2177955.9,
                            "detector_x":1021,"detector_y":326,"tth_fwhm":0.08,"chi_fwhm":0.55,
                            "feature_id":124}, … ] }, … ] }
```
`is_recurrent = n_theta >= min_theta`. Also writes `tracks.csv`
(`tracking.TRACK_COLUMNS`, drops `members`).

### b) rocking curves — `xrd-app rocking` (`core/rocking.py`)
`rocking.fit_tracks(tracks, min_points=4, only_recurrent=True)` fits each track's
intensity(θ) to a Gaussian → θ_Bragg + FWHM (mosaicity), plus microstrain
(ε = −cot θ_B·Δθ) and lattice-tilt dχ/dθ. → `Study/rocking_curves.csv`
(`rocking.ROCKING_COLUMNS`): `track_id, reflection, ref_tth, centroid_row,
centroid_col, n_theta, theta_min, theta_max, status, theta_bragg, fwhm,
amplitude, background, r_squared, tth_com, microstrain, strain_breadth_2th,
chi_tilt_rate, chi_span, theta_at_max, theta_centroid, integrated_intensity,
max_intensity`. `status` ∈ `fit / poor_fit / monotonic / too_sparse / empty /
fit_failed:<Exc>`.

### c) prediction report — `xrd-app predict` (`core/prediction.py`)
`prediction.build_report(tracks, features, match_tol, repeat_pair, rocking_rows)`
turns recurrent tracks into a falsifiable prediction and scores it. →
`Study/prediction_report.{json,md}`:
```jsonc
{ "scans":[…], "n_scans":11, "sampled_thetas":[…], "n_features":1266,
  "per_scan_features":{…}, "n_tracks":705, "match_tol_bins":2.0,
  "predicted_vs_observed":{ "n_recurrent_tracks":…,"n_singleton_tracks":…,
                            "tp":…,"fn":…,"fp":…,"recall":…,"precision":…,"f1":…,"gaps":[…]},
  "repeatability_floor":{ "scan_a":"Scan_0203","scan_b":"Scan_0214",
                          "n_a":…,"n_b":…,"matched":…,"reproducibility":…,"recall_a_in_b":…},
  "dose_check":{ "substrate":["ITO"],"per_reflection":{…},
                 "film_retention":…,"substrate_retention":…,"note":"…"},   // beam-damage check
  "chi_smoothness":{…}, "rocking":{…}, "verdict":"recall 0.46 meets/exceeds …" }
```
The `dose_check` is the beam-damage finding surface (perovskite film collapses,
ITO substrate holds).

### d) combined device (cross-θ) — `xrd-app combined-device` (`core/combined_device.py`)
`combined_device.build_combined(device_map_rows, theta_by_scan, intensity_key,
tracks)` fuses the per-θ `device_map` rows into one spatial canvas. →
`Study/combined_device.npz` + `combined_device.summary.json`. NPZ arrays:
- `max_intensity[r,c]` — strongest diffraction at that spot over any θ.
- `argmax_theta[r,c]` — the θ that produced it (a local-orientation map).
- `n_theta_present[r,c]` — recurrence count.
- `layer_intensity[k,r,c]`, `layer_argmax_theta[k,r,c]` — same, split per
  reflection `k` (`reflections`, `thetas` label the axes).
- `track_*` — centroid overlay (`track_id/reflection/centroid_row/centroid_col/
  theta_at_max/is_recurrent`).

`intensity_key` ∈ `integrated` (default) / `intensity` selects the driving
column. This is distinct from the per-scan `*_combined_NxN.h5` (§7/§11) — that
is one scan's peak+shape map; this is all θ fused.

---

## 14. Reciprocal space — q-space & RSM

- **`xrd-app qspace`** (`core/qspace.py`): maps detector pixels (+ sample θ) into
  3-D reciprocal space. `q_vectors(...)` / `q_vectors_from_poni(...)` /
  `recover_geometry(...)` (fits distance/beam-center from the tth map when no
  poni), `annotate_features(...)` tags features with q. `save_qmap(...)` writes
  `Study/qspace/<scan>_qmap.npz` (`qx,qy,qz,q_mag` + geometry: `beam_row,
  beam_col, distance_m, pixel_m, rms_deg, geometry_source`). Constants:
  `DEFAULT_ENERGY_EV=15000`, `DEFAULT_PIXEL_M=75e-6`.
- **`xrd-app rsm`** (`core/rsm.py`): fuses per-scan q-maps into one binned 3-D
  reciprocal-space map. `common_grid(...)` → `accumulate(...)` →
  `projections(...)` → `save_npz(...)` → `Study/rsm.npz` + `rsm.summary.json`.
  NPZ: `volume[128³]`, `counts`, `qx/qy/qz_edges`, `qx/qy/qz_centers`,
  `proj_qx_qy`, `proj_qx_qz`, `proj_qy_qz`, `scans`, `thetas`.

---

## 15. XRF overlay (`core/xrf.py`) — `xrd-app xrf`

ME7 (7-element XSPRESS3) fluorescence, mapped onto the **same** XRD spatial grid
so elements underlay the device view. `element_maps(...)` reads ME7 `scan_*.h5`
(`me7_files`), sums per-channel MCA spectra (`_summed_spectra`, optional
deadtime), assigns each point to a bin via `fileloc_to_bin(grid_mapping)`, and
integrates per-element ROIs. `save_npz(...)` → `Metadata/<scan>/<scan>_xrf.npz`:
- `elements`, `map_<name>` (float32 per-element `[nr,nc]` intensity map),
  `n_points`, `spectrum` (grand-sum MCA, energy = `bin·ev_per_bin + offset_ev`),
  `roi_<name>` (`[line_ev, center_ev, lo_ev, hi_ev, lo_bin, hi_bin, matched]`),
  `n_rows/n_cols`, `channels`, `deadtime`, `ev_per_bin`, `offset_ev`, `dropped`.
- ROI auto-refinement: `find_spectrum_peaks` + `match_lines_to_peaks` +
  `refine_rois` snap element windows to observed lines; `EMISSION_LINES` /
  `DEFAULT_ELEMENTS` seed the fit. `--save-points` also writes
  `<scan>_xrf_points.npz` (per-frame summed spectra) for fast re-ROI without
  re-reading raw ME7.

---

## Appendix A — command → core function → output

| `xrd-app` command | core entry point | writes |
|---|---|---|
| `scan-detect` | `io.discover_scans` / `validate_scan` | `Raw/scans.json` |
| `create-positions` | `positions.build_positions_csv[_from_h5]` | positions CSV |
| `convert-poni` | `geometry.convert_poni_file` | `Metadata/tth.tiff` |
| `grid` | `io.generate_grid_mapping` | `Metadata/<scan>/grid_mapping_NxN[_v].h5` |
| `bin` / `make-bins` | `io.build_bins` (+ `generate_grid_mapping`) | `Binned/<scan>/xrd_NxN_bins[_v].h5` |
| `reflection-sum` | `reflection_sum.compute_and_save` | `Metadata/<scan>/reflection_sum.npz` |
| `peaks` | `processing.run_peaks` | `Labels/<scan>/<algo>_peaks_NxN[_v].h5` |
| `shapes` | `processing.run_shapes` | `Labels/<scan>/<algo>_shapes_NxN[_v].h5` |
| `run-combined` | `processing.run_combined` | `Labels/<scan>/<algo>_combined_NxN.h5` |
| `run-pipeline` | `run_peaks` + `run_shapes` | peaks + shapes |
| `batch` | grid→bin→peaks→shapes | all of the above, many scans |
| `territory-grid` | `territory.build_territory_mapping` | `grid_mapping_1x1_territory.h5` |
| `territory-build` | territory-grid→bin→peaks→shapes | `territory_shapes_1x1_territory.h5` |
| `hd-device-map` | `hd_map.sample_hd_intensity` | `Labels/<scan>/<algo>_hdmap_NxN.h5` |
| `scan-table` | `scan_table.scan_table_rows` | `Study/scan_summary.csv` |
| `aggregate` | `aggregate.aggregate` | `Study/features.csv`, `device_map.csv`, `study.db` |
| `track` | `tracking.build_tracks` | `Study/tracks.h5` + `.csv` |
| `rocking` | `rocking.fit_tracks` | `Study/rocking_curves.csv` |
| `predict` | `prediction.build_report` | `Study/prediction_report.{md,json}` |
| `combined-device` | `combined_device.build_combined` | `Study/combined_device.npz` + summary |
| `qspace` | `qspace.q_vectors*` / `save_qmap` | `Study/qspace/<scan>_qmap.npz` |
| `rsm` | `rsm.accumulate` / `save_npz` | `Study/rsm.npz` + summary |
| `xrf` | `xrf.element_maps` / `save_npz` | `Metadata/<scan>/<scan>_xrf.npz` |
| `run-study` | aggregate→track→rocking→predict→combined-device→rsm | all of `Study/` + `studies.json` |
| `save-algorithm` | `save_algorithm.save_algorithm` | `PeakAlgorithms/<stem>.py` + `catalog.json` |
| `detectors` / `lineage` / `status` | `DataManager` / `catalogs` / `lineage` | (read-only) |

## Appendix B — the bin-key convention (one gotcha to memorize)

A bin key is `"row_col"` and dereferences **bin → global frame index → [file,
frame] → raw h5**:

```python
gm      = io.load_grid_mapping(dm.grid_mapping(bin_size=3, scan="Scan_0203"))
frames  = gm["bins"]["3_7"]              # [global_frame_index, ...]
file_i, frame_i = gm["frame_map"][frames[0]]
path    = gm["xrd_files"][file_i]        # raw HDF5 file; dataset entry/data/data
```

Exceptions to keep straight:
- **Territorial** mappings key by `"<tid>_0"`; the territory holds *many* frames
  and real adjacency is `territories["<tid>_0"]["neighbors"]`, not row±1/col±1.
- **`device_map.csv`** parses `row/col` from the bin key **only when both parts
  are digits** — territory keys collapse to `row=0`.
- **HD profiles** use 1×1 sub-cell keys from `io.subbin_keys(bin_key, bin_size)`.
