# Pipeline Walkthrough — GUI vs CLI, defaults, and linkage

End-to-end map of the nano-XRD pipeline **from an empty project to the device
maps and territory JSONs**, comparing what the GUI and the CLI can each do, what
the defaults are, and confirming every step is wired correctly. XRF and the
rocking/cross-scan study steps are intentionally out of scope here.

> **The one architectural rule holds:** the CLI *is* the engine. Every GUI "Run"
> button shells out to `python -m xrd_app.cli …` via the embedded `JobConsole`
> (`tabs/_console.py:126`) and parses its `PROGRESS i/n` markers. The GUI and CLI
> read/write the *same* files, so anything the GUI shows is reproducible headless.

This document was verified against a scratch project built from **real
Scan_0203** data (see [Verification](#verification-what-was-actually-run)).

---

## The pipeline at a glance

```
init ─► scan-detect ─► link (tth/refl/pos) ─► [create-positions] ─► grid ─► bin ─► peaks ─► shapes ─► Device View
                                                                     └──────────── make-bins ───────┘
   territory-grid ─► bin ─► peaks ─► shapes  (all --variant territory, 1×1) ─► Territory Map (popup from Device View)
   └──────────────────────── territory-build (one command / one button) ──────────────┘
```

| # | Step | CLI command | GUI control | Output |
|---|------|-------------|-------------|--------|
| 1 | Create project | `init` | Setup → **New project…** | `config.yaml` + tree + seeded `reflections.json` |
| 2 | Register scans | `scan-detect` | Setup → **Select scan folder…** / **Select scan set…** | `Raw/scans.json` |
| 3 | Link calibration | `link` / `convert-poni` | Setup → **Load tth.tiff…** / **Convert .poni → tth…** / **Load positions…** | `config.data_sources`, `Metadata/tth.tiff` |
| 4 | Real positions | `create-positions` | *(auto, inside grid)* / Setup → **Load positions…** | `Metadata/<scan>/positions.csv` |
| 5 | Grid mapping | `grid` | *(folded into Create bins)* | `Metadata/<scan>/grid_mapping_NxN[_variant].h5` |
| 6 | Build bins | `bin` / `make-bins` | Programs → **Create bins** | `Binned/<scan>/xrd_NxN_bins.h5` |
| 7 | Peaks (Phase 1) | `peaks` | Programs → Peak Finding **Run** | `Labels/<scan>/<algo>_peaks_NxN.h5` |
| 8 | Shapes (Phase 2) | `shapes` | Programs → Shape Finding **Run** | `Labels/<scan>/<algo>_shapes_NxN.h5` + kept/filtered CSVs |
| — | peaks→shapes (1 scan) | `run-pipeline` | Shape Finding, Peaks = "run peak algo above first" | both of the above |
| — | many scans | `batch` | Programs multi-select scans × algos (fans out `run-pipeline`) | per-scan outputs |
| 9 | Device maps | *(none — a viewer)* | **Device View** tab | reads the shapes catalog |
| 10 | Territorial reference | `territory-grid` → `bin`/`peaks`/`shapes --variant territory`, **or** `territory-build` | Programs → **Build territorial reference** | `grid_mapping_1x1_territory.h5` + territory peaks/shapes HDF5 |
| 11 | Territory map | *(none — a viewer)* | Device View → **Territorial reference available →** (popup) | reads the territory catalog |

---

## Step-by-step: command, GUI control, defaults, linkage

### 1. `init` — create the project
- **CLI:** `xrd-app init --name <NAME> [--scan-number N] [--root DIR]`. Creates
  the standard tree (`Raw/ Binned/ Metadata/ Labels/ Figures/ CVEvolve/`), writes
  `config.yaml`, **and seeds an editable default reflection set** into
  `Metadata/reflections.json`.
- **GUI:** Setup → **New project…** (a workspace must be set first). This calls
  `workspace.create_project()`, not the `init` command.
- **Linkage — FIXED:** the GUI path previously created the tree/config but
  **skipped the reflection seeding**, so a GUI-made project silently fell back to
  the hidden bundled set (no "Project default" in the Reflections selector).
  `workspace.create_project()` now seeds `reflections.json` exactly like
  `init`, so both front doors produce identical projects.

### 2. `scan-detect` — discover & register scans
- **CLI:** `xrd-app scan-detect --scans-dir <dir>` (or `--scan-file <hdf5>`).
  **Fast by default** — samples the first file per scan, so frame counts are
  estimates (`~` prefix); `--deep` opens every file for exact counts + corruption
  checks (slow on WSL/OneDrive). `--scans "203,204"` keeps a subset.
- **GUI:** Setup → **Select scan folder…** (one `Scan_NNNN/`) or **Select scan
  set…** (a parent of `Scan_*/`, then a checklist). Setup → **Choose scans to
  show…** curates which registered scans appear in the header selector
  (`visible_scans` in config).
- **Guardrail (verified):** a dir with **no XRD frames** (e.g. an ME7-only
  folder) is correctly rejected — `No scans found under …` — so incomplete scans
  never register.

### 3. `link` / `convert-poni` — record calibration & roots
- **CLI:** `xrd-app link --tth <tiff> --reflections <json> --detector <py>
  --raw-root <dir> --position-root <dir> --position-csv <csv> --poni <poni>`.
  Records absolute paths in `config.data_sources` (symlinks by default; `--copy`
  to copy). `convert-poni` turns a pyFAI `.poni` into `Metadata/tth.tiff` (needs
  the optional `[poni]` extra).
- **GUI:** Setup → **Load tth.tiff…** (`link --tth`), **Convert .poni → tth…**
  (`convert-poni`), **Load positions…** (single CSV → `link --position-csv`, or a
  folder → `link --position-root`), **Load reflections…** / **Manual
  reflections…** (per-scan reflection selection + editor).
- **Linkage:** `DataManager` resolves every input with **override → config
  `data_sources` → conventional default → bundled asset** precedence
  (`config.py`). `tth`/`reflections` additionally fall back per-scan
  (`Metadata/<scan>/…`) then project then bundled.

### 4. `create-positions` — real (X,Y) from SOCKETSERVER
- **CLI:** `xrd-app create-positions [--method averaging|basic] [--reduction N]`.
  Reduces the interferometer stream to one true stage position per trigger →
  `Metadata/<scan>/positions.csv`.
- **GUI:** no dedicated button — **`grid` runs this automatically** when no real
  CSV exists but a SOCKETSERVER stream is present. You can also supply a CSV via
  Setup → **Load positions…**.
- **Guardrail:** if there's neither a real CSV nor a SOCKETSERVER stream, `grid`
  **hard-fails** rather than silently reconstructing from the file-per-row layout
  (that silent fallback is what skewed rocking 203–214).

### 5. `grid` — frames → spatial bin grid
- **CLI:** `xrd-app grid --bin-size 3 [--deskew-method auto] [--variant TAG]
  [--shape ROWSxCOLS] [--rawgrid]`. **Default `--deskew-method auto`**:
  `positions_xy` at 1×1 (both axes snapped to true (X,Y) — skew-free), `faithful`
  at ≥2×2. Records `coordinate_source` + `positions_real` in the output JSON.
- **GUI:** no standalone button — **folded into Programs → Create bins**
  (`make-bins` runs `grid` then `bin`). The GUI always uses `deskew-method auto`;
  `--rawgrid`, `--deskew-method`, `--variant`, and `--shape` are **CLI-only**.
- **Guardrail (verified):** with no raw frames, `grid` fails cleanly with
  `raw frames directory not found` + guidance.

### 6. `bin` / `make-bins` — build the binned HDF5
- **CLI:** `xrd-app bin --bin-size 3 [--variant TAG] [--compression zstd]`.
  `make-bins` = `grid` + `bin` in one, and also refreshes the reflection grand-sum
  (`reflection_sum.npz`) so the Setup histogram is instant. **Defaults:** bin-size
  **3**, compression **zstd**.
- **GUI:** Programs → **Data Prep → Create bins** (spin box sets N; type any size).
  Depends only on the spin box, not the "Existing bins" dropdown. `--compression`
  is CLI-only (GUI uses zstd).
- **Guardrail:** `bin` **refuses** any grid mapping whose `positions_real` is not
  true (layout-reconstructed / synthetic / legacy), so nothing is ever binned on
  the serpentine skew.

### 7. `peaks` — Phase 1, per-bin detection
- **CLI:** `xrd-app peaks --bin-size 3 [--algorithm NAME] [--snr 4.0]
  [--variant TAG]`. **Defaults:** snr **4.0**; algorithm = **highest-scoring
  bundled detector** for the bin (`5x5_tophat_band_adaptive_snr` here). Writes
  `<algo>_peaks_NxN.h5` + records a lineage block.
- **GUI:** Programs → **Peak Finding → Run**. Multi-select detectors
  (Ctrl/Shift-click); the highest-scoring is pre-selected. Fans out over
  (selected scans × selected detectors). **`--snr` is not exposed** — the GUI
  always uses 4.0. Per-frame (unbinned) detectors are skipped with a note.

### 8. `shapes` — Phase 2, link + gaussian filter
- **CLI:** `xrd-app shapes --bin-size 3 --peak-algo NAME [--algorithm gaussian]
  [--link-tolerance 5] [--coordinate/--grid-link] [--variant TAG]`. **Defaults:**
  algorithm **gaussian**, link-tolerance **5 px**. Linking mode defaults to
  **grid** at ≥2×2 and **coordinate** (gridless, true-(X,Y) neighbors) at 1×1;
  at 1×1 a `gaussian` request maps to `territory` (same gaussian verification,
  coordinate linking) and the output is tagged `_coord`. Writes
  `<algo>_shapes_NxN.h5` + `kept_peaks_NxN.csv` + `filtered_peaks_NxN.csv`.
- **GUI:** Programs → **Shape Finding → Run**. The **Peaks** dropdown selects the
  input peak set; **"⟵ run peak algorithm above first"** chains peaks→shapes via
  `run-pipeline` (required for multi-scan, since saved peak sets are per-scan).
  **`--link-tolerance` and `--coordinate/--grid-link` are not exposed** — the GUI
  uses defaults (which are the recommended values).

### 9. Device View — the spatial maps
- **No CLI** (it's a viewer; the CLI equivalent is reading the shapes JSON).
- **GUI:** **Device View** tab. Pick **Bin → Feature catalog**; only bins/catalogs
  that exist are listed. Switchable metrics (integrated intensity, Δ2θ, χ
  orientation, χ FWHM, Δ2θ FWHM), χ-range filter. Status line shows the catalog's
  lineage (scan + source peak set).
- **Linkage (verified):** the tab discovers catalogs via `core.catalogs`
  (`feature_sources`/`default_feature_source`); a freshly-run
  `gaussian_shapes_3x3.h5` is auto-listed and its provenance read from the
  in-file lineage block.

### 10–11. Territorial reference + Territory Map
- **CLI (was 4 steps):** `territory-grid --target-size 9` → `bin`/`peaks`/`shapes`
  all `--variant territory` (1×1). **Now one command — FIXED:**
  `xrd-app territory-build [--target-size 9] [--snr 4.0]` chains all four.
- **GUI — FIXED:** Programs → **Build territorial reference** (target-frames spin,
  default 9) shells out to `territory-build`. Previously the territorial reference
  was **CLI-only**: the Territory Map showed a "run this CLI command" placeholder
  with no way to build it from the GUI.
- **Territory Map is popup-only (by design):** there is **no Territory Map tab**.
  It opens from **Device View → "Territorial reference available →"**, a button
  that appears only when `grid_mapping_1x1_territory.h5` exists
  (`has_territory_reference`). Territorial catalogs are deliberately excluded from
  the Device View dropdown (their irregular cells can't render on the fixed
  row/col grid) and routed to the popup instead.

---

## Defaults reference

| Command | Key defaults |
|---|---|
| `init` | scan-number `None`; seeds `Metadata/reflections.*` |
| `scan-detect` | fast (sampled counts); `--deep` for exact |
| `grid` | bin-size **3**, deskew-method **auto** (positions_xy @1×1 / faithful @≥2×2) |
| `bin` / `make-bins` | bin-size **3**, compression **zstd** |
| `peaks` | bin-size **3**, snr **4.0**, algorithm = best bundled detector |
| `shapes` | bin-size **3**, algorithm **gaussian**, link-tolerance **5**, mode = grid @≥2×2 / coordinate @1×1 |
| `territory-grid` / `territory-build` | target-size **9**, snr **4.0**, compression **zstd** |
| `batch` | grid→bin→peaks→shapes, bin-size **3**, snr **4.0** |

The GUI uses each command's defaults verbatim for every knob it does not expose
(see below), so GUI results equal `xrd-app <cmd>` with no extra flags.

---

## GUI vs CLI capability comparison

**Same both ways (GUI = a button over the CLI):** init, scan-detect, link,
convert-poni, make-bins/bin, peaks, shapes, run-pipeline (chain),
run-combined, lineage, build-holdout/run-cvevolve, **territory-build** (new).

**GUI adds convenience the CLI doesn't:** workspace/project picker, visible-scan
curation, multi-scan × multi-algorithm fan-out (queues `run-pipeline` jobs),
live progress + cancel, the interactive viewers (Device / HD Device /
Orientation), and the reflection editor.

**CLI-only (not surfaced in the GUI), by design — defaults are recommended:**

| Capability | CLI flag | Why it's fine as a default |
|---|---|---|
| SNR threshold | `peaks --snr` | 4.0 is the tuned working value |
| Link tolerance | `shapes --link-tolerance` | 5 px matches the grid |
| Linking mode | `shapes --coordinate/--grid-link` | auto by bin size (skew-free @1×1) |
| Deskew method | `grid --deskew-method` | `auto` is the skew-free choice |
| Compression | `bin --compression` | zstd is the standard |
| Coordinate variants / raster | `grid --variant/--shape/--rawgrid` | advanced / debugging only |
| Cross-scan study | `aggregate/track/rocking/predict/run-study` | Rocking-Study tab (out of scope here) |

If per-run control of SNR / link-tolerance is ever wanted in the GUI, the clean
place is an "Advanced" expander in the Peak/Shape boxes that appends the flags —
no engine changes needed.

---

## Breakpoints found and fixed

A GUI-first user going empty-project → device maps → territory now hits **no
dead-ends**. Two were fixed:

1. **GUI "New project" didn't seed reflections** (`workspace.create_project` vs
   `cli.py init`). GUI-made projects fell back to the bundled set and showed no
   "Project default" in the Reflections selector. → `create_project` now seeds
   `reflections.{json,py}` identically to `init`.

2. **Territorial reference was CLI-only.** The Territory Map / Device-View popup
   only appear *after* `territory-grid` + three `--variant territory` steps, which
   had no GUI control. → added the `territory-build` CLI command (one command for
   the whole chain) and a Programs → **Build territorial reference** button that
   shells out to it.

Non-breakpoints left as-is (work via sensible defaults): SNR, link-tolerance,
linking mode, deskew method, compression are CLI-only knobs.

---

## Verification (what was actually run)

Scratch project `/home/takaji/xrd_walkthrough`, real **Scan_0203** inputs
(binned h5 + grid mappings + positions symlinked from `rocking_203_214`; raw XRD
frames live on `/net/micdata` and aren't mounted here, so the front-end
`scan-detect→grid→bin` was exercised via its guardrails rather than a fresh raw
build).

- `init` → tree + config + seeded reflections ✓
- `link --tth` + `status` → all inputs resolve; default detector =
  `5x5_tophat_band_adaptive_snr` ✓
- `peaks --bin-size 3` → **7551 peaks in 2379 bins** (2m20s) ✓
- `shapes --bin-size 3` → **167 shapes kept, 1200 filtered** + kept/filtered CSVs ✓
- `lineage` → shapes → peaks → detector chain with defaults recorded
  (snr=4.0, link_tol=5, gaussian) ✓
- Device View resolution (`core.catalogs`) auto-lists `gaussian_shapes_3x3.h5` ✓
- Territory view: `has_territory_reference=True`, territory catalog recognized &
  routed to the popup (1302 features), territory bins resolve ✓
- Guardrails: `scan-detect` rejects ME7-only dirs; `grid` fails cleanly with no
  raw frames ✓
- **Fixes:** `workspace.create_project` seeds reflections ✓; `territory-build`
  registers and its chain enters at `territory-grid` ✓; full app window +
  Setup/Programs/Device View tabs build headless (Qt offscreen) with the new
  **Build territorial reference** button and the Device-View territory popup
  button both present ✓.

> Not runtime-verified end-to-end (needs raw frames on `/net/micdata` and ~20 min
> over 25k territory bins): a *full* `territory-build` compute. Its wiring, option
> surface, and first-step guardrail were verified; the downstream `bin/peaks/
> shapes --variant territory` steps are the same code paths exercised above.
