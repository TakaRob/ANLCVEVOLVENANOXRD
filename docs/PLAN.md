# XRD App — Single-GUI Workflow Tool (PackageDraft2)

> Fleshed-out plan derived from `NewFullProcessThing.md` (your notes are the
> source of intent; this is the worked-out design). One easy-to-use Python app
> with a **single GUI** where every "big button" is a thin wrapper over a **CLI
> command**, and the old separate GUIs become **tabs** in one window.

---

## 0. Design decisions (locked with you)

1. **Hybrid GUI.** Setup + Programs buttons *shell out to CLI commands*
   (`grid`/`bin`/`peaks`/`shapes`/`batch`). The interactive tools
   (View/Label, Shape/Verify, Device, Orientation) are **embedded as tabs** that
   reuse the existing GUI modules (`labeling.py`, `viewer.py`, `device_map.py`,
   `orientation.py`) rather than rewriting them.
2. **Peak finding vs. shape finding = the two existing pipeline phases.**
   - **Peak Finding** = Phase 1: run a detector over every bin → per-bin peaks.
   - **Shape Finding** = Phase 2: link peaks across neighboring bins (Union-Find),
     keep those with a gaussian-like cross-bin profile, characterize them
     (`rocking_fwhm`, `strain_breadth`, `chi_deg`). A "shape" is a peak that holds
     up across multiple bins under the gaussian filter (your definition).
   - Shape finding *consumes* peaks: fresh peak-finding output or a saved
     `*_peaks.json`.
3. **New package in `PackageDraft2/`, reusing existing modules** (wrap `core/`
   and `gui/`, don't rewrite).
4. **A project holds many scans.** Setup picks an *active* scan; batch/aggregate/
   CVEvolve run across all. Selection is file-driven (§3).

---

## 1. Architecture

```
Single GUI (PyQt5, tabbed)
  Setup │ Programs │ View/Label │ Shape/Verify │ Device │ Orientation
        │  buttons call ↓
CLI  (xrd-app …)   the engine; fully usable headless
  init · link · scan-detect · grid · bin · peaks · shapes ·
  batch · aggregate · save-algorithm · run-cvevolve · gui
        │  imports ↓
core/ (reused)  io · processing · aggregate · algorithms
gui/  (reused)  labeling · viewer · device_map · orientation
```

**GUI↔CLI contract.** Buttons launch CLI subprocesses and stream stdout into a
status pane. CLI emits parseable progress (`PROGRESS 42/138`) → progress bar +
cancel. Anything the GUI does is reproducible from the terminal.

---

## 2. File structure (refined from your sketch)

```
PackageDraft2/
└── xrd_app/
    ├── pyproject.toml          # entry point: xrd-app = xrd_app.cli:main
    ├── cli.py  config.py  app.py
    ├── tabs/                   # setup, programs, view_label, shape_verify,
    │                           #   device, orientation  (wrap existing GUIs)
    ├── core/                   # io, processing, aggregate, algorithms (reused)
    │   ├── geometry.py         # NEW: .poni → tth map (gap #1)
    │   └── save_algorithm.py   # NEW: generate wrapper .py files
    ├── PeakAlgorithms/  1x1/ 3x3/ 4x4/ 5x5/  + catalog.json
    ├── ShapeAlgorithms/        # Phase-2 link/gaussian param sets
    ├── CombinedAlgorithms/     # peak+noise-reduction wrappers (saved)
    └── Projects/<ProjectName>/
        ├── config.yaml
        ├── Raw/scans.json              # {Scan_0203: /abs/path, …} one registry
        ├── Binned/Scan_0203/xrd_3x3_bins.h5
        ├── Metadata/                   # tth.tiff, calibration.poni,
        │     reflections.json (+ reflections.py loader), gui_state.json
        │     reflections may also live per-scan: Metadata/<Scan>/reflections.json
        ├── Labels/Scan_0203/
        │     <algo>_peaks.json  <algo>_shapes.json  manual_labels.json
        ├── Figures/
        └── CVEvolve/
```

- `Raw/scans.json` = "all scan links in one file" (your idea) and feeds the GUI
  scan switcher.
- The existing bundled `detectors/` + `catalog.json` (holdout F1/F2) maps onto
  `PeakAlgorithms/` directly — nothing thrown away.
- Per-scan namespacing matches today's `DataManager`.

---

## 3. Data loading (Setup) — file-driven

- **Pick a single `.hdf5`** → walk up one dir → register that scan →
  `Scan_0203 / 151 hdf5 files`.
- **Pick the `Scans/` parent dir** (`Scans/Scan#/scan#_#.hdf5`) → discover all
  `Scan_*`, validate, report `200 scans detected ✓` (flag bad ones).
- **Validation** (fast, no full load): readable frames, frame shape matches
  detector (confirm 256×256 vs 1062×1028), consistent frame count (warn on
  outliers), tth/poni + reflections present or flagged TODO.

---

## 4. Pages (each has a `--help`-style "General" math/visualization panel)

- **Setup** — Project Name (`init`); Load Data (§3); **Load tth.tiff** (primary)
  with a **"Convert .poni → tth" button** for users who bring a poni; **Manual
  reflection popup**: radial 2θ histogram on the fully-summed #×# image with 2θ
  overlay, `[Name][Bragg Angle][Width]` → `[Create]` saves the reflection set
  (per-scan `reflections.json`), **emits a tth.tiff** if none was loaded, and
  PNGs to `Figures/`. `Width` becomes the detector's band restriction.
- **Programs** — Peak Finding `[algo][bin][Run][status]` → `peaks`; Shape Finding
  `[algo][bin][saved-peaks|current][Run][status]` → `shapes`; **Scan all** →
  `batch --all`; **[Use CVEvolve] popup** → Pick Scan Directory, Pick Development
  Set, open prompt `.md`s + add context files → `run-cvevolve`; new algorithms
  return to the library auto-named and become selectable.
- **View/Label** (wraps `labeling.py`) — interactive labeling; **verified tag**
  stamped into `manual_labels.json`; **Save Algorithm** button (§5).
- **Shape/Verification** (wraps `viewer.py`) — review shapes; **adjustable
  shape-finding sensitivity** + same Save Algorithm + verification.
- **Device View** (wraps `device_map.py`) — spatial metric maps, χ filter.
- **Orientation Map** (wraps `orientation.py`) — KDE sector chart.

---

## 5. "Save Algorithm" (your key reusable idea)

`[Save Algorithm]` writes a new `.py` into the library that (1) sets the chosen
sensitivity, (2) calls the chosen noise-reduction module, (3) delegates to the
base algorithm — conforming to the **detector contract** (`precompute_tth`,
`detect_in_band(...)`) that `core/processing.py` already imports. Template in
`core/save_algorithm.py`.

**Naming conventions** (filling your empty section):
- Manual: `<base>__sens<NN>__nr-<module>` (e.g. `tophat__sens97__nr-tv`).
- CVEvolve: `Scan0203_5x5_<desc>_F2-0.21`.
- Registered in `catalog.json`:
  `{name, bin_size, role, holdout_f1, holdout_f2, source}`.
- BYO: drop a contract-compliant `.py` into `PeakAlgorithms/<NxN>/`; an
  `ALGORITHMS.md` documents the two required functions.

---

## 6. Verification labels (CVEvolve-ready)

`Labels/<scan>/manual_labels.json`, per bin, with
`{peaks:[…], verified:true, reviewer, reviewed_at}`. `verified:true` → eligible
for the CVEvolve holdout. Human labels stay separate from algorithm output for
clear provenance.

---

## 7. WSL/IO

Binned HDF5 is the IO-heavy artifact. Keep `Binned/` symlinked to native WSL or a
mounted fast disk (config records the real path). Setup detects WSL and warns if
`Binned/` resolves under `/mnt/c/...`.

---

## 8. Gaps / changes your draft didn't fully address

**Resolved (your call):**

1. ✅ **2θ map = `tth.tiff` only for now.** Pipeline keeps consuming `tth.tiff`.
   Leave a **"Convert .poni → tth" button** for users who arrive with a poni
   (`geometry.py`, pyFAI, generated lazily). Manual/"other" reflection setup
   also **emits a `.tiff`** so everything downstream is uniform.
2. ✅ **Bin sizes 1×1/3×3/4×4/5×5 first-class.** `SelectBinSize` filters
   algorithms by size; **1×1 shows F2**, others show **F1**.
4. ✅ **Reflections stored as JSON, loaded by `reflections.py`.** GUI writes
   `reflections.json` (data); `reflections.py` is a thin loader. **Per-scan** —
   different scans can carry different angles, so reflections resolve per scan
   (project default + per-scan override). See §2 / §4.1.
9. ✅ **Drop `xrd-tools`.** It was a prototype, not in use — no alias, clean
   `xrd-app` namespace.
10. ✅ **Reflection `Width` drives the detection band.** The band you draw in the
    manual popup *is* the detector's band restriction (draw == detect).

**Resolved (your call, round 2):**

3. ✅ **Long jobs = real terminal.** Each job (`peaks`/`shapes`/`batch`/
   `run-cvevolve`) launches in an **auto-opened terminal window** (whichever is
   simplest to start/stop; small embedded terminal view is an acceptable
   alternative). **Closing the terminal kills the process**; a **Cancel button**
   in the GUI also kills it. CLI prints running counts like `(200/400) peaks`
   (and `PROGRESS i/n`) so progress is visible in the terminal and parseable.
5. ✅ **Concurrency.** Jobs run as their own subprocess; active-scan outputs are
   write-locked while a job touches them (no half-written catalogs in a viewer).
6. ✅ **Plugin tabs.** Contract: module exposes `make_tab(project, scan) ->
   QWidget` + `TAB_META`; app auto-discovers via entry points → "link a gui, it
   appears as a tab."
7. ✅ **Standalone tabs.** Each tab runs alone with a minimal Setup header, then
   the identical UI.
8. ✅ **State persistence.** `gui_state.json`: active scan, bin size, per-tab
   sensitivity/algorithm/colormap/selection; save on change (debounced), restore
   on open.
11. ✅ **CVEvolve dev set = bin selection.** "Pick Development Set" selects **by
    bins** — either the `verified` bins, or **a given algorithm's output set**
    (peak *or* shape). **The source determines the algorithm kind CVEvolve
    evolves**: a peak set → peak algorithm, a shape set → shape algorithm (and a
    combined source → combined). Then a single **holdout percentage** slider
    splits dev/holdout, with a live **`(200/400) peaks`** count. Splits write in
    the format CVEvolve's holdout already expects (see new gap #14).

**Newly surfaced — resolved (your call, round 3):**

12. ✅ **Adaptive frame shape.** Read the frame shape from the data at load time
    and store it in `config.yaml`; never hard-code `1062×1028`. §3 validation
    checks against the detected shape so other detectors just work.
13. ✅ **pyFAI deferred.** Don't implement `.poni→tth` yet — ship a **dummy/stub**
    `geometry.py` + a disabled/placeholder "Convert .poni" button. Real pyFAI
    conversion is a later drop-in. Core app installs without pyFAI.
14. ✅ **Reuse the existing holdout writer.** The dev-set split reuses the old
    labeling GUI's **"Update Holdout Set"** mechanism: writes
    `test_data/bin_annotations.json` (dev) + `holdout_data/bin_annotations.json`
    (holdout) with `empty_bins.json` + `grid_mapping.json`. Seed the split for
    reproducibility (matches seeded `evaluate.py`).
15. ✅ **"Combined" = full end-to-end pipeline, not a joint engine mode.** Per
    `cvevolve_1x1/prompt.md`, one evolved algorithm contains all three stages
    (per-frame detection → spatial linking → Voigt/Gaussian shape verification)
    and is scored on the **final shapes**. So:
    - **PeakAlgorithms/** = Stage-1 detectors (incl. saved peak+noise-reduction
      wrappers from Save Algorithm).
    - **ShapeAlgorithms/** = Stage-2/3 linking+verification (incl. saved
      shape-sensitivity wrappers).
    - **CombinedAlgorithms/** = end-to-end algorithms doing all stages (e.g. the
      cvevolve_1x1 output).
    `run-cvevolve` needs **no joint mode** — the algorithm *kind*
    (peak / shape / combined) is set by the **prompt + which label set** you feed
    it, tagged `kind:` on the output. The 1×1 path naturally produces a combined
    algorithm.

---

## 9. CLI surface

```
xrd-app init           --name --root
xrd-app link           --scan-file <hdf5> | --scans-dir <dir> | --poni | --tth | --reflections
xrd-app scan-detect    --scans-dir <dir>          # validate, report "N scans detected"
xrd-app grid           --scan --bin-size [--shape]
xrd-app bin            --scan --bin-size [--compression]
xrd-app peaks          --scan --bin-size --algorithm [--sensitivity]      # Phase 1
xrd-app shapes         --scan --bin-size --algorithm --from-peaks <json>  # Phase 2
xrd-app batch          --all|--scans --bin-size --algorithm [--shape] [--skip-existing]
xrd-app aggregate      --scans --bin-size --format
xrd-app save-algorithm --base --sensitivity --noise-reduction --name [--kind peak|shape]
xrd-app run-cvevolve   --config [--engine podman|docker|local]
xrd-app gui
```

---

## 10. Build order

1. Package skeleton + `cli.py` re-exporting existing core (init/link/grid/bin).
2. Split `process` → `peaks` + `shapes`; wire the `Labels/` JSON layout.
3. `scan-detect` + `Raw/scans.json` + Setup data-loading UX.
4. `app.py` single window + tab loader; embed the four existing GUIs.
5. `save-algorithm` + catalog + naming; CVEvolve return-to-library hookup.
6. Verification stamping + holdout export.
7. `.poni → tth` (`geometry.py`) + manual reflection popup.
8. Plugin-tab contract + standalone entry points.
9. Polish: progress streaming, cancel, WSL warnings, state persistence.
