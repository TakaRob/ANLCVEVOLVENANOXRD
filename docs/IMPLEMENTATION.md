# XRD App — Implementation Plan

Concrete, phase-by-phase build of the design in `PLAN.md`. Each phase lists the
files to create, what to reuse from the existing `src/xrd_tools/` code, key
signatures, and an **acceptance check** (how you know it's done). Build in order;
each phase leaves the app runnable.

Source reuse map (existing → new):
- `src/xrd_tools/config.py`     → `xrd_app/config.py` (adapt dir names)
- `src/xrd_tools/core/*.py`     → `xrd_app/core/*.py` (copy, refactor `processing`)
- `src/xrd_tools/gui/*.py`      → `xrd_app/gui/*.py` (refactor to embeddable widgets)
- `src/xrd_tools/detectors/`    → `xrd_app/PeakAlgorithms/` (+ catalog.json)

> Note: `xrd-tools` is a dropped prototype — we **copy** its modules into the new
> package and edit freely; no back-compat, no alias.

---

## Phase 0 — Package skeleton (runnable empty CLI)

**Create**
```
PackageDraft2/xrd_app/
  __init__.py
  pyproject.toml          # name=xrd-app, entry: xrd-app = xrd_app.cli:main
  cli.py                  # click group `main`, just `--help` for now
  config.py               # copied, edited in Phase 1
  core/__init__.py        # copy io.py, processing.py, aggregate.py, algorithms.py
  gui/__init__.py         # copy the 4 GUI modules (edited later)
```
- `pyproject.toml` deps: `click pyyaml numpy scipy h5py hdf5plugin tifffile
  PyQt5`. **No pyFAI** (deferred, Phase 7).
- `cli.py`: `@click.group() def main(): ...` and register subcommands as we go.

**Acceptance:** `pip install -e PackageDraft2/xrd_app` then `xrd-app --help`
prints the (empty) command group.

---

## Phase 1 — Project config + tree + init/link

**Edit `xrd_app/config.py`** (from `xrd_tools/config.py`):
- `PROJECT_DIRS = ["Raw", "Binned", "Metadata", "Labels", "Figures", "CVEvolve"]`
  under `Projects/<name>/`.
- `default_config()`: add `detector: {shape: null}` (filled adaptively, Phase 3),
  `scans: {}` registry, keep `data_sources`, `bins`, `tracking`.
- `DataManager` path methods retargeted to the new tree:
  - `binned_h5(scan, bin_size)` → `Binned/<scan>/xrd_NxN_bins.h5`
  - `labels_dir(scan)` → `Labels/<scan>/`
  - `peaks_json(scan, algo, bin_size)` → `Labels/<scan>/<algo>_peaks.json`
  - `shapes_json(scan, algo, bin_size)` → `Labels/<scan>/<algo>_shapes.json`
  - `reflections(scan)` → per-scan `Metadata/<scan>/reflections.json` →
    project `Metadata/reflections.json` → bundled default
  - `tth_map(scan)` → `Metadata/tth.tiff` (or per-scan) → bundled default
  - `scans_registry()` → `Raw/scans.json`
- Keep the bundled-detector resolution logic, pointing at `PeakAlgorithms/`.

**CLI commands** (`cli.py`):
- `init --name --root` → create tree + `config.yaml`.
- `link --scan-file | --scans-dir | --tth | --reflections | --poni` → records
  paths; writes/updates `Raw/scans.json` (Phase 3 does the heavy discovery).

**Acceptance:** `xrd-app init --name Demo` creates the `Projects/Demo/` tree and
`config.yaml`; `xrd-app status` lists resolved paths with ✓/✗.

---

## Phase 2 — Split `process` → `peaks` + `shapes`

The existing `core/processing.py` already separates the phases internally:
- Phase 1: `run_detection_all_bins()` → per-bin peaks.
- Phase 2: `link_peaks()` + `characterize_features()` → kept (shapes)/filtered.

**Refactor `core/processing.py`** into two public entry points:
```python
def run_peaks(bins_h5, tth_path, detector_path, reflections_path,
              bin_size, snr_threshold, sensitivity=None,
              progress=None, log=print) -> dict:
    """Phase 1 only. Returns {scan, bin_size, detector, peaks_by_bin:{...}}.
       `progress(i, n)` is called per bin so the CLI can print (i/n)."""

def run_shapes(peaks, tth_path, grid_mapping, bin_size,
               link_tolerance, sensitivity=None,
               progress=None, log=print) -> dict:
    """Phase 2: link + profile-keep + characterize.
       `peaks` is a run_peaks() result OR a loaded *_peaks.json.
       Returns {kept:[...shapes...], filtered:[...]}."""
```
- Keep `run_analysis()` as a thin wrapper (`run_peaks`→`run_shapes`) for batch.
- `progress` callback drives the `(i/n)` count for long jobs (Phase 9).

**JSON schemas** (write helpers in `core/io.py`):
- `<algo>_peaks.json`: `{scan, bin_size, detector, created_at,
  peaks_by_bin: {"<bin_key>": [{row,col,x,y,snr,intensity,reflection}]}}`
- `<algo>_shapes.json`: `{scan, bin_size, peak_source, shape_algo, created_at,
  kept:[{center_row,center_col,detector_x,detector_y,peak_intensity,mean_snr,
  n_bins,rocking_fwhm,strain_breadth,chi_deg,reflection}], filtered:[...]}`

**CLI**
- `peaks --scan --bin-size --algorithm [--sensitivity]` → `run_peaks` → write
  `Labels/<scan>/<algo>_peaks.json`. Prints `(<k>/<n>) peaks`.
- `shapes --scan --bin-size --algorithm [--from-peaks <json>|--peak-algo <name>]`
  → load peaks → `run_shapes` → write `<algo>_shapes.json`.
- `batch --all|--scans --bin-size --algorithm [--shape] [--skip-existing]` →
  grid→bin→peaks→shapes per scan (reuse existing `batch` control flow).

**Acceptance:** on Scan_0203, `xrd-app peaks …` then `xrd-app shapes …` produce
the two JSONs; counts printed; `shapes` accepts a saved `_peaks.json`.

---

## Phase 3 — `scan-detect` + Setup data loading (adaptive shape)

**`core/io.py` additions**
```python
def discover_scans(path) -> list[ScanInfo]:
    """If `path` is an .hdf5 file → its parent scan dir (one scan).
       If a dir of Scan_*/ → all scans. ScanInfo: name, dir, n_frames, shape."""

def validate_scan(info) -> list[str]:
    """Fast checks: readable frame, frame shape, consistent count. Returns warnings."""

def detect_frame_shape(scan_dir) -> tuple[int,int]:
    """Read ONE frame's shape; do NOT hard-code 1062×1028."""
```
- `scan-detect --scans-dir <dir>` → discover + validate, print
  `N scans detected ✓ (warnings: …)`, write `Raw/scans.json`
  `{ "Scan_0203": {"dir": "...", "n_frames": 151, "shape": [H,W]} }`, and store
  `detector.shape` in `config.yaml` from the first valid scan.

**Acceptance:** pointing `scan-detect` at a single `.hdf5` registers one scan
(`Scan_0203 / 151 hdf5 files`); pointing at a `Scans/` dir reports the full count
and flags bad scans; `config.yaml` records the detected frame shape.

---

## Phase 4 — Single-window GUI + embed the 4 existing GUIs

**The key refactor:** each existing GUI builds its own `QMainWindow`. To embed as
a tab, extract the central widget.

**For each `gui/{labeling,viewer,device_map,orientation}.py`:**
- Refactor the window class so the UI is built in a `QWidget` subclass
  `XxxWidget(project_root, scan, bin_size)`; keep a tiny `launch_gui()` that wraps
  it in a `QMainWindow` for standalone use (Phase 8).
- Replace module-level `_DM/_BIN_SIZE` globals with instance state seeded from the
  passed `DataManager`.

**Create `xrd_app/app.py`**
```python
class MainWindow(QMainWindow):
    # header: project label + scan switcher (from Raw/scans.json) + bin-size
    # QTabWidget: Setup, Programs, View/Label, Shape/Verify, Device, Orientation
    # changing the active scan re-points every tab's DataManager
```
- Tabs built via the Phase-6/8 contract `make_tab(project, scan) -> QWidget`.
- `gui` CLI command → launches `MainWindow`.
- **gui_state.json**: load on open, save on change (debounced) — active scan, bin
  size, per-tab sensitivity/algorithm/colormap/selection.

**Create `xrd_app/tabs/setup.py` + `programs.py`**
- Setup: project open/create, data load (calls `scan-detect`), tth load + stubbed
  poni button, manual-reflection popup (Phase 7 fills it in).
- Programs: Peak/Shape rows (algorithm + bin-size dropdowns filtered by size;
  F2 column for 1×1, F1 otherwise), Run buttons → launch CLI jobs (Phase 9 wires
  the terminal/console), [Use CVEvolve] popup (Phase 6).

**Acceptance:** `xrd-app gui` opens one window; all four legacy GUIs work as tabs
against the active scan; switching scans updates every tab; state persists across
restarts.

---

## Phase 5 — Save Algorithm + library catalog

**Create `core/save_algorithm.py`**
```python
def save_algorithm(base, *, kind, sensitivity, noise_reduction, name,
                   bin_size, dest_root) -> Path:
    """Render a .py from a template that (1) sets sensitivity, (2) applies the
       noise-reduction module, (3) delegates to `base`, conforming to the detector
       contract (precompute_tth, detect_in_band). Register in catalog.json."""
```
- Template lives beside it (`_wrapper_template.py.txt`).
- Naming: manual `<base>__sens<NN>__nr-<module>`; CVEvolve
  `Scan0203_5x5_<desc>_F2-0.21`.
- `catalog.json` entry: `{name, bin_size, role, kind, holdout_f1, holdout_f2,
  source}`. Writes peak wrappers to `PeakAlgorithms/<NxN>/`, shape wrappers to
  `ShapeAlgorithms/`.
- CLI `save-algorithm --base --kind peak|shape --sensitivity --noise-reduction
  --name --bin-size`.
- Wire **[Save Algorithm]** buttons into View/Label (peak) and Shape/Verify
  (shape) tabs, prefilled with the current sliders.

**Acceptance:** dialing sensitivity + noise reduction in a tab and clicking Save
produces a runnable `.py` that `xrd-app peaks --algorithm <new>` can use, and it
appears in the dropdown.

---

## Phase 6 — Verification tag + CVEvolve dev-set popup

**Verification**
- `manual_labels.json` per scan/bin with `{peaks:[...], verified, reviewer,
  reviewed_at}` (schema in `PLAN.md` §6). Stamp `verified:true` from View/Label
  and Shape/Verify on review.

**Reuse the holdout writer** — extract the old labeling GUI's "Update Holdout Set"
logic into `core/holdout.py`:
```python
def build_split(source, holdout_pct, *, seed, dest) -> dict:
    """source = verified bins OR an algorithm's peak/shape set (sets `kind`).
       Seeded split → writes test_data/bin_annotations.json (dev) +
       holdout_data/bin_annotations.json (holdout) + empty_bins.json +
       grid_mapping.json. Returns counts {dev, holdout, total}."""
```
**[Use CVEvolve] popup (Programs tab)**
- Pick Scan Directory; pick **source = verified | <peak algo set> | <shape algo
  set>** (source decides `kind` = peak/shape/combined).
- **Holdout %** slider with live `(200/400) peaks` label (from `build_split`).
- Open prompt `.md`s + attach context files; then `run-cvevolve` (reuse existing
  Podman wrapper). New algorithm returns to the library auto-named (Phase 5).

**Acceptance:** popup builds a seeded dev/holdout split in the CVEvolve-expected
layout, shows the count, and launches `run-cvevolve`; the resulting algorithm
appears in the dropdowns tagged with its `kind`.

---

## Phase 7 — poni stub + manual reflection popup

- **`core/geometry.py`**: stub `poni_to_tth(poni_path, shape) -> np.ndarray` that
  raises `NotImplementedError("pyFAI conversion not yet implemented")`; the Setup
  "Convert .poni" button is present but disabled/placeholder.
- **Reflections**: `reflections.json` = `[{name, two_theta, width}]`;
  `reflections.py` is a thin loader resolving per-scan → project → bundled.
- **Manual reflection popup** (Setup): sum all frames at #×# binning → radial 2θ
  histogram; show summed image + histogram with 2θ overlay; inputs
  `[Name][Bragg Angle][Width]`; **[Create]** appends to `reflections.json`,
  **emits `tth.tiff`** if none loaded, saves PNGs to `Figures/`. The `Width`
  becomes the detector band restriction used by `peaks`.

**Acceptance:** creating a reflection writes `reflections.json`, renders + saves
the histogram/overlay PNG, and a subsequent `peaks` run restricts detection to
the drawn band width.

---

## Phase 8 — Plugin tabs + standalone launch

- **Contract**: a tab module exposes `make_tab(project, scan) -> QWidget` and
  `TAB_META = {title, order, takes_bin_size}`. `app.py` discovers built-in tabs
  plus any registered via entry point group `xrd_app.tabs`.
- **Standalone**: each tab module gets `if __name__ == "__main__"` →
  `python -m xrd_app.tabs.device --project <p> --scan <s>` opens just that tab in
  a minimal window with a Setup header.

**Acceptance:** dropping a third-party module exposing the contract makes a new
tab appear; `python -m xrd_app.tabs.orientation --project … --scan …` runs it
alone.

---

## Phase 9 — Long jobs, concurrency, polish

- **Long jobs**: run CLI subprocesses via `QProcess` into an **embedded console
  pane** (QPlainTextEdit) — simplest cross-platform/WSL option, easy start/stop.
  **Cancel** = `QProcess.kill()`; closing the pane/window kills the process. Parse
  `(i/n)` / `PROGRESS i/n` lines into a progress bar. (A pop-out OS terminal is an
  optional alternative.)
- **Concurrency**: jobs are isolated subprocesses; lock the active scan's output
  files (e.g. a `.lock` next to the catalog) so a viewer tab never reads a
  half-written file; refresh tabs when a job completes.
- **WSL warning**: Setup warns if `Binned/` resolves under `/mnt/c/...`.
- **Resume**: `batch --skip-existing`.

**Acceptance:** a long `batch` shows live progress in the console pane, Cancel
stops it cleanly, closing the window kills it, and viewer tabs don't error on a
catalog mid-write.

---

## Cross-cutting

- **Help/General panels**: each tab renders the matching `xrd-app <cmd> --help`
  text (math/algorithm description) in a toggle panel — keep CLI help and GUI
  "General" text from one source string.
- **Tests**: unit-test `discover_scans`, `validate_scan`, `run_peaks`/`run_shapes`
  on a tiny fixture HDF5; smoke-test the CLI end-to-end on Scan_0203.
- **Migration**: copy the bundled `detectors/` + `catalog.json` into
  `PeakAlgorithms/` as the seed library.

## Suggested sequencing for review
Phases 0–3 (CLI engine + data model) are independent of Qt and worth landing and
verifying first. Phases 4–9 (GUI) build on a working CLI. Each phase is a natural
commit/PR boundary.
