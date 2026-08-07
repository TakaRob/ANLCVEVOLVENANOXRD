# CLAUDE.md - xrd-app

App-specific guidance for `xrd_app/`, the nano-XRD Bragg-peak analysis
application for ISN 26-ID-C at APS. The repository-root `../CLAUDE.md` contains
the scientific context, computing constraints, and headless analysis rules.

## Architecture

**The CLI is the engine; the GUI is a face over it.** Every substantial
persistent or long-running GUI action must have an equivalent CLI command. Put
analysis and I/O in `core/`, expose orchestration through a thin Click command,
and have GUI controls invoke it for persistent work. Read-only rendering and
lightweight summaries may call the same pure `core` functions directly. Do not
put scientific analysis logic only in GUI modules.

Dependency direction:

```
core/         analysis, I/O, catalogs, config; no PyQt or Click
cli.py        Click command surface over core/
tabs/         one QWidget builder per app tab
gui/          reusable viewers and widgets
app.py        tabbed application shell and lazy tab discovery
workspace.py  project/workspace creation and selection
```

Keep imports flowing from the UI toward the CLI/core layers, never from core
into PyQt. `xrf_cli.py`, `xrf_project.py`, and `xrf_gui.py` provide the separate
`xrf-app` workflow while sharing selected XRF logic under `core/`.

## Current CLI workflow

Use `xrd-app --help` and `xrd-app COMMAND --help` as the command authority. The
standard explicit pipeline is:

```
init -> link + scan-detect -> [create-positions] -> grid -> bin -> peaks -> shapes
```

Important command compositions:

- `make-bins`: `archive-unbinned -> grid -> bin`, then refreshes the detector
  `reflection_sum.npz` best-effort.
- `run-pipeline`: `peaks -> shapes` for one scan whose bins already exist.
- `batch`: archive/grid/bin/peaks/shapes across registered scans.
- `territory-build`: archive plus `territory-grid -> bin -> peaks -> shapes` at
  nominal 1x1 with `--variant territory`.
- `reflection-sum`: full detector sum used by reflection histograms and reports.
- `hd-device-map`: caches 1x1 detector-peak intensity under binned features.
- `report`: produces the multi-scan PDF analysis deck.
- `aggregate -> track -> rocking -> predict -> combined-device`: cross-scan
  rocking analysis; `run-study` orchestrates the study workflow.
- `qspace -> rsm`: reciprocal-space products.
- `scan-table`: per-scan feature count, area, coverage, orientation, and shape
  summary.
- `xrf`: integrated ME7 processing into per-element maps on an existing XRD
  grid, stored under per-scan `Metadata/`.
- `xrf-app`: separate XRF-selection workflow using an optional project `XRF/`
  add-on for raw ME7 registration, calibration, material cuts, canonical
  selection HDF5 files, and ME7/XRD frame linking.

## Coordinate and binning rules

- Real measured `(X, Y)` positions are required for production binning. `grid`
  uses a linked position CSV/HDF5 or creates one from SOCKETSERVER data; it must
  fail rather than silently reconstructing the known skewed file-row lattice.
- `grid --deskew-method auto` uses true-position behavior appropriate to bin
  size. Keep row/column versus physical `(x, y)` explicit.
- `bin` rejects a grid mapping without `positions_real=true`.
- Use `--normalize-frames` for the production final-analysis bins so cells with
  different contributing frame counts are comparable.
- At 1x1, `shapes` defaults to gridless coordinate-neighbor linking and maps a
  gaussian request to the coordinate-capable territory implementation. At bin
  sizes 2x2 and above, it defaults to grid linking.
- Territorial products are irregular true-coordinate cells. Keep the
  `territory` variant consistent through mapping, bins, peaks, and shapes.
  `--target-size 1` is the lossless one-frame-per-territory focus product.
- `archive-unbinned` is the lossless acquisition-order cache. Reuse it instead
  of repeatedly reading large raw frame trees.

## Project layout and resolution

`xrd-app init` creates:

```
<project>/
  config.yaml
  Raw/         scans.json registry
  Binned/      <scan>/xrd_NxN_bins[_variant].h5 and unbinned archives
  Metadata/    tth.tiff, reflections.json, per-scan positions/grids/sums/XRF
  Labels/      per-scan peaks, shapes, combined, HD maps, lineage, and CSVs
  Figures/     exported figures and reports
  CVEvolve/    optimizer sessions
  Algorithms/  project-owned detector modules and catalogs
```

Commands also create `Study/` or other named study directories for cross-scan
analyses. `xrf-app` may create `XRF/` with `xrf_config.yaml`, `Raw/`, `Metadata/`,
`Cache/`, `Processed/`, and `Figures/`. Do not hardcode paths in app code. Resolve
them through `ProjectConfig`, `DataManager`, and `core.catalogs` helpers.

Input precedence is explicit command override, then configured `data_sources`,
then the conventional project location, then a bundled asset where supported.
For reflections and 2-theta maps, preserve the per-scan `Metadata/<scan>/`,
project `Metadata/`, bundled fallback chain. A multi-scan `DataManager` uses its
scan override; never silently default an unresolved scan to 203.

## Catalog and terminology rules

- A **peak** is a per-spatial-bin detector detection.
- A **shape** is a linked, verified physical feature spanning neighboring bins.
- A **track** links shapes across the sample-theta scan series.
- Use `core.catalogs` for catalog discovery, identity validation, naming, and
  HDF5 result loading/saving. Preserve lineage fields when producing results.
- Variant, scan, bin-size, and source-algorithm identity must stay consistent
  across a pipeline. Validate these before combining products.
- `chi_fwhm` and `tth_fwhm` are within-shape detector-azimuth and radial
  2-theta breadths. Do not call either a rocking curve; rocking curves are track
  intensity versus sample theta and are fitted by the `rocking` stage.
- Validate detections against expected 2-theta reflection bands. CVEvolve's
  primary optimization metric is mean F2, and final scores require full-setting
  confirmation.

## GUI contract

A tab module under `tabs/` exposes:

```python
TAB_META = {"title": ..., "order": N, ...}
def make_tab(project_root=".", scan=None, bin_size=3) -> QWidget: ...
```

`app.py` imports built-ins and `xrd_app.tabs` entry points lazily. A tab must
handle missing project data without crashing the whole app. Keep PyQt imports
lazy where practical so CLI/core imports work headlessly. Preserve established
design-system and header-selector behavior; supported header bin sizes are
`1, 3, 4, 5` unless intentionally changed across the app.

## Run and verify

From the repository root:

```bash
pip install -e .
xrd-app --help
xrd-app status --root /path/to/project --scan 203 --bin-size 3
python3 -m pytest xrd_app/tests
python3 -c "import py_compile,glob; [py_compile.compile(f,doraise=True) for f in glob.glob('xrd_app/**/*.py',recursive=True)]"
```

Use the project `.venv` when available. Do not launch `xrd-app gui` to verify
headless work; it requires a display and blocks. Test the underlying CLI command
or focused tests instead. Only open the GUI when the user is driving it.

The environment is WSL2 and project paths contain spaces, so quote every path in
shell commands. Raw detector data is large: stream it, crop where possible, preserve sensible
dtypes, and avoid repeated reads over network-mounted storage.
