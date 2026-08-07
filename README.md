# xrd-app

A GUI and scriptable CLI for reproducible nano-XRD Bragg-peak analysis at the
ISN beamline, APS. Persistent and long-running workflows are available through
the CLI; the GUI provides project setup, execution controls, and result views.

## Install

```bash
python3 -m pip install -e .                    # core application
python3 -m pip install -e '.[poni]'            # + pyFAI calibration support
python3 -m pip install -e '.[qspace,poni,gl]'  # full reciprocal-space support
```

## Quick start

```bash
xrd-app init --name MyProject --root "/path/to/project"
xrd-app scan-detect --root "/path/to/project" --scans-dir "/path/to/Scans"
xrd-app link --root "/path/to/project" --tth "/path/to/tth.tiff" \
  --reflections "/path/to/reflections.json" --position-root "/path/to/positions"
xrd-app make-bins --root "/path/to/project" --scan 203 --bin-size 3 \
  --normalize-frames
xrd-app run-pipeline --root "/path/to/project" --scan 203 --bin-size 3
xrd-app gui --root "/path/to/project"
```

Every command supports `--help`, and defaults are shown there. Commands return a
nonzero exit status for usage errors and failed work, making them suitable for
shell scripts and schedulers. Paths supplied as command options are interpreted
relative to the current working directory unless the option explicitly says
otherwise. Production binning requires measured real `(X,Y)` positions. `grid`
can use a linked position CSV/HDF5 or build one from supported SOCKETSERVER data;
it fails rather than silently reconstructing the skew-prone file-row lattice.

## Pipeline

- **Peak finding** (Phase 1): run a detector over every spatial bin → per-bin peaks.
- **Shape finding** (Phase 2): link peaks across neighboring spatial bins,
  validate their intensity profiles, and characterize `chi_deg`, `chi_fwhm`, and
  `tth_fwhm`. These within-shape breadths are not rocking-curve widths; true
  rocking FWHM comes from `track` followed by `rocking` across sample theta.
- **CVEvolve**: build a seeded dev/holdout split (`build-holdout`) from verified
  labels or an algorithm's peak/shape set, then evolve a detector (`run-cvevolve`).

## Commands

Prefer the composite commands for routine work; the individual stages remain
available for inspection and recovery.

| Area | Commands |
|---|---|
| Project setup | `init`, `link`, `status`, `scan-detect`, `convert-poni`, `whole-frame-reflections` |
| Binning | `create-positions`, `grid`, `territory-grid`, `archive-unbinned`, `bin`, `make-bins`, `territory-build` |
| Peak and shape analysis | `detectors`, `peaks`, `shapes`, `run-combined`, `run-pipeline`, `batch` |
| Manual ROI analysis | `roi-detect`, `roi-shapes`, `roi-save`, `roi-cvevolve-init` |
| Cross-scan studies | `aggregate`, `track`, `rocking`, `predict`, `combined-device`, `scan-table`, `register-study`, `list-studies`, `run-study` |
| Linked XRF/XRD | `xrf`, `linked-xrd-track` |
| Reciprocal space | `qspace`, `rsm` |
| Optimization | `build-holdout`, `cvevolve-init`, `run-cvevolve`, `register-cvevolve`, `save-algorithm` |
| Reporting | `reflection-sum`, `hd-device-map`, `report` |
| Interfaces | `gui`, `roi`, `roifeature` |
| Provenance | `lineage` |

Run `xrd-app --help` for the complete installed command list and
`xrd-app COMMAND --help` for options and defaults. The GUI's per-tab General
panels mirror the relevant command help.

## Project layout

```
<project>/
  config.yaml
  Raw/         scans.json registry of external scan locations
  Binned/      per-scan unbinned archives and NxN/variant bin files
  Metadata/    calibration, measured positions, grids, sums, and XRF maps
  Labels/      peak, shape, ROI, combined, HD-map, lineage, and CSV products
  Figures/     exported figures and PDF reports
  CVEvolve/    optimizer sessions and dev/holdout data
  Algorithms/  project-owned detector modules and catalogs
  Study*/      cross-scan products, created by study commands
  XRF/         optional xrf-app add-on
```

## Tutorials

The tracked [notebooks](notebooks/README.md) are CLI-first, percent-format Python
notebooks for new users. They explain each scientific stage, print commands and
results, and block detector-frame work until `RUN_HEAVY = True` is set explicitly.

## Layout of this repo

```
xrd_app/    application package (CLI, core, GUI, tabs, algorithms)
notebooks/  tracked CLI-first tutorials
```

See [xrd_app/docs/BINNING_STORAGE.md](xrd_app/docs/BINNING_STORAGE.md) for
storage choices and [QSPACE.md](notebookwalkthroughthrough/QSPACE.md) for reciprocal-space workflows.
