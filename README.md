# xrd-app

A single-GUI workflow tool for nano-XRD Bragg-peak analysis (ISN 26-ID-C, APS),
built as a friendly face over a scriptable CLI: every "big button" in the GUI
shells out to a CLI command, and the previously-separate GUIs are combined into
one tabbed window.

## Install

```bash
pip install -e .            # core
pip install -e '.[poni]'    # + pyFAI for .poni → tth conversion
```

## Quick start

```bash
xrd-app init --name MyProject --root /path/to/project
xrd-app scan-detect --root /path/to/project --scans-dir /path/to/Scans
xrd-app link --root /path/to/project --tth tth.tiff --reflections reflections.json
xrd-app make-bins --root /path/to/project --scan 203 --bin-size 3
xrd-app run-pipeline --root /path/to/project --scan 203 --bin-size 3
xrd-app gui --root /path/to/project
```

Every command supports `--help`, and defaults are shown there. Commands return a
nonzero exit status for usage errors and failed work, making them suitable for
shell scripts and schedulers. Paths supplied as command options are interpreted
relative to the current working directory unless the option explicitly says
otherwise.

## Pipeline

- **Peak finding** (Phase 1): run a detector over every spatial bin → per-bin peaks.
- **Shape finding** (Phase 2): link peaks across neighboring bins (Union-Find),
  keep gaussian-like features, characterize `rocking_fwhm` / `strain_breadth` /
  `chi_deg`. A *shape* is a peak that holds up across bins.
- **CVEvolve**: build a seeded dev/holdout split (`build-holdout`) from verified
  labels or an algorithm's peak/shape set, then evolve a detector (`run-cvevolve`).

## Commands

Prefer the composite commands for routine work; the individual stages remain
available for inspection and recovery.

| Area | Commands |
|---|---|
| Project setup | `init`, `link`, `status`, `scan-detect`, `convert-poni` |
| Binning | `create-positions`, `grid`, `territory-grid`, `archive-unbinned`, `bin`, `make-bins`, `territory-build` |
| Peak and shape analysis | `detectors`, `peaks`, `shapes`, `run-combined`, `run-pipeline`, `batch`, `reflection-sum` |
| Manual ROI analysis | `roi-detect`, `roi-shapes`, `roi-save`, `roi-cvevolve-init` |
| Cross-scan studies | `aggregate`, `track`, `rocking`, `predict`, `scan-table`, `register-study`, `list-studies`, `run-study` |
| Derived maps | `hd-device-map`, `combined-device`, `qspace`, `rsm`, `xrf` |
| Optimization | `build-holdout`, `cvevolve-init`, `run-cvevolve`, `save-algorithm` |
| Interfaces | `gui`, `roi`, `roifeature` |
| Provenance | `lineage` |

Run `xrd-app --help` for the complete installed command list and
`xrd-app COMMAND --help` for options and defaults. The GUI's per-tab General
panels mirror the relevant command help.

## Project layout

```
<project>/
  Raw/        scans.json registry + links to external scan dirs
  Binned/     pre-binned xrd_NxN_bins.h5 (per scan)
  Metadata/   tth.tiff, reflections.json (+ generated .py), grid maps, gui_state
  Labels/     per-scan algorithm outputs (*_peaks/*_shapes) + manual labels
  Figures/    saved PNGs
  CVEvolve/   dev/holdout splits + sessions
```

## Layout of this repo

```
xrd_app/            the package (CLI, app, core/, gui/, tabs/, algorithms)
cvevolve_*/         CVEvolve configs, prompts, and holdout sets per bin size
docs/               research and deployment reference material
```

Workflow details: [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md) and
[xrd_app/PATHWAYS.md](xrd_app/PATHWAYS.md). Domain vocabulary is defined in
[TERMINOLOGY.md](TERMINOLOGY.md).
