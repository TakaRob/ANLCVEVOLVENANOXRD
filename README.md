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
xrd-app init --name MyProject              # create the project tree
xrd-app scan-detect --scans-dir <Scans/>   # discover + register scans (fast)
xrd-app link --tth tth.tiff --reflections reflections.json
xrd-app peaks  --scan 203 --bin-size 3     # Phase 1: per-bin detection
xrd-app shapes --scan 203 --bin-size 3 --peak-algo <name>   # Phase 2: link + gaussian filter
xrd-app gui                                # the single-window app
```

## Pipeline

- **Peak finding** (Phase 1): run a detector over every spatial bin → per-bin peaks.
- **Shape finding** (Phase 2): link peaks across neighboring bins (Union-Find),
  keep gaussian-like features, characterize `rocking_fwhm` / `strain_breadth` /
  `chi_deg`. A *shape* is a peak that holds up across bins.
- **CVEvolve**: build a seeded dev/holdout split (`build-holdout`) from verified
  labels or an algorithm's peak/shape set, then evolve a detector (`run-cvevolve`).

## Commands

`init · link · status · scan-detect · grid · bin · peaks · shapes · batch ·
detectors · save-algorithm · convert-poni · build-holdout · run-cvevolve · gui`

Run any with `--help`. The GUI's per-tab "General" panels mirror this help.

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
xrd_app/            the package (cli, app, config, core/, gui/, tabs/, PeakAlgorithms/ …)
cvevolve_*/         CVEvolve configs / prompts / holdout sets per bin size
docs/               PLAN.md, IMPLEMENTATION.md, original design notes
```

Design + build details: [docs/PLAN.md](docs/PLAN.md),
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
