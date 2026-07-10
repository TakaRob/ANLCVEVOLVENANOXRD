# CLAUDE.md — nano-XRD scientific computing

General working notes for this repo: nano-XRD data analysis for perovskite /
halide thin films at the **ISN 26-ID-C beamline, APS (Argonne)**. The active
application lives in `xrd_app/` and has its own `xrd_app/CLAUDE.md` with app
specifics — read that when editing the app. This file is the science + computing
context that applies everywhere.

> **Helping a researcher analyze data through chat** (not coding)? Read
> [`ANALYSIS_PLAYBOOK.md`](ANALYSIS_PLAYBOOK.md) first — it maps plain-language
> questions ("how many features? how large? what orientation spread?") to the
> exact `xrd-app` commands and result-file fields that answer them, and tells you
> which commands are cheap (run freely) vs. heavy (confirm first). Don't launch
> the GUI to answer a data question.

## What the science is

- **Goal:** map crystal grain orientation / strain across a sample surface by
  detecting Bragg peaks in nano-XRD detector images, scan position by position.
- **Instrument:** 15 keV beam, 75 µm pixel. A raw
  scan is thousands of CCD frames (one per spatial position) in HDF5.
- **Reflections** of interest (perovskite/halide + substrate): PbI2,
  (001)/(011)/(111)/(002), ITO, (012)/(112), each at a known 2-theta. These
  define the radial bands where peaks are expected.
- **Peak vs. shape:** a *peak* is a detection in a single spatial bin; a *shape*
  is a peak that persists when linked across neighboring bins (Union-Find) and
  is characterized by `rocking_fwhm` / `strain_breadth` / `chi_deg`. A shape is
  the physically real feature; an isolated peak may be noise.
- See `TERMINOLOGY.md` for the full glossary — prefer its vocabulary in code,
  comments, and UI labels so everything stays consistent.

## Computing conventions

- **Detector frames are big; be memory-aware.** Stream/sum frames, crop to the
  ROI / 2-theta bands rather than holding full stacks. Watch dtype (raw is often
  uint16/int32 — don't silently upcast everything to float64).
- **Coordinates:** pixel ↔ 2-theta uses the `tth` map (`tth.tiff`); spatial
  position comes from the scan position CSV / grid mapping. Keep
  (row, col) vs (x, y) straight and document which a function expects — most
  bugs here are axis-order or off-by-one.
- **Reproducibility:** seed any subsampling; record what was run. MLflow tracking
  is wired (`mlruns/`) — log runs there rather than inventing ad-hoc logs.
- **Validate against physics, not just code:** detected peaks should fall in the
  expected 2-theta bands; a "great" score with peaks off-band is a bug. Sanity
  -check counts and positions before trusting a metric.
- **CVEvolve** (automated detector optimization) uses **mean F2**
  (recall-weighted, β=2) as the primary metric, *not* F1 — ground truth
  undercounts peaks, so recall is what we chase. Always confirm a final score at
  full settings, not a dev subset.

## Stack & environment

- Python: numpy, scipy, h5py + hdf5plugin (compressed HDF5), tifffile, pandas,
  matplotlib, pyqtgraph/PyQt5; pyFAI only for `.poni → tth` (optional).
- **WSL2 on Windows, files under OneDrive.** The working path contains spaces
  (`OneDrive - Argonne National Laboratory/…`) — **always quote paths in Bash**.
  Filesystem is slow and partly case-insensitive; avoid churn-y mass file ops.
- Use the project `.venv` / `python3`. GUI work needs an X display.
- Don't commit large data (raw scans, `.tiff`, `.h5`, `mlruns/`); check
  `.gitignore` before adding files.

## Data / project locations

Processed data lives in several **project trees** (each a `config.yaml` +
`Raw/ Binned/ Metadata/ Labels/ Study/`). Analysis scripts and `xrd-app` point at
one via its project root. The three we actively pull data from:

1. `TakaTest/TakaProject` (in-repo) — sandbox project; only **Scan_0203**
   processed, but with many detector/bin/variant catalog combos.
2. `/mnt/z/isn/2026-1/2026-1-Luo/ANLCVEVOLVENANOXRD` — a full working copy on the
   `/mnt/z` share. Holds **two multi-scan projects**: `179-201/` (Scan_0179…0201)
   and `215-226/` (Scan_0215…0226). Point at the sub-dir, not the top level.
3. `/home/takaji/rocking_203_214` — the rocking-study project, scans **203–214**
   (Scan_0203, 0204, 0205, 0207, …), with `Study/` + `Study_1x1/` outputs.

**Local WSL copies of the priority scans (as of 2026-07-10):** the six focus
scans now live on the fast local disk under `/home/takaji/`, each tree mirroring
the `/mnt/z` layout so `xrd-app` can point straight at it — prefer these over the
share for these scans:

- `/home/takaji/179-201/` — Scan_0179, Scan_0182 (Binned: 1x1 + 3x3 + territory)
- `/home/takaji/215-226/` — Scan_0215, Scan_0218 (Binned: 1x1 + 3x3)
- `/home/takaji/rocking_203_214/` — Scan_0203, Scan_0207 (+ full rocking study)

These are copies (verified byte-for-byte); originals remain on `/mnt/z`.

Notes: `/mnt/z` is the slow WAN-mounted beamline share (~23 MB/s) — copy/cache
locally for heavy reads rather than churning over it. Each tree is keyed by the
`Scan_NNNN` dir name; use the app's own discovery (`workspace` /
`DataManager.discover_scans()` / `core.catalogs`) rather than hardcoding scan
paths. See also the memory notes on beamline data layout and the rocking study.

## Working style here

- The user is a beamline scientist comfortable with notebooks and CLI batch
  runs. Lead with the result and the physics check; keep code idiomatic to the
  surrounding module.
- For non-trivial changes inside `xrd_app/`, follow `xrd_app/CLAUDE.md`
  (CLI-is-the-engine architecture).
- Don't reorganize the repo or touch unrelated scaffolding (`cvevolve_*/`,
  `src/`, `analysis/`, loose scripts) unless asked.
