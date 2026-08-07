# CLAUDE.md — nano-XRD scientific computing

General working notes for nano-XRD data analysis of perovskite and halide thin
films at the ISN beamline, APS. The active application lives in `xrd_app/`; read
`xrd_app/CLAUDE.md` before editing it. For binning and storage behavior, use
`xrd_app/docs/BINNING_STORAGE.md`.

## Analysis through chat

- Do not launch the GUI to answer a data question. Use `xrd-app` commands and
  inspect persisted project results headlessly.
- Cheap inspection includes `status`, `lineage`, `detectors`, `list-studies`,
  `Raw/scans.json`, and existing result metadata. Run these freely.
- Commands that read detector frames or build products can take minutes to hours,
  especially over network storage. Confirm before running `archive-unbinned`,
  `bin`, `make-bins`, `peaks`, `run-combined`, `territory-build`,
  `reflection-sum`, `hd-device-map`, `qspace`, `rsm`, `xrf`, `batch`,
  `run-study`, or `report` when it must calculate missing data.
- Lead with the scientific result and physics check. State the project, scan,
  bin size, variant, algorithm/catalog, and relevant command parameters.
- Use circular statistics for `chi_deg` near the -180/180-degree boundary.
- For complete shape characterization, read the shape HDF5 catalog through
  `xrd_app.core.result_store`; the kept/filtered CSV files are limited summaries.

## What the science is

- **Goal:** map crystal grain orientation / strain across a sample surface by
  detecting Bragg peaks in nano-XRD detector images, scan position by position.
- **Instrument:** 15 keV beam, 75 µm pixel. A raw
  scan is thousands of CCD frames (one per spatial position) in HDF5.
- **Reflections:** use the resolved `reflections.json` as authoritative. Its
  configured 2-theta bands define where peaks are expected.
- A **peak** is a detector detection in one spatial bin. A **shape** or
  **feature** is a linked, spatially verified cluster of peaks across neighboring
  bins. A **track** links shapes across scans at different sample-theta values.
- `chi_deg` is detector azimuth. `chi_fwhm` is within-shape azimuthal breadth and
  `tth_fwhm` is within-shape radial 2-theta breadth. Neither is a rocking-curve
  width or calibrated strain. True rocking FWHM comes from a track's intensity
  versus sample theta; `rocking_curves.csv:microstrain` is the current
  calibration/reference-dependent cross-scan strain estimate.

## Computing conventions

- **Detector frames are big; be memory-aware.** Stream/sum frames, crop to the
  ROI / 2-theta bands rather than holding full stacks. Watch dtype (raw is often
  uint16/int32 — don't silently upcast everything to float64).
- **Coordinates:** pixel ↔ 2-theta uses the `tth` map (`tth.tiff`); spatial
  position comes from the scan position CSV / grid mapping. Keep
  (row, col) vs (x, y) straight and document which a function expects — most
  bugs here are axis-order or off-by-one.
- **Reproducibility:** seed subsampling and preserve catalog lineage, algorithm
  identity, scan, bin size, variant, and command parameters. MLflow is not an app
  dependency; `mlruns/` is only ignored if external tooling creates it.
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

## Projects and data

- A project is rooted by `config.yaml`. `xrd-app init` creates `Raw/`, `Binned/`,
  `Metadata/`, `Labels/`, `Figures/`, `CVEvolve/`, and `Algorithms/`. Study
  commands create `Study/` or another named study directory later; `xrf-app` may
  create an optional `XRF/` add-on.
- Use `workspace`, `ProjectConfig`, `DataManager`, and `core.catalogs` discovery
  instead of hardcoding project, scan, or result paths.
- Network-mounted detector data is slow. Prefer a verified local cache for heavy
  reads when available, but never assume a machine-specific path exists.

## Working style here

- The user is a beamline scientist comfortable with notebooks and CLI batch
  runs. Lead with the result and the physics check; keep code idiomatic to the
  surrounding module.
- For non-trivial changes inside `xrd_app/`, follow `xrd_app/CLAUDE.md`
  (CLI-is-the-engine architecture).
- Do not revive or modify archived material under `Ignore/` unless explicitly
  requested. The active code surface is `xrd_app/`, the tracked `notebooks/`
  tutorials, and root packaging, instructions, and workflow files.
