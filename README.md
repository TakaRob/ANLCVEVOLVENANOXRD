# XRD-Tools

A Python package with a CLI and interactive GUIs for nano-XRD data analysis,
spatial Bragg-peak feature detection, and CVEvolve algorithm integration.

It turns the original collection of hard-coded analysis scripts into a
project-centric workflow: you create a project, point it at your raw data once,
and every command and GUI resolves paths from that project's `config.yaml`.

---

## Installation

From the repository root (the folder containing `pyproject.toml`):

```bash
python -m pip install -e .
```

This installs the `xrd-tools` command. Dependencies: numpy, scipy, matplotlib,
h5py, hdf5plugin, tifffile, pandas, Pillow, PyQt5, click, pyyaml.
(Notebook extras: `pip install -e ".[notebook]"`.)

Verify:

```bash
xrd-tools --help
```

---

## Quickstart — a new scan on the shared machine (203, 204, 205, …)

The package **ships with the detector `tth.tiff` map and `reflections.py`**
(the per-experiment physics inputs), so on the machine where the scans live you
only need to point at each scan's **raw frames** and **position CSV**:

```bash
xrd-tools init   --project-name Scan_0205 --scan-number 205
xrd-tools link   --raw       /data/scans/Scan_0205 \
                 --positions /data/scans/Scan_0205_position.csv \
                 --detector  /path/to/your_algorithm.py
xrd-tools grid   --bin-size 3      # spatial grid from raw + positions
xrd-tools bin    --bin-size 3      # build binned detector images
xrd-tools process --bin-size 3     # run the detector -> feature catalog
xrd-tools view                     # inspect
```

`tth` and `reflections` are resolved automatically from the bundled defaults.
Override them per-scan only if the detector geometry or material differs:
`xrd-tools link --tth /path/tth.tiff --reflections /path/reflections.py`.
A project-local `data/holdout/tth.tiff` or `reflections.py`, if present, always
wins over the bundled default.

No detector yet? Evolve one with CVEvolve (see below), then
`xrd-tools link --detector <evolved>.py` and re-run `process`.

### All scans at once

One project handles many scans — bins, grid, and results are namespaced per
scan (`data/bins/<scan>/`, `data/holdout/<scan>/`, `results/<scan>/`), while
`tth`/`reflections` stay shared.

```bash
# Point at the shared roots once (beamline layout):
xrd-tools link --raw-root      /net/.../raw_scans \            # holds Scan_0203/, Scan_0204/, …
               --position-root /net/.../Processed/SOCKETSERVER \ # scan_NNNN_position.csv
               --detector      /path/to/detector.py

# Run grid -> bin -> process for a set of scans (or --all to discover them):
xrd-tools batch --scans 203,204,205 --bin-size 3
xrd-tools batch --all --bin-size 3 --skip-existing
```

Any single step also takes `--scan`, e.g. `xrd-tools process --scan 204`,
`xrd-tools view --scan 204`.

### Comparing across scans

Once scans are processed, combine them into one dataset for analysis:

```bash
xrd-tools aggregate            # -> results/summary/features.csv, device_map.csv, analysis.db
```

- `features.csv` — one row per feature: intensity, prevalence (`n_bins`),
  shape (`rocking_fwhm`, `strain_breadth`), orientation (`chi_deg`), per reflection/scan.
- `device_map.csv` — long/tidy per-bin intensities (scan, reflection, bin).
- `analysis.db` — SQLite with both tables, e.g.:
  ```sql
  SELECT reflection, COUNT(*), AVG(peak_intensity), AVG(n_bins), AVG(rocking_fwhm)
  FROM features GROUP BY reflection;
  ```

### No position CSV?

`grid` uses the scan's position CSV when present (auto-found under
`--position-root` as `scan_NNNN_position.csv`). If a scan has none, synthesize a
regular serpentine grid from the raster shape:

```bash
xrd-tools grid --scan 207 --bin-size 3 --shape 52x74   # ROWSxCOLS (or just COLS)
xrd-tools batch --scans 207 --bin-size 3 --shape 74    # same, in batch
```

This assumes a uniform step raster; for irregular/fly scans, provide the
position CSV instead.

---

## Concepts

A **project** is a directory with a `config.yaml` and a standard tree:

```
<project>/
├── config.yaml          # project settings + resolved data paths
├── data/
│   ├── raw_scans/       # raw per-frame H5 scans (or links to them)
│   ├── bins/            # pre-binned xrd_NxN_bins.h5 files
│   └── holdout/         # tth.tiff, grid_mapping.json, reflections.py, positions
├── hutch/               # detector / evolved-algorithm .py scripts
├── runs/                # CVEvolve session outputs
├── mlruns/              # MLflow tracking
└── results/<scan>/      # feature catalogs + CSV tables
```

**Path resolution.** Every command resolves each input in this order:

1. An explicit `--*-path` / `--*` flag you pass on the command line.
2. The matching entry under `data_sources:` in `config.yaml` (set by `link`).
3. A conventional default location inside the project tree.
4. For `tth.tiff` and `reflections.py` only: the **defaults bundled with the
   package** (`xrd_tools/assets/`), used when nothing else is found.

Run `xrd-tools status` at any time to see what each path resolves to and whether
it exists (✓ / ✗).

---

## End-to-end workflow

```bash
# 1. Create a project (optionally name the scan)
xrd-tools init --project-name MyExperiment --scan-number 203 --root ./my_project
cd my_project

# 2. Point the project at your data (symlinks where possible, else records paths)
xrd-tools link \
  --raw         /path/to/raw_scans/Scan_0203 \
  --positions   "/path/to/Scan_0203_position.csv" \
  --detector    /path/to/detector.py \
  --tth         /path/to/tth.tiff \        # optional: bundled default used otherwise
  --reflections /path/to/reflections.py    # optional: bundled default used otherwise
#   (use --copy to physically duplicate instead of symlink)

# 3. Build the spatial grid mapping for a bin size
xrd-tools grid --bin-size 3            # writes + links data/holdout/grid_mapping_3x3.json

# 4. Pre-build the binned detector images
xrd-tools bin  --bin-size 3            # writes + links data/bins/xrd_3x3_bins.h5

# 5. Run the feature-detection pipeline
xrd-tools process --bin-size 3 --snr 4.0
#   -> results/<scan>/feature_catalog_3x3.json
#      results/<scan>/kept_peaks_3x3.csv
#      results/<scan>/filtered_peaks_3x3.csv

# 6. Explore / label interactively
xrd-tools view          # feature viewer
xrd-tools label         # labeling tool (annotate ground truth)
xrd-tools device-map    # device-level reflection map
xrd-tools orientation   # orientation / rocking-curve map

# 7. Evolve a better detector with CVEvolve (runs in a Podman container)
xrd-tools run-cvevolve --config /path/to/cvevolve_3x3/config.yaml \
                       --prompt /path/to/cvevolve_3x3/prompt.md
```

See **Running CVEvolve in a container** below for build/isolation details.

Each GUI also runs standalone, e.g. `python -m xrd_tools.gui.viewer --project-root . --bin-size 3`.

---

## Commands

| Command | Purpose |
|---|---|
| `init` | Create the project tree and `config.yaml`. |
| `link` | Record data paths: per-scan (`--raw/--positions/--tth/--grid/--reflections/--detector`) or shared roots (`--raw-root/--position-root`); symlinks into `data/`, or `--copy`. |
| `status` | Print the config and every resolved path with ✓/✗ (`--scan` to target a scan). |
| `grid` | Build `grid_mapping_NxN.json` from raw frames + positions, or `--shape RxC` to synthesize. |
| `bin` | Sum each bin's frames into `xrd_NxN_bins.h5` (per scan). |
| `process` | Detect → link → filter → write the feature catalog + CSVs. |
| `batch` | Run `grid → bin → process` over many scans (`--scans` or `--all`). |
| `aggregate` | Combine all scans' catalogs into `results/summary/` — `features.csv`, `device_map.csv`, `analysis.db` (SQLite). |
| `label` / `view` / `device-map` / `orientation` | Launch the interactive GUIs. |
| `run-cvevolve` | Run CVEvolve, by default inside a Podman container. |

All commands take `--root` (project directory, default `.`); the per-scan
commands also take `--scan <number|name>` (defaults to the project's configured
scan). Run any command with `--help` for its full options, including explicit
path overrides.

---

## Running CVEvolve in a container

CVEvolve's agent **executes LLM-generated code**, so it should run inside a
container — that's the real isolation boundary (per CVEvolve's own docs).
`xrd-tools run-cvevolve` defaults to `--engine podman`.

```bash
# One-time: build the image from the CVEvolve checkout
xrd-tools run-cvevolve --build --cvevolve-dir /path/to/CVEvolve \
                       --config  /path/to/cvevolve_3x3/config.yaml \
                       --prompt  /path/to/cvevolve_3x3/prompt.md

# Subsequent runs (image already built)
xrd-tools run-cvevolve --config /path/to/cvevolve_3x3/config.yaml \
                       --prompt /path/to/cvevolve_3x3/prompt.md
```

How it works:
- The container mounts your host directory **at the same absolute path**
  (`-v <dir>:<dir>`), so the absolute paths inside `config.yaml`
  (`workspace.root_dir`, `data_dir`, `holdout_data_dir`) resolve unchanged.
  `--mount` defaults to the project root; pass it (repeatable) to add more.
- The Argo API key is passed through with `-e ARGO_API_KEY` (override the set
  with `--env NAME`, repeatable). Make sure `ARGO_API_KEY` is exported.
- `--engine docker` uses Docker instead; `--engine local` runs CVEvolve directly
  in a Python environment (use `--cvevolve-dir` to pick its venv) with **no
  isolation** — only for trusted/offline debugging.

Notes:
- `cvevolve run` requires `--prompt` (the task description markdown).
- The image tag defaults to `cvevolve` (`--image` to change).

## Package layout

```
src/xrd_tools/
├── cli.py            # Click CLI (entry point: xrd-tools)
├── config.py         # ProjectConfig + DataManager (path resolution)
├── core/
│   ├── io.py         # loaders + grid generation + binning
│   ├── processing.py # spatial feature analysis pipeline
│   └── algorithms.py # noise-reduction / background models
├── gui/
│   ├── labeling.py     # label
│   ├── viewer.py       # view
│   ├── device_map.py   # device-map
│   └── orientation.py  # orientation
└── assets/             # bundled shared defaults
    ├── tth.tiff        #   2θ-per-pixel detector map
    └── reflections.py  #   expected Bragg peak 2θ + labels
```

## Features
- Project-centric config: no hard-coded paths; one `link`, then everything resolves.
- Per-scan and per-bin-size workflow (1×1 / 3×3 / 4×4 / 5×5, …).
- Noise-reduction models: Gaussian, Split Gaussian, Skewed Gaussian, Fourier low-pass.
- Spatial peak linking (Union-Find) + Gaussian-profile filtering with rocking-curve / strain metrics.
- Interactive PyQt5 GUIs that load evolved detectors and stay in sync with the project config.
- CVEvolve workflow support for evolving the detector algorithm.
