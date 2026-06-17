# xrd-tools — Structure & Command Reference

A concise reference for the directories/files `xrd-tools` creates and what each
command does. For the narrative guide see `README.md`.

---

## Directory structure created by `init`

`xrd-tools init` creates this tree plus `config.yaml`:

```
<project>/
├── config.yaml          # project settings + resolved data paths
├── data/
│   ├── raw_scans/       # raw per-frame H5 scans (or symlinks); per-scan: <scan>/XRD/*.h5
│   ├── bins/            # binned HDF5 output (per scan)
│   └── holdout/         # shared inputs + per-scan grid mappings
├── hutch/               # detector / evolved-algorithm .py scripts
├── runs/                # CVEvolve session outputs
├── mlruns/              # MLflow tracking
└── results/             # feature catalogs + CSV tables (per scan)
```

## Files created by the pipeline (per scan)

`<scan>` is the canonical name, e.g. `Scan_0203`. `N` is the bin size.

```
<project>/
├── data/
│   ├── holdout/
│   │   └── <scan>/
│   │       └── grid_mapping_NxN.json     # created by `grid`
│   └── bins/
│       └── <scan>/
│           └── xrd_NxN_bins.h5           # created by `bin`
└── results/
    └── <scan>/
        ├── feature_catalog_NxN.json      # created by `process` (kept features)
        ├── kept_peaks_NxN.csv            # created by `process`
        └── filtered_peaks_NxN.csv        # created by `process`
```

Shared inputs (one per project, not per scan), resolved from `data/holdout/`,
a linked path, or the bundled package defaults:

```
data/holdout/tth.tiff          # 2θ-per-pixel detector map   (bundled default exists)
data/holdout/reflections.py    # expected Bragg peak 2θ + labels (bundled default exists)
```

## Bundled package assets (ship with the install)

```
src/xrd_tools/assets/
├── tth.tiff          # default 2θ map used when a project has none
└── reflections.py    # default reflections used when a project has none
```

## `config.yaml` schema

```yaml
name: <project>
scan:
  number: 203
  name: Scan_0203
paths:
  data_dir: data
  results_dir: results
  runs_dir: runs
  mlruns_dir: mlruns
  hutch_dir: hutch
data_sources:           # absolute paths recorded by `link` (null = use default)
  raw_root: null        # parent dir holding many Scan_NNNN/ dirs (multi-scan)
  position_root: null   # dir holding scan_NNNN_position.csv files (multi-scan)
  raw_scan_dir: null    # single-scan raw dir
  position_csv: null    # single-scan position CSV
  tth_map: null
  grid_mapping: null
  reflections: null
  detector_script: null
bins: {}
tracking:
  enabled: true
  mlflow_tracking_uri: file://<project>/mlruns
```

## Path resolution order

Each input resolves as: **`--flag` → `config.yaml` `data_sources` → project
default**. For `tth.tiff` / `reflections.py` only, a 4th fallback is the
**bundled package asset**. Per-scan paths (`bins`, `grid`, `results`) are
namespaced by `<scan>`, so one project holds many scans without collision.

---

## Command reference

Every command takes `--root <dir>` (project root, default `.`). Per-scan
commands also take `--scan <number|name>` (default: the project's configured
scan). Run any command with `--help` for the full option list.

### `xrd-tools init`
Create the project tree and `config.yaml`.
- `--project-name <name>` — project name (prompted if omitted).
- `--scan-number <int>` — sets `scan.number`/`scan.name` (e.g. 203 → Scan_0203).
- **Creates:** the directory tree above + `config.yaml`.

### `xrd-tools link`
Record external data locations in `config.yaml` (symlinks into `data/` where
possible, else stores absolute paths; `--copy` duplicates instead).
- Per-scan: `--raw <dir>`, `--positions <csv>`, `--tth <tiff>`, `--grid <json>`,
  `--reflections <py>`, `--detector <py>`.
- Multi-scan roots: `--raw-root <dir>` (parent of `Scan_NNNN/`),
  `--position-root <dir>` (holds `scan_NNNN_position.csv`).
- **Writes:** updates `config.yaml` `data_sources`.

### `xrd-tools status`
Print the config and every resolved path with `✓` (exists) / `✗` (missing).
- `--scan <id>`, `--bin-size <N>` — which scan/bin to resolve for.
- **Writes:** nothing.

### `xrd-tools grid`
Assign raw frames to a spatial bin grid → `grid_mapping_NxN.json`.
- Uses the position CSV when present; otherwise `--shape ROWSxCOLS` (or `COLS`)
  synthesizes a regular serpentine grid with no positions.
- `--bin-size <N>`, `--scan <id>`, `--xrd-dir`, `--positions`, `--output`.
- **Creates:** `data/holdout/<scan>/grid_mapping_NxN.json`.

### `xrd-tools bin`
Sum each bin's raw frames into a single binned HDF5.
- `--bin-size <N>`, `--scan <id>`, `--grid-mapping`, `--output`,
  `--compression [gzip|lz4|none]`.
- **Creates:** `data/bins/<scan>/xrd_NxN_bins.h5`.

### `xrd-tools process`
Run the detection pipeline: per-bin detect → spatial link → Gaussian filter →
write catalog.
- `--bin-size <N>`, `--scan <id>`, `--snr <float>`, `--link-tolerance <px>`,
  and explicit overrides `--h5-path/--tth-path/--detector-script/--reflections/--grid-mapping/--output-dir`.
- **Creates:** `results/<scan>/feature_catalog_NxN.json`,
  `kept_peaks_NxN.csv`, `filtered_peaks_NxN.csv`.

### `xrd-tools batch`
Run `grid → bin → process` for many scans, each in its own per-scan dirs.
- `--scans "203,204,205"` or `--all` (discover `Scan_NNNN` under `raw-root`).
- `--bin-size <N>`, `--snr <float>`, `--shape ROWSxCOLS`,
  `--compression`, `--skip-existing`.
- **Creates:** the per-scan grid/bins/results files above for each scan.

### `xrd-tools run-cvevolve`
Run CVEvolve (algorithm evolution), by default inside a Podman container.
- `--config <yaml>` (required), `--prompt <md>`,
  `--engine [podman|docker|local]`, `--build`, `--cvevolve-dir <dir>`,
  `--image <tag>`, `--mount <dir>` (repeatable), `--env <NAME>` (repeatable).
- **Creates:** CVEvolve session outputs (under the paths in its own config).

### `xrd-tools label` / `view` / `device-map` / `orientation`
Launch the interactive GUIs (`--scan`, `--bin-size`). Read-only viewers/editors
over the files above.
