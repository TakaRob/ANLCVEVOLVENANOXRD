# Getting Started -- Nano-XRD Analysis Workspace

> **Project:** 2026-1 Luo | ISN 26-ID-C | APS, Argonne National Laboratory

This guide covers two workflows:
1. **WSL development** -- running analysis notebooks and CVEvolve locally
2. **Docker standalone** -- building a reproducible container for CVEvolve agent runs

---

## Directory Layout

After reorganization, the workspace is structured as:

```
2026-1_Luo/
  analysis/                    # Notebooks & XRD processing library
    nano_xrd_analysis.ipynb    # <-- Main analysis notebook (start here)
    xrd_proc.py                # Core hotspot detection library
    data_view*.ipynb           # Legacy exploration notebooks
    cdte_plotting.ipynb        # XRF elemental mapping
    data_proc_per_pixel.ipynb  # Per-pixel detector stacking

  preprocessing/               # Raw data preprocessing scripts
    data_preprocess_isn_26c1.py
    data_preprocess_isn_26c1_updated.py   # CLI version (recommended)
    data_preprocess_MP_isn_26c1.py        # Multiprocessing version
    process_dp_velo_21c1.py               # VelociProbe legacy
    append_detectors_links.py             # HDF5 external linking
    ptychi_recon.py                        # Ptychographic reconstruction
    ptychi_recon_jd.py                     # Ptycho recon (2xfm variant)

  raw_scans/                   # Raw detector data
    Scan_0179/                 # ME7, XRD, TETRAMM1, SOCKETSERVER
    Scan_0180/
    scan_203_sum.tiff          # Pre-summed CCD image
    tth.tiff                   # Per-pixel 2-theta map

  results/                     # Preprocessed data & analysis outputs
    scan002/, scan004/, ...    # Preprocessed diffraction stacks
    scan203/                   # Scan 203 ROI detection results
      per_pixel_rois/          # Per-position ROI intensity CSVs

  cvevolve_hotspot/            # CVEvolve agent workspace
    config.yaml                # Agent configuration
    prompt.md                  # Task prompt for the agent
    test_data/                 # XRD image, 2-theta, annotations, baseline
    validation_data/           # Holdout validation set

  CVEvolve/                    # CVEvolve framework source code
    src/cvevolve/              # Python package
    Dockerfile                 # Container build file
    pyproject.toml             # Dependencies & build config

  docs/                        # Documentation & reference PDFs
    PROJECT_OVERVIEW.md
    PREPROCESSING_GUIDE.md
    CVEvolve.pdf
    Vibe Coding with Argo.pdf

  python_scripts/              # Original scripts (kept for reference)
  hotspot_detection_split_configs/  # Original hotspot configs (kept)

  NANO_XRD_FEATURES.md        # Feature reference for all scripts
  GETTING_STARTED.md           # This file
  tth.tiff                    # Root copy for notebook compatibility
```

---

## Option 1: WSL Development

### Prerequisites

1. **WSL2 with Ubuntu** (already set up based on your environment)
2. **Python 3.11+** with pip
3. **uv** (recommended for CVEvolve)

### Step 1: Install Python dependencies

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment for analysis
cd "/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo"
python3 -m venv .venv
source .venv/bin/activate

# Core analysis dependencies
pip install numpy matplotlib scipy h5py tifffile jupyter ipykernel pyyaml pandas

# HDF5 compression plugin (needed for some raw data files)
pip install hdf5plugin

# Register the kernel for Jupyter
python -m ipykernel install --user --name nxrd --display-name "Nano-XRD"
```

### Step 2: Launch Jupyter

```bash
cd "/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo"
source .venv/bin/activate
jupyter notebook --notebook-dir=analysis/
```

Open `nano_xrd_analysis.ipynb` -- this is the main unified notebook.

### Step 3: Preprocess raw scans (if needed)

```bash
source .venv/bin/activate

# Preprocess scan 179 (CLI version with flexible parameters)
python preprocessing/data_preprocess_isn_26c1_updated.py 179 \
  --det-npixel 256 \
  --crop-center 396 816

# Or use the multiprocessing version for batch processing
# (edit scan list in the script first)
python preprocessing/data_preprocess_MP_isn_26c1.py
```

### Step 4: Set up CVEvolve

```bash
cd CVEvolve
uv sync

# Set the Argo API key (required for agent runs)
export OPENAI_API_KEY="your-argo-api-key-here"

# The config already points to the Argo API endpoint:
#   api_base: https://apps-test.inside.anl.gov/argoapi/v1
#   model_name: claudeopus46
```

### Step 5: Run CVEvolve agent

```bash
cd "/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo"

# Edit cvevolve_hotspot/config.yaml to set correct paths for your environment:
#   workspace.root_dir: ./cvevolve_hotspot/sessions
#   workspace.data_dir: ./cvevolve_hotspot/test_data
#   workspace.holdout_data_dir: ./cvevolve_hotspot/validation_data

# Run the agent
cd CVEvolve
uv run cvevolve run \
  --config ../cvevolve_hotspot/config.yaml \
  --prompt ../cvevolve_hotspot/prompt.md

# Resume an interrupted session
uv run cvevolve resume --session ../cvevolve_hotspot/sessions/<session-name>
```

### Argo API Notes

The project uses Argonne's Argo API gateway as an OpenAI-compatible proxy for Claude models:

| Setting | Value |
|---------|-------|
| API endpoint | `https://apps-test.inside.anl.gov/argoapi/v1` |
| Model name | `claudeopus46` (Claude Opus via Argo) |
| API key env var | `OPENAI_API_KEY` |
| Auth | Use your Argo API key (get from Argo portal) |

The `model.api_base` in `config.yaml` tells CVEvolve's LangChain OpenAI client to route requests through Argo instead of directly to OpenAI. The `model_name` maps to the Argo-side model alias.

---

## Option 2: Docker Standalone

Use this when you want a reproducible, isolated environment for CVEvolve agent runs.

### Step 1: Build the image

```bash
cd "/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo/CVEvolve"
docker build -t cvevolve .
```

This builds from `Dockerfile`:
- Base: `python:3.11-slim`
- Installs: `uv`, `build-essential`, `git`, `curl`
- Syncs CVEvolve dependencies via `uv sync --frozen --no-dev`
- Entry: `cvevolve --help`

### Step 2: Prepare the workspace

```bash
# Create a workspace directory that will be mounted into the container
mkdir -p /tmp/cvevolve-workspace

# Copy the hotspot detection data and config
cp -r cvevolve_hotspot/test_data /tmp/cvevolve-workspace/
cp -r cvevolve_hotspot/validation_data /tmp/cvevolve-workspace/
cp cvevolve_hotspot/prompt.md /tmp/cvevolve-workspace/
```

Create a Docker-compatible config (paths inside the container):

```bash
cat > /tmp/cvevolve-workspace/config.yaml << 'EOF'
name: hotspot_detection_docker
num_workers_generate: 3
num_workers_tune: 3
model:
  model_name: claudeopus46
  api_key_env_var: OPENAI_API_KEY
  api_base: https://apps-test.inside.anl.gov/argoapi/v1
  max_retries: 15
  rate_limit_resend_attempts: 30
  rate_limit_sleep_seconds: 60
workspace:
  root_dir: /workspace/sessions
  data_dir: /workspace/test_data
  holdout_data_dir: /workspace/validation_data
  require_dangerous_command_approval: true
metric:
  name_hint: f1 score
  direction_hint: maximize
  target_value: null
  description_hint: >
    If a detected point is within 40 pixels to a ground truth point,
    consider that a match. Use this to calculate the f1 score.
branching:
  warmup_rounds: 3
  tune_every: 3
  evolve_every: 2
  lineage_selection_temperature: 1
stopping:
  max_rounds: 20
  patience_rounds: 10
  min_improvement: 0.0
tracking:
  enabled: false
EOF
```

### Step 3: Run the container

```bash
docker run --rm -it \
  -e OPENAI_API_KEY="your-argo-api-key-here" \
  -v /tmp/cvevolve-workspace:/workspace \
  -w /workspace \
  cvevolve \
  bash
```

Inside the container:

```bash
# Verify setup
cvevolve --help

# Run the agent
cvevolve run --config ./config.yaml --prompt ./prompt.md

# Results will be in /workspace/sessions/<name>/
```

### Step 4: Retrieve results

After the container exits, results are in your mounted workspace:

```bash
ls /tmp/cvevolve-workspace/sessions/*/reports/
# final_report.md, final_summary.json, best_candidate.py

ls /tmp/cvevolve-workspace/sessions/*/exports/
# candidates.csv, metrics.csv, rounds.csv, etc.
```

---

## Quick Reference: Common Tasks

| Task | Command / File |
|------|---------------|
| **View single scan** | `analysis/nano_xrd_analysis.ipynb` Section 1 |
| **Sum CCD frames** | `analysis/nano_xrd_analysis.ipynb` Section 2 |
| **Detect hotspots** | `analysis/nano_xrd_analysis.ipynb` Section 4 |
| **Per-pixel maps** | `analysis/nano_xrd_analysis.ipynb` Section 5 |
| **Run CVEvolve** | `cd CVEvolve && uv run cvevolve run --config ... --prompt ...` |
| **Preprocess new scan** | `python preprocessing/data_preprocess_isn_26c1_updated.py <scan_number>` |
| **Evaluate vs ground truth** | `analysis/nano_xrd_analysis.ipynb` Section 6 |
| **XRF element maps** | `analysis/cdte_plotting.ipynb` |

## Key Parameters to Adjust

| Parameter | Where | Typical values |
|-----------|-------|----------------|
| Scan number | Notebook Section 1 | 179, 180, 203, ... |
| Crop center (x, y) | Notebook Section 1 | (396, 816) or (517, 801) |
| `hotspot_percentile` | Section 4 | 94-99 (higher = stricter) |
| `line_tol` | Section 4 | 0.03-0.3 degrees |
| `min_pixels` | Section 4 | 4-10 |
| `pad` | Section 4 | 6-13 pixels |
| Target reflections | Section 4 | (001), (011), (111), (002), (012), (112) |
