# %% [markdown]
# # 00 - Open a project and inspect existing results
#
# This is the safest place to start. The cells run metadata-only CLI commands;
# they do not read detector frames or create analysis products.
#
# A project is a directory containing `config.yaml`, plus `Raw/`, `Binned/`,
# `Metadata/`, `Labels/`, and related result directories. Set `XRD_PROJECT` in
# your environment or edit `PROJECT_ROOT` below.

# %% Configuration
import os
from pathlib import Path

try:
    from notebooks._cli import project_ready, run_command
except ModuleNotFoundError:  # Plain `python notebooks/00_project_and_results.py`.
    from _cli import project_ready, run_command

PROJECT_ROOT = Path(os.environ.get("XRD_PROJECT", "CHANGE_ME"))
SCAN = os.environ.get("XRD_SCAN", "203")
BIN_SIZE = int(os.environ.get("XRD_BIN_SIZE", "3"))

print(f"Project:  {PROJECT_ROOT}")
print(f"Scan:     {SCAN}")
print(f"Bin size: {BIN_SIZE}x{BIN_SIZE}")

# %% [markdown]
# ## Creating a project
#
# Project creation is intentionally not automatic here. Run these commands in a
# terminal, replacing the example paths:
#
# ```bash
# xrd-app init --name MyProject --root /path/to/project
# xrd-app scan-detect --root /path/to/project --scans-dir /path/to/scans
# xrd-app link --root /path/to/project --tth /path/to/tth.tiff \
#   --reflections /path/to/reflections.json --position-root /path/to/positions
# ```
#
# `scan-detect --deep` reads every source file to obtain exact frame counts. The
# default fast scan may report estimates.

# %% Project status
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "status", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--bin-size", BIN_SIZE,
    )

# %% [markdown]
# `status` confirms project and path resolution. For exact registered file and
# frame counts, inspect `Raw/scans.json`; `status` does not print those counts.

# %% Available algorithms
if project_ready(PROJECT_ROOT):
    run_command("xrd_app.cli", "detectors", "--root", PROJECT_ROOT, "--kind", "peak")
    run_command("xrd_app.cli", "detectors", "--root", PROJECT_ROOT, "--kind", "shape")

# %% Result lineage
if project_ready(PROJECT_ROOT):
    run_command("xrd_app.cli", "lineage", "--root", PROJECT_ROOT, "--scan", SCAN)

# %% Registered cross-scan studies
if project_ready(PROJECT_ROOT):
    run_command("xrd_app.cli", "list-studies", "--root", PROJECT_ROOT)

# %% [markdown]
# ## Reading the output
#
# - A **peak** is one detector detection in one spatial bin.
# - A **shape** or **feature** links and validates neighboring peaks in sample
#   space.
# - A **track** links shapes across scans at different sample-theta values.
# - `chi_fwhm` and `tth_fwhm` are within-shape detector breadths, not rocking
#   widths. True rocking FWHM is fitted from track intensity versus sample theta.
#
# Multiple catalogs can exist for one scan. Record the scan, bin size, variant,
# peak detector, shape algorithm, and lineage whenever reporting a result.
