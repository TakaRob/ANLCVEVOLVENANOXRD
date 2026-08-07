# %% [markdown]
# # 05 - XRF material selections and linked XRD
#
# xrd-app provides two related XRF workflows:
#
# 1. `xrd-app xrf` integrates configured element ranges onto an existing XRD grid.
# 2. `xrf-app` manages an optional project `XRF/` add-on with raw ME7 discovery,
#    complete spectra, material cuts, canonical selection HDF5 files, and ME7/XRD
#    frame registration.
#
# Raw XRF processing reads many spectra and is never automatic in this notebook.

# %% Configuration
import os
from pathlib import Path

try:
    from notebooks._cli import project_ready, run_command
except ModuleNotFoundError:  # Plain `python notebooks/05_xrf_and_linked_xrd.py`.
    from _cli import project_ready, run_command

PROJECT_ROOT = Path(os.environ.get("XRD_PROJECT", "CHANGE_ME"))
SCAN = os.environ.get("XRD_SCAN", "24")
ME7_SOURCE = Path(os.environ.get("XRF_SOURCE", "CHANGE_ME"))
DEFINITIONS = Path(os.environ.get("XRF_DEFINITIONS", "CHANGE_ME"))
RUN_SETUP = False
RUN_HEAVY = False

# %% Explain the dedicated XRF command surface (cheap)
run_command("xrd_app.xrf_cli", "--help")

# %% Create the optional XRF add-on (persistent, opt-in)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.xrf_cli", "init", "--name", "XRF analysis",
        "--root", PROJECT_ROOT, execute=RUN_SETUP,
    )

# %% Register raw ME7 data (metadata work, opt-in)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.xrf_cli", "load-data", "--root", PROJECT_ROOT,
        "--source", ME7_SOURCE, "--scan", SCAN, execute=RUN_SETUP,
    )

# %% Build the complete spectrum (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.xrf_cli", "process-raw", "--root", PROJECT_ROOT, "--scan", SCAN,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% Register corresponding ME7/XRD frames and integrate material ranges (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.xrf_cli", "link-dataset", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--definitions", DEFINITIONS,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% Inspect canonical selection status (cheap)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.xrf_cli", "status", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--json-output",
    )

# %% [markdown]
# ## Linked-XRD interpretation
#
# The current `linked-xrd-track` command is a separate compatibility workflow: it
# consumes an explicit XRF-to-XRD link-table HDF5 rather than the canonical
# selection written by `xrf-app`. Do not pass a canonical selection as `--links`.
# Before interpreting a linked-XRD result:
#
# - verify energy calibration, ROI bounds, and material threshold;
# - verify link-table source file/local frame indices resolve inside the original
#   XRD datasets;
# - report the material selection and link-table provenance with XRD results.
#
# For XRF maps directly aligned to an existing grid, inspect `xrd-app xrf --help`.
# For compatibility link-table tracking, inspect `linked-xrd-track --help`.

# %% Related XRD commands (cheap help)
run_command("xrd_app.cli", "xrf", "--help")
run_command("xrd_app.cli", "linked-xrd-track", "--help")
