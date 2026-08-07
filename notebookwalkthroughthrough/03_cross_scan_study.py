# %% [markdown]
# # 03 - Cross-scan tracks, rocking curves, and study summaries
# This Stage is still untested and is hidden in the GUI
# A single-scan shape describes a spatially coherent diffraction feature. A
# **track** links compatible shapes across scans at different sample-theta values.
# The `rocking` stage fits track intensity versus theta; this is where true rocking
# FWHM is measured.

# %% Configuration
import os
from pathlib import Path

try:
    from notebooks._cli import project_ready, run_command
except ModuleNotFoundError:  # Plain `python notebooks/03_cross_scan_study.py`.
    from _cli import project_ready, run_command

PROJECT_ROOT = Path(os.environ.get("XRD_PROJECT", "CHANGE_ME"))
SCANS = os.environ.get("XRD_SCANS", "203,204,205")
BIN_SIZE = int(os.environ.get("XRD_BIN_SIZE", "3"))
STUDY_DIR = os.environ.get("XRD_STUDY", "Study")
RUN_HEAVY = False

# %% Discover existing studies first (cheap)
if project_ready(PROJECT_ROOT):
    run_command("xrd_app.cli", "list-studies", "--root", PROJECT_ROOT)

# %% [markdown]
# ## Build a complete study
#
# `run-study` orchestrates aggregate -> track -> rocking -> predict ->
# combined-device and registers the result for the GUI. It reads many shape
# catalogs and can calculate substantial products, so it is opt-in here.

# %% Build/register the study (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "run-study", "--root", PROJECT_ROOT,
        "--scans", SCANS, "--bin-size", BIN_SIZE, "--out", STUDY_DIR,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% Print per-scan summary from existing catalogs (usually cheap)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "scan-table", "--root", PROJECT_ROOT,
        "--scans", SCANS, "--bin-size", BIN_SIZE,
    )

# %% [markdown]
# ## Reading rocking results
#
# In `rocking_curves.csv`:
#
# - `fwhm` is the fitted rocking width in sample-theta degrees.
# - Check fit `status` and `r_squared`; sparse, monotonic, or poor fits are not
#   trustworthy Gaussian widths.
# - `microstrain` is calculated from track-level 2-theta center relative to the
#   reflection reference. Its accuracy depends on detector calibration and the
#   reference value.
# - `chi_fwhm` and `tth_fwhm` from shape catalogs remain within-shape detector
#   breadths and must not be substituted for rocking FWHM.
#
# Track matching handles chi as circular data. When making custom summaries, do
# not use an arithmetic mean across values that straddle -180/+180 degrees.
