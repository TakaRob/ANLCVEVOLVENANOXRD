# %% [markdown]
# # 01 - Build and inspect a single-scan peak/shape pipeline
#
# This notebook explains the production sequence and prints every command before
# running it. Detector-frame work is blocked until `RUN_HEAVY = True`.
#
# The pipeline is:
#
# `measured positions + raw frames -> archive -> grid -> bins -> peaks -> shapes`
#
# Production grids require measured real `(X,Y)` positions. The app does not
# silently reconstruct the known skew-prone file-row lattice.

# %% Configuration
import os
from pathlib import Path

try:
    from notebooks._cli import project_ready, run_command
except ModuleNotFoundError:  # Plain `python notebooks/01_single_scan_pipeline.py`.
    from _cli import project_ready, run_command

PROJECT_ROOT = Path(os.environ.get("XRD_PROJECT", "CHANGE_ME"))
SCAN = os.environ.get("XRD_SCAN", "203")
BIN_SIZE = int(os.environ.get("XRD_BIN_SIZE", "3"))
DETECTOR = os.environ.get("XRD_DETECTOR", "5x5_tophat_band_adaptive_snr")
SNR = float(os.environ.get("XRD_SNR", "4"))
RUN_HEAVY = False

# %% Check inputs first (cheap)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "status", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--bin-size", BIN_SIZE,
    )

# %% [markdown]
# ## Build normalized bins
#
# `make-bins` archives frames losslessly, creates a real-position grid, writes
# the requested materialized bins, and attempts a detector-sum refresh. Frame
# normalization stores a mean per contributing frame, avoiding brightness changes
# caused only by unequal cell occupancy.

# %% Archive, grid, and bin (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "make-bins", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--bin-size", BIN_SIZE, "--normalize-frames",
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% [markdown]
# ## Detect peaks and link shapes
#
# `peaks` searches only configured reflection bands unless the project uses the
# explicit whole-frame reflection mode. `shapes` links detections in sample space
# and rejects clusters without a plausible spatial intensity profile.

# %% Peak and shape pipeline (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "run-pipeline", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--bin-size", BIN_SIZE,
        "--algorithm", DETECTOR, "--snr", SNR,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% Inspect what was produced (cheap)
if project_ready(PROJECT_ROOT):
    run_command("xrd_app.cli", "lineage", "--root", PROJECT_ROOT, "--scan", SCAN)
    run_command(
        "xrd_app.cli", "scan-table", "--root", PROJECT_ROOT,
        "--scans", SCAN, "--bin-size", BIN_SIZE,
    )

# %% [markdown]
# ## Physics checks before interpretation
#
# 1. Peak 2-theta positions should lie in the resolved `reflections.json` bands.
# 2. Compare peak count, kept shape count, and rejected cluster count; a detector
#    with many isolated detections may be fitting noise.
# 3. Treat `chi_deg` as circular data near -180/+180 degrees.
# 4. `chi_fwhm` is within-shape azimuthal breadth and `tth_fwhm` is within-shape
#    radial breadth. Neither is a rocking-curve width or calibrated strain.
# 5. Use the shape HDF5 catalog for complete characterization. The kept/filtered
#    CSV files are intentionally limited summaries.
