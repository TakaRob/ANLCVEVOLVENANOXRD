# %% [markdown]
# # 02 - True-coordinate territory maps and detector ROIs
#
# This notebook combines two complementary spatial views:
#
# - A **territory** partitions measured sample coordinates into irregular cells
#   and links shapes through physical neighbors. `target-size=1` is the lossless
#   one-frame-per-territory reference when occupancy checks confirm one frame per
#   cell.
# - A **detector ROI** integrates a fixed detector rectangle at every sample
#   position. It answers a different question from adaptive peak/shape detection.
#
# Every persistent operation stays in the CLI. Heavy cells are opt-in.

# %% Configuration
import os
from pathlib import Path

try:
    from notebooks._cli import project_ready, run_command
except ModuleNotFoundError:  # Plain `python notebooks/02_territory_and_roi_maps.py`.
    from _cli import project_ready, run_command

PROJECT_ROOT = Path(os.environ.get("XRD_PROJECT", "CHANGE_ME"))
SCAN = os.environ.get("XRD_SCAN", "203")
DETECTOR = os.environ.get("XRD_DETECTOR", "5x5_tophat_band_adaptive_snr")
SNR = float(os.environ.get("XRD_SNR", "4"))
RUN_HEAVY = False

# %% Build the territorial reference (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "territory-build", "--root", PROJECT_ROOT,
        "--scan", SCAN, "--target-size", 1,
        "--algorithm", DETECTOR, "--snr", SNR,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% Inspect territorial lineage (cheap)
if project_ready(PROJECT_ROOT):
    run_command("xrd_app.cli", "lineage", "--root", PROJECT_ROOT, "--scan", SCAN)

# %% [markdown]
# Territorial products use nominal bin size 1 and variant `territory`. Keep that
# identity consistent across grid, bins, peaks, and shapes. Before calling the
# product lossless, verify measured positions are real and each territory has one
# contributing frame.
#
# ## Manual detector ROI workflow
#
# Detector rectangles use `(x0, y0, x1, y1)` where x is detector column, y is
# detector row, and upper bounds are exclusive. Start with the detector sum to
# identify a region, then detect/save ROI products through the CLI.

# %% Detector sum used for ROI selection (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "reflection-sum", "--root", PROJECT_ROOT, "--scan", SCAN,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% Show ROI command help before choosing bounds (cheap)
run_command("xrd_app.cli", "roi-detect", "--help")
run_command("xrd_app.cli", "roi-shapes", "--help")
run_command("xrd_app.cli", "roi-save", "--help")

# %% [markdown]
# Run `roi-detect` to propose detector rectangles, review or edit them, then use
# `roi-shapes` and `roi-save` to create exact sample-space maps and a persisted ROI
# catalog. Keep fixed-ROI intensity separate from adaptive Bragg shape metrics in
# scientific reporting; agreement is a useful cross-check, not an identity.
