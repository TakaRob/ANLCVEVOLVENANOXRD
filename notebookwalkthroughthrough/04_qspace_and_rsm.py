# %% [markdown]
# # 04 - Reciprocal-space maps
# This Stage is still untested and is hidden from the GUI
# `qspace` maps detector pixels and sample theta to `(qx, qy, qz)`. `rsm` then
# accumulates retained detector intensity into a shared 3D reciprocal-space
# volume. Read `QSPACE.md` for geometry conventions and calibration caveats.
#
# The default path fits a flat detector to `tth.tiff`; `--poni` uses a supplied
# calibrated geometry including tilt. Stored `q_mag` comes from the selected 3D
# geometry, while direct radial `|Q|` can be computed from the original 2-theta
# map for comparison.

# %% Configuration
import os
from pathlib import Path

try:
    from notebooks._cli import project_ready, run_command
except ModuleNotFoundError:  # Plain `python notebooks/04_qspace_and_rsm.py`.
    from _cli import project_ready, run_command

PROJECT_ROOT = Path(os.environ.get("XRD_PROJECT", "CHANGE_ME"))
SCANS = os.environ.get("XRD_SCANS", "203,204,205")
BIN_SIZE = int(os.environ.get("XRD_BIN_SIZE", "3"))
QSPACE_DIR = os.environ.get("XRD_QSPACE_DIR", "Study/qspace")
RSM_PATH = os.environ.get("XRD_RSM_PATH", "Study/rsm.npz")
PONI_PATH = os.environ.get("XRD_PONI", "")
RUN_HEAVY = False

# %% Build per-scan q maps (heavy)
if project_ready(PROJECT_ROOT):
    args = [
        "qspace", "--root", PROJECT_ROOT, "--scans", SCANS,
        "--bin-size", BIN_SIZE, "--out-dir", QSPACE_DIR,
    ]
    if PONI_PATH:
        args.extend(["--poni", PONI_PATH])
    run_command("xrd_app.cli", *args, heavy=True, allow_heavy=RUN_HEAVY)

# %% Fuse q maps into an RSM (heavy)
if project_ready(PROJECT_ROOT):
    run_command(
        "xrd_app.cli", "rsm", "--root", PROJECT_ROOT,
        "--in-dir", QSPACE_DIR, "--out", RSM_PATH,
        heavy=True, allow_heavy=RUN_HEAVY,
    )

# %% [markdown]
# ## Physics checks
#
# - Supply `--theta` for scans absent from the built-in theta table; otherwise the
#   CLI warns and uses zero degrees.
# - Compare feature `q_mag` with `4*pi*sin(tth_pixel/2)/wavelength`. A systematic
#   default-path difference can indicate flat-fit residual or unmodeled tilt.
# - `rsm` voxel `counts` records retained detector pixels after filtering and
#   thresholding; it is not a threshold-independent geometric coverage map.
# - The fused volume integrates the illuminated sample map. Use
#   `<scan>_features_q.csv` for per-feature reciprocal-space points.
#
# The Reciprocal Space GUI tab displays 2D projections and, with the `gl` extra,
# a 3D volume. Data creation remains in the CLI commands above.
