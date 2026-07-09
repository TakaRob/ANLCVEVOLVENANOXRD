#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_contours_hd_215-226.sh
#
# Same as run_contours_hd.sh, preset for the 215-226 project: contour finding
# (peaks -> shapes) + HD Device View (hd-device-map) at 3x3 for scans 215-226,
# writing JSON/CSV to the project's Labels/<scan>/.
#
# Usage:
#   ./run_contours_hd_215-226.sh [PROJECT]     # default: ./215-226
#
# All run_contours_hd.sh env overrides still apply (DRY_RUN, SKIP_EXISTING,
# DO_HD, WIN, SNR, and SCANS if you want a subset).
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT="${PROJECT:-${1:-./215-226}}"
export SCANS="${SCANS:-$(seq 215 226)}"

exec "$HERE/run_contours_hd.sh" "$PROJECT"
