#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_binning_179-226.sh
#
# Build TWO new binning projects from the ISN rocking data, exactly like the
# 203-214 run — reusing run_binning.sh as the engine, once per project:
#
#     179-201   scans 179-201   (1x1 + 3x3)
#     215-226   scans 215-226   (1x1 + 3x3)
#
# Projects are created next to the existing 203-214 project (in Takaji/ under the
# data root). Meant to run over SSH on the LAN host (sec2llm) from inside the
# ANLCVEVOLVENANOXRD/ repo checkout. Binning only (grid -> bin); positions are
# auto-recreated from the file-per-row layout, so no tth/reflection setup.
# Missing/incomplete scans (no XRD frames) are skipped automatically.
#
# Usage (from the ANLCVEVOLVENANOXRD repo dir, with the data dir alongside):
#   ./run_binning_179-226.sh [DATA_ROOT]
#     DATA_ROOT  dir containing Raw/  (default: /mnt/isn/2026-1/2026-1-Luo,
#                then the usual micdata probe in run_binning.sh if that's absent)
#
# Env overrides (forwarded to run_binning.sh):
#   OUTPUT_BASE=<dir>    where the two project dirs are created
#                        (default: <DATA_ROOT>/Takaji if it exists, else $PWD)
#   BIN_SIZES="3 1"      bin sizes; 3x3 first = small fast working layer first
#   COMPRESSION=gzip     gzip | lz4 | none
#   SKIP_EXISTING=1      skip a (scan,bin) whose bins .h5 already exists
#   DRY_RUN=1            print the commands without running them
#   ONLY=A|B             build just one project (A=179-201, B=215-226)
#
# Cost note: each bin size re-reads the full raw once (~283 GB/scan). Two
# projects x ~35 scans x 2 bin sizes is a LOT of I/O — run it on the LAN host,
# not the laptop, and rsync the Binned/ trees down afterwards.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="$HERE/run_binning.sh"

DATA_ROOT_DEFAULT="/mnt/isn/2026-1/2026-1-Luo"
DATA_ROOT="${1:-$DATA_ROOT_DEFAULT}"

# Default the two new projects into Takaji/ (next to the existing 203-214),
# falling back to $PWD if that dir isn't there.
if [ -z "${OUTPUT_BASE:-}" ]; then
  if [ -d "$DATA_ROOT/Takaji" ]; then OUTPUT_BASE="$DATA_ROOT/Takaji"; else OUTPUT_BASE="$PWD"; fi
fi
export BIN_SIZES="${BIN_SIZES:-3 1}"        # 3x3 first (fast), then 1x1
export COMPRESSION="${COMPRESSION:-gzip}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"

hr() { printf -- '===============================================================\n'; }

if [ ! -x "$ENGINE" ] && [ ! -f "$ENGINE" ]; then
  echo "❌ engine not found: $ENGINE"
  echo "   run this script from inside the repo checkout (needs run_binning.sh)."
  exit 2
fi

# If the caller gave an explicit DATA_ROOT (or the default exists), forward it as
# run_binning.sh's positional arg. Otherwise let the engine auto-probe micdata.
DATA_ARG=()
if [ -d "$DATA_ROOT" ]; then
  DATA_ARG=("$DATA_ROOT")
else
  echo "⚠  DATA_ROOT '$DATA_ROOT' not present on this host — letting run_binning.sh auto-probe."
fi

hr
echo "Two-project rocking binning   ($(hostname 2>/dev/null || echo host))"
echo "data root : ${DATA_ROOT:-<auto-probe>}"
echo "output    : $OUTPUT_BASE   bins: [$BIN_SIZES]   compression: $COMPRESSION"
[ "$DRY_RUN" = "1" ] && echo "*** DRY RUN — no commands executed ***"
hr

# one project = one call into the engine, with its own PROJECT dir + name + scans
bin_project() {
  local name="$1" scans="$2"
  echo ""; hr
  echo ">>> PROJECT $name   scans: $(echo $scans | tr '\n' ' ')"
  hr
  PROJECT="$OUTPUT_BASE/$name" \
  PROJECT_NAME="$name" \
  SCANS="$scans" \
    bash "$ENGINE" "${DATA_ARG[@]}"
}

status=0
if [ "$ONLY" != "B" ]; then
  bin_project "179-201" "$(seq 179 201)" || status=1
fi
if [ "$ONLY" != "A" ]; then
  bin_project "215-226" "$(seq 215 226)" || status=1
fi

hr
if [ "$DRY_RUN" != "1" ]; then
  echo "Projects created under: $OUTPUT_BASE"
  for n in 179-201 215-226; do
    [ -d "$OUTPUT_BASE/$n/Binned" ] && \
      echo "  $n: $(find "$OUTPUT_BASE/$n/Binned" -name 'xrd_*x*_bins.h5' 2>/dev/null | wc -l) bin files"
  done
fi
echo "Next: rsync the two Binned/ trees to the laptop, then run peaks/shapes"
echo "(same Phases as the 203-214 study)."
hr
exit $status
