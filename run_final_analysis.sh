#!/usr/bin/env bash
# Run frame-normalized peak detection and shaping on the final project.
#
# Focus scans (FOCUS_SCANS) get a LOSSLESS pair: a territorial 1x1 built at
# --target-size 1 (one frame per cell, no summing) plus the standard 3x3 mapping
# — this is what Device View / HD Device View render. All other device scans get
# the standard 3x3 mapping only. Before detection, summed bins are rebuilt as the
# mean per contributing frame. A missing HDF5 grid mapping is restored from the
# scan's grid_mapping_*.json when present (pre-HDF5 trees only carry the JSON,
# which the current pipeline cannot read), otherwise rebuilt from the scan's real
# positions.csv. This script does not run cross-scan studies.
#
# Example:
#   nohup ./run_final_analysis.sh > run_final_analysis.log 2>&1 &
#
# ROOT defaults to the directory this script lives in. If your bins live in a
# sub-project (e.g. .../ANLCVEVOLVENANOXRD/Scans179-226Perovskite), point ROOT
# at that sub-dir, NOT the parent:
#   ROOT=/full/path/Scans179-226Perovskite ./run_final_analysis.sh
#
# Optional overrides:
#   ROOT=/path/to/project XRD_APP=/path/to/xrd-app DETECTOR=name SNR=4 WORKERS=4 LINK_TOLERANCE=5

set -uo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
XRD_APP="${XRD_APP:-xrd-app}"
DETECTOR="${DETECTOR:-5x5_tophat_band_adaptive_snr}"
SNR="${SNR:-4}"
WORKERS="${WORKERS:-4}"
LINK_TOLERANCE="${LINK_TOLERANCE:-5}"

# RAW_ROOT: optional. When set, the focus scans are re-registered from this raw
# tree before the lossless build, so the scan registry points at THIS host's
# paths. Needed because Raw/scans.json bakes in absolute dirs — a tree registered
# under WSL's /mnt/z won't resolve on a beamline host where the same share is
# /net/micdata/data1. scan-detect merges, so only the focus entries change.
#   RAW_ROOT=/net/micdata/data1/isn/2026-1/2026-1-Luo/Raw ./run_final_analysis.sh
RAW_ROOT="${RAW_ROOT:-}"

FOCUS_SCANS=(179 182 203 207 215 218)
GRID_SCANS=(
  180 181 183 184 185 186 187 188 189 190 191 192 193 194 195 196
  197 198 199 200 201 204 205 208 209 210 211 212 213 214 216 217
  219 220 221 222 223 224 225 226
)

failures=()

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

is_current() {
  local output=$1
  local input=$2
  [[ -s "$output" && "$output" -nt "$input" ]]
}

is_frame_normalized() {
  local bins=$1
  python3 - "$bins" <<'PY'
import sys

import h5py

with h5py.File(sys.argv[1], "r") as handle:
    valid = (
        handle.attrs.get("aggregation") == "mean_per_frame"
        and handle.attrs.get("normalized_by") == "contributing_frame_count"
    )
raise SystemExit(0 if valid else 1)
PY
}

ensure_normalized_bins() {
  local scan=$1
  local bin_size=$2
  local variant=${3:-}
  local bins=$4
  local args=(
    bin
    --root "$ROOT"
    --scan "$scan"
    --bin-size "$bin_size"
    --normalize-frames
  )

  if [[ -n "$variant" ]]; then
    args+=(--variant "$variant")
  fi

  if is_frame_normalized "$bins"; then
    printf '[skip] bins are normalized by contributing frame count: %s\n' "$bins"
    return 0
  fi

  printf '[normalize] rebuilding bins as mean per contributing frame: %s\n' "$bins"
  "$XRD_APP" "${args[@]}" || return 1

  if ! is_frame_normalized "$bins"; then
    printf 'NORMALIZATION PROVENANCE MISSING: %s\n' "$bins" >&2
    return 1
  fi
}

ensure_grid() {
  # Ensure the HDF5 grid mapping exists. Pre-HDF5 project trees carry only the
  # old grid_mapping_*.json; the current pipeline resolves and requires the .h5
  # form, so 'bin --normalize-frames' hard-fails without it. The JSON already
  # holds the full frame->bin assignment, so convert it in place (no raw frames
  # needed). Only when no JSON exists do we rebuild from the scan's real
  # positions.csv via the given 'xrd-app' subcommand.
  local grid=$1
  shift
  local build=("$@")
  local json="${grid%.h5}.json"

  if [[ -s "$grid" ]]; then
    return 0
  fi

  if [[ -s "$json" ]]; then
    printf '[grid] converting JSON grid mapping to HDF5: %s\n' "$json"
    if python3 - "$json" "$grid" <<'PY'
import json, sys
from xrd_app.core import io
src, dst = sys.argv[1], sys.argv[2]
io.save_grid_mapping(dst, json.loads(open(src).read()))
io.load_grid_mapping(dst)  # verify it reads back
PY
    then
      [[ -s "$grid" ]] && return 0
    fi
    printf 'GRID JSON CONVERSION FAILED, trying a raw rebuild: %s\n' "$json" >&2
  fi

  printf '[grid] building missing grid mapping: %s\n' "$grid"
  "$XRD_APP" "${build[@]}" || return 1

  if [[ ! -s "$grid" ]]; then
    printf 'GRID BUILD PRODUCED NO OUTPUT: %s\n' "$grid" >&2
    return 1
  fi
}

run_grid_scan() {
  local scan=$1
  local scan_name
  local bins peaks shapes grid
  scan_name="Scan_$(printf '%04d' "$scan")"
  bins="$ROOT/Binned/$scan_name/xrd_3x3_bins.h5"
  peaks="$ROOT/Labels/$scan_name/${DETECTOR}_peaks_3x3.h5"
  shapes="$ROOT/Labels/$scan_name/gaussian_shapes_3x3.h5"
  grid="$ROOT/Metadata/$scan_name/grid_mapping_3x3.h5"

  printf '\n========== %s 3x3 (%s) ==========\n' "$scan_name" "$(timestamp)"
  if ! ensure_grid "$grid" grid --root "$ROOT" --scan "$scan" --bin-size 3; then
    failures+=("grid3:$scan")
    return
  fi
  if [[ ! -s "$bins" ]]; then
    # No pre-built bins: build them fresh as mean-per-frame. 'bin' reads the
    # unbinned archive when present (fast) else the raw frames; if neither
    # exists it fails and we report it rather than silently skipping the scan.
    printf '[bin] building 3x3 bins (mean per frame): %s\n' "$bins"
    if ! "$XRD_APP" bin --root "$ROOT" --scan "$scan" --bin-size 3 --normalize-frames; then
      printf 'MISSING BINS and could not build them (no archive/raw?): %s\n' "$bins" >&2
      failures+=("bins3:$scan")
      return
    fi
  fi
  if ! ensure_normalized_bins "$scan" 3 "" "$bins"; then
    failures+=("normalize3:$scan")
    return
  fi

  if is_current "$peaks" "$bins"; then
    printf '[skip] peaks are current: %s\n' "$peaks"
  elif ! "$XRD_APP" peaks \
      --root "$ROOT" \
      --scan "$scan" \
      --bin-size 3 \
      --algorithm "$DETECTOR" \
      --snr "$SNR" \
      --workers "$WORKERS"; then
    failures+=("peaks3:$scan")
    return
  fi

  if is_current "$shapes" "$peaks"; then
    printf '[skip] shapes are current: %s\n' "$shapes"
  elif ! "$XRD_APP" shapes \
      --root "$ROOT" \
      --scan "$scan" \
      --bin-size 3 \
      --algorithm gaussian \
      --peak-algo "$DETECTOR" \
      --link-tolerance "$LINK_TOLERANCE" \
      --grid-link; then
    failures+=("shapes3:$scan")
  fi
}

run_territory_scan() {
  local scan=$1
  local scan_name
  local bins peaks shapes grid
  scan_name="Scan_$(printf '%04d' "$scan")"
  bins="$ROOT/Binned/$scan_name/xrd_1x1_bins_territory.h5"
  peaks="$ROOT/Labels/$scan_name/${DETECTOR}_peaks_1x1_territory.h5"
  shapes="$ROOT/Labels/$scan_name/territory_shapes_1x1_territory_coord.h5"
  grid="$ROOT/Metadata/$scan_name/grid_mapping_1x1_territory.h5"

  printf '\n========== %s territory 1x1 (%s) ==========\n' "$scan_name" "$(timestamp)"
  if [[ ! -s "$bins" ]]; then
    printf 'MISSING TERRITORY BINS: %s\n' "$bins" >&2
    failures+=("bins-territory:$scan")
    return
  fi
  if ! ensure_grid "$grid" territory-grid --root "$ROOT" --scan "$scan"; then
    failures+=("grid-territory:$scan")
    return
  fi
  if ! ensure_normalized_bins "$scan" 1 territory "$bins"; then
    failures+=("normalize-territory:$scan")
    return
  fi

  if is_current "$peaks" "$bins"; then
    printf '[skip] territory peaks are current: %s\n' "$peaks"
  elif ! "$XRD_APP" peaks \
      --root "$ROOT" \
      --scan "$scan" \
      --bin-size 1 \
      --variant territory \
      --algorithm "$DETECTOR" \
      --snr "$SNR" \
      --workers "$WORKERS"; then
    failures+=("peaks-territory:$scan")
    return
  fi

  if is_current "$shapes" "$peaks"; then
    printf '[skip] territory shapes are current: %s\n' "$shapes"
  elif ! "$XRD_APP" shapes \
      --root "$ROOT" \
      --scan "$scan" \
      --bin-size 1 \
      --variant territory \
      --algorithm territory \
      --peak-algo "$DETECTOR" \
      --link-tolerance "$LINK_TOLERANCE" \
      --coordinate; then
    failures+=("shapes-territory:$scan")
  fi
}

if ! command -v "$XRD_APP" >/dev/null 2>&1; then
  printf 'xrd-app executable not found: %s\n' "$XRD_APP" >&2
  printf 'Set XRD_APP=/full/path/to/xrd-app and retry.\n' >&2
  exit 2
fi
if ! python3 -c 'import h5py' >/dev/null 2>&1; then
  printf 'Python cannot import h5py; normalization provenance cannot be checked.\n' >&2
  exit 2
fi

printf 'Project: %s\n' "$ROOT"
printf 'Binning: mean per contributing frame (required and provenance-checked)\n'
printf 'Detector: %s, SNR: %s, workers: %s, link tolerance: %s\n' \
  "$DETECTOR" "$SNR" "$WORKERS" "$LINK_TOLERANCE"
printf 'Started: %s\n' "$(timestamp)"

for scan in "${GRID_SCANS[@]}"; do
  run_grid_scan "$scan"
done

# ── Focus-scan lossless build (appended) ────────────────────────────────
# The six focus scans are the ones we inspect in Device View / HD Device View,
# so they get a LOSSLESS pair instead of the summed territorial reference:
#   (1) territorial 1x1 at --target-size 1 — one frame per cell, no summing,
#       built via 'territory-build' (which also builds the unbinned archive)
#   (2) the standard 3x3 mapping (run_grid_scan, which reads that same archive
#       so binning is fast)
# Scans whose raw frames are gone in this tree (no archive) can't be rebuilt
# losslessly — they are reported as failures, never silently summed.
# Set BUILD_FOCUS_LOSSLESS=0 to fall back to the old summed territorial pass.
BUILD_FOCUS_LOSSLESS="${BUILD_FOCUS_LOSSLESS:-1}"

territory_reregister_focus() {
  # Optional: re-point the registry at THIS host's raw tree (RAW_ROOT) so
  # 'archive-unbinned' resolves. scan-detect merges, so only the focus scans
  # are rewritten; the 40 grid-scan entries are left as-is.
  [[ -n "$RAW_ROOT" ]] || return 0
  local focus_csv
  focus_csv="$(IFS=,; printf '%s' "${FOCUS_SCANS[*]}")"
  printf '[raw] re-registering focus scans (%s) from %s\n' "$focus_csv" "$RAW_ROOT"
  if ! "$XRD_APP" scan-detect --root "$ROOT" --scans-dir "$RAW_ROOT" --scans "$focus_csv"; then
    printf 'RAW RE-REGISTRATION reported problems from %s (some focus builds may not find raw)\n' \
      "$RAW_ROOT" >&2
  fi
}

territory_is_lossless() {
  # True only when the territory grid mapping was built at --target-size 1.
  local grid=$1
  [[ -s "$grid" ]] || return 1
  python3 - "$grid" <<'PY'
import sys
from xrd_app.core import io
try:
    gm = io.load_grid_mapping(sys.argv[1])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if int(gm.get("target_size", 0)) == 1 else 1)
PY
}

run_focus_lossless_scan() {
  local scan=$1
  local scan_name grid shapes
  scan_name="Scan_$(printf '%04d' "$scan")"
  grid="$ROOT/Metadata/$scan_name/grid_mapping_1x1_territory.h5"
  shapes="$ROOT/Labels/$scan_name/territory_shapes_1x1_territory_coord.h5"

  printf '\n========== %s focus lossless 1x1 (%s) ==========\n' "$scan_name" "$(timestamp)"
  if territory_is_lossless "$grid" && [[ -s "$shapes" ]]; then
    printf '[skip] lossless territorial 1x1 already built (target-size 1): %s\n' "$grid"
  else
    printf '[territory] building lossless territorial reference (target-size 1): %s\n' "$scan_name"
    if ! "$XRD_APP" territory-build \
        --root "$ROOT" \
        --scan "$scan" \
        --target-size 1 \
        --algorithm "$DETECTOR" \
        --snr "$SNR"; then
      printf 'LOSSLESS TERRITORY BUILD FAILED (raw frames may be absent): %s\n' "$scan_name" >&2
      failures+=("focus-territory:$scan")
      return
    fi
    if ! territory_is_lossless "$grid"; then
      printf 'TERRITORY GRID IS NOT TARGET-SIZE 1 AFTER BUILD: %s\n' "$grid" >&2
      failures+=("focus-territory:$scan")
      return
    fi
  fi

  # Standard 3x3 for the same scan (records its own failures).
  run_grid_scan "$scan"
}

if [[ "$BUILD_FOCUS_LOSSLESS" == 1 ]]; then
  territory_reregister_focus
  for scan in "${FOCUS_SCANS[@]}"; do
    run_focus_lossless_scan "$scan"
  done
else
  for scan in "${FOCUS_SCANS[@]}"; do
    run_territory_scan "$scan"
  done
fi

printf '\n========== FINISHED (%s) ==========\n' "$(timestamp)"
if ((${#failures[@]})); then
  printf 'Failures (%d): %s\n' "${#failures[@]}" "${failures[*]}" >&2
  exit 1
fi

printf 'All peak and shape stages completed successfully.\n'
