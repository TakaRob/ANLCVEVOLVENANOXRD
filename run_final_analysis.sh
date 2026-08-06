#!/usr/bin/env bash
# Start or resume a representative nano-XRD project and run the complete
# focus-scan workflow: calibration/scan setup, lossless 1x1 territory products,
# normalized 3x3 bins, peaks, shapes, HD device maps, full detector sums, and PDF
# reports.
#
# The six focus scans are Scan_0179, 0182, 0203, 0207, 0215, and 0218. As in
# the established Scans179-226 runner, the remaining device scans are processed
# at 3x3 by default; set INCLUDE_GRID_SCANS=0 for a focus-only run. Every stage
# is resumable, and current products are skipped where practical.
#
# Fresh-project example:
#   ROOT=/path/to/new-project \
#   RAW_ROOT=/path/to/parent-containing-Scan_NNNN \
#   POSITION_ROOT=/path/to/position-csvs \
#   TTH_FILE=/path/to/tth.tiff \
#   REFLECTIONS_FILE=/path/to/reflections.json \
#   nohup ./run_final_analysis.sh > run_final_analysis.log 2>&1 &
#
# ROOT defaults to the directory containing this script. TTH_FILE defaults to
# its tth.tiff and REFLECTIONS_FILE to the app's bundled perovskite reflections.
# A fresh ROOT is initialized automatically. Existing projects are never
# reinitialized, but their calibration, reflections, and optional external roots
# are updated through `xrd-app link`.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$SCRIPT_DIR}"
XRD_APP="${XRD_APP:-xrd-app}"
PROJECT_NAME="${PROJECT_NAME:-Nano-XRD focus analysis}"
TTH_FILE="${TTH_FILE:-${TTH:-$SCRIPT_DIR/tth.tiff}}"
RAW_ROOT="${RAW_ROOT:-}"
POSITION_ROOT="${POSITION_ROOT:-}"
REFLECTIONS_FILE="${REFLECTIONS_FILE:-$SCRIPT_DIR/xrd_app/assets/reflections.json}"
DETECTOR="${DETECTOR:-5x5_tophat_band_adaptive_snr}"
SNR="${SNR:-4}"
WORKERS="${WORKERS:-16}"
LINK_TOLERANCE="${LINK_TOLERANCE:-5}"
HD_WIN="${HD_WIN:-4}"
INCLUDE_GRID_SCANS="${INCLUDE_GRID_SCANS:-1}"
BUILD_FOCUS_LOSSLESS="${BUILD_FOCUS_LOSSLESS:-1}"
REPORT_DIR="${REPORT_DIR:-$ROOT/Figures}"

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

run_focus_hd_map() {
  local scan=$1
  local scan_name shapes hd_map
  scan_name="Scan_$(printf '%04d' "$scan")"
  shapes="$ROOT/Labels/$scan_name/gaussian_shapes_3x3.h5"
  hd_map="$ROOT/Labels/$scan_name/gaussian_hdmap_3x3.h5"

  printf '\n========== %s HD device map 3x3 (%s) ==========\n' "$scan_name" "$(timestamp)"
  if [[ ! -s "$shapes" ]]; then
    printf 'MISSING 3x3 SHAPES; cannot build HD device map: %s\n' "$shapes" >&2
    failures+=("hd3:$scan")
  elif is_current "$hd_map" "$shapes"; then
    printf '[skip] HD device map is current: %s\n' "$hd_map"
  elif ! "$XRD_APP" hd-device-map \
      --root "$ROOT" \
      --scan "$scan" \
      --bin-size 3 \
      --catalog "$shapes" \
      --name gaussian \
      --win "$HD_WIN"; then
    failures+=("hd3:$scan")
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
if [[ ! -f "$TTH_FILE" ]]; then
  printf '2-theta calibration file not found: %s\n' "$TTH_FILE" >&2
  printf 'Set TTH_FILE=/full/path/to/tth.tiff and retry.\n' >&2
  exit 2
fi
if [[ ! -f "$REFLECTIONS_FILE" ]]; then
  printf 'Reflection definition file not found: %s\n' "$REFLECTIONS_FILE" >&2
  printf 'Set REFLECTIONS_FILE=/full/path/to/reflections.json and retry.\n' >&2
  exit 2
fi

if [[ ! -f "$ROOT/config.yaml" ]]; then
  if [[ -z "$RAW_ROOT" ]]; then
    printf 'A fresh project requires RAW_ROOT to discover the focus scans.\n' >&2
    exit 2
  fi
  printf '[setup] initializing project: %s\n' "$ROOT"
  "$XRD_APP" init --root "$ROOT" --name "$PROJECT_NAME" || exit 2
fi

link_args=(
  link
  --root "$ROOT"
  --tth "$TTH_FILE"
  --reflections "$REFLECTIONS_FILE"
  --copy
)
[[ -n "$RAW_ROOT" ]] && link_args+=(--raw-root "$RAW_ROOT")
[[ -n "$POSITION_ROOT" ]] && link_args+=(--position-root "$POSITION_ROOT")
printf '[setup] linking 2-theta calibration, reflections, and external data roots\n'
"$XRD_APP" "${link_args[@]}" || exit 2

if [[ -n "$RAW_ROOT" ]]; then
  scans_to_register=("${FOCUS_SCANS[@]}")
  if [[ "$INCLUDE_GRID_SCANS" == 1 ]]; then
    scans_to_register+=("${GRID_SCANS[@]}")
  fi
  scan_csv="$(IFS=,; printf '%s' "${scans_to_register[*]}")"
  printf '[setup] discovering selected scans (%s)\n' "$scan_csv"
  "$XRD_APP" scan-detect --root "$ROOT" --scans-dir "$RAW_ROOT" --scans "$scan_csv" || exit 2
fi

printf 'Project: %s\n' "$ROOT"
printf '2-theta map: %s\n' "$TTH_FILE"
printf 'Reflections: %s\n' "$REFLECTIONS_FILE"
printf 'Focus report reflections: (001), (111)\n'
printf 'Binning: mean per contributing frame (required and provenance-checked)\n'
printf 'Detector: %s, SNR: %s, workers: %s, link tolerance: %s, HD window: %s\n' \
  "$DETECTOR" "$SNR" "$WORKERS" "$LINK_TOLERANCE" "$HD_WIN"
printf 'Started: %s\n' "$(timestamp)"

if [[ "$INCLUDE_GRID_SCANS" == 1 ]]; then
  for scan in "${GRID_SCANS[@]}"; do
    run_grid_scan "$scan"
  done
fi

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
  for scan in "${FOCUS_SCANS[@]}"; do
    run_focus_lossless_scan "$scan"
  done
else
  for scan in "${FOCUS_SCANS[@]}"; do
    run_territory_scan "$scan"
    run_grid_scan "$scan"
  done
fi

# Cache the high-definition intensity layer used by HD Device View. The 3x3
# shapes and lossless 1x1 source needed here were built immediately above.
for scan in "${FOCUS_SCANS[@]}"; do
  run_focus_hd_map "$scan"
done

# Reports use the full scan detector sum, not a preview subset. Recompute it
# after binning so stale or max-bin-limited artifacts cannot enter the PDFs.
for scan in "${FOCUS_SCANS[@]}"; do
  scan_name="Scan_$(printf '%04d' "$scan")"
  printf '\n========== %s full detector sum (%s) ==========\n' "$scan_name" "$(timestamp)"
  if ! "$XRD_APP" reflection-sum \
      --root "$ROOT" \
      --scan "$scan" \
      --max-bins 0 \
      --overwrite; then
    failures+=("reflection-sum:$scan")
  fi
done

report_targets=()
for scan in "${FOCUS_SCANS[@]}"; do
  report_targets+=(--target "Scan_$(printf '%04d' "$scan"):3")
done

printf '\n========== Full focus-scan report (%s) ==========\n' "$(timestamp)"
if ! "$XRD_APP" report \
    --root "$ROOT" \
    "${report_targets[@]}" \
    --output "$REPORT_DIR/focus_scans_full_report.pdf" \
    --summed-images \
    --all-reflections \
    --features-by-reflection \
    --top-features \
    --top-count 5 \
    --top-scope total \
    --reflection '(001)' \
    --reflection '(111)' \
    --source-images \
    --roi-images \
    --calculate-rois \
    --territory-maps; then
  failures+=("report:full")
fi

printf '\n========== Summed image and device-map report (%s) ==========\n' "$(timestamp)"
if ! "$XRD_APP" report \
    --root "$ROOT" \
    "${report_targets[@]}" \
    --output "$REPORT_DIR/focus_scans_detector_and_device_maps.pdf" \
    --summed-images \
    --all-reflections \
    --no-features-by-reflection \
    --no-top-features \
    --no-source-images \
    --no-roi-images \
    --no-territory-maps; then
  failures+=("report:detector-device")
fi

printf '\n========== FINISHED (%s) ==========\n' "$(timestamp)"
if ((${#failures[@]})); then
  printf 'Failures (%d): %s\n' "${#failures[@]}" "${failures[*]}" >&2
  exit 1
fi

printf 'Complete focus-scan analysis and both PDF reports finished successfully.\n'
