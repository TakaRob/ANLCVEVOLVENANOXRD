#!/usr/bin/env bash
# Run frame-normalized peak detection and shaping on the final project.
#
# Focus scans use the lossless territorial 1x1 variant. All other device scans
# use the standard 3x3 mapping. Before detection, summed bins are rebuilt as the
# mean per contributing frame. This script does not rebuild grids or run
# cross-scan studies.
#
# Example:
#   nohup ./run_final_analysis.sh > run_final_analysis.log 2>&1 &
#
# Optional overrides:
#   XRD_APP=/path/to/xrd-app DETECTOR=name SNR=4 WORKERS=4 LINK_TOLERANCE=5

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XRD_APP="${XRD_APP:-xrd-app}"
DETECTOR="${DETECTOR:-5x5_tophat_band_adaptive_snr}"
SNR="${SNR:-4}"
WORKERS="${WORKERS:-4}"
LINK_TOLERANCE="${LINK_TOLERANCE:-5}"

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

run_grid_scan() {
  local scan=$1
  local scan_name
  local bins peaks shapes
  scan_name="Scan_$(printf '%04d' "$scan")"
  bins="$ROOT/Binned/$scan_name/xrd_3x3_bins.h5"
  peaks="$ROOT/Labels/$scan_name/${DETECTOR}_peaks_3x3.h5"
  shapes="$ROOT/Labels/$scan_name/gaussian_shapes_3x3.h5"

  printf '\n========== %s 3x3 (%s) ==========\n' "$scan_name" "$(timestamp)"
  if [[ ! -s "$bins" ]]; then
    printf 'MISSING BINS: %s\n' "$bins" >&2
    failures+=("bins3:$scan")
    return
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
  local bins peaks shapes
  scan_name="Scan_$(printf '%04d' "$scan")"
  bins="$ROOT/Binned/$scan_name/xrd_1x1_bins_territory.h5"
  peaks="$ROOT/Labels/$scan_name/${DETECTOR}_peaks_1x1_territory.h5"
  shapes="$ROOT/Labels/$scan_name/territory_shapes_1x1_territory_coord.h5"

  printf '\n========== %s territory 1x1 (%s) ==========\n' "$scan_name" "$(timestamp)"
  if [[ ! -s "$bins" ]]; then
    printf 'MISSING TERRITORY BINS: %s\n' "$bins" >&2
    failures+=("bins-territory:$scan")
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

for scan in "${FOCUS_SCANS[@]}"; do
  run_territory_scan "$scan"
done

for scan in "${GRID_SCANS[@]}"; do
  run_grid_scan "$scan"
done

printf '\n========== FINISHED (%s) ==========\n' "$(timestamp)"
if ((${#failures[@]})); then
  printf 'Failures (%d): %s\n' "${#failures[@]}" "${failures[*]}" >&2
  exit 1
fi

printf 'All peak and shape stages completed successfully.\n'
