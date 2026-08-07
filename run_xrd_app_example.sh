#!/usr/bin/env bash
# Build the Luo final-analysis project from scratch using only xrd-app commands.
#
# This linear workflow shows every project and analysis stage without inspecting
# or modifying artifacts with Python/HDF5 helpers. Shell variables and loops only
# organize the CLI commands. Use a new ROOT: `xrd-app init` deliberately refuses
# an existing project.
#
# Analysis scope:
#   - scans 179-201, 203-205, 207-214, and 215-226 at normalized 3x3
#   - lossless territorial 1x1 products for scans 179, 182, 203, 207, 215, 218
#   - peaks, shapes, focus-scan HD maps, full detector sums, and two PDF reports
#
# Usage:
#   ROOT=/home/takaji/luo-final-cli \
#   RAW_ROOT=/mnt/z/isn/2026-1/2026-1-Luo/Raw \
#   ./run_xrd_app_example.sh
#
# Print the complete walkthrough without running it:
#   DRY_RUN=1 ./run_xrd_app_example.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XRD_APP="${XRD_APP:-xrd-app}"
ROOT="${ROOT:-$SCRIPT_DIR/LuoFinalAnalysisCLI}"
PROJECT_NAME="${PROJECT_NAME:-Nano-XRD focus analysis}"
RAW_ROOT="${RAW_ROOT:-/mnt/z/isn/2026-1/2026-1-Luo/Raw}"
POSITION_ROOT="${POSITION_ROOT:-}"
TTH_FILE="${TTH_FILE:-$SCRIPT_DIR/tth.tiff}"
REFLECTIONS_FILE="${REFLECTIONS_FILE:-$SCRIPT_DIR/xrd_app/assets/reflections.json}"
REPORT_DIR="${REPORT_DIR:-$ROOT/Figures}"

DETECTOR="${DETECTOR:-5x5_tophat_band_adaptive_snr}"
SNR="${SNR:-4}"
WORKERS="${WORKERS:-16}"
LINK_TOLERANCE="${LINK_TOLERANCE:-5}"
HD_WIN="${HD_WIN:-4}"
COMPRESSION="${COMPRESSION:-zstd}"

FOCUS_SCANS=(179 182 203 207 215 218)
# Scans 202 and 206 are intentionally absent from the production set.
ALL_SCANS=(
  179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194
  195 196 197 198 199 200 201
  203 204 205 207 208 209 210 211 212 213 214
  215 216 217 218 219 220 221 222 223 224 225 226
)
SCAN_CSV="$(IFS=,; printf '%s' "${ALL_SCANS[*]}")"

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  [[ "${DRY_RUN:-0}" == 1 ]] || "$@"
}

step() {
  printf '\n========== %s ==========\n' "$*"
}

if [[ "${DRY_RUN:-0}" != 1 ]] && ! command -v "$XRD_APP" >/dev/null 2>&1; then
  printf 'xrd-app executable not found: %s\n' "$XRD_APP" >&2
  printf 'Set XRD_APP=/full/path/to/xrd-app and retry.\n' >&2
  exit 2
fi

step "1/9 initialize a new project"
run "$XRD_APP" init --root "$ROOT" --name "$PROJECT_NAME"

step "2/9 link calibration and shared data roots"
link_args=(
  "$XRD_APP" link
  --root "$ROOT"
  --raw-root "$RAW_ROOT"
  --tth "$TTH_FILE"
  --reflections "$REFLECTIONS_FILE"
  --copy
)
[[ -n "$POSITION_ROOT" ]] && link_args+=(--position-root "$POSITION_ROOT")
run "${link_args[@]}"

step "3/9 discover the same Luo scan set"
run "$XRD_APP" scan-detect \
  --root "$ROOT" \
  --scans-dir "$RAW_ROOT" \
  --scans "$SCAN_CSV"

step "4/9 build normalized 3x3 bins, peaks, and shapes"
for scan in "${ALL_SCANS[@]}"; do
  step "Scan_$(printf '%04d' "$scan") normalized 3x3"
  run "$XRD_APP" make-bins \
    --root "$ROOT" \
    --scan "$scan" \
    --bin-size 3 \
    --compression "$COMPRESSION" \
    --normalize-frames
  run "$XRD_APP" peaks \
    --root "$ROOT" \
    --scan "$scan" \
    --bin-size 3 \
    --algorithm "$DETECTOR" \
    --snr "$SNR" \
    --workers "$WORKERS"
  run "$XRD_APP" shapes \
    --root "$ROOT" \
    --scan "$scan" \
    --bin-size 3 \
    --algorithm gaussian \
    --peak-algo "$DETECTOR" \
    --link-tolerance "$LINK_TOLERANCE" \
    --grid-link
done

step "5/9 build lossless territorial 1x1 focus products"
for scan in "${FOCUS_SCANS[@]}"; do
  run "$XRD_APP" territory-build \
    --root "$ROOT" \
    --scan "$scan" \
    --target-size 1 \
    --algorithm "$DETECTOR" \
    --snr "$SNR" \
    --compression "$COMPRESSION"
done

step "6/9 build high-definition device maps for focus scans"
for scan in "${FOCUS_SCANS[@]}"; do
  run "$XRD_APP" hd-device-map \
    --root "$ROOT" \
    --scan "$scan" \
    --bin-size 3 \
    --catalog gaussian_shapes_3x3.h5 \
    --name gaussian \
    --win "$HD_WIN"
done

step "7/9 recompute full detector sums for focus scans"
for scan in "${FOCUS_SCANS[@]}"; do
  run "$XRD_APP" reflection-sum \
    --root "$ROOT" \
    --scan "$scan" \
    --max-bins 0 \
    --overwrite
done

report_targets=()
for scan in "${FOCUS_SCANS[@]}"; do
  report_targets+=(--target "Scan_$(printf '%04d' "$scan"):3")
done

step "8/9 create the full focus-scan report"
run "$XRD_APP" report \
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
  --territory-maps

step "9/9 create the detector and device-map report"
run "$XRD_APP" report \
  --root "$ROOT" \
  "${report_targets[@]}" \
  --output "$REPORT_DIR/focus_scans_detector_and_device_maps.pdf" \
  --summed-images \
  --all-reflections \
  --no-features-by-reflection \
  --no-top-features \
  --no-source-images \
  --no-roi-images \
  --no-territory-maps

step "complete"
run "$XRD_APP" status --root "$ROOT" --scan 203 --bin-size 3
printf '\nProject created at: %s\n' "$ROOT"
