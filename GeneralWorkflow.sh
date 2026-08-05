#!/usr/bin/env bash
# GeneralWorkflow.sh — a runnable, editable TEMPLATE for the xrd-app CLI.
#
# It walks a brand-new project through the whole pipeline, one command at a
# time, so you can see how the CLI fits together — then copy and edit it for
# your own project. Every "big button" in the GUI maps to one of these
# `xrd-app` commands, so this is the same pipeline the GUI drives, just scripted.
#
# Pipeline order:
#   init → link → scan-detect → (positions) → grid → bin → peaks → shapes
#   Phase 1 = peaks  (per-bin detection)
#   Phase 2 = shapes (link peaks across neighbouring bins → physical features)
#
# Usage:
#   1. Edit the CONFIG block below to point at your data.
#   2. Run it:                    ./GeneralWorkflow.sh
#      Preview commands only:     DRY_RUN=1 ./GeneralWorkflow.sh
#
# This is a teaching template — it favours a clear, linear read over the
# hardened skip/normalize/retry logic in run_final_analysis.sh. For production
# reruns of an existing project, prefer that script.

set -euo pipefail

# ── CONFIG — edit these ─────────────────────────────────────────────────
ROOT="${ROOT:-./MyProject}"                    # project root (created by 'init')
PROJECT_NAME="${PROJECT_NAME:-MyProject}"
SCAN="${SCAN:-203}"                            # scan number → Scan_0203

# External inputs (recorded by 'link' / 'scan-detect'):
SCANS_DIR="${SCANS_DIR:-/path/to/raw/scans}"   # parent dir containing Scan_NNNN/
TTH="${TTH:-/path/to/tth.tiff}"                # 2θ-per-pixel map (required by peaks)
REFLECTIONS="${REFLECTIONS:-}"                 # reflections.json (blank = use init's default set)

# Detection knobs:
DETECTOR="${DETECTOR:-5x5_tophat_band_adaptive_snr}"  # bundled name OR path to a .py detector
BIN_SIZE="${BIN_SIZE:-3}"                      # spatial binning (1, 3, 4, 5)
SNR="${SNR:-4}"                                # detection threshold
WORKERS="${WORKERS:-4}"                        # detector worker processes
LINK_TOLERANCE="${LINK_TOLERANCE:-5}"          # px tolerance when linking peaks into shapes

XRD_APP="${XRD_APP:-xrd-app}"                  # override with a full path if not on PATH
# ────────────────────────────────────────────────────────────────────────

# run: print the command, then run it (unless DRY_RUN=1). Keeps the CLI visible.
run()  { printf '\n$ %s\n' "$*"; [[ "${DRY_RUN:-0}" == 1 ]] || "$@"; }
step() { printf '\n========== %s ==========\n' "$*"; }

if ! command -v "$XRD_APP" >/dev/null 2>&1; then
  printf 'xrd-app not found: %s\n  Set XRD_APP=/full/path/to/xrd-app and retry.\n' "$XRD_APP" >&2
  exit 2
fi

# ── 1. init — create the project tree + config.yaml ─────────────────────
# Safe to re-run: it refuses to overwrite an existing config.yaml.
step "1/8  init"
if [[ -f "$ROOT/config.yaml" ]]; then
  echo "config.yaml already exists at $ROOT — skipping init."
else
  run "$XRD_APP" init --name "$PROJECT_NAME" --scan-number "$SCAN" --root "$ROOT"
fi

# ── 2. link — record shared calibration inputs ──────────────────────────
# Copies/records the 2θ map (and optional reflections) into Metadata/. init
# already seeds a default perovskite reflection set, so --reflections is
# optional. To use a custom evolved detector, add: --detector /path/to/algo.py
step "2/8  link"
link_args=(link --root "$ROOT" --tth "$TTH")
[[ -n "$REFLECTIONS" ]] && link_args+=(--reflections "$REFLECTIONS")
run "$XRD_APP" "${link_args[@]}"

# ── 3. scan-detect — discover raw scans → Raw/scans.json ────────────────
# --scans keeps just the scan(s) you want from a big directory. Drop it to
# register every Scan_*/ found under --scans-dir.
step "3/8  scan-detect"
run "$XRD_APP" scan-detect --root "$ROOT" --scans-dir "$SCANS_DIR" --scans "$SCAN"

# ── 4. positions — the REAL per-frame (X, Y) stage positions ────────────
# 'grid' auto-builds these from the SOCKETSERVER interferometry stream when
# missing, so this step is usually optional. Run it explicitly to (re)build,
# or link a CSV you already have instead:
#   xrd-app link --root "$ROOT" --scan "$SCAN" --position-csv /path/positions.csv
step "4/8  create-positions (optional)"
run "$XRD_APP" create-positions --root "$ROOT" --scan "$SCAN" \
  || echo "  create-positions skipped/failed — 'grid' will try SOCKETSERVER itself."

# ── 5. grid — map raw frames onto a spatial bin grid ────────────────────
# Writes Metadata/<scan>/grid_mapping_NxN.h5. Requires REAL positions (CSV or
# SOCKETSERVER); it hard-fails rather than reconstructing a skewed lattice.
step "5/8  grid"
run "$XRD_APP" grid --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE"

# ── 6. bin — build the binned detector HDF5 ─────────────────────────────
# --normalize-frames stores the MEAN per contributing frame (recommended) so
# bins with different frame counts are comparable. Writes Binned/<scan>/.
step "6/8  bin"
run "$XRD_APP" bin --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE" --normalize-frames

# ── 7. peaks — Phase 1: per-bin detection ───────────────────────────────
# Runs the detector over every bin → Labels/<scan>/<detector>_peaks_NxN.h5.
step "7/8  peaks (Phase 1)"
run "$XRD_APP" peaks --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE" \
  --algorithm "$DETECTOR" --snr "$SNR" --workers "$WORKERS"

# ── 8. shapes — Phase 2: link peaks across bins into shapes ─────────────
# --grid-link links neighbours on the N×N grid; a shape is the physically real
# feature (an isolated peak may be noise). Writes gaussian_shapes_NxN.h5.
step "8/8  shapes (Phase 2)"
run "$XRD_APP" shapes --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE" \
  --algorithm gaussian --peak-algo "$DETECTOR" --link-tolerance "$LINK_TOLERANCE" --grid-link

# ── inspect ─────────────────────────────────────────────────────────────
step "done — inspect results"
run "$XRD_APP" status --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE"
echo
echo "Explore interactively with:  $XRD_APP gui --root \"$ROOT\""

# ════════════════════════════════════════════════════════════════════════
# Recipes to copy — uncomment / adapt as needed
# ════════════════════════════════════════════════════════════════════════
#
# A) Multi-scan: loop the per-scan steps over several scans
# --------------------------------------------------------
# for SCAN in 203 204 205 207; do
#   xrd-app grid   --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE"
#   xrd-app bin    --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE" --normalize-frames
#   xrd-app peaks  --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE" \
#                  --algorithm "$DETECTOR" --snr "$SNR" --workers "$WORKERS"
#   xrd-app shapes --root "$ROOT" --scan "$SCAN" --bin-size "$BIN_SIZE" \
#                  --algorithm gaussian --peak-algo "$DETECTOR" \
#                  --link-tolerance "$LINK_TOLERANCE" --grid-link
# done
#
# B) Skew-free territorial 1x1 variant (high-resolution "focus" scans)
# --------------------------------------------------------
# Groups frames by true (X, Y) territories instead of the serpentine N×N grid.
# Note the --variant territory tag threads through every step, and shapes uses
# the territory algorithm + --coordinate linking.
# xrd-app territory-grid --root "$ROOT" --scan "$SCAN"
# xrd-app bin    --root "$ROOT" --scan "$SCAN" --bin-size 1 --variant territory --normalize-frames
# xrd-app peaks  --root "$ROOT" --scan "$SCAN" --bin-size 1 --variant territory \
#                --algorithm "$DETECTOR" --snr "$SNR" --workers "$WORKERS"
# xrd-app shapes --root "$ROOT" --scan "$SCAN" --bin-size 1 --variant territory \
#                --algorithm territory --peak-algo "$DETECTOR" \
#                --link-tolerance "$LINK_TOLERANCE" --coordinate
#
# C) Datasets with no known Bragg reflections (search the whole frame)
# --------------------------------------------------------
# xrd-app whole-frame-reflections --root "$ROOT" --scan "$SCAN"
