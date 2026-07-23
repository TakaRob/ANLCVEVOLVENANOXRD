#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_contours_hd.sh
#
# For an already-binned project, run the "contour finding" (peaks -> shapes) and
# then precompute the HD Device View (hd-device-map) for each scan, at 3x3. All
# outputs are JSON/CSV under the project's Labels/<scan>/ — i.e. they land in
# whatever project you point --root at (e.g. the network project).
#
# Per scan:
#   xrd-app run-pipeline  --bin-size 3   (peaks -> gaussian shapes = contours)
#   xrd-app hd-device-map --bin-size 3   (1x1 intensity beneath the 3x3 features)
#
# RUN THIS ON THE LAN HOST (sec2llm), not the laptop: hd-device-map does random
# raw-frame reads and is ~10x slower over the slow /mnt/z 9p mount. Pointing
# --root at the network project still writes the JSONs to the network drive.
#
# Usage:
#   ./run_contours_hd.sh /path/to/project        # project dir (has config.yaml)
#   PROJECT=/path/to/project ./run_contours_hd.sh
#
# Env overrides:
#   SCANS="179 .. 201"   scans to process (default 179-201)
#   BIN=3                source feature-map bin size
#   SKIP_EXISTING=1      skip a step whose output JSON already exists
#   DO_HD=1              also run hd-device-map (set 0 for contours only)
#   WIN=4                hd-device-map detector-peak half-window (px)
#   SNR=4.0              peak-detection SNR threshold
#   DRY_RUN=1            print commands without running
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT="${PROJECT:-${1:-}}"
SCANS="${SCANS:-$(seq 179 201)}"
BIN="${BIN:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DO_HD="${DO_HD:-1}"
WIN="${WIN:-4}"
SNR="${SNR:-4.0}"
DRY_RUN="${DRY_RUN:-0}"

say() { printf '%s\n' "$*"; }
hr()  { printf -- '---------------------------------------------------------------\n'; }
run() { say "  \$ $*"; [ "$DRY_RUN" = "1" ] && return 0; "$@"; }

if [ -z "$PROJECT" ]; then
  say "❌ no project given. Usage: ./run_contours_hd.sh /path/to/project"; exit 2
fi
if [ ! -f "$PROJECT/config.yaml" ]; then
  say "❌ '$PROJECT' is not an xrd-app project (no config.yaml)."; exit 2
fi
# --- resolve the CLI so this works over a fresh SSH shell -----------------
# Prefer an installed console script; fall back to the module form. Auto-activate
# a repo .venv if present, and put common user-install bins on PATH first.
[ -f "$HERE/.venv/bin/activate" ] && . "$HERE/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
if command -v xrd-app >/dev/null 2>&1; then
  XRD=(xrd-app)
elif python3 -c "import xrd_app" >/dev/null 2>&1; then
  XRD=(python3 -m xrd_app.cli)
elif python -c "import xrd_app" >/dev/null 2>&1; then
  XRD=(python -m xrd_app.cli)
else
  say "❌ can't find 'xrd-app' or the xrd_app package on PATH."
  say "   Activate the env you used for 'pip install -e .', e.g. ONE of:"
  say "     source .venv/bin/activate            # a repo virtualenv"
  say "     conda activate <env>                 # a conda env"
  say "     export PATH=\"\$HOME/.local/bin:\$PATH\"    # after 'pip install --user -e .'"
  say "   …then re-run. (This script already tries .venv and ~/.local/bin.)"
  exit 3
fi
say "cli: ${XRD[*]}"

SUF="${BIN}x${BIN}"
hr
say "Contours (+HD) for project: $PROJECT"
say "scans: $(echo $SCANS | tr '\n' ' ')"
say "bin: $SUF   do_hd: $DO_HD   skip_existing: $SKIP_EXISTING   snr: $SNR   win: $WIN"
[ "$DRY_RUN" = "1" ] && say "*** DRY RUN ***"
hr

ok=0; failed=""; total_start=$SECONDS
for n in $SCANS; do
  s=$(printf 'Scan_%04d' "$n")
  ldir="$PROJECT/Labels/$s"
  bins3="$PROJECT/Binned/$s/xrd_${SUF}_bins.h5"
  if [ ! -f "$bins3" ]; then
    say "  - $s: no ${SUF} bins — skip"; continue
  fi
  say ""; say ">> $s"

  # --- contours: peaks -> shapes ---------------------------------------
  shapes_json="$ldir/gaussian_shapes_${SUF}.json"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$shapes_json" ]; then
    say "  = shapes exist — skip ($shapes_json)"
  else
    t0=$SECONDS
    if run "${XRD[@]}" run-pipeline --root "$PROJECT" --scan "$s" --bin-size "$BIN" --snr "$SNR"; then
      say "   ✓ contours $s (${SUF}) in $(( SECONDS - t0 ))s"; ok=$((ok+1))
    else
      say "   ✗ contours $s FAILED — skipping hd for this scan"; failed="$failed ${s}:contours"; continue
    fi
  fi

  # --- HD device map ---------------------------------------------------
  [ "$DO_HD" != "1" ] && continue
  hd_json="$ldir/gaussian_hdmap_${SUF}.json"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$hd_json" ]; then
    say "  = hd-map exists — skip ($hd_json)"; continue
  fi
  t0=$SECONDS
  if run "${XRD[@]}" hd-device-map --root "$PROJECT" --scan "$s" --bin-size "$BIN" --win "$WIN"; then
    say "   ✓ hd-map $s (${SUF}) in $(( SECONDS - t0 ))s"; ok=$((ok+1))
  else
    say "   ✗ hd-map $s FAILED — continuing"; failed="$failed ${s}:hd"
  fi
done

hr
say "Done in $(( SECONDS - total_start ))s.   steps ok: $ok"
[ -n "$failed" ] && say "failed:$failed"
say "Outputs under: $PROJECT/Labels/<scan>/  (gaussian_shapes_${SUF}.json, gaussian_hdmap_${SUF}.json)"
hr
