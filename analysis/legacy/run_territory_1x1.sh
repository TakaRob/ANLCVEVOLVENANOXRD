#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_territory_1x1.sh
#
# Build the skew-free, true-(X,Y) 1x1 "territory" maps for a project and run the
# full detection pipeline on them, per scan:
#
#   territory-grid  ->  bin(--variant territory)  ->  peaks  ->  shapes
#
# Outputs (all under the project, i.e. the network drive via --root):
#   Metadata/<scan>/grid_mapping_1x1_territory.json
#   Binned/<scan>/xrd_1x1_bins_territory.h5
#   Labels/<scan>/*_peaks_1x1_territory.json
#   Labels/<scan>/territory_shapes_1x1_territory.json
#
# RUN THIS ON THE LAN HOST (sec2llm). The `bin` step reads raw detector pixels;
# over the slow /mnt/z mount it will crawl. On the host, territory-grid also
# resolves the raw frames via the project registry (/net/micdata) and stores
# machine-correct paths — which is why we (re)build the grid here rather than
# reuse a map made elsewhere (a laptop-built map carries /mnt/z paths that the
# host can't read).
#
# Usage:
#   ./run_territory_1x1.sh /path/to/project      # project dir (has config.yaml)
#   PROJECT=/path/to/project ./run_territory_1x1.sh
#
# Env overrides:
#   SCANS="179 180 ..."  scans to process (default: auto-discover every
#                        Metadata/Scan_* that has a positions.csv)
#   TARGET=1             frames per territory (1 = full 1x1 resolution)
#   SNR=4.0              peak-detection SNR threshold
#   RAW_ROOT=<dir>       parent of Scan_NNNN/XRD, if the project registry can't
#                        resolve raw here (passed as territory-grid --xrd-dir)
#   SKIP_EXISTING=1      skip a step whose output already exists
#   REGRID=0             force territory-grid rebuild even if bins exist
#   DRY_RUN=1            print commands without running
# ---------------------------------------------------------------------------
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT="${PROJECT:-${1:-}}"
TARGET="${TARGET:-1}"
SNR="${SNR:-4.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
REGRID="${REGRID:-0}"
DRY_RUN="${DRY_RUN:-0}"
RAW_ROOT="${RAW_ROOT:-}"

say() { printf '%s\n' "$*"; }
hr()  { printf -- '---------------------------------------------------------------\n'; }
run() { say "  \$ $*"; [ "$DRY_RUN" = "1" ] && return 0; "$@"; }

[ -z "$PROJECT" ]                 && { say "❌ usage: ./run_territory_1x1.sh /path/to/project"; exit 2; }
[ ! -f "$PROJECT/config.yaml" ]   && { say "❌ '$PROJECT' is not an xrd-app project (no config.yaml)."; exit 2; }

# --- resolve the CLI (works over a fresh SSH shell) -----------------------
[ -f "$HERE/.venv/bin/activate" ] && . "$HERE/.venv/bin/activate"
export PATH="$HOME/.local/bin:$PATH"
if   command -v xrd-app >/dev/null 2>&1;              then XRD=(xrd-app)
elif python3 -c "import xrd_app" >/dev/null 2>&1;     then XRD=(python3 -m xrd_app.cli)
elif python  -c "import xrd_app" >/dev/null 2>&1;     then XRD=(python -m xrd_app.cli)
else say "❌ can't find xrd-app / xrd_app. Activate the env (conda/venv/~.local) first."; exit 3
fi

# --- scans: explicit list, else discover from Metadata --------------------
if [ -z "${SCANS:-}" ]; then
  SCANS=""
  for d in "$PROJECT"/Metadata/Scan_*/; do
    [ -f "$d/positions.csv" ] && SCANS="$SCANS $(basename "$d" | sed 's/Scan_0*//')"
  done
fi

hr
say "Territory 1x1 pipeline: $PROJECT"
say "cli: ${XRD[*]}   target: $TARGET   snr: $SNR   skip_existing: $SKIP_EXISTING"
say "scans: $(echo $SCANS | tr '\n' ' ')"
[ "$DRY_RUN" = "1" ] && say "*** DRY RUN ***"
hr

ok=0; failed=""; t_all=$SECONDS
for n in $SCANS; do
  s=$(printf 'Scan_%04d' "$n")
  bins="$PROJECT/Binned/$s/xrd_1x1_bins_territory.h5"
  pos="$PROJECT/Metadata/$s/positions.csv"
  [ ! -f "$pos" ] && { say ">> $s: no positions.csv — skip"; continue; }
  say ""; say ">> $s"

  # 1+2. (re)build territory grid on THIS host, then bin -------------------
  if [ "$SKIP_EXISTING" = "1" ] && [ "$REGRID" != "1" ] && [ -f "$bins" ]; then
    say "  = territory bins exist — skip grid+bin ($bins)"
  else
    xdir_opt=()
    [ -n "$RAW_ROOT" ] && xdir_opt=(--xrd-dir "$RAW_ROOT/$s/XRD")
    t0=$SECONDS
    if run "${XRD[@]}" territory-grid --root "$PROJECT" --scan "$s" --target-size "$TARGET" "${xdir_opt[@]}" \
       && run "${XRD[@]}" bin --bin-size 1 --variant territory --root "$PROJECT" --scan "$s"; then
      say "   ✓ grid+bin $s ($(( SECONDS - t0 ))s)"; ok=$((ok+1))
    else
      say "   ✗ grid/bin $s FAILED — skipping peaks/shapes"; failed="$failed ${s}:bin"; continue
    fi
  fi

  # 3. peaks --------------------------------------------------------------
  peaks_glob=$(ls "$PROJECT/Labels/$s/"*_peaks_1x1_territory.json 2>/dev/null | head -1)
  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$peaks_glob" ]; then
    say "  = peaks exist — skip"
  else
    t0=$SECONDS
    if run "${XRD[@]}" peaks --bin-size 1 --variant territory --root "$PROJECT" --scan "$s" --snr "$SNR"; then
      say "   ✓ peaks $s ($(( SECONDS - t0 ))s)"; ok=$((ok+1))
    else
      say "   ✗ peaks $s FAILED — skipping shapes"; failed="$failed ${s}:peaks"; continue
    fi
  fi

  # 4. shapes (coordinate linking is the 1x1 default → writes a *_coord.json) -
  sj=$(ls "$PROJECT/Labels/$s/"territory_shapes_1x1_territory*.json 2>/dev/null | head -1)
  if [ "$SKIP_EXISTING" = "1" ] && [ -n "$sj" ]; then
    say "  = shapes exist — skip ($sj)"
  else
    pj=$(ls "$PROJECT/Labels/$s/"*_peaks_1x1_territory.json 2>/dev/null | head -1)
    if [ -z "$pj" ] && [ "$DRY_RUN" != "1" ]; then
      say "   ✗ shapes $s: no territory peaks json to link"; failed="$failed ${s}:shapes"; continue
    fi
    t0=$SECONDS
    if run "${XRD[@]}" shapes --bin-size 1 --variant territory --algorithm territory --from-peaks "$pj" --root "$PROJECT" --scan "$s"; then
      say "   ✓ shapes $s ($(( SECONDS - t0 ))s)"; ok=$((ok+1))
    else
      say "   ✗ shapes $s FAILED"; failed="$failed ${s}:shapes"
    fi
  fi
done

hr
say "Done in $(( SECONDS - t_all ))s.   steps ok: $ok"
[ -n "$failed" ] && say "failed:$failed"
say "Outputs: $PROJECT/{Binned,Labels}/<scan>/*territory*"
hr
