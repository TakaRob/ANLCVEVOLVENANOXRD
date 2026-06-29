#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_binning.sh
#
# Bin the rocking-series scans (203-214) at 1x1 AND 3x3, on a machine with fast
# LAN access to micdata (run check_lan_access.sh first — see LAN_ACCESS_CHECK.md).
#
# Binning only (grid -> bin). No peaks/shapes here; positions are auto-recreated
# from the file-per-row layout, so no tth/reflection setup is needed. Incomplete
# scans (no XRD frames) are skipped automatically.
#
# Usage:
#   ./run_binning.sh [DATA_ROOT]
#     DATA_ROOT  dir containing Raw/ (or the Raw/ dir itself, or .../2026-1-Luo).
#                Auto-probed if omitted (same logic as check_lan_access.sh).
#
# Env overrides:
#   PROJECT=<dir>        output project (default: $PWD/xrd_study) — NOT OneDrive
#   BIN_SIZES="1 3"      bin sizes to build (space-separated)
#   SCANS="203 .. 214"   scan numbers to process
#   COMPRESSION=gzip     gzip | lz4 | none
#   SKIP_EXISTING=1      skip a (scan,bin) whose bins .h5 already exists
#   DRY_RUN=1            print the commands without running them
#
# NOTE: each bin size re-reads the full raw once (~283 GB/scan). Building BOTH
# 1x1 and 3x3 reads every scan TWICE. Reorder BIN_SIZES="3 1" to get the small,
# fast 3x3 working layer first.
# ---------------------------------------------------------------------------
set -uo pipefail

REL='isn/2026-1/2026-1-Luo'
PROJECT="${PROJECT:-$PWD/xrd_study}"
BIN_SIZES="${BIN_SIZES:-1 3}"
# 206 omitted: incomplete (3/151 XRD frames) as of 2026-06-29. Add it back if recollected.
SCANS="${SCANS:-203 204 205 207 208 209 210 211 212 213 214}"
COMPRESSION="${COMPRESSION:-gzip}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
PROJECT_NAME="${PROJECT_NAME:-rocking_203_214}"

say() { printf '%s\n' "$*"; }
hr()  { printf -- '---------------------------------------------------------------\n'; }
run() {  # echo + (optionally) execute a command
  say "  \$ $*"
  [ "$DRY_RUN" = "1" ] && return 0
  "$@"
}

hr
say "Binning rocking series 203-214   ($(hostname 2>/dev/null || echo host))"
say "project: $PROJECT   bins: [$BIN_SIZES]   compression: $COMPRESSION"
[ "$DRY_RUN" = "1" ] && say "*** DRY RUN — no commands executed ***"
hr

# --- locate the data (same resolution as check_lan_access.sh) ------------
candidates=()
[ "$#" -gt 0 ] && candidates+=("$@")
candidates+=(
  "/mnt/z/${REL}" "/micdata/data1/${REL}" "/mnt/micdata/data1/${REL}"
  "/net/micdata/data1/${REL}" "/Volumes/data1/${REL}" "$HOME/micdata/data1/${REL}"
)
resolve_raw() {
  local c="$1"
  [ -d "$c/Raw" ] && { printf '%s\n' "$c/Raw"; return; }
  [ "$(basename "$c")" = "Raw" ] && [ -d "$c" ] && { printf '%s\n' "$c"; return; }
  if [ -d "$c" ] && compgen -G "$c/Scan_*" >/dev/null 2>&1; then printf '%s\n' "$c"; fi
}
RAW=""
for c in "${candidates[@]}"; do
  for g in $c; do r="$(resolve_raw "$g" 2>/dev/null || true)"; [ -n "$r" ] && { RAW="$r"; break; }; done
  [ -n "$RAW" ] && break
done
if [ -z "$RAW" ]; then
  say "❌ could not find Raw/ on this machine. Pass it explicitly:"
  say "   ./run_binning.sh /path/to/data1/${REL}"
  say "   (see LAN_ACCESS_CHECK.md for mounting the share)"
  hr; exit 2
fi
say "✓ data root: $RAW"

# --- prerequisites ------------------------------------------------------
if ! command -v xrd-app >/dev/null 2>&1; then
  say "❌ 'xrd-app' not found. Install the app from the repo first:"
  say "     pip install -e ."
  [ "$DRY_RUN" = "1" ] || { hr; exit 3; }
fi

# --- project init + scan discovery --------------------------------------
mkdir -p "$PROJECT"
if [ ! -f "$PROJECT/config.yaml" ]; then
  run xrd-app init --name "$PROJECT_NAME" --root "$PROJECT"
else
  say "  (project exists: $PROJECT/config.yaml)"
fi
run xrd-app scan-detect --root "$PROJECT" --scans-dir "$RAW"

# --- bin each scan at each size -----------------------------------------
ok=0; skipped=0; failed=""
total_start=$SECONDS
for n in $SCANS; do
  s=$(printf 'Scan_%04d' "$n")
  if [ ! -d "$RAW/$s/XRD" ] && [ ! -d "$RAW/$s" ]; then
    say "  - $s: not present — skip"; skipped=$((skipped+1)); continue
  fi
  for b in $BIN_SIZES; do
    out="$PROJECT/Binned/$s/xrd_${b}x${b}_bins.h5"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out" ]; then
      say "  = $s ${b}x${b}: exists — skip ($out)"; ok=$((ok+1)); continue
    fi
    say ""
    say ">> $s  ${b}x${b}"
    t0=$SECONDS
    if run xrd-app make-bins --root "$PROJECT" --scan "$s" \
           --bin-size "$b" --compression "$COMPRESSION"; then
      dt=$(( SECONDS - t0 ))
      if [ "$DRY_RUN" != "1" ] && [ -f "$out" ]; then
        sz=$(du -h "$out" 2>/dev/null | cut -f1)
        say "   ✓ $s ${b}x${b} done in ${dt}s -> $out ($sz)"
      else
        say "   ✓ $s ${b}x${b} (${dt}s)"
      fi
      ok=$((ok+1))
    else
      say "   ✗ $s ${b}x${b} FAILED (likely incomplete scan) — continuing"
      failed="$failed ${s}:${b}x${b}"
    fi
  done
done

# --- summary ------------------------------------------------------------
hr
total=$(( SECONDS - total_start ))
say "Done in ${total}s.   built/exists: $ok   scans-skipped: $skipped"
[ -n "$failed" ] && say "failed:$failed"
say "Binned output under: $PROJECT/Binned/"
if [ "$DRY_RUN" != "1" ]; then
  say ""; say "Produced bins:"
  find "$PROJECT/Binned" -name 'xrd_*x*_bins.h5' 2>/dev/null \
    | sort | while read -r f; do say "  $(du -h "$f" 2>/dev/null | cut -f1)  $f"; done
fi
hr
say "Next: rsync $PROJECT/Binned down to the laptop, then run peaks/shapes"
say "(see ROCKING_STUDY_203-214.md, Phases 2-6)."
