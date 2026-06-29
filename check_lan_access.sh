#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# check_lan_access.sh
#
# Run this on ANY candidate computer (APS workstation, lab desktop, etc.) to
# answer one question: can this machine read the micdata XRD data fast enough
# to bin here, instead of pulling 283 GB/scan to a laptop over the ~23 MB/s WAN?
#
# It (1) finds the data, (2) confirms the study scans exist, and (3) measures
# real sequential read bandwidth on a ~1.2 GB XRD frame file — the exact thing
# binning does. No data is written anywhere (reads to /dev/null).
#
# Usage:
#   ./check_lan_access.sh [DATA_ROOT]
#     DATA_ROOT  optional: a dir containing Raw/  (or the Raw/ dir itself,
#                or the .../2026-1-Luo project dir, or a single Scan_* dir).
#                If omitted, common mount points are auto-probed.
#
# Windows users: see LAN_ACCESS_CHECK.md for the PowerShell equivalent.
# ---------------------------------------------------------------------------
set -uo pipefail

SHARE_UNC='\\micdata\data1'        # what Windows maps as Z:
REL='isn/2026-1/2026-1-Luo'        # project path under the share
WAN_BASELINE=23                    # MB/s measured laptop -> micdata (WAN ceiling)
RAW_GB_PER_SCAN=283                # ~size of one Scan_*/XRD/
N_STUDY_SCANS=12                   # scans 203-214

say()  { printf '%s\n' "$*"; }
hr()   { printf -- '---------------------------------------------------------------\n'; }

hr
say "micdata LAN access + bandwidth check   ($(hostname 2>/dev/null || echo host))"
say "share: ${SHARE_UNC}   project: ${REL}"
hr

# --- 1. locate the data --------------------------------------------------
# Build candidate list: explicit arg(s) first, then common mounts.
candidates=()
[ "$#" -gt 0 ] && candidates+=("$@")
candidates+=(
  "/mnt/z/${REL}"
  "/micdata/data1/${REL}"
  "/mnt/micdata/data1/${REL}"
  "/net/micdata/data1/${REL}"
  "/Volumes/data1/${REL}"            # macOS SMB mount
  "$HOME/micdata/data1/${REL}"
)

resolve_raw() {  # echo a Raw dir for a candidate, or nothing
  local c="$1"
  [ -d "$c/Raw" ] && { printf '%s\n' "$c/Raw"; return; }
  [ "$(basename "$c")" = "Raw" ] && [ -d "$c" ] && { printf '%s\n' "$c"; return; }
  # candidate may itself be a folder of Scan_* dirs
  if [ -d "$c" ] && compgen -G "$c/Scan_*" >/dev/null 2>&1; then
    printf '%s\n' "$c"; return
  fi
}

RAW=""
for c in "${candidates[@]}"; do
  # expand globs (gvfs etc.)
  for g in $c; do
    r="$(resolve_raw "$g" 2>/dev/null || true)"
    if [ -n "$r" ]; then RAW="$r"; break; fi
  done
  [ -n "$RAW" ] && break
done

if [ -z "$RAW" ]; then
  say "RESULT: ❌ could not find the data on this machine."
  say ""
  say "Tried:"; for c in "${candidates[@]}"; do say "  - $c"; done
  say ""
  say "Fixes:"
  say "  • Mount the share, then re-run with the path, e.g.:"
  say "      ./check_lan_access.sh /path/to/data1/${REL}"
  say "  • Linux SMB mount example:"
  say "      sudo mkdir -p /mnt/micdata"
  say "      sudo mount -t cifs //micdata.xray.aps.anl.gov/data1 /mnt/micdata \\"
  say "           -o username=YOURUSER,domain=ANL,uid=\$(id -u),ro,vers=3.0"
  say "      ./check_lan_access.sh /mnt/micdata/${REL}"
  hr
  exit 2
fi

say "✓ found data root: $RAW"

# --- 2. confirm the study scans -----------------------------------------
nscan=$(compgen -G "$RAW/Scan_*" 2>/dev/null | wc -l | tr -d ' ')
say "  Scan_* dirs present: ${nscan}"

present=0; missing=""
for n in $(seq 203 214); do
  s=$(printf 'Scan_%04d' "$n")
  if [ -d "$RAW/$s" ]; then present=$((present+1)); else missing="$missing $s"; fi
done
say "  study scans 203-214 present: ${present}/12"
[ -n "$missing" ] && say "    missing:${missing}"

# pick one XRD .h5 to benchmark (prefer a study scan)
pick=""
for n in 203 207 211 214 $(seq 203 214); do
  s=$(printf 'Scan_%04d' "$n")
  d="$RAW/$s/XRD"
  if [ -d "$d" ]; then
    f=$(compgen -G "$d/*.h5" 2>/dev/null | head -1 || true)
    [ -n "$f" ] && { pick="$f"; break; }
  fi
done
# fallback: any XRD .h5 anywhere
[ -z "$pick" ] && pick=$(find "$RAW" -path '*/XRD/*.h5' 2>/dev/null | head -1 || true)

if [ -z "$pick" ]; then
  say "RESULT: ⚠ data root found but no XRD/*.h5 frame files located — cannot benchmark."
  hr; exit 3
fi
say "  benchmark file: ${pick#$RAW/}"

# --- 3. measure sequential read bandwidth -------------------------------
say ""
say "Reading the file to /dev/null (this is the real binning read)..."
sz=$(wc -c < "$pick" 2>/dev/null || echo 0)
mbps=""

if command -v python3 >/dev/null 2>&1; then
  mbps=$(python3 - "$pick" <<'PYEOF'
import sys, time
f = sys.argv[1]; n = 0; t = time.time()
with open(f, "rb") as fh:
    while True:
        b = fh.read(8 * 1024 * 1024)
        if not b: break
        n += len(b)
dt = time.time() - t
print(f"{(n/1e6)/dt:.1f}" if dt > 0 else "9999")
PYEOF
)
else
  # bash fallback (integer-second resolution)
  SECONDS=0
  dd if="$pick" of=/dev/null bs=8M 2>/dev/null
  dur=$SECONDS; [ "$dur" -lt 1 ] && dur=1
  mbps=$(( sz / 1000000 / dur ))
fi

# --- 4. verdict ----------------------------------------------------------
hr
say "MEASURED READ SPEED:  ${mbps} MB/s     (laptop WAN baseline ≈ ${WAN_BASELINE} MB/s)"
# integer MB/s for portable comparisons / estimates (no awk dependency)
mbps_int=${mbps%%.*}; case "$mbps_int" in ''|*[!0-9]*) mbps_int=0;; esac
# est. time to bin all 12 study scans (binning reads full raw once per scan)
if [ "$mbps_int" -gt 0 ]; then
  est=$(( RAW_GB_PER_SCAN * 1000 * N_STUDY_SCANS / mbps_int / 3600 ))
  say "Est. time to bin all 12 scans here: ~${est} h   (laptop ≈ 42 h)"
fi
say ""
# classify
if   [ "$mbps_int" -ge 80 ]; then verdict="FAST"
elif [ "$mbps_int" -ge $(( 2 * WAN_BASELINE )) ]; then verdict="BETTER"
else verdict="NOFASTER"
fi
case "$verdict" in
  FAST)
    say "✅ FAST LAN access. Bin the scans HERE, then download only the small"
    say "   binned .h5 files (≥3×3 → <1 h total). This is the recommended path."
    ;;
  BETTER)
    say "🟡 Faster than the laptop, but not local-class. Binning here still beats"
    say "   pulling raw to the laptop — worth doing if no faster machine exists."
    ;;
  NOFASTER)
    say "❌ No faster than the laptop WAN link (~${WAN_BASELINE} MB/s). This machine"
    say "   is not closer to the data — find one on the micdata LAN, or accept"
    say "   the ~42 h binning on the laptop."
    ;;
esac
hr
say "Report the MEASURED READ SPEED line back to continue planning."
