#!/usr/bin/env python3
"""Regrid the grid mappings (all bin sizes) to the REAL (X, Y) coordinates.

Rocking 203-214 were binned on the ``file_per_row`` layout (the one-file-per-row
reconstruction), not on the true stage positions — so every ``grid_mapping_
NxN.json`` is skewed. This script rebuilds each mapping from the **real
coordinate CSV** the beamline wrote to the SOCKETSERVER directory
(``Scan_NNNN_position.csv``), using the app's own de-skew (``assign_grid_
coordinate_faithful``) so the result is identical to::

    xrd-app grid --bin-size N --deskew-method faithful

For each scan the per-frame (row, col) is computed once from the true positions,
then binned at every bin size present (auto-detected from the existing
``grid_mapping_NxN.json`` files, or restricted with ``--bin-sizes``).

The spatial regrid needs no raw frames — the existing mapping already carries the
coordinate-independent ``frame_map`` / ``xrd_files`` / ``n_total``. The heavy
binned HDF5 (``xrd_NxN_bins.h5``) still has to be rebuilt from the raw frames;
pass ``--rebin`` to do that here when the raw frames are reachable, otherwise
re-run ``xrd-app bin --bin-size N`` wherever the raw data lives.

Idempotent: a mapping already regridded to the requested method is skipped unless
``--force``. Scans with no ``Scan_NNNN_position.csv`` are skipped and reported.

Usage
-----
    python regrid_bins.py                      # regrid every bin size for 203-214
    python regrid_bins.py --bin-sizes 3        # just the 3x3 mappings
    python regrid_bins.py --rebin              # also rebuild the HDF5 (needs raw frames)
    python regrid_bins.py --dry-run            # report what would change, write nothing
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# --- locate the xrd_app package (installed via `pip install -e .`, else add the
#     directory holding this script's parent to sys.path so `import xrd_app` works) ---
try:
    from xrd_app.core import io
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from xrd_app.core import io
    except ModuleNotFoundError as e:  # pragma: no cover - environment issue
        sys.exit(f"Cannot import xrd_app ({e}). Run `pip install -e .` in the "
                 "xrd_app directory, or run this script from beside it.")

import numpy as np  # noqa: E402  (after the path fix-up above)

# ── defaults for THIS dataset (override on the CLI) ──────────────────────────
DEFAULT_PROJECT_ROOT = "/home/takaji/rocking_203_214"
DEFAULT_POSITION_ROOT = "/mnt/z/isn/2026-1/2026-1-Luo/Processed/SOCKETSERVER"
DEFAULT_SCANS = "203-214"

# faithful column_mode -> coordinate_source label (matches io.generate_grid_mapping)
_METHODS = {
    "faithful": ("square", "positions_faithful"),
    "faithful_native": ("native", "positions_faithful_native"),
}
_MAPPING_RE = re.compile(r"grid_mapping_(\d+)x(\d+)\.json$")


def parse_scans(spec: str) -> list[int]:
    """'203-214' or '203,204,207' or a mix -> sorted unique scan numbers."""
    out: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def find_position_csv(position_root: Path, num: int) -> Path | None:
    """Locate scan ``num``'s real position CSV, tolerating naming variants."""
    for name in (f"Scan_{num:04d}_position.csv", f"scan_{num:04d}_position.csv",
                 f"Scan_{num}_position.csv", f"scan_{num}_position.csv"):
        cand = position_root / name
        if cand.exists():
            return cand
    return None


def discover_bin_sizes(scan_dir: Path) -> list[int]:
    """Bin sizes N for which a real (non-backup) grid_mapping_NxN.json exists."""
    sizes = set()
    for p in scan_dir.glob("grid_mapping_*x*.json"):
        if ".bak" in p.name or "pre-regrid" in p.name:
            continue
        m = _MAPPING_RE.search(p.name)
        if m and m.group(1) == m.group(2):
            sizes.add(int(m.group(1)))
    return sorted(sizes)


def regrid_scan(num: int, project_root: Path, position_root: Path,
                bin_sizes: list[int] | None, column_mode: str, coord_source: str,
                rebin: bool, force: bool, dry_run: bool) -> dict:
    """Regrid one scan across bin sizes. Returns a status dict for the report."""
    scan = f"Scan_{num:04d}"
    st: dict = {"scan": scan, "status": "skipped", "reason": "", "sizes": {}}
    scan_dir = project_root / "Metadata" / scan
    if not scan_dir.is_dir():
        st["reason"] = f"no Metadata dir {scan_dir}"
        return st

    sizes = bin_sizes if bin_sizes is not None else discover_bin_sizes(scan_dir)
    if not sizes:
        st["reason"] = "no existing grid_mapping_NxN.json to regrid"
        return st

    csv_path = find_position_csv(position_root, num)
    if csv_path is None:
        st["reason"] = f"no Scan_{num:04d}_position.csv in {position_root}"
        return st
    if io.is_recreated_csv(csv_path):
        st["reason"] = f"{csv_path.name} is a recreated (synthetic) CSV, not real positions"
        return st

    # Source the coordinate-independent frame layout from any existing mapping
    # (prefer the finest, 1x1). It is identical across bin sizes for a scan.
    src_map_path = min(
        (scan_dir / f"grid_mapping_{n}x{n}.json" for n in sizes
         if (scan_dir / f"grid_mapping_{n}x{n}.json").exists()),
        key=lambda p: p.stat().st_size, default=None)
    if src_map_path is None:
        st["reason"] = "no readable source mapping for frame_map"
        return st
    src = io.load_grid_mapping(src_map_path)
    frame_map = src["frame_map"]
    xrd_files = src["xrd_files"]
    n_total = src.get("n_total_frames", len(frame_map))
    h5_dataset = src.get("h5_dataset", io.H5_DATASET)

    frame_x, frame_y = io.load_positions_xy(csv_path, n_total)
    if not np.isfinite(frame_y).any():
        st["reason"] = f"{csv_path.name} has no Y_Position column (need true 2-D X,Y)"
        return st

    # De-skew once at 1x1 resolution; bin sizes just regroup this per-frame grid.
    grid_row, grid_col, n_rows, n_cols = io.assign_grid_coordinate_faithful(
        frame_x, frame_y, frame_map, column_mode=column_mode, log=print)
    grid_to_frames: dict = {}
    for gi in range(n_total):
        grid_to_frames.setdefault((int(grid_row[gi]), int(grid_col[gi])), []).append(gi)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st["status"] = "ok"
    for b in sizes:
        gm_path = scan_dir / f"grid_mapping_{b}x{b}.json"
        info: dict = {"path": gm_path.name}
        if gm_path.exists() and not force:
            existing = io.load_grid_mapping(gm_path)
            if (existing.get("positions_real") and
                    existing.get("coordinate_source") == coord_source):
                info["result"] = "already regridded (use --force to redo)"
                st["sizes"][b] = info
                continue

        bins, n_bin_rows, n_bin_cols = io.build_bin_mapping(n_rows, n_cols, b, grid_to_frames)
        prev = (io.load_grid_mapping(gm_path).get("coordinate_source", "?")
                if gm_path.exists() else "none")
        new_gm = {
            "bin_size": b,
            "coordinate_source": coord_source,
            "positions_csv": str(csv_path),
            "positions_real": True,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_bin_rows": n_bin_rows,
            "n_bin_cols": n_bin_cols,
            "n_total_frames": n_total,
            "n_bins": len(bins),
            "h5_dataset": h5_dataset,
            "xrd_files": xrd_files,
            "bins": bins,
            "frame_map": frame_map,
        }
        info["result"] = f"{prev} -> {coord_source} ({n_rows}x{n_cols}, {len(bins)} bins)"

        if dry_run:
            info["result"] = "would regrid: " + info["result"]
            st["sizes"][b] = info
            continue

        if gm_path.exists():
            backup = gm_path.with_name(f"grid_mapping_{b}x{b}.pre-regrid-{stamp}.bak.json")
            shutil.copy2(gm_path, backup)
            info["backup"] = backup.name
        io.atomic_write_json(gm_path, new_gm, indent=None)

        if rebin:
            first = Path(xrd_files[0]) if xrd_files else None
            archive = project_root / "Binned" / scan / "xrd_unbinned_archive.h5"
            if archive.exists() or (first is not None and first.exists()):
                out_h5 = project_root / "Binned" / scan / f"xrd_{b}x{b}_bins.h5"
                io.build_bins(new_gm, out_h5, bin_size=b, log=print,
                              archive=archive if archive.exists() else None)
                info["h5"] = f"rebuilt -> {out_h5.name}"
            else:
                info["h5"] = f"SKIPPED (archive/raw frames not reachable: {first})"
        st["sizes"][b] = info

    # Drop the real CSV beside the mappings so future `xrd-app` runs resolve it.
    if not dry_run:
        shutil.copy2(csv_path, scan_dir / "positions.csv")
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT,
                    help=f"App project root (default: {DEFAULT_PROJECT_ROOT})")
    ap.add_argument("--position-root", default=DEFAULT_POSITION_ROOT,
                    help=f"Dir of Scan_NNNN_position.csv (default: {DEFAULT_POSITION_ROOT})")
    ap.add_argument("--scans", default=DEFAULT_SCANS,
                    help=f"Scan numbers, e.g. '203-214' or '203,207' (default: {DEFAULT_SCANS})")
    ap.add_argument("--bin-sizes", default=None,
                    help="Comma list of bin sizes to regrid (default: auto-detect per scan)")
    ap.add_argument("--method", choices=list(_METHODS), default="faithful",
                    help="De-skew column mode (default: faithful = square-pixel)")
    ap.add_argument("--rebin", action="store_true",
                    help="Also rebuild xrd_NxN_bins.h5 (needs raw frames reachable)")
    ap.add_argument("--force", action="store_true",
                    help="Regrid even mappings already tagged with this method")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; write nothing")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    position_root = Path(args.position_root)
    column_mode, coord_source = _METHODS[args.method]
    scans = parse_scans(args.scans)
    bin_sizes = ([int(x) for x in args.bin_sizes.split(",")]
                 if args.bin_sizes else None)

    print(f"Regrid mappings  method={args.method} ({coord_source})  "
          f"bin-sizes={bin_sizes or 'auto'}")
    print(f"  project-root : {project_root}")
    print(f"  position-root: {position_root}")
    print(f"  scans        : {scans[0]}-{scans[-1]} ({len(scans)} scans)"
          f"{'   [DRY RUN]' if args.dry_run else ''}\n")

    results = []
    for num in scans:
        print(f"── Scan_{num:04d} ──")
        try:
            st = regrid_scan(num, project_root, position_root, bin_sizes,
                             column_mode, coord_source, args.rebin, args.force,
                             args.dry_run)
        except Exception as e:  # keep going; report per-scan failures
            st = {"scan": f"Scan_{num:04d}", "status": "error", "reason": repr(e),
                  "sizes": {}}
        results.append(st)
        if st["status"] == "ok":
            for b, info in sorted(st["sizes"].items()):
                line = f"   {b}x{b}: {info['result']}"
                if "h5" in info:
                    line += f"   [h5: {info['h5']}]"
                print(line)
        else:
            print(f"   {st['status']}: {st['reason']}")
        print()

    # ── summary ──
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errored = [r for r in results if r["status"] == "error"]
    print("═══ summary ═══")
    print(f"  regridded: {len(ok)}   skipped: {len(skipped)}   errors: {len(errored)}")
    for r in skipped:
        print(f"    SKIP  {r['scan']}: {r['reason']}")
    for r in errored:
        print(f"    ERROR {r['scan']}: {r['reason']}")
    if not args.rebin and ok and not args.dry_run:
        print("\n  Mappings regridded. The binned HDF5 files still need rebuilding")
        print("  from raw frames — re-run with --rebin where raw data is reachable,")
        print("  or: xrd-app bin --bin-size N --scan <NNN>   (per scan, per size)")
    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main())
