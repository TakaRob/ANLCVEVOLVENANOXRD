#!/usr/bin/env python3
"""Convert existing SUMMED bins to mean-per-frame WITHOUT re-reading raw frames.

``run_final_analysis.sh`` requires frame-normalized bins (mean per contributing
frame). The normal way to get them is ``xrd-app bin --normalize-frames``, which
re-reads every detector frame from raw — cheap for scans with a local unbinned
archive, but ~1 s/bin (tens of minutes/scan) for device scans whose frames live
as loose per-row HDF5 on the slow WAN share.

But the old summed bins already ARE the sum of their contributing frames, and the
grid mapping records how many frames went into each bin. So:

    mean_per_frame[bin] = summed[bin] / n_frames(bin)

is an exact reconstruction — no raw reads. This takes a device scan from tens of
minutes down to seconds (it only re-reads the already-binned HDF5).

The output is byte-for-byte equivalent in structure to what
``xrd-app bin --normalize-frames`` writes: float32, ``np.clip(x, 0, 1e9)``, the
same zstd compression, per-dataset ``n_frames`` attr, and the file attrs
``aggregation="mean_per_frame"`` / ``normalized_by="contributing_frame_count"``.
Because those two attrs are exactly what run_final_analysis.sh's
``is_frame_normalized`` checks, a converted scan is automatically SKIPPED on the
next run of the .sh — no .sh change needed.

CORRECTNESS CAVEAT: old summed bins carry no ``aggregation`` attr, so they cannot
self-certify as sums. They were sums by the old ``build_bins`` default. If a file
is ALREADY normalized (has the mean_per_frame attrs) it is skipped (dividing a
mean by the frame count again would be wrong) — do not pass --force to a file
that is already mean-per-frame. Use --verify on one scan first to confirm the
shortcut matches a true raw rebuild before trusting it across a project.

Usage:
    # Convert one scan's 3x3 bins in place (atomic):
    python3 normalize_bins_from_sum.py <root> --scan 195 --bin-size 3

    # Convert every summed 3x3 bins file found under Binned/ (skips normalized):
    python3 normalize_bins_from_sum.py <root> --bin-size 3

    # Territorial 1x1 focus variant:
    python3 normalize_bins_from_sum.py <root> --scan 203 --bin-size 1 --variant territory

    # Sanity check: rebuild one scan from raw and compare to the shortcut
    # (writes NOTHING to the real file):
    python3 normalize_bins_from_sum.py <root> --scan 226 --bin-size 3 --verify

    # Preview only:
    python3 normalize_bins_from_sum.py <root> --bin-size 3 --dry-run
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from xrd_app.config import DataManager
from xrd_app.core import io


def _bin_datasets(handle):
    """Yield (key, dataset) for every 2-D bin dataset in an open bins file."""
    for key, obj in handle.items():
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            yield key, obj


def frame_counts(grid_path: Path) -> dict:
    """Per-bin contributing-frame count from the grid mapping."""
    grid = io.load_grid_mapping(grid_path)
    return {key: len(frames) for key, frames in grid.get("bins", {}).items()}


def is_normalized(bins_path: Path) -> bool:
    with h5py.File(bins_path, "r") as h:
        return (h.attrs.get("aggregation") == "mean_per_frame"
                and h.attrs.get("normalized_by") == "contributing_frame_count")


def convert(bins_path: Path, grid_path: Path, out_path: Path) -> str:
    """Divide each summed bin by its frame count → mean-per-frame at out_path.

    Writes atomically when out_path == bins_path (tmp file + os.replace).
    """
    counts = frame_counts(grid_path)
    comp_kwargs, comp_label = io.get_compression_kwargs("zstd")

    in_place = out_path.resolve() == bins_path.resolve()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp") if in_place else out_path
    tmp.parent.mkdir(parents=True, exist_ok=True)

    n_bins = 0
    missing = []
    with h5py.File(bins_path, "r") as fin, h5py.File(tmp, "w") as fout:
        # Carry over file attrs (bin_size, n_bin_rows/cols, detector_shape, …),
        # then stamp the normalization provenance the pipeline checks for.
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        fout.attrs["aggregation"] = "mean_per_frame"
        fout.attrs["normalized_by"] = "contributing_frame_count"

        for key, dset in _bin_datasets(fin):
            count = counts.get(key)
            if not count:
                missing.append(key)
                # No frames recorded for this bin: it must be all-zero already.
                data = dset[()].astype(np.float32)
            else:
                arr = dset[()].astype(np.float64) / float(count)
                np.clip(arr, 0, 1e9, out=arr)
                data = arr.astype(np.float32)
            out = fout.create_dataset(key, data=data, **comp_kwargs)
            out.attrs["n_frames"] = int(count or 0)
            n_bins += 1

        fout.attrs["n_bins"] = n_bins

    if in_place:
        os.replace(tmp, bins_path)

    note = f"OK ({n_bins} bins, {comp_label})"
    if missing:
        note += f"  [WARN {len(missing)} bins had 0 frames in grid: {missing[:3]}…]"
    return note


def compare(path_a: Path, path_b: Path) -> str:
    """Stream-compare two bins files; report worst absolute / relative diff."""
    max_abs = 0.0
    max_rel = 0.0
    n_checked = 0
    with h5py.File(path_a, "r") as a, h5py.File(path_b, "r") as b:
        keys_a = {k for k, _ in _bin_datasets(a)}
        keys_b = {k for k, _ in _bin_datasets(b)}
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        for key in sorted(keys_a & keys_b):
            va = a[key][()].astype(np.float64)
            vb = b[key][()].astype(np.float64)
            diff = np.abs(va - vb)
            m = float(diff.max())
            if m > max_abs:
                max_abs = m
            denom = np.maximum(np.abs(vb), 1e-6)
            r = float((diff / denom).max())
            if r > max_rel:
                max_rel = r
            n_checked += 1
    verdict = "MATCH" if max_abs <= 1e-3 else "MISMATCH"
    msg = (f"{verdict}: {n_checked} bins compared, "
           f"max|Δ|={max_abs:.3e}, max relΔ={max_rel:.3e}")
    if only_a or only_b:
        msg += f"  [key mismatch: only_a={len(only_a)}, only_b={len(only_b)}]"
    return msg


def verify_scan(dm: DataManager, scan: int, bin_size: int, variant: str) -> int:
    """Build the scan from raw (--normalize-frames) into a temp, run the shortcut
    into another temp, and compare. Writes nothing to the real bins file."""
    bins_path = dm.binned_h5(bin_size, scan=scan, variant=variant or None)
    grid_path = dm.grid_mapping(bin_size=bin_size, scan=scan, variant=variant or None)
    if not bins_path.exists():
        print(f"  no summed bins to shortcut: {bins_path}", file=sys.stderr)
        return 2
    if not grid_path.exists():
        print(f"  no grid mapping: {grid_path}", file=sys.stderr)
        return 2
    if is_normalized(bins_path):
        print("  bins are ALREADY mean-per-frame — cannot verify the shortcut "
              "against them (need the original sums). Pick a still-summed scan.",
              file=sys.stderr)
        return 2

    xrd_app = os.environ.get("XRD_APP", "xrd-app")
    with tempfile.TemporaryDirectory() as td:
        raw_ref = Path(td) / "raw_mean.h5"
        shortcut = Path(td) / "shortcut_mean.h5"

        print(f"  [1/2] rebuilding from raw via {xrd_app} bin --normalize-frames …")
        cmd = [xrd_app, "bin", "--root", str(dm.root), "--scan", str(scan),
               "--bin-size", str(bin_size), "--normalize-frames",
               "--output", str(raw_ref)]
        if variant:
            cmd += ["--variant", variant]
        proc = subprocess.run(cmd)
        if proc.returncode != 0 or not raw_ref.exists():
            print("  raw rebuild failed", file=sys.stderr)
            return 1

        print("  [2/2] running the sum→mean shortcut …")
        convert(bins_path, grid_path, shortcut)

        print("  " + compare(shortcut, raw_ref))
    return 0


def discover_scans(dm: DataManager, bin_size: int, variant: str) -> list:
    """Scan names under Binned/ that have a summed bins file for this bin_size."""
    tag = f"_{variant}" if variant else ""
    name = f"xrd_{bin_size}x{bin_size}_bins{tag}.h5"
    root = dm.binned_dir_root
    found = sorted(p.parent.name for p in root.glob(f"Scan_*/{name}"))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Project root (contains Binned/ Metadata/ …)")
    ap.add_argument("--scan", type=int, help="Scan number (omit to convert all found)")
    ap.add_argument("--bin-size", type=int, default=3)
    ap.add_argument("--variant", default="", help="e.g. 'territory' for focus 1x1")
    ap.add_argument("--force", action="store_true",
                    help="Convert even if already normalized (DANGER: double-divides)")
    ap.add_argument("--dry-run", action="store_true", help="List targets, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="Rebuild one --scan from raw and compare; writes nothing")
    args = ap.parse_args()

    variant = args.variant or ""
    dm = DataManager(args.root)

    if args.verify:
        if args.scan is None:
            print("--verify needs a single --scan", file=sys.stderr)
            return 2
        print(f"Verify Scan_{args.scan:04d} "
              f"({args.bin_size}x{args.bin_size}{'/' + variant if variant else ''})")
        return verify_scan(dm, args.scan, args.bin_size, variant)

    if args.scan is not None:
        scans = [f"Scan_{args.scan:04d}"]
    else:
        scans = discover_scans(dm, args.bin_size, variant)
        if not scans:
            print("No summed bins files found for those parameters.", file=sys.stderr)
            return 2
        print(f"Discovered {len(scans)} scan(s) with "
              f"{args.bin_size}x{args.bin_size}{'/' + variant if variant else ''} bins.\n")

    failures = 0
    converted = 0
    skipped = 0
    for scan_name in scans:
        bins_path = dm.binned_h5(args.bin_size, scan=scan_name, variant=variant or None)
        grid_path = dm.grid_mapping(bin_size=args.bin_size, scan=scan_name,
                                    variant=variant or None)
        if not bins_path.exists():
            print(f"  {scan_name}: MISSING bins {bins_path}")
            failures += 1
            continue
        if not grid_path.exists():
            print(f"  {scan_name}: MISSING grid {grid_path.name}")
            failures += 1
            continue
        if is_normalized(bins_path) and not args.force:
            print(f"  {scan_name}: skip (already mean-per-frame)")
            skipped += 1
            continue
        if args.dry_run:
            counts = frame_counts(grid_path)
            print(f"  {scan_name}: would convert "
                  f"({len(counts)} bins in grid) -> {bins_path.name}")
            continue
        try:
            status = convert(bins_path, grid_path, bins_path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  {scan_name}: FAIL {exc}")
            failures += 1
            continue
        print(f"  {scan_name}: {status}")
        converted += 1

    if not args.dry_run:
        print(f"\nDone. {converted} converted, {skipped} skipped, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
