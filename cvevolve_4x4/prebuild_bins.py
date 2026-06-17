"""
Pre-compute spatially-binned XRD images into a single HDF5 file.

Reads raw scan H5 frames via the grid mapping, sums each bin, and
writes the result as float32 datasets keyed by "row_col".

Usage:
    python prebuild_bins.py
    python prebuild_bins.py --bin-size 3 --compression lz4
    python prebuild_bins.py --grid-mapping path/to/grid_mapping.json --output /home/takaji/xrd_5x5_bins.h5

The output HDF5 structure:
    /0_0   -> float32 array (1062, 1028)
    /0_1   -> float32 array (1062, 1028)
    ...
    attrs: bin_size, n_bin_rows, n_bin_cols, n_bins, detector_shape
"""

import argparse
import json
import os
import time

import h5py
import numpy as np


def get_compression_kwargs(compression):
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    elif compression == "lz4":
        import hdf5plugin
        return hdf5plugin.LZ4()
    elif compression == "none":
        return {}
    else:
        raise ValueError(f"Unknown compression: {compression}")


def main():
    parser = argparse.ArgumentParser(description="Pre-build all bin images into a single HDF5")
    parser.add_argument("--grid-mapping", default="test_data/grid_mapping.json")
    parser.add_argument("--output", default=None,
                        help="Output HDF5 path (default: /home/takaji/xrd_{N}x{N}_bins.h5)")
    parser.add_argument("--bin-size", type=int, default=None,
                        help="Bin size (read from grid_mapping.json if not specified)")
    parser.add_argument("--compression", default="gzip",
                        choices=["gzip", "lz4", "none"],
                        help="Compression filter (default: gzip)")
    args = parser.parse_args()

    with open(args.grid_mapping) as f:
        gm = json.load(f)

    bin_size = args.bin_size or gm["bin_size"]
    if args.output is None:
        args.output = f"/home/takaji/xrd_{bin_size}x{bin_size}_bins.h5"

    bins = gm["bins"]
    frame_map = gm["frame_map"]
    xrd_files = gm["xrd_files"]
    h5_dataset = gm["h5_dataset"]
    n_bins = len(bins)

    comp_kwargs = get_compression_kwargs(args.compression)
    print(f"Building {n_bins} bin images ({bin_size}x{bin_size}) -> {args.output}")
    print(f"  Compression: {args.compression}")

    h5_handles = {}
    out = h5py.File(args.output, "w")
    out.attrs["bin_size"] = bin_size
    out.attrs["n_bin_rows"] = gm["n_bin_rows"]
    out.attrs["n_bin_cols"] = gm["n_bin_cols"]
    out.attrs["n_bins"] = n_bins
    out.attrs["detector_shape"] = [1062, 1028]

    t0 = time.time()
    for i, (bin_key, frame_indices) in enumerate(sorted(bins.items())):
        by_file = {}
        for gi in frame_indices:
            fi, fj = frame_map[gi]
            by_file.setdefault(fi, []).append(fj)

        summed = None
        for fi, frame_list in by_file.items():
            if fi not in h5_handles:
                h5_handles[fi] = h5py.File(xrd_files[fi], "r")
            ds = h5_handles[fi][h5_dataset]
            for fj in frame_list:
                frame = ds[fj].astype(np.float64)
                if summed is None:
                    summed = frame
                else:
                    summed += frame

        if summed is not None:
            summed[summed < 0] = 0
            summed[summed > 1e9] = 0
            out.create_dataset(bin_key, data=summed.astype(np.float32),
                               **comp_kwargs)

        if (i + 1) % 100 == 0 or (i + 1) == n_bins:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_bins - i - 1) / rate
            print(f"  [{i+1}/{n_bins}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    for fh in h5_handles.values():
        fh.close()
    out.close()

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nDone! {args.output}: {size_mb:.0f} MB ({size_mb/1024:.1f} GB)")


if __name__ == "__main__":
    main()
