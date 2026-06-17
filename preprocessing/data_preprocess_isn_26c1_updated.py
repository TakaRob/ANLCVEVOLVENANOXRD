#!/usr/bin/env python3
"""Preprocess diffraction patterns for ISN 26c1

Creates cropped/stacked diffraction dataset and saves positions and geometry.

Based on `process_dp_velo_21c1.py` but updated for the ISN 26c1 file layout.
"""
import hdf5plugin
import argparse
import glob
import h5py
import numpy as np
import os
import csv
import sys


def load_positions(path):
    # CSV format: Trigger, X_Position (um), Y_Position (um)
    ppX = []
    ppY = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            parts = ','.join(row).split(',')
            try:
                x =  float(parts[1]) * 1e-6   # X_Position, convert um to m
                y =  float(parts[2]) * 1e-6   # Y_Position, convert um to m
            except Exception:
                continue  # skips header row and any malformed lines
            ppY.append(y)
            ppX.append(x)
    ppX = np.array(ppX).reshape(-1, 1)
    ppY = np.array(ppY).reshape(-1, 1)
    return ppX, ppY


def main():
    p = argparse.ArgumentParser(description='Preprocess ISN 26c1 diffraction data')
    p.add_argument('scan', type=int, help='scan number to process (integer)')
    p.add_argument('--dataset', default='entry/data/data', help='h5 dataset path inside each file')
    p.add_argument('--det-npixel', type=int, default=256, help='final detector pixel size (pad/crop to this)')
    p.add_argument('--crop-center', type=int, nargs=2, metavar=('CX','CY'), help='center x,y for cropping (int)')
    p.add_argument('--crop-size', type=int, nargs=2, metavar=('SX','SY'), help='crop size (width,height)')
    args = p.parse_args()

    scanNo = args.scan

    data_dir = f'/net/micdata/data1/isn/2026-1/2026-1-Luo/Raw/Scan_{scanNo:04d}/PTYCHO'
    file_pattern = os.path.join(data_dir, f'scan_{scanNo:04d}_*.h5')
    files = sorted(glob.glob(file_pattern))
    if len(files) == 0:
        print('No diffraction files found for pattern:', file_pattern)
        sys.exit(1)

    print(f'Found {len(files)} files; reading first file to infer shape')
    with h5py.File(files[0], 'r') as h5f:
        if args.dataset not in h5f:
            # try common alternatives
            possible = list(h5f.keys())
            print('Dataset', args.dataset, 'not found. Top-level keys:', possible)
            # try to find a 2D dataset inside file
            def find_first_2d(group):
                for k,v in group.items():
                    if isinstance(v, h5py.Dataset) and v.ndim == 2:
                        return f"{k}"
                    if isinstance(v, h5py.Group):
                        sub = find_first_2d(v)
                        if sub:
                            return f"{k}/{sub}"
                return None
            found = find_first_2d(h5f)
            if found is None:
                print('Could not find a 2D dataset in file. Exiting.')
                sys.exit(1)
            args.dataset = found
            print('Using dataset:', args.dataset)

    # read first to get shape
    with h5py.File(files[0], 'r') as h5f:
        dp0 = h5f[args.dataset][()]
    if dp0.ndim == 2:
        n_per_file = 1
        full_h, full_w = dp0.shape
        print(f'2D dataset detected: {n_per_file} pattern per file, frame shape {full_h}x{full_w}')
    elif dp0.ndim == 3:
        n_per_file, full_h, full_w = dp0.shape
        print(f'3D dataset detected: {n_per_file} patterns per file, frame shape {full_h}x{full_w}')
    else:
        print('Unexpected dataset ndim; got', dp0.shape)
        sys.exit(1)

    # determine crop
    if args.crop_size:
        sx, sy = args.crop_size
    else:
        sx = sy = min(full_w, full_h, args.det_npixel)


    cx = 416-20
    cy = 796+20

    x_lb = int(cx - sx//2)
    x_ub = x_lb + sx
    y_lb = int(cy - sy//2)
    y_ub = y_lb + sy

    # prepare stack
    

    print(f'Cropping region x[{x_lb}:{x_ub}] y[{y_lb}:{y_ub}] -> resize/pad to {args.det_npixel}x{args.det_npixel}')

    N_total = len(files) * n_per_file
    dp_stack = np.zeros((N_total, args.det_npixel, args.det_npixel), dtype='float32')

    idx = 0
    for i, fn in enumerate(files):
        with h5py.File(fn, 'r') as h5f:
            arr = h5f[args.dataset][()]

        # normalise to 3D: (n_frames, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        for frame in arr:
            crop = frame[y_lb:y_ub, x_lb:x_ub]
            ch, cw = crop.shape
            target = np.zeros((args.det_npixel, args.det_npixel), dtype='float32')
            off_y = (args.det_npixel - ch) // 2
            off_x = (args.det_npixel - cw) // 2
            target[off_y:off_y + ch, off_x:off_x + cw] = crop.astype('float32')
            target[target < 0] = 0
            target[target > 1e7] = 0
            dp_stack[idx] = target
            idx += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f'  Read {i + 1}/{len(files)} files ({idx} patterns so far)')

    N = idx                  # actual number of patterns loaded
    dp_stack = dp_stack[:N]  # trim any pre-allocated excess
    print(f'Total patterns loaded: {N}')
    # load positions
    pos_file = f'/net/micdata/data1/isn/2026-1/2026-1-Luo/Processed/SOCKETSERVER/scan_{scanNo:04d}_position.csv'
    if os.path.exists(pos_file):
        ppX, ppY = load_positions(pos_file)
        print(f'Loaded {ppX.shape[0]} positions from {pos_file}')
        if ppX.shape[0] >= N:
            ppX = ppX[:N]
            ppY = ppY[:N]
        else:
            pad_n = N - ppX.shape[0]
            print(f'  Warning: only {ppX.shape[0]} positions for {N} patterns — zero-padding {pad_n} entries')
            ppX = np.vstack([ppX, np.zeros((pad_n, 1))])
            ppY = np.vstack([ppY, np.zeros((pad_n, 1))])
    else:
        print('Position file not found:', pos_file)
        ppX = np.zeros((N, 1))
        ppY = np.zeros((N, 1))

    # geometry constants (copied from original; adapt if needed)
    energy = 15
    det_sample_dist = 6.16
    det_pixel_size = 75e-6
    lam = 1.23984193e-9/energy
    dx = lam*det_sample_dist/det_pixel_size/args.det_npixel

    # output
    out_dir = '/net/micdata/data1/isn/2026-1/2026-1-Luo/results'
    os.makedirs(out_dir, exist_ok=True)
    scan_dir = os.path.join(out_dir, f"scan{scanNo:03d}")
    os.makedirs(scan_dir, exist_ok=True)

    roi = f'1_Ndp{args.det_npixel}'
    dp_name = os.path.join(scan_dir, f'data_roi{roi}_dp.hdf5')
    para_name = os.path.join(scan_dir, f'data_roi{roi}_para.hdf5')

    print('Saving diffraction stack to', dp_name)
    with h5py.File(dp_name, 'w') as f:
        f.create_dataset('dp', data=dp_stack, compression='gzip')

    print('Saving parameters to', para_name)
    with h5py.File(para_name, 'w') as f:
        f.create_dataset('lambda', data=np.array([lam], dtype='float64'))
        f.create_dataset('dx', data=np.array([dx], dtype='float64'))
        f.create_dataset('ppY', data=ppY.astype('float64'))
        f.create_dataset('ppX', data=ppX.astype('float64'))
        f.create_dataset('N_files', data=np.array([N], dtype='int'))

    print(f'ppX: {ppX[100]}')
    print(f'ppY: {ppY[100]}')
    print('Done.')


if __name__ == '__main__':
    main()
