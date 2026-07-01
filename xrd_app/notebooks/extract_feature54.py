#!/usr/bin/env python3
"""Extract feature #54 (Scan_0203, 3x3 faithful) into a small, self-contained HDF5
so the binning toy notebook runs on REAL data without touching the giant raw files.

What it pulls out, for the (001) reflection feature #54 (detector pixel ~917,446):
  * its real **spatial footprint** -- intensity of the (001) band at every scan
    position the feature lights up (this is the "image" the binning toy uses), and
  * a detector **ROI crop** around the spot for each member bin (the real (001)
    Bragg blob), so the notebook can show what one "scan" actually looks like.

Source of truth:
  Labels/Scan_0203/gaussian_shapes_3x3_faithful.json   -> feature #54 record
  Metadata/Scan_0203/grid_mapping_3x3_faithful.json    -> bins{R_C->[gi]}, frame_map
  Binned/Scan_0203/xrd_3x3_bins_faithful.h5            -> per-bin summed detector images

--source binned (default): read the local, already-summed 3x3 bin images. Fast.
--source raw            : read the raw per-frame HDF5s instead. NOTE: on this
    machine the raw Scan_0203 files are OneDrive cloud placeholders and error on
    read (errno 5) until hydrated ("Always keep on this device"). Use binned until
    then; the code path is here for when the raw frames are local.

Writes  xrd_app/notebooks/feature54_scans.h5  (a few tens of MB; do NOT commit).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import h5py

FEATURE_ID = 54
ROI_HALF = 48           # detector ROI is (2*ROI_HALF) square, centred on the spot


def _project_base(root: Path) -> Path:
    """Find the TakaProject tree (holds Labels/ Metadata/ Binned/)."""
    for cand in [root / "TakaTest" / "TakaProject", root]:
        if (cand / "Labels" / "Scan_0203").is_dir():
            return cand
    raise SystemExit(f"Could not find TakaProject/Labels/Scan_0203 under {root}")


def load_feature(base: Path, fid: int) -> dict:
    sj = base / "Labels" / "Scan_0203" / "gaussian_shapes_3x3_faithful.json"
    S = json.loads(sj.read_text())
    feat = next((f for f in S["kept"] if f.get("feature_id") == fid), None)
    if feat is None:
        raise SystemExit(f"feature_id {fid} not in {sj}")
    return feat


def roi_bounds(det_x: int, det_y: int, half: int, shape) -> tuple:
    """ROI as (r0, r1, c0, c1); array indexing is [row=y, col=x]. Clipped to frame."""
    H, W = shape
    r0, r1 = max(0, det_y - half), min(H, det_y + half)
    c0, c1 = max(0, det_x - half), min(W, det_x + half)
    return r0, r1, c0, c1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--source", choices=["binned", "raw"], default="binned")
    ap.add_argument("--roi-half", type=int, default=ROI_HALF)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    base = _project_base(root)
    out = Path(args.out) if args.out else (root / "xrd_app" / "notebooks" / "feature54_scans.h5")
    out.parent.mkdir(parents=True, exist_ok=True)

    feat = load_feature(base, FEATURE_ID)
    det_x, det_y = int(feat["detector_x"]), int(feat["detector_y"])
    ref_tth = float(feat["ref_tth"])
    prof = feat["intensity_profile"]            # {bin_key: {intensity, integrated, det_x, det_y, tth, chi}}
    members = list(prof.keys())
    print(f"feature #{FEATURE_ID}: {feat['reflection']} @ det ({det_x},{det_y}), "
          f"ref_tth={ref_tth}, {len(members)} member bins")

    gm = json.loads((base / "Metadata" / "Scan_0203" / "grid_mapping_3x3_faithful.json").read_text())
    bins_gi = gm["bins"]                        # bin_key -> [global frame indices]

    # footprint bounding box in BIN coordinates
    rs = [int(k.split("_")[0]) for k in members]
    cs = [int(k.split("_")[1]) for k in members]
    r_lo, r_hi, c_lo, c_hi = min(rs), max(rs), min(cs), max(cs)
    Hb, Wb = r_hi - r_lo + 1, c_hi - c_lo + 1
    print(f"footprint bin bbox: rows {r_lo}..{r_hi}, cols {c_lo}..{c_hi}  -> {Hb}x{Wb} grid")

    binned_h5 = base / "Binned" / "Scan_0203" / "xrd_3x3_bins_faithful.h5"

    # JSON-stored intensity per member bin (ground-truth footprint)
    foot_json = np.full((Hb, Wb), np.nan, np.float32)
    for k, rec in prof.items():
        r, c = (int(x) for x in k.split("_"))
        foot_json[r - r_lo, c - c_lo] = rec["intensity"]

    # measured footprint = ROI sum around the spot, read from real detector images
    foot_meas = np.full((Hb, Wb), np.nan, np.float32)
    member_rc, member_acq, crops = [], [], []

    if args.source == "binned":
        with h5py.File(binned_h5, "r") as h:
            shape = h[members[0]].shape if members[0] in h else (1062, 1028)
            r0, r1, c0, c1 = roi_bounds(det_x, det_y, args.roi_half, shape)
            for k in members:
                if k not in h:
                    continue
                crop = np.asarray(h[k][r0:r1, c0:c1], np.float32)
                r, c = (int(x) for x in k.split("_"))
                foot_meas[r - r_lo, c - c_lo] = float(crop.sum())
                member_rc.append((r, c))
                member_acq.append(int(min(bins_gi.get(k, [10**9]))))  # acq order proxy
                crops.append(crop)
    else:  # raw -- requires hydrated raw frames + 1x1 faithful per-frame coords
        raise SystemExit(
            "raw source not run: the raw Scan_0203 HDF5s are OneDrive placeholders "
            "(read -> errno 5). Hydrate raw_scans/Scan_0203/XRD/ first, then re-run "
            "with --source raw. Use --source binned in the meantime.")

    crops = np.asarray(crops, np.float32)
    member_rc = np.asarray(member_rc, np.int32)
    member_acq = np.asarray(member_acq, np.int64)
    r0, r1, c0, c1 = roi_bounds(det_x, det_y, args.roi_half, crops.shape[1:][::-1] if crops.ndim == 3 else (1062, 1028))

    # quick fidelity check: measured ROI sum vs stored intensity
    a = foot_meas[~np.isnan(foot_meas)]; b = foot_json[~np.isnan(foot_meas)]
    if a.size > 2:
        print(f"footprint corr(measured ROI-sum, stored intensity) = {np.corrcoef(a, b)[0,1]:.3f}")

    with h5py.File(out, "w") as o:
        o.create_dataset("footprint_measured", data=foot_meas, compression="gzip")
        o.create_dataset("footprint_intensity", data=foot_json, compression="gzip")
        o.create_dataset("member_bin_rc", data=member_rc)         # absolute bin (row,col)
        o.create_dataset("member_acq", data=member_acq)           # acquisition-order proxy (min gi)
        o.create_dataset("roi_stack", data=crops, compression="gzip")  # (N, h, w) real spot crops
        o.attrs.update(dict(feature_id=FEATURE_ID, reflection=str(feat["reflection"]),
                            detector_x=det_x, detector_y=det_y, ref_tth=ref_tth,
                            roi_r0=r0, roi_r1=r1, roi_c0=c0, roi_c1=c1,
                            bin_row_lo=r_lo, bin_col_lo=c_lo, bin_size=3,
                            n_members=len(member_rc), source=args.source, scan="Scan_0203"))
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(member_rc)} member bins, "
          f"roi {crops.shape[1]}x{crops.shape[2]})")


if __name__ == "__main__":
    main()
