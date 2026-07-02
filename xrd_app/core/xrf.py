"""X-ray fluorescence (XRF) element maps from the ME7 XSPRESS3 detector.

XRF is collected simultaneously with the XRD frames and lives in the scan's
``ME7/`` directory — one ``scan_NNNN_MMMMM.h5`` per scan row, dataset
``entry/data/data`` of shape ``(points, 7 channels, 4096 MCA bins)`` (uint32).
Because it is the *same* raster as the XRD, an XRF point maps onto the *same*
de-skewed ``(row, col)`` bin as the corresponding XRD frame — so this module
reuses the existing ``grid_mapping`` for registration (no re-interpolation),
giving element maps pixel-aligned to the XRD device maps.

Pipeline per point: deadtime-correct each channel (``× CHAN{n}DTFactor``), sum the
enabled channels into one MCA spectrum, then integrate energy-ROI windows (one per
element) into scalar element intensities. Energy is calibrated linearly
(``eV = bin × ev_per_bin + offset_ev``) and ROIs are defined by physical emission
energy ± a window — **no hardcoded bin indices**. Points are accumulated into grid
bins via the ``(file_index, local_index) → bin`` map derived from the grid
mapping's ``frame_map`` (robust to the small per-file frame-count differences
between XRD and ME7).

Config (channels, calibration, elements) is a small JSON, defaulting to the
perovskite/halide element set below; see :func:`default_config`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

H5_DATASET = "entry/data/data"
DT_ATTR = "entry/instrument/NDAttributes/CHAN{ch1}DTFactor"  # ch1 is 1-indexed
N_CHANNELS = 7          # ME7 = 7-element XSPRESS3
N_BINS = 4096           # MCA length
DEFAULT_EV_PER_BIN = 10.0
DEFAULT_OFFSET_EV = 0.0
DEFAULT_HALF_WIDTH_EV = 150.0

# Standard emission lines (eV) excited at 15 keV for these perovskite/halide
# samples. I/Sn/Cs K lines are above the excitation energy, so their L lines are
# used; Br K and Pb/Au L lines are directly excited. Adjust half_width per line.
DEFAULT_ELEMENTS = [
    {"name": "Pb_La", "line_ev": 10551.0, "half_width_ev": DEFAULT_HALF_WIDTH_EV},
    {"name": "I_La",  "line_ev": 3938.0,  "half_width_ev": DEFAULT_HALF_WIDTH_EV},
    {"name": "Sn_La", "line_ev": 3444.0,  "half_width_ev": DEFAULT_HALF_WIDTH_EV},
    {"name": "Cs_La", "line_ev": 4286.0,  "half_width_ev": DEFAULT_HALF_WIDTH_EV},
    {"name": "Br_Ka", "line_ev": 11924.0, "half_width_ev": DEFAULT_HALF_WIDTH_EV},
    {"name": "Au_La", "line_ev": 9713.0,  "half_width_ev": DEFAULT_HALF_WIDTH_EV},
]


# ─────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────
# Defaults for the data-driven ROI refinement (see refine_rois). The incident
# beam is 15 keV; the elastic + Compton scatter sits just below/at that energy and
# must be ignored by the peak finder, as must the extreme low-energy noise floor.
DEFAULT_REFINEMENT = {
    "prominence_frac": 0.01,     # peak prominence as a fraction of the spectrum max
    "min_width_bins": 2,         # minimum peak width (bins)
    "match_tol_ev": 200.0,       # max |observed − theoretical| to accept a match
    "noise_floor_ev": 1200.0,    # ignore peaks below this energy
    "incident_ev": 15000.0,      # elastic/Compton scatter region to ignore …
    "exclude_incident_lo_ev": 800.0,   # … from incident−lo …
    "exclude_incident_hi_ev": 400.0,   # … to incident+hi
}


def default_config() -> dict:
    """Fresh default XRF config (all channels on, 10 eV/bin, standard lines).

    Each element takes a symmetric ``half_width_ev`` or an asymmetric
    ``window_ev: [lo_offset, hi_offset]`` (signed eV around the center) to trim
    crosstalk between neighbours (e.g. the I/Sn/Cs cluster at 3.4–4.3 keV).
    """
    return {
        "detector": "ME7",
        "channels": list(range(N_CHANNELS)),   # enabled channel indices (0-based)
        "deadtime_correction": True,
        "calibration": {"ev_per_bin": DEFAULT_EV_PER_BIN,
                        "offset_ev": DEFAULT_OFFSET_EV},
        "refinement": dict(DEFAULT_REFINEMENT),
        "elements": [dict(e) for e in DEFAULT_ELEMENTS],
    }


def read_config(path) -> dict:
    """Read an XRF config JSON, filling any missing keys from the default."""
    cfg = default_config()
    p = Path(path)
    if p.exists():
        with open(p) as f:
            user = json.load(f)
        cfg.update({k: user[k] for k in user})
    return cfg


def write_config(cfg: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def energy_to_bin_range(line_ev: float, half_width_ev: float,
                        ev_per_bin: float, offset_ev: float,
                        n_bins: int = N_BINS):
    """Convert an emission line ± window (eV) to an inclusive-exclusive bin slice.

    ``eV = bin × ev_per_bin + offset_ev`` → ``bin = (eV − offset) / ev_per_bin``.
    Returned ``(lo, hi)`` is clipped to ``[0, n_bins]`` with ``lo < hi``.
    """
    lo = (float(line_ev) - float(half_width_ev) - offset_ev) / ev_per_bin
    hi = (float(line_ev) + float(half_width_ev) - offset_ev) / ev_per_bin
    lo_b = int(max(0, np.floor(lo)))
    hi_b = int(min(n_bins, np.ceil(hi)))
    if hi_b <= lo_b:  # window entirely outside the spectrum → empty but valid
        hi_b = min(n_bins, lo_b + 1)
    return lo_b, hi_b


def roi_bins(center_ev: float, element: dict, ev_per_bin: float,
             offset_ev: float, n_bins: int = N_BINS):
    """Bin slice for an element's window, centered on ``center_ev``.

    Uses the element's asymmetric ``window_ev = [lo_off, hi_off]`` (signed eV) if
    present, else the symmetric ``half_width_ev``. Returns
    ``(lo_bin, hi_bin, lo_ev, hi_ev)`` with the eV edges snapped to the bin grid.
    """
    win = element.get("window_ev")
    if win is not None and len(win) == 2:
        lo_ev, hi_ev = center_ev + float(win[0]), center_ev + float(win[1])
    else:
        hw = float(element.get("half_width_ev", DEFAULT_HALF_WIDTH_EV))
        lo_ev, hi_ev = center_ev - hw, center_ev + hw
    lo = int(max(0, np.floor((lo_ev - offset_ev) / ev_per_bin)))
    hi = int(min(n_bins, np.ceil((hi_ev - offset_ev) / ev_per_bin)))
    if hi <= lo:
        hi = min(n_bins, lo + 1)
    return lo, hi, lo * ev_per_bin + offset_ev, hi * ev_per_bin + offset_ev


# ─────────────────────────────────────────────────────────────────────
# file discovery + point→bin registration
# ─────────────────────────────────────────────────────────────────────
def me7_files(me7_dir) -> List[Path]:
    """Sorted list of ME7 per-row HDF5 files (``scan_*.h5``)."""
    return sorted(Path(me7_dir).glob("scan_*.h5"))


def fileloc_to_bin(grid_mapping: dict) -> Dict[tuple, str]:
    """Map ``(file_index, local_index) → bin_key`` from a grid mapping.

    The grid mapping assigns each *global* XRD frame to a bin (``bins``) and
    records each frame's ``(file_index, local_index)`` (``frame_map``). Composing
    them keys the assignment by (file, local) so ME7 points — which share the raw
    one-file-per-row layout — register to the same de-skewed bins even though XRD
    and ME7 have slightly different per-file frame counts.
    """
    bins = grid_mapping["bins"]
    frame_map = grid_mapping["frame_map"]
    frame_bin: Dict[int, str] = {}
    for bk, frames in bins.items():
        for g in frames:
            frame_bin[int(g)] = bk
    out: Dict[tuple, str] = {}
    for g, fl in enumerate(frame_map):
        bk = frame_bin.get(g)
        if bk is not None:
            out[(int(fl[0]), int(fl[1]))] = bk
    return out


def _bin_key_to_rc(bk: str):
    r, c = bk.split("_")
    return int(r), int(c)


# ─────────────────────────────────────────────────────────────────────
# spectra → element intensities
# ─────────────────────────────────────────────────────────────────────
def _summed_spectra(h5, channels: Sequence[int], deadtime: bool) -> np.ndarray:
    """Deadtime-correct + channel-sum one ME7 file → ``(points, n_bins)`` float64.

    ``corrected = raw × CHAN{n}DTFactor`` per channel per point (XSPRESS3
    convention; DTFactor ≥ 1), then the enabled channels are summed.
    """
    data = h5[H5_DATASET]  # (points, N_CHANNELS, n_bins), uint32
    n_pts = data.shape[0]
    acc = np.zeros((n_pts, data.shape[2]), dtype=np.float64)
    for ch in channels:
        spec = data[:, ch, :].astype(np.float64)
        if deadtime:
            attr = DT_ATTR.format(ch1=ch + 1)
            if attr in h5:
                fac = np.asarray(h5[attr][:], dtype=np.float64)
                spec *= fac[:, None]
        acc += spec
    return acc


# ─────────────────────────────────────────────────────────────────────
# data-driven ROI refinement (correct minor detector/energy drift)
# ─────────────────────────────────────────────────────────────────────
def grand_sum_spectrum(me7_dir, channels: Sequence[int], deadtime: bool = True,
                       max_files: Optional[int] = None,
                       log: Callable[[str], None] = lambda *_: None) -> np.ndarray:
    """Master MCA spectrum: enabled channels summed over *all* spatial points.

    Accumulated file-by-file (memory-light). ``max_files`` subsamples rows for a
    faster spectrum when exactness isn't needed.
    """
    import h5py
    files = me7_files(me7_dir)
    if not files:
        raise FileNotFoundError(f"No ME7 files (scan_*.h5) in {me7_dir}")
    if max_files and max_files < len(files):
        step = max(1, len(files) // max_files)
        files = files[::step]
    acc = np.zeros(N_BINS, dtype=np.float64)
    for fp in files:
        with h5py.File(fp, "r") as h5:
            if H5_DATASET in h5:
                acc += _summed_spectra(h5, channels, deadtime).sum(axis=0)
    log(f"  grand sum over {len(files)} files")
    return acc


def find_spectrum_peaks(spectrum: np.ndarray, ev_per_bin: float, offset_ev: float,
                        prominence_frac: float = 0.01, min_width_bins: float = 2,
                        noise_floor_ev: float = 1200.0, incident_ev: float = 15000.0,
                        exclude_incident_lo_ev: float = 800.0,
                        exclude_incident_hi_ev: float = 400.0):
    """Locate emission peaks in the master spectrum with ``scipy.signal.find_peaks``.

    The extreme low-energy noise floor and the elastic/Compton scatter band around
    the incident energy are masked out first so the finder returns only true
    fluorescence lines. Returns ``(peak_bins, prominences)``.
    """
    from scipy.signal import find_peaks

    spec = np.asarray(spectrum, dtype=np.float64).copy()
    ev = np.arange(spec.size) * ev_per_bin + offset_ev
    mask = ev < noise_floor_ev
    mask |= (ev > incident_ev - exclude_incident_lo_ev) & \
            (ev < incident_ev + exclude_incident_hi_ev)
    spec[mask] = 0.0
    peak = float(spec.max())
    if peak <= 0:
        return np.array([], dtype=int), np.array([])
    peaks, props = find_peaks(spec, prominence=prominence_frac * peak,
                              width=min_width_bins)
    return peaks, props.get("prominences", np.array([]))


def match_lines_to_peaks(peak_bins, elements, ev_per_bin: float, offset_ev: float,
                         tol_ev: float = 200.0) -> dict:
    """Match each theoretical element line to the nearest observed peak within
    ``tol_ev``. Returns ``{name: {observed_ev, observed_bin, shift_ev} | None}``."""
    peak_bins = np.asarray(peak_bins)
    peak_ev = peak_bins * ev_per_bin + offset_ev
    out = {}
    for el in elements:
        line = float(el["line_ev"])
        if peak_ev.size == 0:
            out[el["name"]] = None
            continue
        i = int(np.argmin(np.abs(peak_ev - line)))
        if abs(peak_ev[i] - line) <= tol_ev:
            out[el["name"]] = {"observed_ev": float(peak_ev[i]),
                               "observed_bin": int(peak_bins[i]),
                               "shift_ev": float(peak_ev[i] - line)}
        else:
            out[el["name"]] = None
    return out


def detect_roi_overlaps(rois: dict) -> List[tuple]:
    """Return element-name pairs whose ``[lo_bin, hi_bin)`` windows overlap."""
    items = sorted(rois.items(), key=lambda kv: kv[1]["lo_bin"])
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a["lo_bin"] < b["hi_bin"] and b["lo_bin"] < a["hi_bin"]:
                out.append((items[i][0], items[j][0]))
    return out


def refine_rois(me7_dir, config: dict,
                log: Callable[[str], None] = lambda *_: None):
    """Data-driven ROI refinement: center each element on its observed peak.

    Builds the grand-sum spectrum, finds peaks (ignoring the noise floor + the
    elastic/Compton band), matches them to the theoretical lines, and injects the
    observed centroid as ``observed_ev`` on each matched element. Returns
    ``(refined_config, diagnostics)`` where diagnostics has per-element shift info
    and any ROI ``overlaps`` (e.g. the tight I/Sn/Cs cluster).
    """
    cal = config["calibration"]
    evpb, off = float(cal["ev_per_bin"]), float(cal["offset_ev"])
    channels = list(config.get("channels", list(range(N_CHANNELS))))
    deadtime = bool(config.get("deadtime_correction", True))
    ref = {**DEFAULT_REFINEMENT, **config.get("refinement", {})}

    spec = grand_sum_spectrum(me7_dir, channels, deadtime, log=log)
    peaks, _ = find_spectrum_peaks(
        spec, evpb, off,
        prominence_frac=ref["prominence_frac"], min_width_bins=ref["min_width_bins"],
        noise_floor_ev=ref["noise_floor_ev"], incident_ev=ref["incident_ev"],
        exclude_incident_lo_ev=ref["exclude_incident_lo_ev"],
        exclude_incident_hi_ev=ref["exclude_incident_hi_ev"])
    matches = match_lines_to_peaks(peaks, config["elements"], evpb, off,
                                   tol_ev=ref["match_tol_ev"])

    new_elements, diag_els, rois = [], [], {}
    for el in config["elements"]:
        m = matches.get(el["name"])
        e2 = dict(el)
        if m:
            e2["observed_ev"] = m["observed_ev"]
            e2["matched"] = True
        else:
            e2["matched"] = False
        new_elements.append(e2)
        center = float(e2.get("observed_ev", el["line_ev"]))
        lo, hi, _, _ = roi_bins(center, e2, evpb, off)
        rois[el["name"]] = {"lo_bin": lo, "hi_bin": hi}
        diag_els.append({
            "name": el["name"], "line_ev": float(el["line_ev"]),
            "observed_ev": (m["observed_ev"] if m else None),
            "shift_ev": (m["shift_ev"] if m else None),
            "matched": bool(m), "lo_bin": lo, "hi_bin": hi})

    refined = dict(config)
    refined["elements"] = new_elements
    overlaps = detect_roi_overlaps(rois)
    diagnostics = {"n_peaks_found": int(len(peaks)),
                   "elements": diag_els, "overlaps": overlaps}
    return refined, diagnostics


# ─────────────────────────────────────────────────────────────────────
# main entry: element maps on the grid
# ─────────────────────────────────────────────────────────────────────
def element_maps(me7_dir, grid_mapping: dict, config: Optional[dict] = None,
                 log: Callable[[str], None] = lambda *_: None) -> dict:
    """Build per-element spatial maps on the grid from ME7 XRF.

    Returns a dict with ``elements`` (names), ``maps`` (``{name: (nr, nc)}`` summed
    ROI intensity), ``n_points`` (``(nr, nc)`` contributing points per bin),
    ``rois`` (``{name: {line_ev, lo_ev, hi_ev, lo_bin, hi_bin}}``), ``shape``,
    ``channels``, ``deadtime`` and ``dropped`` (points with no bin).
    """
    cfg = config or default_config()
    cal = cfg["calibration"]
    ev_per_bin = float(cal["ev_per_bin"])
    offset_ev = float(cal["offset_ev"])
    channels = list(cfg.get("channels", list(range(N_CHANNELS))))
    deadtime = bool(cfg.get("deadtime_correction", True))
    elements = cfg["elements"]

    nr = int(grid_mapping["n_bin_rows"])
    nc = int(grid_mapping["n_bin_cols"])
    loc_bin = fileloc_to_bin(grid_mapping)

    # precompute ROI bin slices — centered on the observed (refined) energy when
    # refine_rois has injected "observed_ev", else the theoretical line.
    rois = {}
    for el in elements:
        line = float(el["line_ev"])
        center = float(el.get("observed_ev", line))
        lo, hi, lo_ev, hi_ev = roi_bins(center, el, ev_per_bin, offset_ev)
        rois[el["name"]] = {"line_ev": line, "center_ev": center,
                            "shift_ev": center - line,
                            "matched": bool(el.get("matched", False)),
                            "lo_ev": lo_ev, "hi_ev": hi_ev,
                            "lo_bin": lo, "hi_bin": hi}

    maps = {el["name"]: np.zeros((nr, nc), dtype=np.float64) for el in elements}
    npoints = np.zeros((nr, nc), dtype=np.int64)

    import h5py
    files = me7_files(me7_dir)
    if not files:
        raise FileNotFoundError(f"No ME7 files (scan_*.h5) in {me7_dir}")
    dropped = 0
    for fi, fp in enumerate(files):
        with h5py.File(fp, "r") as h5:
            if H5_DATASET not in h5:
                log(f"  {fp.name}: no {H5_DATASET}; skipped")
                continue
            spectra = _summed_spectra(h5, channels, deadtime)  # (points, n_bins)
        # per-element ROI sums for every point in this file
        el_vals = {name: spectra[:, r["lo_bin"]:r["hi_bin"]].sum(axis=1)
                   for name, r in rois.items()}
        for li in range(spectra.shape[0]):
            bk = loc_bin.get((fi, li))
            if bk is None:
                dropped += 1
                continue
            r, c = _bin_key_to_rc(bk)
            if not (0 <= r < nr and 0 <= c < nc):
                dropped += 1
                continue
            for name in maps:
                maps[name][r, c] += el_vals[name][li]
            npoints[r, c] += 1
        log(f"  {fp.name}: {spectra.shape[0]} points")

    return {
        "elements": [el["name"] for el in elements],
        "maps": maps,
        "n_points": npoints,
        "rois": rois,
        "shape": (nr, nc),
        "channels": channels,
        "deadtime": deadtime,
        "calibration": {"ev_per_bin": ev_per_bin, "offset_ev": offset_ev},
        "dropped": dropped,
    }


def save_npz(path, result: dict) -> Path:
    """Write element maps + metadata to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "elements": np.array(result["elements"]),
        "n_points": result["n_points"],
        "n_rows": result["shape"][0], "n_cols": result["shape"][1],
        "channels": np.array(result["channels"]),
        "deadtime": bool(result["deadtime"]),
        "ev_per_bin": result["calibration"]["ev_per_bin"],
        "offset_ev": result["calibration"]["offset_ev"],
        "dropped": int(result["dropped"]),
    }
    for name, m in result["maps"].items():
        payload[f"map_{name}"] = m.astype(np.float32)
    for name, r in result["rois"].items():
        payload[f"roi_{name}"] = np.array(
            [r["line_ev"], r.get("center_ev", r["line_ev"]),
             r["lo_ev"], r["hi_ev"], r["lo_bin"], r["hi_bin"],
             float(r.get("matched", False))], float)
    np.savez_compressed(path, **payload)
    return path


def summary(result: dict) -> dict:
    """Small JSON-able summary (per-element intensity totals + coverage)."""
    nr, nc = result["shape"]
    filled = int(np.count_nonzero(result["n_points"]))
    return {
        "shape": [nr, nc],
        "channels": list(result["channels"]),
        "deadtime_correction": bool(result["deadtime"]),
        "calibration": result["calibration"],
        "dropped_points": int(result["dropped"]),
        "bins_filled": filled,
        "fill_fraction": filled / float(nr * nc) if nr * nc else 0.0,
        "elements": {
            name: {
                "line_ev": result["rois"][name]["line_ev"],
                "center_ev": result["rois"][name].get("center_ev"),
                "shift_ev": result["rois"][name].get("shift_ev"),
                "matched": result["rois"][name].get("matched"),
                "roi_ev": [result["rois"][name]["lo_ev"], result["rois"][name]["hi_ev"]],
                "total": float(result["maps"][name].sum()),
                "max_bin": float(result["maps"][name].max()),
            }
            for name in result["elements"]
        },
    }
