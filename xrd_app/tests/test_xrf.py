"""Headless tests for ME7 XSPRESS3 XRF → element maps (core/xrf.py).

Synthetic ME7 HDF5 files (with the real dataset + deadtime attributes) exercise
the actual load/deadtime/channel-sum/ROI/registration paths — no network data.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from xrd_app.core import xrf


def _write_me7(path, spectra, dtfactors=None):
    """Write a synthetic ME7 file. spectra: (P, 7, 4096). dtfactors: (7, P) or None."""
    P = spectra.shape[0]
    with h5py.File(path, "w") as f:
        f.create_dataset(xrf.H5_DATASET, data=spectra.astype(np.uint32))
        for ch in range(xrf.N_CHANNELS):
            fac = np.ones(P) if dtfactors is None else np.asarray(dtfactors[ch])
            f.create_dataset(xrf.DT_ATTR.format(ch1=ch + 1), data=fac.astype(np.float64))


def _spike(P, channel, bin_idx, value):
    """(P,7,4096) zeros with `value` at one MCA bin of one channel for all points."""
    s = np.zeros((P, xrf.N_CHANNELS, xrf.N_BINS), dtype=np.uint32)
    s[:, channel, bin_idx] = value
    return s


def _gauss(P, channel, center_bin, amp, sigma=2.0):
    """(P,7,4096) with a realistic ~few-bin-wide Gaussian line (find_peaks needs width≥2)."""
    s = np.zeros((P, xrf.N_CHANNELS, xrf.N_BINS), dtype=np.uint32)
    bins = np.arange(xrf.N_BINS)
    prof = (amp * np.exp(-0.5 * ((bins - center_bin) / sigma) ** 2)).astype(np.uint32)
    s[:, channel, :] = prof
    return s


# grid mapping: 2 files × 2 points → a 1×2 grid (col 0 = file/point pairs 0-1, col 1 = 2-3)
def _grid_mapping():
    return {
        "n_bin_rows": 1, "n_bin_cols": 2,
        "bins": {"0_0": [0, 1], "0_1": [2, 3]},
        "frame_map": [[0, 0], [0, 1], [1, 0], [1, 1]],
    }


def test_energy_to_bin_range():
    lo, hi = xrf.energy_to_bin_range(10000.0, 50.0, ev_per_bin=10.0, offset_ev=0.0)
    assert (lo, hi) == (995, 1005)  # 9950..10050 eV / 10
    # offset shifts the window
    lo2, hi2 = xrf.energy_to_bin_range(10000.0, 50.0, ev_per_bin=10.0, offset_ev=100.0)
    assert (lo2, hi2) == (985, 995)
    # clipped to [0, N_BINS]
    lo3, hi3 = xrf.energy_to_bin_range(50.0, 100.0, 10.0, 0.0)
    assert lo3 == 0 and hi3 >= 1


def test_position_offset_uses_nearest_then_interpolates(tmp_path):
    path = tmp_path / "position_offset.json"
    path.write_text(json.dumps({"theta": [2, 0, 4], "y_offset": [-20, 0, -60]}))

    assert xrf.position_offset_at_theta(path, 2.005) == -20
    assert xrf.position_offset_at_theta(path, 3.0) == -40


def test_fileloc_to_bin():
    m = xrf.fileloc_to_bin(_grid_mapping())
    assert m[(0, 0)] == "0_0" and m[(0, 1)] == "0_0"
    assert m[(1, 0)] == "0_1" and m[(1, 1)] == "0_1"


def _cfg(channels=None, deadtime=False, line_ev=10000.0, hw=50.0):
    c = xrf.default_config()
    c["deadtime_correction"] = deadtime
    if channels is not None:
        c["channels"] = channels
    c["elements"] = [{"name": "Test", "line_ev": line_ev, "half_width_ev": hw}]
    return c


def test_element_maps_accumulates_to_bins(tmp_path):
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, channel=0, bin_idx=1000, value=100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, channel=0, bin_idx=1000, value=100))
    res = xrf.element_maps(tmp_path, _grid_mapping(), _cfg(deadtime=False))
    m = res["maps"]["Test"]
    assert m.shape == (1, 2)
    # 2 points/bin × 100 counts each = 200 per bin
    assert np.allclose(m, [[200.0, 200.0]])
    assert np.array_equal(res["n_points"], [[2, 2]])
    assert res["dropped"] == 0


def test_deadtime_scales_counts(tmp_path):
    dt = np.full((7, 2), 1.0)
    dt[0] = 2.0  # channel 0 deadtime factor ×2
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100), dtfactors=dt)
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 100), dtfactors=dt)
    on = xrf.element_maps(tmp_path, _grid_mapping(), _cfg(deadtime=True))["maps"]["Test"]
    off = xrf.element_maps(tmp_path, _grid_mapping(), _cfg(deadtime=False))["maps"]["Test"]
    assert np.allclose(on, off * 2.0)


def test_channel_mask_excludes_dead_channel(tmp_path):
    # counts only in channel 3
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, channel=3, bin_idx=1000, value=50))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, channel=3, bin_idx=1000, value=50))
    gm = _grid_mapping()
    with_ch3 = xrf.element_maps(tmp_path, gm, _cfg(channels=[0, 1, 2, 3]))["maps"]["Test"]
    without = xrf.element_maps(tmp_path, gm, _cfg(channels=[0, 1, 2]))["maps"]["Test"]
    assert with_ch3.sum() == 200.0     # 2 files × 2 pts × 50
    assert without.sum() == 0.0


def test_points_without_a_bin_are_dropped(tmp_path):
    # file 1 exists but the grid only maps file 0 → file-1 points dropped
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 100))
    gm = {"n_bin_rows": 1, "n_bin_cols": 1,
          "bins": {"0_0": [0, 1]}, "frame_map": [[0, 0], [0, 1]]}
    res = xrf.element_maps(tmp_path, gm, _cfg())
    assert res["maps"]["Test"].sum() == 200.0  # only file 0's 2 points
    assert res["dropped"] == 2                  # file 1's 2 points


def test_config_roundtrip_and_defaults(tmp_path):
    cfg = xrf.default_config()
    names = [e["name"] for e in cfg["elements"]]
    assert names == ["Pb_La", "I_La", "Sn_La", "Cs_La", "Br_Ka", "Au_La"]
    assert cfg["calibration"]["ev_per_bin"] == 10.0
    assert cfg["channels"] == list(range(7))
    p = tmp_path / "xrf_elements.json"
    xrf.write_config(cfg, p)
    back = xrf.read_config(p)
    assert [e["name"] for e in back["elements"]] == names


def test_roi_bins_asymmetric_window():
    # window_ev [-50, +20] around 4000 eV at 10 eV/bin
    el = {"name": "X", "line_ev": 4000.0, "window_ev": [-50.0, 20.0]}
    lo, hi, lo_ev, hi_ev = xrf.roi_bins(4000.0, el, 10.0, 0.0)
    assert lo == 395 and hi == 402          # 3950..4020 eV
    # symmetric fallback when no window_ev
    el2 = {"name": "Y", "line_ev": 4000.0, "half_width_ev": 100.0}
    lo2, hi2, _, _ = xrf.roi_bins(4000.0, el2, 10.0, 0.0)
    assert lo2 == 390 and hi2 == 410


def test_find_peaks_excludes_noise_floor_and_incident():
    bins = np.arange(xrf.N_BINS)
    def g(center, amp, sigma=2.0):
        return amp * np.exp(-0.5 * ((bins - center) / sigma) ** 2)
    spec = g(100, 1000) + g(1000, 800) + g(1500, 5000)  # noise / line / elastic
    peaks, _ = xrf.find_spectrum_peaks(spec, 10.0, 0.0, prominence_frac=0.02)
    assert 1000 in peaks                       # the real line (10000 eV) is kept
    assert not any(p < 120 for p in peaks)     # noise floor (<1200 eV) excluded
    assert not any(1450 < p < 1550 for p in peaks)  # elastic/Compton excluded


def test_match_lines_to_peaks_within_tolerance():
    # observed peak at bin 1057 = 10570 eV; Pb line 10551 → within 200 eV
    m = xrf.match_lines_to_peaks([1057, 394], [{"name": "Pb_La", "line_ev": 10551.0},
                                               {"name": "I_La", "line_ev": 3938.0}],
                                 10.0, 0.0, tol_ev=200.0)
    assert m["Pb_La"]["observed_ev"] == 10570.0
    assert abs(m["Pb_La"]["shift_ev"] - 19.0) < 1e-6
    # I line 3938 vs nearest peak 3940 → matched
    assert m["I_La"] is not None
    # nothing within tolerance
    m2 = xrf.match_lines_to_peaks([100], [{"name": "Pb_La", "line_ev": 10551.0}], 10.0, 0.0)
    assert m2["Pb_La"] is None


def test_detect_roi_overlaps_flags_the_cluster():
    rois = {
        "Sn_La": {"lo_bin": 329, "hi_bin": 360},
        "I_La":  {"lo_bin": 355, "hi_bin": 385},   # overlaps Sn (355<360)
        "Cs_La": {"lo_bin": 413, "hi_bin": 444},   # separate
        "Pb_La": {"lo_bin": 1040, "hi_bin": 1071},
    }
    ov = xrf.detect_roi_overlaps(rois)
    assert ("Sn_La", "I_La") in ov
    assert not any("Cs_La" in pair for pair in ov)


def test_refine_rois_centers_on_observed_peak(tmp_path):
    # grand sum will have a peak at bin 1057 (10570 eV), 19 eV above Pb line 10551
    _write_me7(tmp_path / "scan_0001_00001.h5", _gauss(3, channel=0, center_bin=1057, amp=500))
    cfg = xrf.default_config()
    cfg["deadtime_correction"] = False
    cfg["elements"] = [{"name": "Pb_La", "line_ev": 10551.0, "half_width_ev": 150.0}]
    cfg["refinement"] = {**xrf.DEFAULT_REFINEMENT, "prominence_frac": 0.05}
    refined, diag = xrf.refine_rois(tmp_path, cfg)
    el = refined["elements"][0]
    assert el["matched"] is True
    assert el["observed_ev"] == 10570.0          # centered on the observed peak
    assert diag["elements"][0]["shift_ev"] == 19.0
    assert diag["overlaps"] == []


def test_save_npz_and_summary(tmp_path):
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 100))
    res = xrf.element_maps(tmp_path, _grid_mapping(), _cfg())
    out = tmp_path / "xrf.npz"
    xrf.save_npz(out, res)
    d = np.load(out, allow_pickle=False)
    assert d["map_Test"].shape == (1, 2)
    assert list(d["elements"]) == [b"Test"] or list(d["elements"].astype(str)) == ["Test"]
    s = xrf.summary(res)
    assert s["elements"]["Test"]["total"] == 400.0
    assert s["shape"] == [1, 2]


def test_element_maps_accumulates_grand_sum_spectrum(tmp_path):
    # 2 files × 2 points, spike value 100 at bin 1000 → grand sum = 4×100 at bin 1000
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 100))
    res = xrf.element_maps(tmp_path, _grid_mapping(), _cfg())
    spec = res["spectrum"]
    assert spec.shape == (xrf.N_BINS,)
    assert spec[1000] == 400.0
    assert spec.sum() == 400.0


def test_point_spectrum_sums_only_that_bins_frames(tmp_path):
    # 2 files × 2 points. bins: 0_0→globals[0,1] (file0), 0_1→globals[2,3] (file1).
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 50))
    gm = _grid_mapping()
    s00 = xrf.point_spectrum(tmp_path, gm, "0_0", [0, 1, 2], deadtime=False)
    s01 = xrf.point_spectrum(tmp_path, gm, "0_1", [0, 1, 2], deadtime=False)
    assert s00[1000] == 200.0      # 2 points × 100 (file 0)
    assert s01[1000] == 100.0      # 2 points × 50  (file 1)
    # a bin with no frames returns None (not an all-zero spectrum)
    assert xrf.point_spectrum(tmp_path, gm, "9_9", [0], deadtime=False) is None


def test_point_spectrum_respects_deadtime(tmp_path):
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100),
               dtfactors=[[2.0, 2.0]] + [[1.0, 1.0]] * 6)
    gm = _grid_mapping()
    on = xrf.point_spectrum(tmp_path, gm, "0_0", [0], deadtime=True)
    off = xrf.point_spectrum(tmp_path, gm, "0_0", [0], deadtime=False)
    assert on[1000] == 2.0 * off[1000]


def test_point_store_matches_raw_and_roundtrips(tmp_path):
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 50))
    gm = _grid_mapping()
    store = xrf.build_point_store(tmp_path, gm, [0, 1, 2], deadtime=False)
    assert store.shape == (4, xrf.N_BINS)   # 4 global frames
    # store lookup == raw ME7 lookup, for every bin
    for bk in ("0_0", "0_1"):
        from_store = xrf.point_spectrum_from_store(store, gm, bk)
        from_raw = xrf.point_spectrum(tmp_path, gm, bk, [0, 1, 2], deadtime=False)
        assert np.array_equal(from_store, from_raw)
    # save/load roundtrip (values preserved even if downcast to uint16)
    p = tmp_path / "x_points.npz"
    xrf.save_point_store(p, store, [0, 1, 2], False,
                         {"ev_per_bin": 10.0, "offset_ev": 0.0})
    d = xrf.load_point_store(p)
    assert np.array_equal(d["frames"], store)
    assert xrf.point_spectrum_from_store(store, gm, "9_9") is None


def test_save_load_product_roundtrip(tmp_path):
    _write_me7(tmp_path / "scan_0001_00001.h5", _spike(2, 0, 1000, 100))
    _write_me7(tmp_path / "scan_0001_00002.h5", _spike(2, 0, 1000, 100))
    res = xrf.element_maps(tmp_path, _grid_mapping(), _cfg())
    out = tmp_path / "Scan_0001_xrf_1x1.npz"
    xrf.save_npz(out, res)
    prod = xrf.load_product(out)
    assert prod["elements"] == ["Test"]
    assert prod["shape"] == (1, 2)
    assert prod["maps"]["Test"].shape == (1, 2)
    assert prod["spectrum"][1000] == 400.0
    # energy axis: bin × ev_per_bin + offset_ev (default 10 eV/bin, 0 offset)
    assert prod["energy_ev"][1000] == pytest.approx(10000.0)
    assert prod["rois"]["Test"]["line_ev"] == 10000.0
