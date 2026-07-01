"""Headless tests for fusing per-scan q-maps into a 3D RSM (core/rsm.py)."""
from __future__ import annotations

import numpy as np

from xrd_app.core import rsm as R


def _qmap(scan, theta, qxc, qyc, qzc, spike=100.0, shape=(8, 8)):
    """A tiny q-map whose pixels sit near (qxc,qyc,qzc), with one bright pixel."""
    qx = np.full(shape, qxc) + np.linspace(-0.01, 0.01, shape[0])[:, None]
    qy = np.full(shape, qyc) + np.linspace(-0.01, 0.01, shape[1])[None, :]
    qz = np.full(shape, qzc)
    inten = np.zeros(shape)
    inten[shape[0] // 2, shape[1] // 2] = spike
    return R.QMap(scan, theta, qx, qy, qz, inten)


def test_common_grid_spans_all_maps():
    a = _qmap("s1", 0.0, 1.0, 0.0, 0.5)
    b = _qmap("s2", 5.0, 2.0, -1.0, 0.5)
    ex, ey, ez = R.common_grid([a, b], nbins=16)
    assert ex[0] <= min(a.qx.min(), b.qx.min())
    assert ex[-1] >= max(a.qx.max(), b.qx.max())
    assert len(ex) == 17 and len(ey) == 17 and len(ez) == 17


def test_accumulate_conserves_intensity_and_locates_spike():
    a = _qmap("s1", 0.0, 1.0, 0.0, 0.5, spike=100.0)
    b = _qmap("s2", 5.0, 2.0, 0.0, 0.5, spike=40.0)
    edges = R.common_grid([a, b], nbins=32)
    vol, counts = R.accumulate([a, b], edges, subtract_median=False)
    # min_intensity default 0 → the flat-zero background is dropped, only the two
    # bright pixels contribute; their intensity is retained.
    assert np.isclose(vol.sum(), 140.0)
    assert counts.sum() == 2
    # the two spikes land in different voxels (different qx centers)
    assert np.count_nonzero(vol > 30) >= 2


def test_median_subtraction_removes_flat_background():
    m = _qmap("s1", 0.0, 1.0, 0.0, 0.0, spike=100.0)
    m.intensity[:] += 7.0  # flat pedestal on every pixel
    edges = R.common_grid([m], nbins=16)
    vol, _ = R.accumulate([m], edges, subtract_median=True, min_intensity=0.0)
    # pedestal (=median=7) is subtracted; the spike 107-7=100 survives, rest →0
    assert np.isclose(vol.sum(), 100.0, atol=1e-6)


def test_min_intensity_threshold():
    m = _qmap("s1", 0.0, 1.0, 0.0, 0.0, spike=100.0)
    m.intensity[0, 0] = 5.0
    edges = R.common_grid([m], nbins=16)
    vol, _ = R.accumulate([m], edges, subtract_median=False, min_intensity=10.0)
    assert np.isclose(vol.sum(), 100.0)  # the 5.0 pixel dropped


def test_projections_shapes():
    m = _qmap("s1", 0.0, 1.0, 0.0, 0.5)
    edges = R.common_grid([m], nbins=(6, 7, 8))
    vol, _ = R.accumulate([m], edges, subtract_median=False)
    p = R.projections(vol)
    assert p["qx_qy"].shape == (6, 7)
    assert p["qx_qz"].shape == (6, 8)
    assert p["qy_qz"].shape == (7, 8)


def test_save_load_roundtrip(tmp_path):
    a = _qmap("Scan_0203", 20.5, 1.0, 0.0, 0.5)
    edges = R.common_grid([a], nbins=10)
    vol, counts = R.accumulate([a], edges, subtract_median=False)
    out = tmp_path / "rsm.npz"
    R.save_npz(out, vol, counts, edges,
               meta={"scans": np.array(["Scan_0203"]), "thetas": np.array([20.5])})
    d = np.load(out, allow_pickle=False)
    assert d["volume"].shape == (10, 10, 10)
    assert np.isclose(d["volume"].sum(), vol.sum())
    assert d["proj_qx_qy"].shape == (10, 10)
    assert len(d["qx_centers"]) == 10
    s = R.summary(vol, counts, edges, ["Scan_0203"], [20.5])
    assert s["n_scans"] == 1 and s["thetas"] == [20.5]
    assert 0.0 <= s["fill_fraction"] <= 1.0


def test_load_rsm_reads_projections(tmp_path):
    a = _qmap("Scan_0203", 20.5, 1.0, 0.0, 0.5)
    edges = R.common_grid([a], nbins=12)
    vol, counts = R.accumulate([a], edges, subtract_median=False)
    out = tmp_path / "rsm.npz"
    R.save_npz(out, vol, counts, edges,
               meta={"scans": np.array(["Scan_0203"]), "thetas": np.array([20.5])})
    d = R.load_rsm(out)
    assert d["proj"]["qx_qy"].shape == (12, 12)
    assert len(d["centers"][0]) == 12
    assert d["scans"] == ["Scan_0203"] and d["thetas"] == [20.5]
    assert d["volume"].shape == (12, 12, 12)


def test_load_feature_cloud_reads_csvs_and_theta(tmp_path):
    import csv
    qdir = tmp_path / "qspace"
    qdir.mkdir()
    cols = ["scan", "reflection", "detector_x", "detector_y",
            "sum_integrated", "qx", "qy", "qz", "q_mag"]
    with open(qdir / "Scan_0203_features_q.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"scan": "Scan_0203", "reflection": "(002)", "detector_x": 484,
                    "detector_y": 21, "sum_integrated": 41.0, "qx": -1.19,
                    "qy": -1.92, "qz": 1.26, "q_mag": 2.59})
    cloud = R.load_feature_cloud(qdir)
    assert cloud["n"] == 1
    assert cloud["reflection"] == ["(002)"]
    assert np.isclose(cloud["q_mag"][0], 2.59)
    assert np.isclose(cloud["intensity"][0], 41.0)
    # θ resolved from the built-in 203–214 table (Scan_0203 → 20.5°)
    assert np.isclose(cloud["theta"][0], 20.5)


def test_load_qmap_reads_qspace_npz(tmp_path):
    # a qspace-style npz (subset of fields) must load into a QMap
    p = tmp_path / "Scan_0207_qmap.npz"
    np.savez_compressed(p, qx=np.ones((4, 4)), qy=np.zeros((4, 4)),
                        qz=np.full((4, 4), 0.5), intensity=np.eye(4).astype("float32"),
                        scan="Scan_0207", theta_deg=5.5)
    m = R.load_qmap(p)
    assert m.scan == "Scan_0207" and m.theta_deg == 5.5
    assert m.intensity is not None and m.qx.shape == (4, 4)
