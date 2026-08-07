"""Tests for the lossless unbinned detector archive."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrd_app.config import DataManager
from xrd_app.core import catalogs, io


def _raw_file(path, frames):
    with h5py.File(path, "w") as f:
        f.create_dataset(io.H5_DATASET, data=np.asarray(frames))


def _project(tmp_path):
    (tmp_path / "Binned" / "Scan_0007").mkdir(parents=True)
    (tmp_path / "Metadata" / "Scan_0007").mkdir(parents=True)
    (tmp_path / "Raw" / "Scan_0007" / "XRD").mkdir(parents=True)
    (tmp_path / "config.yaml").write_text(
        "name: test\nscan:\n  number: 7\n  name: Scan_0007\n"
        "paths:\n  raw_dir: Raw\n  binned_dir: Binned\n"
        "  metadata_dir: Metadata\n  labels_dir: Labels\n"
    )
    return DataManager(tmp_path, scan=7)


def _write_grid(path, bins):
    io.save_grid_mapping(path, {
        "bin_size": 3,
        "n_bin_rows": 1,
        "n_bin_cols": len(bins),
        "xrd_files": [],
        "frame_map": [],
        "bins": bins,
    })


def test_catalog_lineage_selects_matching_tagged_grid_and_h5(tmp_path):
    dm = _project(tmp_path)
    labels = dm.labels_dir(7)
    labels.mkdir(parents=True)
    default_grid = dm.grid_mapping(bin_size=3, scan=7)
    tagged_grid = dm.grid_mapping(bin_size=3, scan=7, variant="faithful")
    _write_grid(default_grid, {"0_0": [0]})
    _write_grid(tagged_grid, {"0_0": [0]})
    catalog = labels / "gaussian_shapes_3x3_export.h5"
    catalogs.save_result(catalog, {
        "lineage": {
            "stage": "shapes", "scan": "Scan_0007", "bin_size": 3,
            "peak_source": {"stage": "peaks", "variant": "faithful"},
        },
        "kept": [{"center_bin": "0_0", "spatial_extent": ["0_0"]}],
        "filtered": [],
    })

    resolved = catalogs.resolve_catalog_sources(dm, catalog, bin_size=3, scan=7)

    assert resolved.variant == "faithful"
    assert resolved.grid_mapping == tagged_grid
    assert resolved.bins_h5 == dm.binned_h5(3, scan=7, variant="faithful")
    assert (resolved.matched, resolved.total) == (1, 1)


def test_tagged_catalog_filename_selects_matching_grid_and_h5(tmp_path):
    dm = _project(tmp_path)
    labels = dm.labels_dir(7)
    labels.mkdir(parents=True)
    tagged_grid = dm.grid_mapping(bin_size=3, scan=7, variant="faithful")
    _write_grid(tagged_grid, {"4_5": [0]})
    catalog = labels / "gaussian_shapes_3x3_faithful.h5"
    catalogs.save_result(catalog, {
        "kept": [{"center_bin": "4_5", "spatial_extent": ["4_5"]}],
        "filtered": [],
    })

    resolved = catalogs.resolve_catalog_sources(dm, catalog, scan=7)

    assert resolved.grid_mapping == tagged_grid
    assert resolved.bins_h5.name == "xrd_3x3_bins_faithful.h5"


def test_best_grid_mapping_rejects_zero_overlap_for_nonempty_catalog(tmp_path):
    grid = tmp_path / "grid_mapping_3x3.h5"
    _write_grid(grid, {"0_0": [0]})
    catalog = tmp_path / "gaussian_shapes_3x3.h5"
    catalogs.save_result(catalog, {
        "kept": [{"center_bin": "9_9", "spatial_extent": ["9_9"]}],
        "filtered": [],
    })

    with pytest.raises(catalogs.CatalogGridMismatch, match="covers only 0/1"):
        catalogs.best_grid_mapping([grid], catalog, default=grid)


def test_best_grid_mapping_rejects_partial_coverage(tmp_path):
    grid = tmp_path / "grid_mapping_3x3.h5"
    _write_grid(grid, {"0_0": [0]})
    catalog = tmp_path / "gaussian_shapes_3x3.h5"
    catalogs.save_result(catalog, {
        "kept": [{"center_bin": "0_0", "spatial_extent": ["0_0", "0_1"]}],
        "filtered": [],
    })

    with pytest.raises(catalogs.CatalogGridMismatch, match="covers only 1/2"):
        catalogs.best_grid_mapping([grid], catalog, default=grid)


def test_unbinned_archive_roundtrip_and_metadata(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    a = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    b = (np.arange(12, dtype=np.uint16) + 100).reshape(1, 3, 4)
    _raw_file(raw / "scan_0007_000.h5", a)
    _raw_file(raw / "scan_0007_001.h5", b)
    positions = tmp_path / "positions.csv"
    positions.write_text(
        "Trigger,X_Position,Y_Position\n0,1.0,4.0\n1,2.0,5.0\n2,3.0,6.0\n"
    )
    archive = tmp_path / "xrd_unbinned_archive.h5"

    io.build_unbinned_archive(
        raw, archive, 7, positions=positions, compression="gzip", log=lambda _: None
    )

    with h5py.File(archive, "r") as f:
        ds = f[io.ARCHIVE_FRAMES]
        assert ds.shape == (3, 3, 4)
        assert ds.dtype == np.uint16
        assert ds.chunks == (1, 3, 4)
        assert np.array_equal(ds[:2], a)
        assert np.array_equal(ds[2], b[0])
        assert np.array_equal(f["metadata/x"][:], [1.0, 2.0, 3.0])
        assert np.array_equal(f["metadata/y"][:], [4.0, 5.0, 6.0])
        assert f.attrs["format"] == io.ARCHIVE_FORMAT
        assert bool(f.attrs["positions_real"])

    _, frame_map, n_frames = io.archive_metadata(archive)
    assert frame_map == [[0, 0], [0, 1], [1, 0]]
    assert n_frames == 3


def test_load_positions_accepts_prefilter_h5_layout(tmp_path):
    positions = tmp_path / "positions.h5"
    with h5py.File(positions, "w") as handle:
        group = handle.create_group("entry/data")
        group.create_dataset("X_Position", data=[1.0, 2.0])
        group.create_dataset("Y_Position", data=[3.0, 4.0])

    x, y = io.load_positions_xy(positions, 3)

    np.testing.assert_allclose(x[:2], [1.0, 2.0])
    np.testing.assert_allclose(y[:2], [3.0, 4.0])
    assert np.isnan(x[2]) and np.isnan(y[2])


def test_archive_source_sums_cells_and_regions_without_raw(tmp_path):
    dm = _project(tmp_path)
    raw = dm.xrd_frames_dir(scan=7)
    frames = np.arange(36, dtype=np.uint16).reshape(3, 3, 4)
    _raw_file(raw / "scan_0007_000.h5", frames)
    archive = dm.unbinned_archive_h5(scan=7)
    io.build_unbinned_archive(raw, archive, 7, compression="gzip", log=lambda _: None)

    gm = {
        "bin_size": 1,
        "n_bin_rows": 1,
        "n_bin_cols": 2,
        "n_total_frames": 3,
        "xrd_files": [str(raw / "scan_0007_000.h5")],
        "frame_map": [[0, 0], [0, 1], [0, 2]],
        "bins": {"0_0": [0, 1], "0_1": [2]},
    }
    gm_path = dm.grid_mapping(bin_size=1, scan=7)
    io.save_grid_mapping(gm_path, gm)
    for path in raw.iterdir():
        path.unlink()

    source = io.open_bin_source(dm, 1, scan=7)
    try:
        assert source.is_archive
        expected = frames[0].astype(float) + frames[1]
        assert np.array_equal(source.image("0_0"), expected)
        assert np.array_equal(source.region("0_0", 1, 3, 1, 4), expected[1:3, 1:4])
    finally:
        source.close()


def test_build_bins_uses_archive_when_raw_paths_are_gone(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frames = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    raw_path = raw / "scan_0007_000.h5"
    _raw_file(raw_path, frames)
    archive = tmp_path / "xrd_unbinned_archive.h5"
    io.build_unbinned_archive(raw, archive, 7, compression="gzip", log=lambda _: None)
    raw_path.unlink()
    gm = {
        "bin_size": 2,
        "n_bin_rows": 1,
        "n_bin_cols": 1,
        "xrd_files": [str(raw_path)],
        "frame_map": [[0, 0], [0, 1]],
        "bins": {"0_0": [0, 1]},
    }
    output = tmp_path / "xrd_2x2_bins.h5"

    io.build_bins(gm, output, compression="gzip", archive=archive, log=lambda _: None)

    with h5py.File(output, "r") as f:
        assert f.attrs["aggregation"] == "sum"
        assert f["0_0"].attrs["n_frames"] == 2
        assert np.array_equal(f["0_0"][:], frames.sum(axis=0).astype(np.float32))


def test_build_bins_can_normalize_by_contributing_frame_count(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frames = np.array([[[2, 4]], [[6, 8]]], dtype=np.uint16)
    raw_path = raw / "scan_0007_000.h5"
    _raw_file(raw_path, frames)
    gm = {
        "bin_size": 1, "n_bin_rows": 1, "n_bin_cols": 1,
        "xrd_files": [str(raw_path)], "frame_map": [[0, 0], [0, 1]],
        "bins": {"0_0": [0, 1]},
    }
    output = tmp_path / "normalized_bins.h5"

    io.build_bins(gm, output, compression="gzip", normalize_frames=True,
                  log=lambda _: None)

    with h5py.File(output, "r") as f:
        assert f.attrs["aggregation"] == "mean_per_frame"
        assert f.attrs["normalized_by"] == "contributing_frame_count"
        assert f["0_0"].attrs["n_frames"] == 2
        assert np.array_equal(f["0_0"][:], np.array([[4, 6]], dtype=np.float32))


def test_summation_clips_saturation_instead_of_zeroing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_path = raw / "scan_0007_000.h5"
    frames = np.array([[[8e8, -2]], [[8e8, 4]]], dtype=np.float64)
    _raw_file(raw_path, frames)
    frame_map = [[0, 0], [0, 1]]

    summed = io.sum_raw_frames([str(raw_path)], frame_map, [0, 1])

    assert np.array_equal(summed, [[1e9, 2]])


def test_build_bins_records_actual_detector_shape_and_clips_saturation(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    raw_path = raw / "scan_0007_000.h5"
    frames = np.full((2, 2, 3), 8e8, dtype=np.float64)
    _raw_file(raw_path, frames)
    gm = {
        "bin_size": 1, "n_bin_rows": 1, "n_bin_cols": 1,
        "xrd_files": [str(raw_path)], "frame_map": [[0, 0], [0, 1]],
        "bins": {"0_0": [0, 1]},
    }
    output = tmp_path / "bins.h5"

    io.build_bins(gm, output, compression="gzip", log=lambda _: None)

    with h5py.File(output, "r") as f:
        assert tuple(f.attrs["detector_shape"]) == (2, 3)
        assert np.all(f["0_0"][:] == np.float32(1e9))


def test_archive_source_cache_is_lru_and_cleared_on_close(tmp_path):
    archive = tmp_path / "archive.h5"
    with h5py.File(archive, "w") as f:
        f.attrs["format"] = io.ARCHIVE_FORMAT
        f.create_dataset(io.ARCHIVE_FRAMES, data=np.arange(10).reshape(10, 1, 1))
    gm = {"bins": {f"0_{i}": [i] for i in range(10)}}
    source = io._ArchiveSource(archive, gm)
    for key in source.keys():
        source.image(key)

    assert len(source._cache) == 8
    assert "0_0" not in source._cache
    source.image("0_2")
    source.image("0_0")
    assert "0_1" not in source._cache
    source.close()
    assert not source._cache


def test_raw_source_cache_is_lru_and_cleared_on_close(monkeypatch):
    source = io._RawSource.__new__(io._RawSource)
    source._bins = {f"0_{i}": [i] for i in range(10)}
    source._xrd_files = []
    source._frame_map = []
    source._cache = io.OrderedDict()
    monkeypatch.setattr(io, "sum_raw_frames", lambda files, mapping, indices:
                        np.array([[indices[0]]], dtype=float))

    for key in source.keys():
        source.image(key)

    assert len(source._cache) == 8
    assert "0_0" not in source._cache
    source.image("0_2")
    source.image("0_0")
    assert "0_1" not in source._cache
    source.close()
    assert not source._cache


def test_grid_rebuild_uses_positions_embedded_in_archive(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frames = np.arange(48, dtype=np.uint16).reshape(4, 3, 4)
    raw_path = raw / "scan_0007_000.h5"
    _raw_file(raw_path, frames)
    positions = tmp_path / "positions.csv"
    positions.write_text(
        "Trigger,X_Position,Y_Position\n"
        "0,0.0,0.0\n1,1.0,0.0\n2,0.0,1.0\n3,1.0,1.0\n"
    )
    archive = tmp_path / "xrd_unbinned_archive.h5"
    io.build_unbinned_archive(
        raw, archive, 7, positions=positions, compression="gzip", log=lambda _: None
    )
    raw_path.unlink()
    positions.unlink()

    gm = io.generate_grid_mapping(
        raw, positions, 1, scan_number=7, deskew_method="positions_xy",
        archive=archive, log=lambda _: None
    )

    assert gm["positions_real"] is True
    assert gm["n_total_frames"] == 4
    assert sorted(i for frames in gm["bins"].values() for i in frames) == [0, 1, 2, 3]


def test_raw_image_broker_returns_summed_bin_without_disk_cache(tmp_path):
    import time

    from xrd_app.core.raw_broker import RawImageBroker

    raw_file = tmp_path / "scan_0007_00001.h5"
    _raw_file(raw_file, np.stack([
        np.full((2, 2), value, dtype=np.uint16) for value in (2, 3)
    ]))
    broker = RawImageBroker({
        "xrd_files": [str(raw_file)], "frame_map": [[0, 0], [0, 1]],
        "bins": {"0_0": [0, 1]},
    })
    try:
        assert broker.request(4, "0_0")
        deadline = time.monotonic() + 5
        responses = []
        while time.monotonic() < deadline and not responses:
            responses = broker.poll()
            time.sleep(0.01)
        assert len(responses) == 1
        generation, key, image, error = responses[0]
        assert (generation, key, error) == (4, "0_0", None)
        np.testing.assert_array_equal(image, np.full((2, 2), 5.0))
    finally:
        broker.close()


def test_unbinned_archive_path_is_separate_from_1x1_bins(tmp_path):
    dm = _project(tmp_path)
    assert dm.unbinned_archive_h5(scan=7).name == "xrd_unbinned_archive.h5"
    assert dm.unbinned_archive_h5(scan=7) != dm.binned_h5(1, scan=7)


def test_configured_grid_only_applies_without_explicit_context(tmp_path):
    dm = _project(tmp_path)
    configured = tmp_path / "Metadata" / "configured_3x3.h5"
    io.save_grid_mapping(configured, {"bin_size": 3, "bins": {}})
    dm.config.data.setdefault("data_sources", {})["grid_mapping"] = str(configured)

    assert dm.grid_mapping() == configured
    assert dm.grid_mapping(bin_size=5, scan=7).name == "grid_mapping_5x5.h5"


def test_open_bin_source_pairs_supplied_variant_grid_with_variant_h5(tmp_path):
    dm = _project(tmp_path)
    gm = dm.grid_mapping(bin_size=3, scan=7, variant="faithful")
    io.save_grid_mapping(gm, {
        "bin_size": 3, "variant": "faithful", "bins": {"9_9": []},
    })
    with h5py.File(dm.binned_h5(3, scan=7), "w") as f:
        f.create_dataset("0_0", data=np.zeros((1, 1)))
    with h5py.File(dm.binned_h5(3, scan=7, variant="faithful"), "w") as f:
        f.create_dataset("9_9", data=np.ones((1, 1)))

    source = io.open_bin_source(dm, 3, scan=7, grid_mapping=gm)
    try:
        assert source.keys() == ["9_9"]
    finally:
        source.close()


def test_roi_source_uses_requested_bin_grid_not_configured_grid(tmp_path):
    dm = _project(tmp_path)
    configured = tmp_path / "Metadata" / "configured_3x3.h5"
    io.save_grid_mapping(configured, {"bin_size": 3, "bins": {"3_3": []}})
    dm.config.data.setdefault("data_sources", {})["grid_mapping"] = str(configured)
    requested = dm.grid_mapping(bin_size=5, scan=7)
    io.save_grid_mapping(requested, {"bin_size": 5, "bins": {"5_5": []}})
    with h5py.File(dm.binned_h5(5, scan=7), "w") as f:
        f.create_dataset("5_5", data=np.ones((1, 1)))

    source = io.open_bin_source(dm, 5, scan=7)
    try:
        assert source.keys() == ["5_5"]
    finally:
        source.close()


def test_open_bin_source_rejects_wrong_h5_bin_size_attr(tmp_path):
    dm = _project(tmp_path)
    gm = dm.grid_mapping(bin_size=3, scan=7)
    _write_grid(gm, {"0_0": [0]})
    with h5py.File(dm.binned_h5(3, scan=7), "w") as f:
        f.attrs["bin_size"] = 5
        f.create_dataset("0_0", data=np.ones((2, 3)))

    with pytest.raises(ValueError, match=r"bin_size is 5x5.*3x3.*stale or mismatched"):
        io.open_bin_source(dm, 3, scan=7)


def test_open_bin_source_rejects_out_of_grid_mapping_key(tmp_path):
    dm = _project(tmp_path)
    gm = dm.grid_mapping(bin_size=3, scan=7)
    io.save_grid_mapping(gm, {
        "bin_size": 3, "n_bin_rows": 1, "n_bin_cols": 1,
        "bins": {"1_0": [0]},
    })
    with h5py.File(dm.binned_h5(3, scan=7), "w") as f:
        f.create_dataset("1_0", data=np.ones((2, 3)))

    with pytest.raises(ValueError, match=r"1_0.*outside n_bin_rows=1"):
        io.open_bin_source(dm, 3, scan=7)


def test_open_bin_source_rejects_non_2d_bin_dataset(tmp_path):
    dm = _project(tmp_path)
    gm = dm.grid_mapping(bin_size=3, scan=7)
    _write_grid(gm, {"0_0": [0]})
    with h5py.File(dm.binned_h5(3, scan=7), "w") as f:
        f.create_dataset("0_0", data=np.ones((1, 2, 3)))

    with pytest.raises(ValueError, match=r"0_0.*must be 2-D, got 3-D"):
        io.open_bin_source(dm, 3, scan=7)


def test_open_bin_source_accepts_valid_sparse_h5_and_metadata(tmp_path):
    dm = _project(tmp_path)
    gm = dm.grid_mapping(bin_size=3, scan=7)
    io.save_grid_mapping(gm, {
        "bin_size": 3, "n_bin_rows": 4, "n_bin_cols": 5,
        "bins": {"0_0": [0], "1_2": [], "3_4": [1]},
    })
    with h5py.File(dm.binned_h5(3, scan=7), "w") as f:
        f.attrs["bin_size"] = 3
        f.attrs["n_bin_rows"] = 4
        f.attrs["n_bin_cols"] = 5
        f.attrs["detector_shape"] = [2, 3]
        f.create_dataset("0_0", data=np.ones((2, 3)))
        f.create_dataset("3_4", data=np.full((2, 3), 2))
        metadata = f.create_group("metadata")
        metadata.create_dataset("calibration", data=np.arange(4))

    source = io.open_bin_source(dm, 3, scan=7)
    try:
        assert source.keys() == ["0_0", "3_4"]
        assert "metadata" not in source
        assert np.array_equal(source.image("3_4"), np.full((2, 3), 2))
    finally:
        source.close()


def test_open_bin_source_preserves_territory_key_space(tmp_path):
    dm = _project(tmp_path)
    gm = dm.grid_mapping(bin_size=1, scan=7, variant="territory")
    io.save_grid_mapping(gm, {
        "bin_size": 1, "coordinate_source": "territory_xy",
        "n_bin_rows": 2, "n_bin_cols": 2, "bins": {"7_0": [0]},
    })
    with h5py.File(dm.binned_h5(1, scan=7, variant="territory"), "w") as f:
        f.attrs["bin_size"] = 1
        f.attrs["n_bin_rows"] = 2
        f.attrs["n_bin_cols"] = 2
        f.create_dataset("7_0", data=np.ones((2, 3)))

    source = io.open_bin_source(dm, 1, scan=7, variant="territory")
    try:
        assert source.keys() == ["7_0"]
    finally:
        source.close()
