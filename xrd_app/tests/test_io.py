"""Tests for the lossless unbinned detector archive."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from xrd_app.config import DataManager
from xrd_app.core import io


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
    gm_path.write_text(json.dumps(gm))
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
        assert np.array_equal(f["0_0"][:], frames.sum(axis=0).astype(np.float32))


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


def test_unbinned_archive_path_is_separate_from_1x1_bins(tmp_path):
    dm = _project(tmp_path)
    assert dm.unbinned_archive_h5(scan=7).name == "xrd_unbinned_archive.h5"
    assert dm.unbinned_archive_h5(scan=7) != dm.binned_h5(1, scan=7)
