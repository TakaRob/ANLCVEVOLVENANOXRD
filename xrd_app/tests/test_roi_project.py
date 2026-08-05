"""Deferred project creation for the standalone ROI-feature workflow."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tifffile

from xrd_app.config import DataManager
from xrd_app.core import io, roi_project


def test_create_from_scan_registers_inputs_and_transfers_sum(tmp_path, monkeypatch):
    scan = tmp_path / "Scan_0042"
    xrd = scan / "XRD"
    xrd.mkdir(parents=True)
    for index in range(2):
        raw = xrd / f"scan_0042_{index:04d}.h5"
        with h5py.File(raw, "w") as handle:
            handle.create_dataset(io.H5_DATASET, data=np.ones((2, 4, 5), dtype=np.uint16))
    tth = tmp_path / "tth.tiff"
    tifffile.imwrite(tth, np.ones((4, 5), dtype=np.float32))
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("xrd_app.workspace.SETTINGS_DIR", tmp_path / ".settings")
    monkeypatch.setattr("xrd_app.workspace.SETTINGS_PATH",
                        tmp_path / ".settings" / "settings.json")

    root, scan_name = roi_project.create_from_scan(
        scan, projects, "Scan_0042-Project", tth, bin_size=2,
        summed_image=np.full((4, 5), 7.0),
    )

    assert scan_name == "Scan_0042"
    dm = DataManager(root, scan=scan_name)
    assert dm.scans_registry()[scan_name]["dir"] == str(scan)
    assert dm.tth_map() == tth
    grid_path = dm.grid_mapping(bin_size=2, scan=scan_name)
    assert grid_path.exists()
    assert io.load_grid_mapping(grid_path)["bin_size"] == 2
    saved = np.load(dm.metadata_scan_dir(scan_name) / "reflection_sum.npz")
    assert np.all(saved["image"] == 7)
    assert Path(root).name == "Scan_0042-Project"
