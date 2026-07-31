"""Create a normal xrd-app project from a standalone ROI-feature scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import workspace
from ..config import DataManager
from . import io, reflection_sum


def create_from_scan(scan_folder, project_parent, name, tth_path, *,
                     positions_path=None, bin_size=3, summed_image=None) -> tuple[Path, str]:
    """Create and prepare a single-scan project for manual ROI mapping.

    The raw scan remains in place and is registered by absolute path. The
    calibration and optional positions file are likewise linked by path. A grid
    mapping for ``bin_size`` is generated now; detector bins are built later by
    the ROI mapper so the GUI can show progress and cancellation.
    """
    scan_folder = Path(scan_folder).resolve()
    project_parent = Path(project_parent).resolve()
    tth_path = Path(tth_path).resolve()
    positions_path = Path(positions_path).resolve() if positions_path else None
    if not scan_folder.is_dir():
        raise FileNotFoundError(f"Scan folder does not exist: {scan_folder}")
    if not tth_path.is_file():
        raise FileNotFoundError(f"2-theta map does not exist: {tth_path}")

    found = io.discover_scans(scan_folder, deep=False)
    if len(found) != 1:
        raise ValueError(f"Expected one scan under {scan_folder}; found {len(found)}")
    info = found[0]
    scan_name = info["name"]
    scan_number = DataManager.scan_number_of(scan_name)
    if scan_number is None:
        raise ValueError(f"Could not determine a scan number from {scan_name!r}")

    root = project_parent / name
    if root.exists():
        raise FileExistsError(f"Project path already exists: {root}")

    grid_data = io.generate_grid_mapping(
        info["frames_dir"], positions_path, int(bin_size), scan_number=scan_number,
        output=None, log=lambda *_: None)

    root = workspace.create_project(name, project_parent, scan_number=scan_number)
    dm = DataManager(root, scan=scan_name)
    registry = {
        scan_name: {key: info[key] for key in
                    ("dir", "frames_dir", "n_files", "n_frames", "shape")}
    }
    dm.write_scans_registry(registry)
    cfg = dm.config
    cfg.data["scans"] = registry
    cfg.data["detector"]["shape"] = info.get("shape")
    cfg.data["data_sources"]["raw_scan_dir"] = str(scan_folder)
    cfg.data["data_sources"]["tth_map"] = str(tth_path)
    if positions_path:
        cfg.data["data_sources"]["position_csv"] = str(positions_path)
    cfg.save()

    if summed_image is not None:
        reflection_sum.save(dm, scan_name, np.asarray(summed_image), is_raw=True)

    io.atomic_write_json(dm.grid_mapping(bin_size=int(bin_size), scan=scan_name),
                         grid_data, indent=None)
    workspace.set_last_project(root)
    return root, scan_name
