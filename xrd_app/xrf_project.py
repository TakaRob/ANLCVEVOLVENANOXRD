"""XRF add-on configuration inside a standard xrd-app project."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import h5py
import yaml

from .config import ProjectConfig, default_config as default_xrd_config

ADDON_DIR = "XRF"
CONFIG_FILENAME = "xrf_config.yaml"
PROJECT_DIRS = ("Raw", "Metadata", "Cache", "Processed", "Figures")
DEFAULT_CALIBRATION = {
    "quadratic_kev": 5.263744e-7,
    "linear_kev": 8.41967e-3,
    "offset_kev": 1.136032,
}


def scan_name(value):
    """Normalize a scan number/name to ``Scan_NNNN``."""
    if isinstance(value, int):
        return f"Scan_{value:04d}"
    text = str(value).strip()
    if text.lower().startswith("scan_"):
        text = text.split("_", 1)[1]
    try:
        return f"Scan_{int(text):04d}"
    except ValueError as exc:
        raise ValueError(f"Invalid scan {value!r}") from exc


def default_config(name):
    return {
        "name": str(name),
        "active_scan": None,
        "scans": {},
        "data_sources": {
            "me7_root": None,
            "position_root": None,
            "position_offset": None,
            "xrd_identity_root": None,
        },
        "calibration": dict(DEFAULT_CALIBRATION),
        "paths": {
            "raw_dir": "Raw",
            "metadata_dir": "Metadata",
            "cache_dir": "Cache",
            "processed_dir": "Processed",
            "figures_dir": "Figures",
        },
    }


class XRFProject:
    """Read and write the ``XRF/`` add-on for an xrd-app project."""

    def __init__(self, root, data=None):
        self.root = Path(root).resolve()
        self.data = data or {}

    @property
    def addon_root(self):
        return self.root / ADDON_DIR

    @property
    def xrd_config_path(self):
        return self.root / "config.yaml"

    @property
    def config_path(self):
        return self.addon_root / CONFIG_FILENAME

    @classmethod
    def load(cls, root="."):
        start = Path(root).resolve()
        project_root = start
        for directory in (start, *start.parents):
            if (directory / "config.yaml").exists():
                project_root = directory
                break
            if directory.name == ADDON_DIR and (directory / CONFIG_FILENAME).exists():
                project_root = directory.parent
                break
            if (directory / ADDON_DIR / CONFIG_FILENAME).exists():
                project_root = directory
                break
        project = cls(project_root)
        if project.config_path.exists():
            with project.config_path.open() as stream:
                project.data = yaml.safe_load(stream) or {}
            project.data.setdefault("calibration", dict(DEFAULT_CALIBRATION))
        return project

    def xrd_exists(self):
        return self.xrd_config_path.exists()

    def exists(self):
        return self.config_path.exists()

    def create(self, name, scan_number=None):
        """Create a normal xrd-app project if needed, then its XRF add-on."""
        if not self.xrd_exists():
            self.root.mkdir(parents=True, exist_ok=True)
            config = ProjectConfig(
                self.root,
                data=default_xrd_config(name, self.root, scan_number=scan_number),
            )
            config.create_tree()
            config.save()
        return self.create_addon(name)

    def create_addon(self, name=None):
        """Create only the XRF add-on under an existing xrd-app project."""
        if not self.xrd_exists():
            raise FileNotFoundError(
                f"No xrd-app config.yaml found at {self.root}; create the parent project first"
            )
        if self.config_path.exists():
            raise FileExistsError(f"XRF add-on already exists: {self.config_path}")
        xrd_config = ProjectConfig.load(self.root)
        addon_name = name or xrd_config.data.get("name") or self.root.name
        self.addon_root.mkdir(parents=True, exist_ok=True)
        for directory in PROJECT_DIRS:
            (self.addon_root / directory).mkdir(exist_ok=True)
        self.data = default_config(addon_name)
        active_scan = (xrd_config.data.get("scan") or {}).get("name")
        if active_scan:
            self.data["active_scan"] = active_scan
        offset = self.path("metadata_dir") / "position_offset.json"
        with offset.open("w") as stream:
            json.dump({"theta": [0.0], "y_offset": [0.0]}, stream, indent=2)
            stream.write("\n")
        self.data["data_sources"]["position_offset"] = "Metadata/position_offset.json"
        self.save()
        return self

    def save(self):
        self.addon_root.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_name(f".{self.config_path.name}.tmp")
        try:
            with temporary.open("w") as stream:
                yaml.safe_dump(self.data, stream, sort_keys=False)
            temporary.replace(self.config_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def path(self, key):
        configured = (self.data.get("paths") or {}).get(key)
        if not configured:
            raise KeyError(f"XRF add-on path {key!r} is not configured")
        path = Path(configured)
        return path if path.is_absolute() else self.addon_root / path

    def metadata_scan_dir(self, scan):
        return self.path("metadata_dir") / scan_name(scan)

    def cache_scan_dir(self, scan):
        return self.path("cache_dir") / scan_name(scan)

    def selection_path(self, scan):
        return self.path("processed_dir") / f"{scan_name(scan)}_xrf_selection.h5"

    @staticmethod
    def inspect_scan_folder(path):
        """Return scan identity and ME7 count for one Scan_NNNN/ME7 folder."""
        path = Path(path).resolve()
        me7 = path / "ME7" if (path / "ME7").is_dir() else path
        files = sorted(me7.glob("scan_*.h5"))
        if not files:
            return None
        match = re.match(r"scan_(\d+)_", files[0].name, re.IGNORECASE)
        if not match:
            return None
        scan = scan_name(int(match.group(1)))
        valid_files = 0
        points = 0
        for file in files:
            try:
                with h5py.File(file, "r") as handle:
                    dataset = handle["entry/data/data"]
                    points += int(dataset.shape[0])
                    valid_files += 1
            except (KeyError, OSError):
                continue
        if not valid_files:
            return None
        return {"name": scan, "me7_dir": me7, "n_files": valid_files, "n_points": points}

    @classmethod
    def discover_scan_folders(cls, path):
        """Discover one scan folder or every Scan_* child containing ME7 data."""
        path = Path(path).resolve()
        direct = cls.inspect_scan_folder(path)
        if direct is not None:
            return [direct]
        scans = []
        for child in sorted(path.glob("Scan_*")):
            if child.is_dir():
                info = cls.inspect_scan_folder(child)
                if info is not None:
                    scans.append(info)
        return scans

    def discover_processed(self):
        """Register every valid canonical selection already in XRF/Processed."""
        from .core import xrf_selection

        discovered = []
        for path in sorted(self.path("processed_dir").glob("Scan_*_xrf_selection.h5")):
            try:
                selection = xrf_selection.load(path)
                info = xrf_selection.summary(selection)
                scan = scan_name(info["scan"])
                self.register_selection(
                    scan, path, info["materials"], info["selection_hash"], save=False
                )
                discovered.append(scan)
            except (KeyError, OSError, ValueError):
                continue
        if discovered:
            self.save()
        return discovered

    def set_calibration(self, calibration):
        self.data["calibration"] = {
            "quadratic_kev": float(calibration["quadratic_kev"]),
            "linear_kev": float(calibration["linear_kev"]),
            "offset_kev": float(calibration["offset_kev"]),
        }
        self.save()

    def position_offset_path(self):
        """Canonical project-owned position-offset calibration."""
        return self.path("metadata_dir") / "position_offset.json"

    def restore_position_offset(self):
        """Migrate a configured external offset into the canonical project path."""
        destination = self.position_offset_path()
        if destination.exists():
            relative = str(destination.relative_to(self.addon_root))
            if (self.data.get("data_sources") or {}).get("position_offset") != relative:
                self.data.setdefault("data_sources", {})["position_offset"] = relative
                self.save()
            return destination
        configured = (self.data.get("data_sources") or {}).get("position_offset")
        if not configured:
            return destination
        source = Path(configured)
        if not source.is_absolute():
            source = self.addon_root / source
        if source.is_file():
            return self.set_position_offset(source)
        return destination

    def set_position_offset(self, path):
        """Copy a validated position-offset JSON to its canonical project path."""
        source = Path(path).resolve()
        destination = self.position_offset_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copyfile(source, destination)
        self.data.setdefault("data_sources", {})["position_offset"] = str(
            destination.relative_to(self.addon_root)
        )
        self.save()
        return destination

    def register_raw_me7(self, scan, path, save=True):
        name = scan_name(scan)
        scans = self.data.setdefault("scans", {})
        scans.setdefault(name, {})["me7_dir"] = str(Path(path).resolve())
        self.data["active_scan"] = name
        if save:
            self.save()

    def register_selection(self, scan, path, materials, selection_hash, save=True):
        name = scan_name(scan)
        scans = self.data.setdefault("scans", {})
        scans.setdefault(name, {})["selection"] = {
            "path": str(Path(path).resolve()),
            "materials": list(materials),
            "selection_hash": str(selection_hash),
        }
        self.data["active_scan"] = name
        if save:
            self.save()
