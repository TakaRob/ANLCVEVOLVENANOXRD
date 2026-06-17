"""
Project configuration and path resolution for xrd-tools.

A project is a directory containing a ``config.yaml`` plus a standard set of
sub-directories created by ``xrd-tools init``. Every path used by the CLI and
the GUIs is resolved through :class:`DataManager`, which applies a consistent
precedence:

    1. An explicit override (CLI argument / GUI selection).
    2. The ``data_sources`` entry in ``config.yaml``.
    3. A conventional default location inside the project tree.

This is the single source of truth for "where does file X live" so that no
module needs hard-coded absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

CONFIG_FILENAME = "config.yaml"

# Standard project sub-directories created by ``xrd-tools init``.
PROJECT_DIRS = [
    "data/raw_scans",   # raw per-frame HDF5 scans (or symlinks to them)
    "data/bins",        # pre-binned xrd_NxN_bins.h5 files
    "data/holdout",     # tth.tiff, grid_mapping.json, reflections.py, annotations
    "hutch",            # detector / evolved-algorithm .py scripts
    "runs",             # CVEvolve session outputs
    "mlruns",           # MLflow tracking
    "results",          # feature catalogs and per-scan outputs
]


def default_config(name: str, root: Path, scan_number: Optional[int] = None) -> dict:
    """Return a fresh config dict for a newly initialized project."""
    root = Path(root).resolve()
    cfg: dict[str, Any] = {
        "name": name,
        "scan": {
            "number": scan_number,
            "name": f"Scan_{scan_number:04d}" if scan_number is not None else None,
        },
        "paths": {
            "data_dir": "data",
            "results_dir": "results",
            "runs_dir": "runs",
            "mlruns_dir": "mlruns",
            "hutch_dir": "hutch",
        },
        # Absolute paths to external inputs; populated by ``xrd-tools link``.
        "data_sources": {
            "raw_root": None,        # parent dir holding many Scan_NNNN/ dirs
            "position_root": None,   # dir holding scan_NNNN_position.csv files
            "raw_scan_dir": None,    # single-scan raw dir (this project's scan)
            "position_csv": None,    # single-scan position CSV
            "tth_map": None,
            "grid_mapping": None,
            "reflections": None,
            "detector_script": None,
        },
        # Map of bin_size (int) -> path to the pre-binned HDF5 file.
        "bins": {},
        "tracking": {
            "enabled": True,
            "mlflow_tracking_uri": f"file://{root / 'mlruns'}",
        },
    }
    return cfg


class ProjectConfig:
    """Load / save a project's ``config.yaml`` and create its directory tree."""

    def __init__(self, root: os.PathLike | str = ".", data: Optional[dict] = None):
        self.root = Path(root).resolve()
        self.data: dict = data if data is not None else {}

    # ----- persistence -------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    @classmethod
    def load(cls, root: os.PathLike | str = ".") -> "ProjectConfig":
        root = Path(root).resolve()
        cfg = cls(root)
        if cfg.config_path.exists():
            with open(cfg.config_path) as f:
                cfg.data = yaml.safe_load(f) or {}
        return cfg

    def save(self) -> None:
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)

    def exists(self) -> bool:
        return self.config_path.exists()

    def create_tree(self) -> None:
        """Create the standard project sub-directories."""
        for d in PROJECT_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ----- convenience accessors --------------------------------------
    def get(self, *keys, default=None):
        """Nested get: ``cfg.get('data_sources', 'tth_map')``."""
        node: Any = self.data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


class DataManager:
    """Resolve project file paths with override -> config -> default precedence.

    The GUIs and CLI commands construct one of these from the project root and
    ask it for paths, e.g. ``dm.tth_map()`` or ``dm.bins_h5(3)``.
    """

    def __init__(self, root: os.PathLike | str = ".",
                 config: Optional[ProjectConfig] = None,
                 scan: Optional[object] = None):
        self.config = config or ProjectConfig.load(root)
        self.root = self.config.root
        # An explicit scan (number or name) overrides config["scan"] for this
        # instance, so one project can be driven across many scans.
        self._scan_override = scan

    # ----- internal helpers -------------------------------------------
    def _abs(self, value: Optional[str]) -> Optional[Path]:
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def _resolve(self, source_key: str, override: Optional[str], default: Path) -> Path:
        """Apply override -> config[data_sources][key] -> default."""
        if override:
            return self._abs(override)
        configured = self.config.get("data_sources", source_key)
        if configured:
            return self._abs(configured)
        return default

    # ----- standard directories ---------------------------------------
    @property
    def data_dir(self) -> Path:
        return self._abs(self.config.get("paths", "data_dir", default="data"))

    @property
    def holdout_dir(self) -> Path:
        return self.data_dir / "holdout"

    @property
    def hutch_dir(self) -> Path:
        return self._abs(self.config.get("paths", "hutch_dir", default="hutch"))

    # ----- scan identity ----------------------------------------------
    @staticmethod
    def scan_name_of(scan: object) -> Optional[str]:
        """Normalize a scan identifier to the canonical ``Scan_NNNN`` name."""
        if scan is None:
            return None
        if isinstance(scan, int):
            return f"Scan_{scan:04d}"
        s = str(scan).strip()
        if s.isdigit():
            return f"Scan_{int(s):04d}"
        return s  # already a name like "Scan_0203"

    @staticmethod
    def scan_number_of(scan: object) -> Optional[int]:
        """Extract the integer scan number from a name or number."""
        if scan is None:
            return None
        if isinstance(scan, int):
            return scan
        digits = "".join(ch for ch in str(scan) if ch.isdigit())
        return int(digits) if digits else None

    def _scan(self, scan: object = None) -> Optional[str]:
        """Resolve the scan name to use: arg -> instance override -> config."""
        for candidate in (scan, self._scan_override, self.config.get("scan", "name")):
            name = self.scan_name_of(candidate)
            if name:
                return name
        return None

    @property
    def scan_name(self) -> Optional[str]:
        return self._scan()

    def scan_number(self, scan: object = None) -> Optional[int]:
        return self.scan_number_of(self._scan(scan)) or self.config.get("scan", "number")

    # ----- per-scan directories ---------------------------------------
    def results_dir(self, scan: object = None) -> Path:
        base = self._abs(self.config.get("paths", "results_dir", default="results"))
        name = self._scan(scan)
        return base / name if name else base

    def bins_dir(self, scan: object = None) -> Path:
        name = self._scan(scan)
        base = self.data_dir / "bins"
        return base / name if name else base

    def holdout_scan_dir(self, scan: object = None) -> Path:
        """Per-scan holdout dir for the grid mapping (tth/reflections are shared)."""
        name = self._scan(scan)
        return (self.holdout_dir / name) if name else self.holdout_dir

    # ----- resolved input files ---------------------------------------
    def raw_scan_dir(self, override: Optional[str] = None, scan: object = None) -> Path:
        name = self._scan(scan) or ""
        # A common parent of all scans (raw_root), else the project data dir.
        raw_root = self.config.get("data_sources", "raw_root")
        if raw_root:
            default = self._abs(raw_root) / name if name else self._abs(raw_root)
        else:
            default = self.data_dir / "raw_scans" / name if name else self.data_dir / "raw_scans"
        # A single linked raw_scan_dir only applies to the config's own scan.
        if not override and name == self.scan_name_of(self.config.get("scan", "name")):
            single = self.config.get("data_sources", "raw_scan_dir")
            if single:
                return self._abs(single)
        return self._abs(override) if override else default

    def xrd_frames_dir(self, override: Optional[str] = None, scan: object = None) -> Path:
        """Directory of raw per-frame H5 files (the ``XRD/`` subdir if present)."""
        base = self.raw_scan_dir(override, scan)
        sub = base / "XRD"
        return sub if sub.is_dir() else base

    def position_csv(self, override: Optional[str] = None, scan: object = None) -> Path:
        if override:
            return self._abs(override)
        name = self._scan(scan)
        num = self.scan_number_of(name)
        # Beamline convention: Processed/SOCKETSERVER/scan_NNNN_position.csv
        pos_root = self.config.get("data_sources", "position_root")
        if pos_root and num is not None:
            cand = self._abs(pos_root) / f"scan_{num:04d}_position.csv"
            if cand.exists():
                return cand
        # Single linked CSV applies to the config's own scan.
        if name == self.scan_name_of(self.config.get("scan", "name")):
            single = self.config.get("data_sources", "position_csv")
            if single:
                return self._abs(single)
        # Project-local conventions.
        if name:
            local = self.holdout_dir / f"{name}_position.csv"
            if local.exists():
                return local
        return self.holdout_dir / "positions.csv"

    def _asset(self, name: str) -> Path:
        """Path to a file bundled with the package (shared defaults)."""
        return Path(__file__).parent / "assets" / name

    def tth_map(self, override: Optional[str] = None) -> Path:
        proj = self.holdout_dir / "tth.tiff"
        default = proj if proj.exists() else self._asset("tth.tiff")
        return self._resolve("tth_map", override, default)

    def grid_mapping(self, override: Optional[str] = None, bin_size: Optional[int] = None,
                     scan: object = None) -> Path:
        if override:
            return self._abs(override)
        sdir = self.holdout_scan_dir(scan)
        if bin_size is not None:
            return sdir / f"grid_mapping_{bin_size}x{bin_size}.json"
        return sdir / "grid_mapping.json"

    def reflections(self, override: Optional[str] = None) -> Path:
        proj = self.holdout_dir / "reflections.py"
        default = proj if proj.exists() else self._asset("reflections.py")
        return self._resolve("reflections", override, default)

    def detector_script(self, override: Optional[str] = None) -> Path:
        return self._resolve("detector_script", override, self.hutch_dir / "detector.py")

    def bins_h5(self, bin_size: int, override: Optional[str] = None, scan: object = None) -> Path:
        if override:
            return self._abs(override)
        return self.bins_dir(scan) / f"xrd_{bin_size}x{bin_size}_bins.h5"
