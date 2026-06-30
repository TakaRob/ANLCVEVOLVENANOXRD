"""
Project configuration and path resolution for xrd-app.

A project is a directory containing a ``config.yaml`` plus a standard set of
sub-directories created by ``xrd-app init``:

    <project>/
      Raw/        scan registry (scans.json) + links to external scan dirs
      Binned/     pre-binned xrd_NxN_bins.h5 (per scan)
      Metadata/   tth.tiff, reflections.json (+ loader), grid mappings, gui_state
      Labels/     per-scan peak/shape algorithm outputs + manual labels
      Figures/    saved PNGs (setup histogram, etc.)
      CVEvolve/   CVEvolve session outputs created from the GUI

Every path used by the CLI and GUI resolves through :class:`DataManager`, which
applies a consistent precedence:

    1. An explicit override (CLI argument / GUI selection).
    2. The ``data_sources`` entry in ``config.yaml``.
    3. A conventional default location inside the project tree.

For ``tth.tiff`` / ``reflections`` a 4th fallback is the bundled package asset.
Algorithms live in the package's ``PeakAlgorithms/`` / ``ShapeAlgorithms/`` /
``CombinedAlgorithms/`` libraries (not per project).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml


@lru_cache(maxsize=8)
def _position_csv_listing(root_str: str) -> tuple:
    """Cached ``*.csv`` filenames in a positions directory (one listing per dir).

    The shared positions dir lives on a slow networked mount, and the loose
    filename fallback in :meth:`DataManager._find_position_in_root` would
    otherwise re-list it for every CSV-less scan in a batch. Cached for the
    process lifetime; restart to pick up newly-arrived files.
    """
    root = Path(root_str)
    if not root.is_dir():
        return ()
    return tuple(sorted(p.name for p in root.iterdir()
                        if p.is_file() and p.suffix.lower() == ".csv"))

CONFIG_FILENAME = "config.yaml"


def format_detector_label(d: dict) -> str:
    """Dropdown label for a catalog detector: ``name (F1 0.79)``.

    Prefers the holdout F1, falls back to F2 (e.g. the recall-first 1x1
    sessions), and marks unscored detectors. Per-frame (unbinned) detectors get
    a trailing tag so it's clear they don't run in the binned pipeline.
    """
    name = d.get("name", "?")
    f1, f2 = d.get("holdout_f1"), d.get("holdout_f2")
    if f1 is not None:
        score = f"F1 {f1:.2f}"
    elif f2 is not None:
        score = f"F2 {f2:.2f}"
    else:
        score = "unscored"
    tag = " · unbinned" if d.get("pipeline") == "perframe" else ""
    return f"{name} ({score}{tag})"

# Standard project sub-directories created by ``xrd-app init``.
PROJECT_DIRS = [
    "Raw",        # scans.json registry + links to external scan dirs
    "Binned",     # pre-binned xrd_NxN_bins.h5 (per scan)
    "Metadata",   # tth.tiff, reflections.json, grid mappings, gui_state.json
    "Labels",     # per-scan algorithm outputs + manual labels
    "Figures",    # saved PNGs
    "CVEvolve",   # CVEvolve sessions created from the GUI
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
        # Filled adaptively from the data by `scan-detect` — never hard-coded.
        "detector": {"shape": None},
        # Registry of scans in this project: name -> {dir, n_frames, shape}.
        # Mirrors Raw/scans.json; kept here for quick access.
        "scans": {},
        "paths": {
            "raw_dir": "Raw",
            "binned_dir": "Binned",
            "metadata_dir": "Metadata",
            "labels_dir": "Labels",
            "figures_dir": "Figures",
            "cvevolve_dir": "CVEvolve",
        },
        # Absolute paths to external inputs; populated by ``xrd-app link``.
        "data_sources": {
            "raw_root": None,        # parent dir holding many Scan_NNNN/ dirs
            "position_root": None,   # dir holding scan_NNNN_position.csv files
            "raw_scan_dir": None,    # single-scan raw dir (this project's scan)
            "position_csv": None,    # single-scan position CSV
            "tth_map": None,
            "reflections": None,
            "grid_mapping": None,
            "detector_script": None,
        },
        # Map of bin_size (int) -> path to the pre-binned HDF5 file.
        "bins": {},
        "tracking": {
            "enabled": True,
            "mlflow_tracking_uri": f"file://{root / 'CVEvolve' / 'mlruns'}",
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
        """Load the nearest project config at or above ``root``.

        Walks up from ``root`` looking for the first directory containing
        ``config.yaml`` (like ``git`` locating ``.git``), so commands and the GUI
        work from any subdirectory of the project. Falls back to ``root`` itself.
        """
        start = Path(root).resolve()
        project_root = start
        for d in (start, *start.parents):
            if (d / CONFIG_FILENAME).exists():
                project_root = d
                break
        cfg = cls(project_root)
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

    The GUI and CLI commands construct one of these from the project root and ask
    it for paths, e.g. ``dm.tth_map()`` or ``dm.binned_h5(3)``.
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

    def _path(self, paths_key: str, default: str) -> Path:
        return self._abs(self.config.get("paths", paths_key, default=default))

    # ----- standard directories ---------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self._path("raw_dir", "Raw")

    @property
    def binned_dir_root(self) -> Path:
        return self._path("binned_dir", "Binned")

    @property
    def metadata_dir(self) -> Path:
        return self._path("metadata_dir", "Metadata")

    @property
    def labels_dir_root(self) -> Path:
        return self._path("labels_dir", "Labels")

    @property
    def figures_dir(self) -> Path:
        return self._path("figures_dir", "Figures")

    @property
    def cvevolve_dir(self) -> Path:
        return self._path("cvevolve_dir", "CVEvolve")

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

    # ----- scan registry (Raw/scans.json) -----------------------------
    def scans_registry_path(self) -> Path:
        return self.raw_dir / "scans.json"

    def scans_registry(self) -> dict:
        """Read Raw/scans.json: {scan_name: {dir, n_frames, shape}}."""
        p = self.scans_registry_path()
        if p.exists():
            with open(p) as f:
                return json.load(f) or {}
        return {}

    def write_scans_registry(self, registry: dict) -> Path:
        p = self.scans_registry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(registry, f, indent=2)
        return p

    def discover_scans(self, usable_only: bool = False) -> list:
        """List ``Scan_NNNN`` names known to this project.

        Prefers the Raw/scans.json registry; falls back to scanning the per-scan
        subdirectories under ``Binned/`` and ``Labels/``. With ``usable_only``,
        drops registry scans that have no frames (incomplete / no ``XRD/`` files)
        so batch runs skip them.
        """
        reg = self.scans_registry()
        if reg:
            names = sorted(reg.keys(), key=lambda n: self.scan_number_of(n) or 0)
            if usable_only:
                names = [n for n in names
                         if (reg[n].get("n_files") or 0) > 0
                         and (reg[n].get("n_frames") or 0) > 0]
            return names
        found = set()
        for base in (self.binned_dir_root, self.labels_dir_root):
            if base.is_dir():
                for p in base.iterdir():
                    if p.is_dir() and re.fullmatch(r"Scan_\d+", p.name):
                        found.add(p.name)
        return sorted(found, key=lambda n: self.scan_number_of(n) or 0)

    # ----- per-scan directories ---------------------------------------
    def labels_dir(self, scan: object = None) -> Path:
        name = self._scan(scan)
        return self.labels_dir_root / name if name else self.labels_dir_root

    def binned_dir(self, scan: object = None) -> Path:
        name = self._scan(scan)
        return self.binned_dir_root / name if name else self.binned_dir_root

    def metadata_scan_dir(self, scan: object = None) -> Path:
        """Per-scan metadata dir (grid mapping, per-scan reflections/tth)."""
        name = self._scan(scan)
        return (self.metadata_dir / name) if name else self.metadata_dir

    # ----- algorithm output files -------------------------------------
    def peaks_json(self, algo: str, bin_size: int, scan: object = None,
                   variant: Optional[str] = None) -> Path:
        tag = f"_{variant}" if variant else ""
        return self.labels_dir(scan) / f"{algo}_peaks_{bin_size}x{bin_size}{tag}.json"

    def shapes_json(self, algo: str, bin_size: int, scan: object = None,
                    variant: Optional[str] = None) -> Path:
        tag = f"_{variant}" if variant else ""
        return self.labels_dir(scan) / f"{algo}_shapes_{bin_size}x{bin_size}{tag}.json"

    def manual_labels_json(self, scan: object = None) -> Path:
        return self.labels_dir(scan) / "manual_labels.json"

    # ----- resolved input files ---------------------------------------
    def raw_scan_dir(self, override: Optional[str] = None, scan: object = None) -> Path:
        """Directory holding a scan's raw per-frame H5 files.

        Resolution: override -> Raw/scans.json registry -> single linked
        raw_scan_dir (config's own scan) -> raw_root/<scan> -> Raw/<scan>.
        """
        if override:
            return self._abs(override)
        name = self._scan(scan) or ""
        reg = self.scans_registry()
        if name and name in reg and reg[name].get("dir"):
            return self._abs(reg[name]["dir"])
        # A single linked raw_scan_dir only applies to the config's own scan.
        if name == self.scan_name_of(self.config.get("scan", "name")):
            single = self.config.get("data_sources", "raw_scan_dir")
            if single:
                return self._abs(single)
        raw_root = self.config.get("data_sources", "raw_root")
        if raw_root:
            return self._abs(raw_root) / name if name else self._abs(raw_root)
        return self.raw_dir / name if name else self.raw_dir

    def xrd_frames_dir(self, override: Optional[str] = None, scan: object = None) -> Path:
        """Directory of raw per-frame H5 files (the ``XRD/`` subdir if present)."""
        base = self.raw_scan_dir(override, scan)
        sub = base / "XRD"
        return sub if sub.is_dir() else base

    def socketserver_dir(self, override: Optional[str] = None, scan: object = None) -> Path:
        """Directory of the scan's SOCKETSERVER interferometry H5 files.

        Sibling of :meth:`xrd_frames_dir` — the real per-frame stage positions are
        derived from here (see ``core.positions``). Falls back to the scan dir
        itself when there is no ``SOCKETSERVER/`` subdir.
        """
        base = self.raw_scan_dir(override, scan)
        sub = base / "SOCKETSERVER"
        return sub if sub.is_dir() else base

    def position_csv(self, override: Optional[str] = None, scan: object = None) -> Path:
        if override:
            return self._abs(override)
        name = self._scan(scan)
        num = self.scan_number_of(name)
        pos_root = self.config.get("data_sources", "position_root")
        if pos_root and num is not None:
            cand = self._find_position_in_root(self._abs(pos_root), num)
            if cand:
                return cand
        if name == self.scan_name_of(self.config.get("scan", "name")):
            single = self.config.get("data_sources", "position_csv")
            if single:
                return self._abs(single)
        if name:
            local = self.metadata_scan_dir(scan) / "positions.csv"
            if local.exists():
                return local
        return self.metadata_dir / "positions.csv"

    @staticmethod
    def _find_position_in_root(root: Path, num: int) -> Optional[Path]:
        """Locate scan ``num``'s position CSV inside a shared positions directory.

        The beamline writes one ``scan_NNNN_position.csv`` per scan into a common
        dir (e.g. ``Processed/SOCKETSERVER/``). Naming varies across exports —
        ``scan_``/``Scan_``, zero-padded or not — and beamline mounts are
        case-sensitive, so we try the common exact names first, then fall back to
        a loose, case-insensitive scan over ``*position*.csv`` files that carry
        this scan number as a delimited token. Returns the match, or None.
        """
        if not root.is_dir():
            return None
        for nm in (f"scan_{num:04d}_position.csv", f"Scan_{num:04d}_position.csv",
                   f"scan_{num}_position.csv", f"Scan_{num}_position.csv"):
            if (root / nm).exists():
                return root / nm
        # Loose fallback: any *position*.csv whose name carries this scan number
        # (0-padded or not) as its own token, matched case-insensitively. Uses a
        # cached directory listing so a batch over many CSV-less scans pays the
        # (slow, networked) listing once, not per scan.
        token = re.compile(rf"(?<!\d)0*{num}(?!\d)")
        for nm in _position_csv_listing(str(root)):
            if "position" in nm.lower() and token.search(nm):
                return root / nm
        return None

    def _asset(self, name: str) -> Path:
        """Path to a file bundled with the package (shared defaults)."""
        return Path(__file__).parent / "assets" / name

    def tth_map(self, override: Optional[str] = None, scan: object = None) -> Path:
        """2θ-per-pixel map: override -> config -> per-scan -> project -> bundled."""
        if override:
            return self._abs(override)
        configured = self.config.get("data_sources", "tth_map")
        if configured:
            return self._abs(configured)
        per_scan = self.metadata_scan_dir(scan) / "tth.tiff"
        if per_scan.exists():
            return per_scan
        proj = self.metadata_dir / "tth.tiff"
        return proj if proj.exists() else self._asset("tth.tiff")

    def reflection_source(self, scan: object = None) -> Optional[Path]:
        """The user-selected reflections source for this scan, if any.

        Stored per scan under ``data_sources.reflections_by_scan`` and chosen via
        the host header "Reflections:" selector or Setup → Load reflections…. The
        value points at a ``reflections.py`` (its sibling ``.json`` is derived).
        """
        name = self._scan(scan)
        by_scan = self.config.get("data_sources", "reflections_by_scan", default={})
        if name and isinstance(by_scan, dict) and by_scan.get(name):
            return self._abs(by_scan[name])
        return None

    def set_reflection_source(self, path, scan: object = None) -> None:
        """Persist the chosen reflections source for ``scan`` (or clear if None)."""
        name = self._scan(scan)
        if not name:
            return
        ds = self.config.data.setdefault("data_sources", {})
        by_scan = ds.setdefault("reflections_by_scan", {})
        if path is None:
            by_scan.pop(name, None)
        else:
            by_scan[name] = str(Path(path))
        self.config.save()

    def clear_reflection_source(self, scan: object = None) -> None:
        self.set_reflection_source(None, scan)

    def reflections_json(self, scan: object = None) -> Path:
        """Reflection data (JSON): per-scan selection -> per-scan -> project."""
        chosen = self.reflection_source(scan)
        if chosen is not None:
            return chosen.with_suffix(".json")
        per_scan = self.metadata_scan_dir(scan) / "reflections.json"
        if per_scan.exists():
            return per_scan
        return self.metadata_dir / "reflections.json"

    def reflections(self, override: Optional[str] = None, scan: object = None) -> Path:
        """Reflections module (.py loader) the pipeline imports.

        override -> per-scan selection -> config -> per-scan
        Metadata/<scan>/reflections.py -> project Metadata/reflections.py ->
        bundled asset.
        """
        if override:
            return self._abs(override)
        chosen = self.reflection_source(scan)
        if chosen is not None:
            return chosen
        configured = self.config.get("data_sources", "reflections")
        if configured:
            return self._abs(configured)
        per_scan = self.metadata_scan_dir(scan) / "reflections.py"
        if per_scan.exists():
            return per_scan
        proj = self.metadata_dir / "reflections.py"
        return proj if proj.exists() else self._asset("reflections.py")

    def grid_mapping(self, override: Optional[str] = None, bin_size: Optional[int] = None,
                     scan: object = None, variant: Optional[str] = None) -> Path:
        if override:
            return self._abs(override)
        if not variant:
            configured = self.config.get("data_sources", "grid_mapping")
            if configured:
                return self._abs(configured)
        sdir = self.metadata_scan_dir(scan)
        tag = f"_{variant}" if variant else ""
        if bin_size is not None:
            return sdir / f"grid_mapping_{bin_size}x{bin_size}{tag}.json"
        return sdir / f"grid_mapping{tag}.json"

    def binned_h5(self, bin_size: int, override: Optional[str] = None,
                  scan: object = None, variant: Optional[str] = None) -> Path:
        if override:
            return self._abs(override)
        tag = f"_{variant}" if variant else ""
        return self.binned_dir(scan) / f"xrd_{bin_size}x{bin_size}_bins{tag}.h5"

    # ----- compatibility shims for the embedded legacy GUIs -----------
    # The viewer/device_map/orientation modules were written against the old
    # DataManager API (results_dir / holdout_dir / bins_h5). Keep thin aliases
    # so they resolve correctly against the new project tree unchanged.
    def results_dir(self, scan: object = None) -> Path:
        return self.labels_dir(scan)

    @property
    def holdout_dir(self) -> Path:
        return self.metadata_dir

    def bins_h5(self, bin_size: int, override: Optional[str] = None,
                scan: object = None) -> Path:
        return self.binned_h5(bin_size, override, scan)

    # ----- detector / algorithm libraries -----------------------------
    def algorithms_dir(self, kind: str = "peak") -> Path:
        """Directory of a bundled algorithm library shipped with the package."""
        sub = {"peak": "PeakAlgorithms", "shape": "ShapeAlgorithms",
               "combined": "CombinedAlgorithms"}.get(kind, "PeakAlgorithms")
        return Path(__file__).parent / sub

    def detectors_dir(self) -> Path:
        """The peak-algorithm library directory (holds catalog.json)."""
        return self.algorithms_dir("peak")

    def combined_dir(self) -> Path:
        """The combined-algorithm library directory (holds catalog.json)."""
        return self.algorithms_dir("combined")

    def shapes_dir(self) -> Path:
        """The shape-algorithm library directory (holds catalog.json)."""
        return self.algorithms_dir("shape")

    def load_catalog(self, kind: str = "peak") -> dict:
        """Read the ``catalog.json`` of an algorithm library (peak/shape/combined)."""
        cat = self.algorithms_dir(kind) / "catalog.json"
        if cat.exists():
            with open(cat) as f:
                return json.load(f)
        return {"detectors": []}

    def load_detector_catalog(self) -> dict:
        return self.load_catalog("peak")

    def list_detectors(self, bin_size: Optional[int] = None) -> list:
        """List bundled *detector* entries (excludes support modules).

        An entry with a null/missing ``bin_size`` applies to any bin size (the
        flat library is bin-agnostic), so it is always included.
        """
        size = f"{bin_size}x{bin_size}" if bin_size else None
        out = []
        for d in self.load_detector_catalog().get("detectors", []):
            if d.get("role") != "detector":
                continue
            entry_size = d.get("bin_size")
            if size and entry_size and entry_size != size:
                continue
            out.append(d)
        return out

    def best_detector(self, bin_size: int) -> Optional[Path]:
        """Path to the highest-scoring bundled detector for ``bin_size``.

        Only binned detectors are eligible — per-frame (unbinned) detectors run
        through a different path and would crash the binned ``peaks`` pipeline.
        """
        def binned(dets):
            return [d for d in dets if d.get("pipeline") != "perframe"]
        candidates = binned(self.list_detectors(bin_size)) or \
            binned(self.list_detectors(3)) or binned(self.list_detectors())
        if not candidates:
            return None
        candidates.sort(
            key=lambda d: (d.get("holdout_f1") if d.get("holdout_f1") is not None else -1,
                           d.get("name") == "5x5_tophat_band_adaptive_snr"),
            reverse=True)
        return self.detectors_dir() / candidates[0]["file"]

    def resolve_detector_name(self, name: str, bin_size: Optional[int] = None) -> Optional[Path]:
        """Resolve a bare detector name from the library."""
        stem = name[:-3] if name.endswith(".py") else name
        matches = [d for d in self.list_detectors() if d["name"] == stem]
        if bin_size:
            sized = [d for d in matches if d.get("bin_size") == f"{bin_size}x{bin_size}"]
            matches = sized or matches
        if matches:
            return self.detectors_dir() / matches[0]["file"]
        return None

    def detector_script(self, override: Optional[str] = None,
                        bin_size: Optional[int] = None) -> Path:
        """Resolve the detector script.

        Precedence: explicit path/name -> config -> best bundled detector.
        """
        if override:
            p = Path(override)
            if p.exists():
                return self._abs(override)
            byname = self.resolve_detector_name(override, bin_size)
            if byname:
                return byname
            return self._abs(override)
        configured = self.config.get("data_sources", "detector_script")
        if configured:
            return self._abs(configured)
        bundled = self.best_detector(bin_size or 3)
        return bundled if bundled else (self.detectors_dir() / "detector.py")

    # ----- combined algorithm library (peak + shape in one pass) -------
    def list_combined(self) -> list:
        """List combined (per-frame) algorithm entries from CombinedAlgorithms."""
        return [d for d in self.load_catalog("combined").get("detectors", [])
                if d.get("role") == "detector"]

    def resolve_combined_name(self, name: str) -> Optional[Path]:
        """Resolve a combined-algorithm name to its script path."""
        stem = name[:-3] if name.endswith(".py") else name
        for d in self.list_combined():
            if d["name"] == stem:
                return self.combined_dir() / d["file"]
        return None

    def combined_script(self, override: str) -> Path:
        """Resolve a combined-algorithm script: explicit path -> library name."""
        p = Path(override)
        if p.exists():
            return self._abs(override)
        byname = self.resolve_combined_name(override)
        return byname if byname else self._abs(override)

    # ----- shape algorithm library (cross-bin link + shape filter) ------
    def list_shapes(self) -> list:
        """List bundled *shape* algorithm entries from ShapeAlgorithms."""
        return [d for d in self.load_catalog("shape").get("detectors", [])
                if d.get("role") == "shape"]

    def resolve_shape_name(self, name: str) -> Optional[Path]:
        """Resolve a bare shape-algorithm name to its script path."""
        stem = name[:-3] if name.endswith(".py") else name
        for d in self.list_shapes():
            if d["name"] == stem:
                return self.shapes_dir() / d["file"]
        return None

    def best_shape(self) -> Optional[Path]:
        """Path to the highest-scoring bundled shape algorithm (default 'gaussian')."""
        shapes = self.list_shapes()
        if not shapes:
            return None
        shapes.sort(
            key=lambda d: (d.get("holdout_f1") if d.get("holdout_f1") is not None else -1,
                           d.get("name") == "gaussian"),
            reverse=True)
        return self.shapes_dir() / shapes[0]["file"]

    def shape_script(self, override: Optional[str] = None) -> Path:
        """Resolve a shape-algorithm script.

        Precedence: explicit path/name -> best bundled shape algorithm.
        """
        if override:
            p = Path(override)
            if p.exists():
                return self._abs(override)
            byname = self.resolve_shape_name(override)
            if byname:
                return byname
            return self._abs(override)
        bundled = self.best_shape()
        return bundled if bundled else (self.shapes_dir() / "gaussian.py")
