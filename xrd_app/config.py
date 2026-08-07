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
Algorithms are discovered from the package's bundled libraries and writable
project-owned ``Algorithms/`` libraries.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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


def safe_component(value: object, *, normalize: bool = False, label: str = "name") -> str:
    """Return a safe filesystem component, optionally normalizing punctuation."""
    text = str(value).strip()
    if normalize:
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    elif (not text or text in {".", ".."} or "/" in text or "\\" in text
          or Path(text).is_absolute()):
        raise ValueError(f"Invalid {label}: expected one non-empty path component")
    if not text:
        raise ValueError(f"Invalid {label}: must contain letters or numbers")
    return text


def format_detector_label(d: dict) -> str:
    """Dropdown label for a catalog detector, displaying F2 when available."""
    name = d.get("name", "?")
    f1, f2 = d.get("holdout_f1"), d.get("holdout_f2")
    if f2 is not None:
        score = f"F2 {f2:.2f}"
    elif f1 is not None:
        score = f"F1 {f1:.2f}"
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
    "Algorithms", # project-owned user detector modules and catalogs
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
            "detector_script": "Algorithms/PeakAlgorithms/default_detector.py",
        },
        # Map of bin_size (int) -> path to the pre-binned HDF5 file.
        "bins": {},
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
        """Create the standard tree and seed project-owned editable defaults."""
        for d in PROJECT_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)

        package = Path(__file__).parent
        metadata = self.root / "Metadata"
        algorithms = self.root / "Algorithms" / "PeakAlgorithms"
        algorithms.mkdir(parents=True, exist_ok=True)
        defaults = (
            (package / "assets" / "reflections.json", metadata / "reflections.json"),
            (package / "assets" / "tth.tiff", metadata / "tth.tiff"),
            (package / "PeakAlgorithms" / "5x5_tophat_band_adaptive_snr.py",
             algorithms / "default_detector.py"),
        )
        for source, destination in defaults:
            if not destination.exists():
                shutil.copy2(source, destination)

        catalog = algorithms / "catalog.json"
        if not catalog.exists():
            with catalog.open("w") as stream:
                json.dump({
                    "description": "Project-owned peak detectors. Copy default_detector.py "
                                   "and add another catalog entry to include an alternative.",
                    "metric": "holdout_f2 is the primary ranking metric",
                    "detectors": [{
                        "name": "default_detector",
                        "file": "default_detector.py",
                        "role": "detector",
                        "bin_size": None,
                        "holdout_f1": None,
                        "holdout_f2": None,
                        "source": "project default",
                        "notes": "Editable copy of the bundled production detector.",
                    }],
                }, stream, indent=2)
                stream.write("\n")

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
        value = str(self.config.get("paths", paths_key, default=default)).strip()
        if not value or "\\" in value:
            raise ValueError(f"Invalid paths.{paths_key}: path must remain under project root")
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (self.root / path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise ValueError(
                f"Invalid paths.{paths_key}: path must remain under project root") from None
        if path == self.root:
            raise ValueError(f"Invalid paths.{paths_key}: path must name a subdirectory")
        return path

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
        """Normalize an integer/``Scan_NNNN`` identifier and reject path names."""
        if scan is None:
            return None
        if isinstance(scan, int):
            if scan < 0:
                raise ValueError("Invalid scan name: scan number must be non-negative")
            return f"Scan_{scan:04d}"
        value = safe_component(scan, label="scan name")
        if value.isdigit():
            return f"Scan_{int(value):04d}"
        match = re.fullmatch(r"Scan_(\d+)", value)
        if match:
            return f"Scan_{int(match.group(1)):04d}"
        raise ValueError("Invalid scan name: expected an integer or Scan_NNNN")

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

    def discover_scans(self, usable_only: bool = False,
                       selected_only: bool = False) -> list:
        """List ``Scan_NNNN`` names known to this project.

        Prefers the Raw/scans.json registry; falls back to scanning the per-scan
        subdirectories under ``Binned/`` and ``Labels/``. With ``usable_only``,
        drops registry scans that have no frames (incomplete / no ``XRD/`` files)
        so batch runs skip them. With ``selected_only``, further narrows to the
        user-curated :meth:`visible_scans` subset (what the GUI Scan selector
        shows) — a stale/empty selection is ignored so the list never goes empty.
        """
        reg = self.scans_registry()
        if reg:
            names = sorted(reg.keys(), key=lambda n: self.scan_number_of(n) or 0)
            if usable_only:
                names = [n for n in names
                         if (reg[n].get("n_files") or 0) > 0
                         and (reg[n].get("n_frames") or 0) > 0]
        else:
            found = set()
            for base in (self.binned_dir_root, self.labels_dir_root):
                if base.is_dir():
                    for p in base.iterdir():
                        if p.is_dir() and re.fullmatch(r"Scan_\d+", p.name):
                            found.add(p.name)
            names = sorted(found, key=lambda n: self.scan_number_of(n) or 0)
        if selected_only:
            vis = self.visible_scans()
            if vis:
                keep = set(vis)
                filtered = [n for n in names if n in keep]
                if filtered:  # ignore a stale selection that hides everything
                    names = filtered
        return names

    def visible_scans(self) -> Optional[list]:
        """User-curated subset of registered scans shown in the GUI selector.

        Stored as a top-level ``visible_scans`` list in ``config.yaml`` and set
        via Setup → "Choose scans to show…". ``None`` (absent or empty) means
        show every registered scan. Names are normalized to ``Scan_NNNN``.
        """
        v = self.config.get("visible_scans")
        if not v:
            return None
        names = [self.scan_name_of(n) for n in v]
        names = [n for n in names if n]
        return names or None

    def set_visible_scans(self, names) -> None:
        """Persist the curated visible-scan subset (None/empty clears → all)."""
        if names:
            norm = []
            for n in names:
                nm = self.scan_name_of(n)
                if nm and nm not in norm:
                    norm.append(nm)
            self.config.data["visible_scans"] = norm
        else:
            self.config.data.pop("visible_scans", None)
        self.config.save()

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

    def local_cache_dir(self, scan: object = None) -> Path:
        """Machine-local cache directory, isolated by project identity and scan."""
        base = Path(os.environ.get("XRD_APP_CACHE_DIR", "~/.cache/xrd-app")).expanduser()
        project_id = hashlib.sha256(str(self.root.resolve()).encode()).hexdigest()[:16]
        return base / project_id / (self._scan(scan) or "project")

    def local_frame_cache_h5(self, scan: object = None) -> Path:
        """Sparse lossless raw-frame cache used by interactive viewers."""
        return self.local_cache_dir(scan) / "xrd_frame_cache.h5"

    def local_grid_mapping(self, bin_size: int, scan: object = None) -> Path:
        """Grid mapping copied beside the machine-local frame cache."""
        return self.local_cache_dir(scan) / f"grid_mapping_{bin_size}x{bin_size}.h5"

    # ----- algorithm output files -------------------------------------
    def peaks_path(self, algo: str, bin_size: int, scan: object = None,
                   variant: Optional[str] = None) -> Path:
        algo = safe_component(algo, label="algorithm name")
        tag = f"_{safe_component(variant, label='variant')}" if variant else ""
        return self.labels_dir(scan) / f"{algo}_peaks_{bin_size}x{bin_size}{tag}.h5"

    def shapes_path(self, algo: str, bin_size: int, scan: object = None,
                    variant: Optional[str] = None) -> Path:
        algo = safe_component(algo, label="algorithm name")
        tag = f"_{safe_component(variant, label='variant')}" if variant else ""
        return self.labels_dir(scan) / f"{algo}_shapes_{bin_size}x{bin_size}{tag}.h5"

    def hd_map_path(self, algo: str, bin_size: int, scan: object = None,
                    variant: Optional[str] = None) -> Path:
        """High-def (1x1) intensity map sampled beneath a binned feature map."""
        algo = safe_component(algo, label="algorithm name")
        tag = f"_{safe_component(variant, label='variant')}" if variant else ""
        return self.labels_dir(scan) / f"{algo}_hdmap_{bin_size}x{bin_size}{tag}.h5"

    def roi_map_path(self, name: str, bin_size: int, scan: object = None) -> Path:
        """Dedicated ROI > Shape catalog, intentionally separate from shapes."""
        tag = safe_component(name, normalize=True, label="catalog name")
        return self.labels_dir(scan) / f"{tag}_roimap_{bin_size}x{bin_size}.h5"

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

    def me7_dir(self, override: Optional[str] = None, scan: object = None) -> Optional[Path]:
        """Directory of the scan's ME7 (XSPRESS3) XRF H5 files, or ``None``.

        Prefers a **local** ``<project>/Raw/<scan>/ME7`` copy (as mirrored for the
        focus scans) over the config's ``data_sources`` raw path, which may point
        at the beamline share/LAN (``/net/micdata`` or ``/mnt/z``) that isn't
        mounted here. Sibling of :meth:`xrd_frames_dir`.
        """
        if override:
            p = self._abs(override)
            return p if p and p.is_dir() else None
        name = self.scan_name_of(scan) if scan is not None else self.scan_name
        # 1) local project copy (fast disk, mirrors the /mnt/z layout)
        if name:
            local = self.root / "Raw" / name / "ME7"
            if local.is_dir():
                return local
        # 2) sibling of the resolved raw XRD dir (share/LAN or elsewhere)
        try:
            xrd = self.xrd_frames_dir(scan=scan)
            cand = (xrd.parent if xrd.name.upper() == "XRD" else xrd) / "ME7"
            if cand.is_dir():
                return cand
        except Exception:
            pass
        return None

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
            mdir = self.metadata_scan_dir(scan)
            # A placed Lozano position h5 wins over a CSV (loaders dispatch on
            # suffix; see io.load_positions_xy).
            local_h5 = mdir / "positions.h5"
            if local_h5.exists():
                return local_h5
            local = mdir / "positions.csv"
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
        """The user-selected JSON reflection source, if any.

        Stored per scan under ``data_sources.reflections_by_scan`` and chosen via
        the host header "Reflections:" selector or Setup → Load reflections….
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
        """Resolve reflection JSON for editors and pipeline consumers."""
        chosen = self.reflection_source(scan)
        if chosen is not None:
            return chosen
        configured = self.config.get("data_sources", "reflections")
        if configured:
            return self._abs(configured)
        per_scan = self.metadata_scan_dir(scan) / "reflections.json"
        if per_scan.exists():
            return per_scan
        project = self.metadata_dir / "reflections.json"
        return project if project.exists() else self._asset("reflections.json")

    def reflections(self, override: Optional[str] = None, scan: object = None) -> Path:
        """Resolve reflection JSON: override, selected, configured, scan, project, bundled."""
        return self._abs(override) if override else self.reflections_json(scan)

    def grid_mapping(self, override: Optional[str] = None, bin_size: Optional[int] = None,
                     scan: object = None, variant: Optional[str] = None) -> Path:
        if override:
            return self._abs(override)
        if bin_size is None and scan is None and variant is None:
            configured = self.config.get("data_sources", "grid_mapping")
            if configured:
                return self._abs(configured)
        sdir = self.metadata_scan_dir(scan)
        tag = f"_{safe_component(variant, label='variant')}" if variant else ""
        stem = (f"grid_mapping_{bin_size}x{bin_size}{tag}"
                if bin_size is not None else f"grid_mapping{tag}")
        return sdir / f"{stem}.h5"

    def unbinned_archive_h5(self, override: Optional[str] = None,
                            scan: object = None) -> Path:
        """Lossless acquisition-order detector-frame archive for a scan."""
        if override:
            return self._abs(override)
        return self.binned_dir(scan) / "xrd_unbinned_archive.h5"

    def binned_h5(self, bin_size: int, override: Optional[str] = None,
                  scan: object = None, variant: Optional[str] = None) -> Path:
        if override:
            return self._abs(override)
        tag = f"_{safe_component(variant, label='variant')}" if variant else ""
        return self.binned_dir(scan) / f"xrd_{bin_size}x{bin_size}_bins{tag}.h5"

    def xrf_product(self, scan: object = None) -> Path:
        """Saved XRF element-map product for a scan.

        Written by ``xrd-app xrf`` into the per-scan Metadata dir as
        ``<scan>_xrf.npz``. One product per scan, computed at 1×1 so it can
        underlay a device map at any bin size (the maps are scaled onto the bin
        grid). The GUI loads this via ``core.xrf.load_product`` — never raw ME7.
        """
        name = self.scan_name_of(scan) if scan is not None else self.scan_name
        return self.metadata_scan_dir(scan) / f"{name}_xrf.npz"

    def xrf_points_product(self, scan: object = None) -> Path:
        """Per-frame XRF spectrum store for a scan (``<scan>_xrf_points.npz``).

        Compact, lossless per-global-frame summed spectra (see
        ``core.xrf.build_point_store``). Lets the Shape/Verify per-frame histogram
        read a small file instead of raw ME7. Loaded lazily — separate from the
        small ``xrf_product`` maps the device-view underlay uses.
        """
        name = self.scan_name_of(scan) if scan is not None else self.scan_name
        return self.metadata_scan_dir(scan) / f"{name}_xrf_points.npz"

    # ----- detector / algorithm libraries -----------------------------
    @staticmethod
    def _algorithm_subdir(kind: str) -> str:
        return {"peak": "PeakAlgorithms", "shape": "ShapeAlgorithms",
                "combined": "CombinedAlgorithms"}.get(kind, "PeakAlgorithms")

    def algorithms_dir(self, kind: str = "peak") -> Path:
        """Directory of a bundled algorithm library shipped with the package."""
        return Path(__file__).parent / self._algorithm_subdir(kind)

    def project_algorithms_dir(self, kind: str = "peak") -> Path:
        """Writable project-owned algorithm library."""
        return self.root / "Algorithms" / self._algorithm_subdir(kind)

    def detectors_dir(self) -> Path:
        """The bundled peak-algorithm library directory."""
        return self.algorithms_dir("peak")

    def combined_dir(self) -> Path:
        """The bundled combined-algorithm library directory."""
        return self.algorithms_dir("combined")

    def shapes_dir(self) -> Path:
        """The bundled shape-algorithm library directory."""
        return self.algorithms_dir("shape")

    def load_catalog(self, kind: str = "peak") -> dict:
        """Merge bundled and project catalogs, with project entries taking precedence."""
        merged = {}
        description = None
        for library in (self.algorithms_dir(kind), self.project_algorithms_dir(kind)):
            cat = library / "catalog.json"
            if not cat.exists():
                continue
            with open(cat) as f:
                data = json.load(f)
            description = data.get("description", description)
            for original in data.get("detectors", []):
                entry = dict(original)
                entry["_library_dir"] = library
                entry["_path"] = library / entry.get("file", "")
                merged[(entry.get("name"), entry.get("bin_size"))] = entry
        return {"description": description, "detectors": list(merged.values())}

    def load_detector_catalog(self) -> dict:
        return self.load_catalog("peak")

    def list_detectors(self, bin_size: Optional[int] = None) -> list:
        """List cataloged modules compatible with the production binned API."""
        from .core.processing import load_detector

        size = f"{bin_size}x{bin_size}" if bin_size else None
        out = []
        for d in self.load_detector_catalog().get("detectors", []):
            if d.get("role") != "detector" or d.get("pipeline") == "perframe":
                continue
            entry_size = d.get("bin_size")
            if size and entry_size and entry_size != size:
                continue
            path = d.get("_path")
            if not path or not Path(path).is_file():
                continue
            try:
                load_detector(path)
            except Exception:
                continue
            out.append(d)
        return out

    @staticmethod
    def _detector_supports_bin(entry: dict, bin_size: int) -> bool:
        declared = entry.get("bin_size")
        return declared is None or declared == f"{bin_size}x{bin_size}"

    def best_detector(self, bin_size: int) -> Optional[Path]:
        """Path to the highest-scoring detector compatible with ``bin_size``.

        Only an exact declared bin size or an explicitly generic (null) catalog
        entry is eligible. Per-frame detectors use a different pipeline.
        """
        candidates = self.list_detectors(bin_size)
        if not candidates:
            return None
        candidates.sort(
            key=lambda d: (d.get("holdout_f2") if d.get("holdout_f2") is not None else -1,
                           d.get("holdout_f1") if d.get("holdout_f1") is not None else -1,
                           d.get("name") == "5x5_tophat_band_adaptive_snr"),
            reverse=True)
        return Path(candidates[0]["_path"])

    def resolve_detector_name(self, name: str, bin_size: Optional[int] = None) -> Optional[Path]:
        """Resolve a compatible bare detector name from the library."""
        stem = name[:-3] if name.endswith(".py") else name
        matches = [d for d in self.list_detectors() if d["name"] == stem]
        if bin_size is not None:
            matches = [d for d in matches
                       if self._detector_supports_bin(d, bin_size)]
        if matches:
            return Path(matches[0]["_path"])
        return None

    def _catalog_entry_for_path(self, path: Path) -> Optional[dict]:
        """Return metadata only when ``path`` is an actual cataloged script."""
        target = path.resolve()
        for entry in self.load_detector_catalog().get("detectors", []):
            catalog_path = entry.get("_path")
            if catalog_path and Path(catalog_path).resolve() == target:
                return entry
        return None

    def _checked_detector_path(self, value: str, bin_size: int) -> Path:
        path = self._abs(value)
        entry = self._catalog_entry_for_path(path) if path.exists() else None
        if entry is None and not path.exists():
            stem = value[:-3] if value.endswith(".py") else value
            matches = [d for d in self.list_detectors() if d["name"] == stem]
            compatible = [d for d in matches
                          if self._detector_supports_bin(d, bin_size)]
            if compatible:
                return Path(compatible[0]["_path"])
            if matches:
                entry = matches[0]
        if entry is not None and not self._detector_supports_bin(entry, bin_size):
            raise ValueError(
                f"Detector {entry.get('name')!r} declares bin_size "
                f"{entry.get('bin_size')!r}; requested {bin_size}x{bin_size}")
        return path

    def detector_script(self, override: Optional[str] = None,
                        bin_size: Optional[int] = None) -> Path:
        """Resolve a detector with explicit catalog bin compatibility.

        Precedence: explicit path/name -> config -> best cataloged detector.
        Uncataloged external scripts are treated as generic because they have no
        declared bin-size constraint.
        """
        if bin_size is None:
            raise ValueError("bin_size is required to resolve a detector script")
        if override:
            return self._checked_detector_path(override, bin_size)
        configured = self.config.get("data_sources", "detector_script")
        if configured:
            return self._checked_detector_path(configured, bin_size)
        bundled = self.best_detector(bin_size)
        if bundled is None:
            raise FileNotFoundError(
                f"No detector supports requested bin size {bin_size}x{bin_size}")
        return bundled

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
