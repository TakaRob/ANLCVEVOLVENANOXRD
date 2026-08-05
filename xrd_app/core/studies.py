"""Discover and catalog rocking-study result sets so the GUI can pick between them.

A *study* is a directory holding the cross-scan artifacts produced by the
``aggregate → track → rocking → predict → combined-device`` pipeline (and,
optionally, the q-space RSM). A project can hold several — e.g. ``Study/`` (3×3)
and ``Study_1x1/`` (1×1), or one per scan subset. This module finds them and
maintains an optional ``studies.json`` registry (name + notes + provenance) at
the project root, so the Reciprocal-Space and Rocking-Study tabs can offer a
study selector instead of a hardcoded ``Study/`` path.

The registry is *advisory*: discovery works with or without it (any directory
carrying a primary artifact is listed), and the registry only overlays a
human-friendly name, notes, and provenance keyed by directory. ``run-study``
writes an entry when it finishes so the newest analysis shows up named.

Pure module — no Qt, no click. ``numpy`` is imported lazily (only to peek at a
grid shape); metadata is read from the ``.summary.json`` sidecars the pipeline
already writes, so this stays cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

REGISTRY_NAME = "studies.json"

# A directory is a "study" if it holds at least one of these. Ordered by how
# defining they are (rsm/rocking/combined are study-only; features is shared).
PRIMARY_ARTIFACTS = (
    "rsm.npz", "rocking_curves.csv", "combined_device.npz",
    "tracks.h5", "features.csv",
)

# logical artifact name -> the file (relative to the study dir) that proves it ran
ARTIFACT_FILES = {
    "features": "features.csv",
    "device_map": "device_map.csv",
    "tracks": "tracks.h5",
    "rocking": "rocking_curves.csv",
    "prediction": "prediction_report.md",
    "combined_device": "combined_device.npz",
    "rsm": "rsm.npz",
}


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def artifacts_present(study_dir) -> dict:
    """Map each logical artifact to whether its file exists under ``study_dir``.

    ``qspace`` is a directory of ``*_qmap.npz``; it's reported as present when the
    directory holds at least one q-map.
    """
    d = Path(study_dir)
    present = {name: (d / fname).exists() for name, fname in ARTIFACT_FILES.items()}
    qdir = d / "qspace"
    present["qspace"] = qdir.is_dir() and any(qdir.glob("*_qmap.npz"))
    return present


def is_study_dir(path) -> bool:
    """True if ``path`` is a directory carrying at least one primary artifact."""
    d = Path(path)
    return d.is_dir() and any((d / a).exists() for a in PRIMARY_ARTIFACTS)


def read_meta(study_dir) -> dict:
    """Best-effort provenance (bin_size, scans, thetas, grid) from sidecar JSONs.

    Reads small summary sidecars and HDF5 track metadata without loading arrays.
    """
    d = Path(study_dir)
    meta: dict = {}

    rsm = _load_json(d / "rsm.summary.json")
    if rsm:
        for k in ("scans", "thetas", "grid_shape", "q_ranges"):
            if rsm.get(k) is not None:
                meta[k] = rsm[k]

    cdv = _load_json(d / "combined_device.summary.json")
    if cdv:
        meta.setdefault("thetas", cdv.get("thetas"))
        if cdv.get("reflections") is not None:
            meta["reflections"] = cdv["reflections"]
        if cdv.get("n_rows") is not None:
            meta["grid_rc"] = [cdv.get("n_rows"), cdv.get("n_cols")]
        if cdv.get("n_tracks") is not None:
            meta["n_tracks"] = cdv["n_tracks"]

    from . import result_store
    tracks_path = d / "tracks.h5"
    tracks = result_store.metadata(tracks_path) if tracks_path.exists() else None
    if tracks:
        if tracks.get("bin_size") is not None:
            meta["bin_size"] = tracks["bin_size"]
        if tracks.get("n_tracks") is not None:
            meta.setdefault("n_tracks", tracks["n_tracks"])

    # drop empties so callers can use `.get(...)` cleanly
    return {k: v for k, v in meta.items() if v not in (None, [], {})}


# ── registry (studies.json) ──────────────────────────────────────────────────
def registry_path(project_root) -> Path:
    return Path(project_root) / REGISTRY_NAME


def load_registry(project_root) -> dict:
    """Load ``studies.json`` (``{"studies": [...]}``), or an empty registry."""
    reg = _load_json(registry_path(project_root))
    if not isinstance(reg, dict) or not isinstance(reg.get("studies"), list):
        return {"studies": []}
    return reg


def save_registry(project_root, reg: dict) -> Path:
    p = registry_path(project_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(reg, f, indent=2)
    return p


def _rel(project_root, path) -> str:
    """Study path relative to the project root (POSIX), for a stable registry key."""
    try:
        return Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def register_study(project_root, path, name=None, notes=None,
                   created=None, extra: Optional[dict] = None) -> dict:
    """Add or update a registry entry keyed by directory (relative to root).

    Only the human-authored fields (name, notes, created, extra) are stored;
    everything else is re-discovered on read, so the registry never goes stale
    against the files on disk.
    """
    reg = load_registry(project_root)
    rel = _rel(project_root, path)
    entry = next((e for e in reg["studies"] if e.get("path") == rel), None)
    if entry is None:
        entry = {"path": rel}
        reg["studies"].append(entry)
    if name is not None:
        entry["name"] = name
    if notes is not None:
        entry["notes"] = notes
    if created is not None:
        entry.setdefault("created", created)
    if extra:
        entry.update(extra)
    save_registry(project_root, reg)
    return entry


# ── discovery + merge ────────────────────────────────────────────────────────
def discover_dirs(project_root, max_depth: int = 1) -> List[Path]:
    """Directories under ``project_root`` that look like studies.

    Scans the root and (up to ``max_depth``) its subdirectories. Skips the
    standard project data dirs, which are never studies.
    """
    root = Path(project_root)
    skip = {"Raw", "Binned", "Metadata", "Labels", "Figures", "CVEvolve",
            "mlruns", "__pycache__", ".git", "qspace"}
    found: List[Path] = []
    if is_study_dir(root):
        found.append(root)

    def _walk(base: Path, depth: int):
        if depth < 0 or not base.is_dir():
            return
        for child in sorted(base.iterdir()):
            if not child.is_dir() or child.name in skip or child.name.startswith("."):
                continue
            if is_study_dir(child):
                found.append(child)
            _walk(child, depth - 1)

    _walk(root, max_depth - 1)
    # de-dup while preserving order (root may re-appear via recursion)
    seen, uniq = set(), []
    for p in found:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def list_studies(project_root) -> List[dict]:
    """Every study in the project, merged with registry name/notes/provenance.

    Each entry is a dict: ``path`` (relative), ``abs_path``, ``name``,
    ``notes``, ``created``, ``artifacts`` (name→bool), plus any provenance from
    :func:`read_meta` (bin_size, scans, thetas, grid…). Registry-only entries
    whose directory has since disappeared are dropped.
    """
    root = Path(project_root)
    reg = {e.get("path"): e for e in load_registry(project_root).get("studies", [])}

    out: List[dict] = []
    for d in discover_dirs(project_root):
        rel = _rel(project_root, d)
        overlay = reg.get(rel, {})
        entry = {
            "path": rel,
            "abs_path": str(d.resolve()),
            "name": overlay.get("name") or (d.name if d != root else root.name),
            "notes": overlay.get("notes", ""),
            "created": overlay.get("created"),
            "artifacts": artifacts_present(d),
        }
        entry.update(read_meta(d))
        # registry-supplied provenance wins over discovery for the human fields
        for k in ("bin_size", "scans", "thetas"):
            if overlay.get(k) is not None:
                entry[k] = overlay[k]
        out.append(entry)
    return out


# ── result loaders (for the Rocking-Study viewers) ───────────────────────────
def _coerce(v):
    """CSV cell → float when it looks numeric, else the original string."""
    if v in ("", None):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def load_rocking_curves(study_dir) -> List[dict]:
    """Read ``rocking_curves.csv`` (one row per track) with numeric coercion."""
    import csv as _csv
    p = Path(study_dir) / "rocking_curves.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return [{k: _coerce(v) for k, v in row.items()} for row in _csv.DictReader(f)]


def load_tracks(study_dir) -> List[dict]:
    """Read the full HDF5 track list with per-theta members."""
    from . import result_store
    data = result_store.load(Path(study_dir) / "tracks.h5")
    return (data or {}).get("tracks", []) if isinstance(data, dict) else []


def load_combined_device(study_dir) -> Optional[dict]:
    """Load ``combined_device.npz`` into a dict of arrays (or None if absent)."""
    import numpy as np
    p = Path(study_dir) / "combined_device.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_prediction_report(study_dir) -> Optional[str]:
    """Return the prediction report markdown text, or None if not written yet."""
    p = Path(study_dir) / "prediction_report.md"
    try:
        return p.read_text() if p.exists() else None
    except Exception:
        return None


def describe(entry: dict) -> str:
    """One-line human summary for a study entry (used in dropdowns/CLI)."""
    bits = []
    if entry.get("bin_size") is not None:
        bits.append(f"{entry['bin_size']}×{entry['bin_size']}")
    scans = entry.get("scans")
    if scans:
        bits.append(f"{len(scans)} scans")
    thetas = entry.get("thetas")
    if thetas:
        vals = [t for t in thetas if isinstance(t, (int, float))]
        if vals:
            bits.append(f"θ {min(vals):g}–{max(vals):g}°")
    have = [k for k in ("rsm", "rocking", "combined_device") if entry["artifacts"].get(k)]
    if have:
        bits.append("+".join(have))
    return "  ·  ".join(bits)
