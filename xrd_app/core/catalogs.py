"""Catalog discovery, lineage resolution, and the per-scan lineage manifest.

Pure logic (no PyQt, no click) shared by the viewer and the GUI tabs so catalog
listing/selection behaves identically everywhere.

Three catalog kinds live flat in a scan's results dir (``Labels/<scan>/``):

* **peaks**   ``<algo>_peaks_<NxN>[_<tag>].h5``
* **shapes**  ``<algo>_shapes_<NxN>[_<tag>].h5``
* **combined** ``<algo>_combined_<NxN>[_<tag>].h5``

The bin size can sit in the *middle* of the name (followed by an optional grid /
experiment ``tag``), so it is parsed from the part after the kind keyword rather
than from the file extension.

Lineage resolution order for any file: **in-file ``lineage`` block → per-scan
manifest entry → ``None`` (caller falls back to manual selection).** The manifest
(:data:`MANIFEST_NAME`) lets future plain-list outputs be tracked without
changing their on-disk format; the CLI appends to it on every write via
:func:`record_catalog`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from . import result_store
from .io import atomic_write_json

# Per-scan sidecar mapping ``filename -> lineage dict``.
MANIFEST_NAME = "catalog_lineage.json"

# kind -> the keyword that precedes the bin in the filename.
_KIND_KEYWORDS = (("peaks", "_peaks_"), ("shapes", "_shapes_"),
                  ("combined", "_combined_"))
# A "<NxN>" bin followed by an optional "_<tag>" — anchored at the start of the
# substring that follows the kind keyword.
_BIN_TAG_RE = re.compile(r"^(\d+)x(\d+)(?:_(.+))?$")


# ── name parsing ───────────────────────────────────────────────────
def _bin_tag(rest: str):
    """Split a ``"3x3"`` / ``"3x3_territory"`` remainder into bin and tag."""
    m = _BIN_TAG_RE.match(rest)
    return (int(m.group(1)), m.group(3) or "") if m else (None, "")


def parse_name(name) -> "dict | None":
    """Parse a catalog filename → ``{algo, kind, bin, tag}`` (or None).

    ``kind`` is peaks, shapes, or combined. The bin is taken from the
    segment after the kind keyword so an algorithm name like
    ``5x5_tophat_band_adaptive_snr`` doesn't masquerade as a bin size.
    """
    stem = Path(name).stem
    for kind, kw in _KIND_KEYWORDS:
        if kw in stem:
            algo, rest = stem.split(kw, 1)
            bin_size, tag = _bin_tag(rest)
            return {"algo": algo, "kind": kind, "bin": bin_size, "tag": tag}
    return None


# ── lineage resolution + manifest ──────────────────────────────────
def load_result(path):
    """Load a numerical HDF5 catalog."""
    try:
        return result_store.load(path)
    except Exception:
        return None


def load_result_metadata(path):
    """Load catalog metadata without reading HDF5 numerical datasets."""
    try:
        return result_store.metadata(path)
    except Exception:
        return None


def save_result(path, data):
    """Persist a numerical catalog in the canonical HDF5 format."""
    return result_store.save(path, data)


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _manifest_path(results_dir) -> Path:
    return Path(results_dir) / MANIFEST_NAME


def read_lineage(path, results_dir=None) -> "dict | None":
    """In-file ``lineage`` block, else the manifest entry, else None."""
    data = load_result_metadata(path)
    if isinstance(data, dict):
        lin = data.get("lineage")
        if isinstance(lin, dict):
            return lin
    rd = Path(results_dir) if results_dir is not None else Path(path).parent
    man = _load_json(_manifest_path(rd))
    if isinstance(man, dict):
        entry = man.get(Path(path).name)
        if isinstance(entry, dict):
            return entry
    return None


def record_catalog(results_dir, filename, lineage) -> Path:
    """Merge ``filename → lineage`` into the per-scan manifest (atomic)."""
    mp = _manifest_path(results_dir)
    man = _load_json(mp)
    if not isinstance(man, dict):
        man = {}
    man[Path(filename).name] = lineage
    return atomic_write_json(mp, man)


def has_lineage(path, results_dir=None) -> bool:
    """Whether a file's lineage is tracked (in-file or manifest)."""
    return read_lineage(path, results_dir) is not None


# ── bin + cross-bin identity ───────────────────────────────────────
def catalog_bin(path, results_dir=None) -> "int | None":
    """Bin size for a catalog, preferring embedded/manifest lineage."""
    lin = read_lineage(path, results_dir)
    if isinstance(lin, dict) and lin.get("bin_size") is not None:
        return lin["bin_size"]
    data = load_result_metadata(path)
    if isinstance(data, dict) and data.get("bin_size") is not None:
        return data["bin_size"]
    info = parse_name(Path(path).name)
    return info.get("bin") if info else None


_IDENTITY_UNSET = object()


def validate_result_identity(path, expected_scan=None, expected_bin_size=None,
                             expected_variant=_IDENTITY_UNSET, results_dir=None) -> dict:
    """Reject known scan/bin/variant mismatches for a result artifact."""
    data = load_result_metadata(path)
    lin = read_lineage(path, results_dir)
    metadata = lin if isinstance(lin, dict) else (data if isinstance(data, dict) else {})
    info = parse_name(Path(path).name) or {}
    actual_scan = metadata.get("scan")
    if actual_scan is None and isinstance(data, dict):
        actual_scan = data.get("scan")
    if actual_scan is None:
        parent = Path(path).parent.name
        actual_scan = parent if re.fullmatch(r"Scan_\d+", parent, re.IGNORECASE) else None
    actual_bin = metadata.get("bin_size")
    if actual_bin is None and isinstance(data, dict):
        actual_bin = data.get("bin_size")
    if actual_bin is None:
        actual_bin = info.get("bin")
    actual_variant = catalog_variant(path, results_dir)

    def _scan_number(value):
        match = re.search(r"(\d+)$", str(value)) if value is not None else None
        return int(match.group(1)) if match else value

    mismatches = []
    if (expected_scan is not None and actual_scan is not None
            and _scan_number(actual_scan) != _scan_number(expected_scan)):
        mismatches.append(f"scan {actual_scan!r} != {expected_scan!r}")
    if (expected_bin_size is not None and actual_bin is not None
            and int(actual_bin) != int(expected_bin_size)):
        mismatches.append(f"bin {actual_bin}x{actual_bin} != "
                          f"{expected_bin_size}x{expected_bin_size}")
    if expected_variant is not _IDENTITY_UNSET and actual_variant != expected_variant:
        expected_label = "plain" if expected_variant is None else repr(expected_variant)
        mismatches.append(f"variant {actual_variant!r} != {expected_label}")
    if mismatches:
        raise ValueError(f"Artifact identity mismatch for {path}: " + ", ".join(mismatches))
    return {"scan": actual_scan, "bin_size": actual_bin, "variant": actual_variant}


def catalog_variant(path, results_dir=None):
    """Coordinate variant for a catalog, preferring its embedded lineage.

    Shape lineage inherits the variant of its upstream peaks. Older catalogs do
    not record a variant, so their filename tag remains the fallback.
    """
    lin = read_lineage(path, results_dir)
    if isinstance(lin, dict):
        for source in (lin, lin.get("peak_source")):
            if isinstance(source, dict):
                variant = source.get("variant") or source.get("tag")
                if variant:
                    return str(variant)
        peak_file = lin.get("peak_source_file")
        if peak_file:
            source_info = parse_name(peak_file) or {}
            if source_info.get("tag"):
                return source_info["tag"]
    info = parse_name(Path(path).name) or {}
    tag = info.get("tag") or None
    # Coordinate linking changes shape connectivity, not the source grid/H5.
    if tag == "coord":
        return None
    if tag and tag.endswith("_coord"):
        return tag[:-len("_coord")]
    return tag


def lineage_key(path):
    """Bin-independent identity used to carry a selection across a bin switch:
    ``(kind, algo, tag)`` from the filename."""
    info = parse_name(Path(path).name) or {}
    return (info.get("kind"), info.get("algo"), info.get("tag"))


# ── discovery ──────────────────────────────────────────────────────
def _iter_catalogs(results_dir):
    """HDF5 catalogs in the scan directory and one subdirectory level."""
    rd = Path(results_dir)
    if not rd.is_dir():
        return
    yield from rd.glob("*.h5")
    for directory in rd.iterdir():
        if directory.is_dir():
            yield from directory.glob("*.h5")


def list_catalogs(results_dir, kind, bin_size=None) -> list:
    """All catalogs of ``kind`` (optionally for one bin), sorted by name."""
    out = []
    for p in _iter_catalogs(results_dir):
        if p.name == MANIFEST_NAME:
            continue
        info = parse_name(p.name)
        if not info or info["kind"] != kind:
            continue
        if bin_size is not None and info["bin"] != bin_size:
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.name)


def available_bins(results_dir, kinds=("peaks", "shapes", "combined")) -> list:
    """Sorted bin sizes that have at least one catalog of the given kinds."""
    bins = set()
    for p in _iter_catalogs(results_dir):
        info = parse_name(p.name)
        if info and info["kind"] in kinds and info["bin"] is not None:
            bins.add(info["bin"])
    return sorted(bins)


def feature_sources(results_dir, bin_size=None) -> list:
    """Shape and combined HDF5 catalogs offered as feature sources."""
    return (list_catalogs(results_dir, "shapes", bin_size)
            + list_catalogs(results_dir, "combined", bin_size))


def default_feature_source(results_dir, bin_size):
    """Newest-named shapes/combined catalog for ``bin_size``, or ``None``."""
    primary = (list_catalogs(results_dir, "shapes", bin_size)
               + list_catalogs(results_dir, "combined", bin_size))
    return primary[-1] if primary else None


def shapes_for_peaks(results_dir, peaks_path) -> list:
    """Feature catalogs derived from a given peaks file.

    A shape catalog must identify the exact upstream peaks filename in lineage.
    """
    pname = Path(peaks_path).name
    pinfo = parse_name(pname) or {}
    return [
        path for path in list_catalogs(results_dir, "shapes", pinfo.get("bin"))
        if (read_lineage(path, results_dir) or {}).get("peak_source_file") == pname
    ]


def match_across_bin(results_dir, kind, ref_path, new_bin) -> "Path | None":
    """The catalog of ``kind`` at ``new_bin`` with the same lineage key, if any."""
    key = lineage_key(ref_path)
    for p in list_catalogs(results_dir, kind, bin_size=new_bin):
        if lineage_key(p) == key:
            return p
    return None


# ── feature loading (format-agnostic) ──────────────────────────────
def peaks_to_features(peaks_by_bin):
    """Convert a peaks-by-bin map into single-bin point-features (no shapes).

    Mirrors the viewer's renderer so a raw peak set can be displayed the same way
    as kept shapes. Returns ``(features, [])``.
    """
    feats = []
    fid = 0
    for bk, peaks in peaks_by_bin.items():
        try:
            r, c = int(bk.split("_")[0]), int(bk.split("_")[1])
        except (ValueError, IndexError):
            continue
        for p in peaks:
            fid += 1
            inten = float(p.get("cleaned_intensity", p.get("intensity", 0)) or 0)
            integ = float(p.get("integrated_intensity", inten) or inten)
            x, y = int(p.get("x", 0)), int(p.get("y", 0))
            feats.append({
                "feature_id": fid,
                "reflection": p.get("label", "unknown"),
                "detector_x": x,
                "detector_y": y,
                "peak_intensity": inten,
                "mean_snr": float(p.get("snr", 0) or 0),
                "n_bins": 1,
                "spatial_extent": [bk],
                "center_bin": bk,
                "center_row": r,
                "center_col": c,
                "intensity_profile": {bk: {
                    "intensity": round(inten, 1),
                    "integrated": round(integ, 1),
                    "det_x": x, "det_y": y,
                }},
                "reason": "raw peak (no shape filtering)",
            })
    return feats, []


def catalog_bin_keys(path, limit=400):
    """A sample of bin keys (center_bin + spatial_extent) a catalog references."""
    kept, _ = load_features_any(path)
    keys = set()
    for f in kept:
        cb = f.get("center_bin")
        if cb:
            keys.add(cb)
        for bk in f.get("spatial_extent", []) or []:
            keys.add(bk)
        if len(keys) >= limit:
            break
    return keys


def _grid_bin_keys(grid_mapping_path):
    from .io import load_grid_mapping
    try:
        g = load_grid_mapping(grid_mapping_path)
    except Exception:
        g = None
    if isinstance(g, dict) and isinstance(g.get("bins"), dict):
        return set(g["bins"].keys())
    return set()


class GridCoverage(NamedTuple):
    path: "Path | None"
    matched: int
    total: int


class CatalogGridMismatch(ValueError):
    """A nonempty catalog is not fully covered by a candidate grid mapping."""


def best_grid_mapping(candidates, feature_catalog_path, default=None,
                      coverage=False, strict=True):
    """Return the grid mapping whose bins best cover a feature catalog.

    ``coverage=True`` returns :class:`GridCoverage`. A nonempty catalog without
    full coverage raises :class:`CatalogGridMismatch` by default; pass
    ``strict=False`` only for callers intentionally inspecting partial coverage.
    """
    cbins = catalog_bin_keys(feature_catalog_path)
    if not cbins:
        result = GridCoverage(Path(default) if default is not None else None, 0, 0)
        return result if coverage else result.path
    best, best_score = (Path(default) if default is not None else None), 0
    target = len(cbins)
    for gm in candidates:
        keys = _grid_bin_keys(gm)
        if not keys:
            continue
        score = sum(1 for b in cbins if b in keys)
        if score > best_score:
            best, best_score = Path(gm), score
        if best_score == target:
            break
    if best_score < target and strict:
        names = ", ".join(Path(p).name for p in candidates) or "(none)"
        raise CatalogGridMismatch(
            f"Catalog {Path(feature_catalog_path).name} references {target} bin(s), "
            f"but the best candidate covers only {best_score}/{target}: {names}")
    result = GridCoverage(best, best_score, target)
    return result if coverage else result.path


class CatalogSources(NamedTuple):
    grid_mapping: Path
    bins_h5: Path
    variant: "str | None"
    matched: int
    total: int


def resolve_catalog_sources(dm, catalog_path, bin_size=None, scan=None):
    """Resolve one catalog to a faithful grid/H5 pair for its scan and variant."""
    results_dir = dm.labels_dir(scan)
    bin_size = bin_size or catalog_bin(catalog_path, results_dir)
    if bin_size is None:
        raise ValueError(f"Cannot determine bin size for catalog {Path(catalog_path).name}")
    expected_scan = dm.scan_name_of(scan) if scan is not None else dm.scan_name
    variant = catalog_variant(catalog_path, results_dir)
    validate_result_identity(
        catalog_path, expected_scan=expected_scan, expected_bin_size=bin_size,
        expected_variant=variant, results_dir=results_dir)
    if variant:
        grid = dm.grid_mapping(bin_size=bin_size, scan=scan, variant=variant)
        if not grid.exists():
            raise FileNotFoundError(
                f"Catalog {Path(catalog_path).name} requires variant {variant!r}, "
                f"but its grid mapping does not exist: {grid}")
        candidates = [grid]
    else:
        mdir = dm.metadata_scan_dir(scan)
        default = dm.grid_mapping(bin_size=bin_size, scan=scan)
        tagged = sorted(
            p for p in mdir.glob(f"grid_mapping_{bin_size}x{bin_size}_*.h5")
            if p != default)
        candidates = [default] + tagged
        grid = default
    cov = best_grid_mapping(candidates, catalog_path, default=grid, coverage=True)
    return CatalogSources(cov.path, dm.binned_h5(bin_size, scan=scan, variant=variant),
                          variant, cov.matched, cov.total)


def load_features_any(path):
    """``(kept, filtered)`` from any catalog kind.

    Shapes/combined dictionaries become feature lists; peaks become point features.
    """
    data = load_result(path)
    if isinstance(data, dict):
        if "kept" in data or "filtered" in data:
            return data.get("kept", []), data.get("filtered", [])
        if "features" in data:                       # combined
            return data.get("features", []), []
        if "peaks_by_bin" in data:
            return peaks_to_features(data["peaks_by_bin"])
    return [], []


def append_features(path, feats) -> list:
    """Append feature dicts into a catalog in place, assigning ``feature_id``.

    A shapes file appends to ``kept`` and a combined file appends to ``features``.
    Returns the assigned IDs. The replacement is atomic.
    """
    path = Path(path)
    data = load_result(path)
    if isinstance(data, dict) and isinstance(data.get("kept"), list):
        target = data["kept"]
    elif isinstance(data, dict) and isinstance(data.get("features"), list):
        target = data["features"]
    else:
        raise ValueError(f"Catalog cannot accept features: {path}")
    next_id = max((f.get("feature_id", 0) for f in target), default=0) + 1
    ids = []
    for feat in feats:
        feat["feature_id"] = next_id
        target.append(feat)
        ids.append(next_id)
        next_id += 1
    save_result(path, data)
    return ids
