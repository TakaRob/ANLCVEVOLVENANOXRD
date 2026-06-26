"""Catalog discovery, lineage resolution, and the per-scan lineage manifest.

Pure logic (no PyQt, no click) shared by the viewer and the GUI tabs so catalog
listing/selection behaves identically everywhere.

Three catalog kinds live flat in a scan's results dir (``Labels/<scan>/``):

* **peaks**   ``<algo>_peaks_<NxN>[_<tag>].json``   — dict, ``peaks_by_bin`` + lineage
* **shapes**  ``<algo>_shapes_<NxN>[_<tag>].json``  — dict, ``kept``/``filtered`` + lineage
* **feature** ``feature_catalog_<NxN>[_<tag>].json`` — a plain list (no lineage)

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

from .io import atomic_write_json

# Per-scan sidecar mapping ``filename -> lineage dict`` for files that cannot
# carry an in-file lineage block (the plain-list feature_catalog, hand-made files).
MANIFEST_NAME = "catalog_lineage.json"

# kind -> the keyword that precedes the bin in the filename.
_KIND_KEYWORDS = (("peaks", "_peaks_"), ("shapes", "_shapes_"),
                  ("combined", "_combined_"))
_FEATURE_PREFIX = "feature_catalog_"
# A "<NxN>" bin followed by an optional "_<tag>" — anchored at the start of the
# substring that follows the kind keyword.
_BIN_TAG_RE = re.compile(r"^(\d+)x(\d+)(?:_(.+))?$")


# ── name parsing ───────────────────────────────────────────────────
def _bin_tag(rest: str):
    """Split a ``"3x3"`` / ``"3x3_perrowOffset151x235"`` remainder → (bin, tag)."""
    m = _BIN_TAG_RE.match(rest)
    if m:
        return int(m.group(1)), (m.group(3) or "")
    # Fallback: find the first "_NxN" anywhere (tolerates odd remainders).
    m = re.search(r"(\d+)x(\d+)", rest)
    return (int(m.group(1)) if m else None), ""


def parse_name(name) -> "dict | None":
    """Parse a catalog filename → ``{algo, kind, bin, tag}`` (or None).

    ``kind`` ∈ {peaks, shapes, combined, feature}. The bin is taken from the
    segment after the kind keyword so an algorithm name like
    ``5x5_tophat_band_adaptive_snr`` doesn't masquerade as a bin size.
    """
    stem = Path(name).stem
    for kind, kw in _KIND_KEYWORDS:
        if kw in stem:
            algo, rest = stem.split(kw, 1)
            bin_size, tag = _bin_tag(rest)
            return {"algo": algo, "kind": kind, "bin": bin_size, "tag": tag}
    if stem.startswith(_FEATURE_PREFIX):
        bin_size, tag = _bin_tag(stem[len(_FEATURE_PREFIX):])
        return {"algo": "", "kind": "feature", "bin": bin_size, "tag": tag}
    return None


# ── lineage resolution + manifest ──────────────────────────────────
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
    data = _load_json(path)
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
    """Bin size for a catalog — filename first (cheap, reliable), lineage fallback."""
    info = parse_name(Path(path).name)
    if info and info.get("bin") is not None:
        return info["bin"]
    lin = read_lineage(path, results_dir)
    if isinstance(lin, dict):
        return lin.get("bin_size")
    return None


def lineage_key(path):
    """Bin-independent identity used to carry a selection across a bin switch:
    ``(kind, algo, tag)`` from the filename."""
    info = parse_name(Path(path).name) or {}
    return (info.get("kind"), info.get("algo"), info.get("tag"))


# ── discovery ──────────────────────────────────────────────────────
def _iter_json(results_dir):
    """JSONs in the scan dir plus one level of subdirs (future Peaks/ Shapes/)."""
    rd = Path(results_dir)
    if not rd.is_dir():
        return
    yield from rd.glob("*.json")
    for d in rd.iterdir():
        if d.is_dir():
            yield from d.glob("*.json")


def list_catalogs(results_dir, kind, bin_size=None) -> list:
    """All catalogs of ``kind`` (optionally for one bin), sorted by name."""
    out = []
    for p in _iter_json(results_dir):
        if p.name == MANIFEST_NAME:
            continue
        info = parse_name(p.name)
        if not info or info["kind"] != kind:
            continue
        if bin_size is not None and info["bin"] != bin_size:
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.name)


def available_bins(results_dir, kinds=("peaks", "shapes", "feature")) -> list:
    """Sorted bin sizes that have at least one catalog of the given kinds."""
    bins = set()
    for p in _iter_json(results_dir):
        info = parse_name(p.name)
        if info and info["kind"] in kinds and info["bin"] is not None:
            bins.add(info["bin"])
    return sorted(bins)


def feature_sources(results_dir, bin_size=None) -> list:
    """Catalogs usable as a feature source (shapes first, then plain feature lists)."""
    return (list_catalogs(results_dir, "shapes", bin_size)
            + list_catalogs(results_dir, "combined", bin_size)
            + list_catalogs(results_dir, "feature", bin_size))


def shapes_for_peaks(results_dir, peaks_path) -> list:
    """Feature catalogs derived from a given peaks file.

    Primary match: a shapes file whose ``lineage.peak_source_file`` names this
    peaks file. Fallback (covers manually-renamed/tagged files where the lineage
    pointer no longer resolves): shapes at the same bin and tag.
    """
    pname = Path(peaks_path).name
    pinfo = parse_name(pname) or {}
    bin_size = pinfo.get("bin")
    shapes = list_catalogs(results_dir, "shapes", bin_size)

    direct = []
    for sp in shapes:
        lin = read_lineage(sp, results_dir)
        if isinstance(lin, dict) and lin.get("peak_source_file") == pname:
            direct.append(sp)
    if direct:
        return direct

    tag = pinfo.get("tag")
    return [sp for sp in shapes if (parse_name(sp.name) or {}).get("tag") == tag]


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


def load_features_any(path):
    """``(kept, filtered)`` from any catalog kind.

    shapes/combined dict → (kept, filtered); peaks dict → point-features;
    plain list (feature_catalog) → (list, []). Missing/odd files → ([], []).
    """
    data = _load_json(path)
    if isinstance(data, list):
        return data, []
    if isinstance(data, dict):
        if "kept" in data or "filtered" in data:
            return data.get("kept", []), data.get("filtered", [])
        if "features" in data:                       # combined
            return data.get("features", []), []
        if "peaks_by_bin" in data:
            return peaks_to_features(data["peaks_by_bin"])
    return [], []
