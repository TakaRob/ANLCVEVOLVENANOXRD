"""Provenance ("lineage") blocks embedded in every result JSON.

Each pipeline output carries a ``lineage`` dict recording how it was made, so a
saved peaks/shapes/combined file is self-describing:

  peaks     → bin size + peak algorithm (+ detector file, snr)
  shapes    → bin size + shape algorithm + the *upstream peaks* lineage it was
              built from (nested under ``peak_source``)
  combined  → bin size + combined algorithm (peak+shape in one per-frame pass)

``format_lineage`` renders the ancestry as readable lines for the CLI / GUI.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from .. import __version__


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _base(stage: str, scan, bin_size) -> dict:
    return {
        "stage": stage,
        "scan": scan,
        "bin_size": bin_size,
        "created": now_iso(),
        "app_version": __version__,
    }


def peak_lineage(scan, bin_size, algorithm, detector_file=None, snr=None) -> dict:
    d = _base("peaks", scan, bin_size)
    d["peak_algorithm"] = algorithm
    if detector_file is not None:
        d["detector_file"] = str(detector_file)
    if snr is not None:
        d["snr"] = snr
    return d


def combined_lineage(scan, bin_size, algorithm, detector_file=None) -> dict:
    d = _base("combined", scan, bin_size)
    d["combined_algorithm"] = algorithm
    if detector_file is not None:
        d["detector_file"] = str(detector_file)
    d["note"] = "peak + shape in one per-frame pass"
    return d


def shape_lineage(scan, bin_size, shape_algorithm, link_tolerance,
                  peak_source, peak_source_file=None) -> dict:
    """Shape lineage. ``peak_source`` is the upstream peaks' lineage dict (or a
    bare name string for legacy peak files)."""
    d = _base("shapes", scan, bin_size)
    d["shape_algorithm"] = shape_algorithm
    if link_tolerance is not None:
        d["link_tolerance"] = link_tolerance
    if peak_source_file is not None:
        d["peak_source_file"] = peak_source_file
    d["peak_source"] = peak_source
    return d


def from_peaks_data(peaks_data: dict, fallback_file=None) -> dict:
    """Recover an upstream peaks lineage from a loaded peaks JSON.

    Prefers an embedded ``lineage`` block; otherwise synthesizes one from the
    legacy top-level fields so older files still chain.
    """
    lin = peaks_data.get("lineage")
    if isinstance(lin, dict):
        return lin
    return peak_lineage(
        scan=peaks_data.get("scan"),
        bin_size=peaks_data.get("bin_size"),
        algorithm=peaks_data.get("algorithm") or peaks_data.get("detector")
        or (str(fallback_file) if fallback_file else None),
        detector_file=peaks_data.get("detector"),
        snr=peaks_data.get("snr"),
    )


def _bin_label(bs) -> str:
    return f"{bs}x{bs}" if isinstance(bs, int) else str(bs)


def format_lineage(d: dict, indent: int = 0) -> list[str]:
    """Render a lineage dict (and any nested parent) as a list of lines."""
    if not isinstance(d, dict):
        return [("  " * indent) + f"← from peaks: {d}"]
    pad = "  " * indent
    stage = d.get("stage")
    bl = _bin_label(d.get("bin_size"))
    scan = d.get("scan") or "—"
    lines = []
    if stage == "peaks":
        snr = d.get("snr")
        snr_s = f", snr={snr}" if snr is not None else ""
        lines.append(f"{pad}peaks: {d.get('peak_algorithm')} ({bl}{snr_s})  scan={scan}")
        if d.get("detector_file"):
            lines.append(f"{pad}  detector: {Path(d['detector_file']).name}")
    elif stage == "combined":
        lines.append(f"{pad}combined: {d.get('combined_algorithm')} ({bl})  "
                     f"scan={scan}  [peak+shape in one pass]")
        if d.get("detector_file"):
            lines.append(f"{pad}  detector: {Path(d['detector_file']).name}")
    elif stage == "shapes":
        lt = d.get("link_tolerance")
        lt_s = f", link_tol={lt}" if lt is not None else ""
        lines.append(f"{pad}shapes: {d.get('shape_algorithm')} ({bl}{lt_s})  scan={scan}")
        src_file = d.get("peak_source_file")
        if src_file:
            lines.append(f"{pad}  ← from peaks file: {src_file}")
        parent = d.get("peak_source")
        lines.extend(format_lineage(parent, indent + 2))
    else:
        lines.append(f"{pad}{stage or 'unknown'}: {d}")
    if d.get("created"):
        lines.append(f"{pad}  created {d['created']}  app v{d.get('app_version', '?')}")
    return lines


def format_text(d: dict) -> str:
    return "\n".join(format_lineage(d))
