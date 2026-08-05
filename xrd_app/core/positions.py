"""
Build a real per-frame position CSV from the SOCKETSERVER interferometry stream.

The beamline's SOCKETSERVER detector writes one HDF5 file per scan row holding
the interferometer encoder samples (many samples per trigger / per frame). This
module reduces that stream to one ``(X, Y)`` per trigger and writes the standard
``Trigger,X_Position,Y_Position`` CSV the rest of the pipeline consumes
(:func:`io.load_positions_xy`) — the *real* stage positions. When a scan has no
SOCKETSERVER stream, ``generate_grid_mapping`` reconstructs the grid directly
from the one-file-per-row layout instead (no positions file is fabricated).

Port of ``mictools.process_data.process_position_data`` (the ``averaging``
method), reimplemented with numpy so ``core`` keeps no pandas/apstools/master-
file dependency. It needs only the SOCKETSERVER files and does not use a
sample-theta baseline.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from . import io

# Column layout of the SOCKETSERVER ``entry/data/data`` table (one row per
# interferometer sample). Mirrors ``mictools.load_data.load_interferometry_data``;
# only the columns we reduce to (X, Y) positions are named here by index.
N_INTERF_COLS = 24
_C_TRIGGER = 2    # Counter3 — the per-frame trigger id
_C_Y = (3, 4, 5)  # I7 (Y ds), I8 (Y us-ob), I9 (Y us-ib)
_C_Z = 6          # I12 (Z)
_C_X = (7, 19, 20)  # I15 (X), I10 (X-us), I11 (X-ds)


def socketserver_files(socket_dir: Union[str, Path], scan_number: int) -> list:
    """Sorted, non-empty ``scan_NNNN_*.h5`` SOCKETSERVER files for a scan.

    Same naming/discovery rules as the raw XRD frames, so it tolerates the
    ``scan_``/``Scan_`` and zero-padding variations across exports.
    """
    return io.scan_h5_files(socket_dir, scan_number)


def has_socketserver(socket_dir: Union[str, Path], scan_number: int) -> bool:
    """True if at least one SOCKETSERVER interferometry file exists for the scan."""
    return len(socketserver_files(socket_dir, scan_number)) > 0


def _load_interferometry(socket_dir: Path, scan_number: int,
                         reduction: int = 1) -> np.ndarray:
    """Stack every SOCKETSERVER file's interferometer samples into one (N, 24) array."""
    import h5py
    files = socketserver_files(socket_dir, scan_number)
    if not files:
        raise FileNotFoundError(
            f"No SOCKETSERVER files (scan_{scan_number:04d}_*.h5) in {socket_dir}.")
    chunks = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            raw = np.asarray(f[io.H5_DATASET][::reduction], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != N_INTERF_COLS:
            raise ValueError(
                f"Expected {N_INTERF_COLS}-column SOCKETSERVER data, got shape "
                f"{raw.shape} in {Path(fp).name}. This may not be an interferometry "
                "stream — check the SOCKETSERVER directory.")
        chunks.append(raw)
    return np.concatenate(chunks, axis=0)


def _group_mean_by_trigger(rows: np.ndarray):
    """Average interferometer samples sharing a trigger → (triggers, mean_rows).

    Equivalent to ``pandas.groupby('Counter3').mean()`` but vectorised: triggers
    come out ascending, matching the original implementation.
    """
    trig = rows[:, _C_TRIGGER]
    order = np.argsort(trig, kind="stable")
    rows_s, trig_s = rows[order], trig[order]
    uniq, start = np.unique(trig_s, return_index=True)
    sums = np.add.reduceat(rows_s, start, axis=0)
    counts = np.diff(np.append(start, len(trig_s)))
    return uniq, sums / counts[:, None]


def compute_positions(socket_dir: Union[str, Path], scan_number: int,
                      reduction: int = 1):
    """Reduce the SOCKETSERVER stream to per-trigger ``(triggers, x_um, y_um)``.

    The three X interferometers are combined with Z and the three Y
    interferometers. The first trigger is dropped because it carries no trigger
    data, and the first kept point defines the origin. Positions are in microns.
    """
    rows = _load_interferometry(Path(socket_dir), scan_number, reduction=reduction)
    triggers, means = _group_mean_by_trigger(rows)
    # Drop the first trigger group — it has no trigger data (matches the source).
    triggers, means = triggers[1:], means[1:]
    if len(triggers) == 0:
        raise ValueError(
            f"Scan {scan_number}: only one trigger group in the SOCKETSERVER "
            "stream — nothing to position. The interferometry data looks empty.")

    rel = means - means[0]  # origin at the first kept trigger
    x_avg = rel[:, list(_C_X)].mean(axis=1)
    y_avg = rel[:, list(_C_Y)].mean(axis=1)
    z = rel[:, _C_Z]
    x_pos = -np.sqrt(x_avg ** 2 + z ** 2)
    y_pos = y_avg

    x_um = x_pos / 1e4
    y_um = y_pos / 1e4
    # Re-anchor to the first point (and flip X to the device-map convention),
    # matching the source so the de-skew/orientation downstream is identical.
    x_um = -1.0 * (x_um - x_um[0])
    y_um = y_um - y_um[0]
    return triggers, x_um, y_um


def build_positions_csv(
    socket_dir: Union[str, Path],
    output: Union[str, Path],
    scan_number: int,
    reduction: int = 1,
    log: Callable[[str], None] = print,
) -> dict:
    """Compute real positions from SOCKETSERVER and write the standard CSV.

    Writes a marker-free ``Trigger,X_Position,Y_Position`` CSV (so loaders treat
    it as *real* positions, unlike the recreated file-per-row lattice). Returns
    ``{path, n_positions, x_span_um, y_span_um}``.
    """
    log(f"Reading SOCKETSERVER interferometry for scan {scan_number} ...")
    triggers, x_um, y_um = compute_positions(
        socket_dir, scan_number, reduction=reduction)
    x_span = float(np.ptp(x_um)) if len(x_um) else 0.0
    y_span = float(np.ptp(y_um)) if len(y_um) else 0.0
    log(f"  {len(triggers)} positions, span {x_span:.2f} x {y_span:.2f} um "
        f"(method={method})")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Trigger", "X_Position", "Y_Position"])
        for t, x, y in zip(triggers, x_um, y_um):
            w.writerow([int(t), f"{x:.6f}", f"{y:.6f}"])
    os.replace(tmp, output)
    log(f"Wrote {len(triggers)} positions -> {output}")
    return {"path": output, "n_positions": int(len(triggers)),
            "x_span_um": x_span, "y_span_um": y_span}


# ─────────────────────────────────────────────────────────────────────
# Lozano position HDF5 — already-reduced (X, Y) per frame, no interferometry
# reduction needed. Discovered by scan number (``Scan_NNNN.h5``) or given
# explicitly. See :data:`io.H5_POSITION_GROUP` for the layout.
# ─────────────────────────────────────────────────────────────────────
def _scan_h5_name_variants(scan_number: int):
    """Filename spellings for a scan's Lozano position h5 (padding + case)."""
    for stem in (f"Scan_{scan_number:04d}", f"scan_{scan_number:04d}",
                 f"Scan_{scan_number}", f"scan_{scan_number}"):
        for ext in (".h5", ".hdf5"):
            yield stem + ext


def find_position_h5(search_dirs, scan_number: int) -> Optional[Path]:
    """Locate a scan's Lozano position h5 across candidate dirs, or ``None``.

    Tries ``Scan_NNNN.h5`` (padding/case variants) in each dir and verifies the
    file actually carries the ``entry/data/Position`` group before accepting it —
    so a same-named *raw XRD* h5 (``entry/data/data``) is not mistaken for a
    position file. ``search_dirs`` is tried in order; first valid match wins.
    """
    import h5py
    for d in search_dirs:
        if d is None:
            continue
        d = Path(d)
        if not d.is_dir():
            continue
        for nm in _scan_h5_name_variants(scan_number):
            cand = d / nm
            if not cand.is_file():
                continue
            try:
                with h5py.File(cand, "r") as f:
                    if io.H5_POSITION_GROUP in f:
                        return cand
            except OSError:
                continue
    return None


def build_positions_csv_from_h5(
    h5_path: Union[str, Path],
    output: Union[str, Path],
    log: Callable[[str], None] = print,
) -> dict:
    """Read a Lozano position h5 and write the standard positions CSV.

    Companion to :func:`build_positions_csv` (SOCKETSERVER stream) for scans that
    ship already-reduced ``entry/data/Position/{X_Position,Y_Position}`` (µm). The
    output is a marker-free ``Trigger,X_Position,Y_Position`` CSV (treated as
    *real* positions downstream). Returns ``{path, n_positions, x_span_um,
    y_span_um}``.
    """
    h5_path = Path(h5_path)
    log(f"Reading Lozano position h5 {h5_path.name} ...")
    x_um, y_um = io._read_positions_h5(h5_path)
    if len(x_um) != len(y_um):
        raise ValueError(
            f"{h5_path.name}: X_Position ({len(x_um)}) and Y_Position "
            f"({len(y_um)}) lengths differ — cannot build positions CSV.")
    if len(x_um) == 0:
        raise ValueError(f"{h5_path.name}: empty {io.H5_POSITION_GROUP} group.")
    x_span = float(np.ptp(x_um))
    y_span = float(np.ptp(y_um))
    log(f"  {len(x_um)} positions, span {x_span:.2f} x {y_span:.2f} um (lozano-h5)")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Trigger", "X_Position", "Y_Position"])
        for t, (x, y) in enumerate(zip(x_um, y_um)):
            w.writerow([t, f"{x:.6f}", f"{y:.6f}"])
    os.replace(tmp, output)
    log(f"Wrote {len(x_um)} positions -> {output}")
    return {"path": output, "n_positions": int(len(x_um)),
            "x_span_um": x_span, "y_span_um": y_span}
