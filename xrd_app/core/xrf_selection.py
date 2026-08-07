"""Canonical XRF material selections exported for read-only use by xrd-app."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

FORMAT = "xrd-app-xrf-selection"
FORMAT_VERSION = 1


def pixel_to_kev(pixel, calibration):
    """Convert MCA pixels with a linear or quadratic calibration."""
    pixel = np.asarray(pixel, dtype=float)
    calibration = calibration or {}
    if "quadratic_kev" in calibration:
        return (
            float(calibration["quadratic_kev"]) * pixel**2
            + float(calibration["linear_kev"]) * pixel
            + float(calibration["offset_kev"])
        )
    if calibration.get("kind") == "polynomial":
        return np.polyval(np.asarray(calibration["coefficients_kev"], dtype=float), pixel)
    return (
        pixel * float(calibration.get("ev_per_bin", 10.0))
        + float(calibration.get("offset_ev", 0.0))
    ) / 1000.0


def kev_to_pixel(energy_kev, calibration):
    """Invert a monotonic linear or quadratic MCA calibration."""
    calibration = calibration or {}
    energy = np.asarray(energy_kev, dtype=float)
    if "quadratic_kev" in calibration:
        a = float(calibration["quadratic_kev"])
        b = float(calibration["linear_kev"])
        c = float(calibration["offset_kev"])
        discriminant = b**2 - 4 * a * (c - energy)
        if np.any(discriminant < 0):
            raise ValueError("Energy is outside the calibration domain")
        return (-b + np.sqrt(discriminant)) / (2 * a)
    if calibration.get("kind") == "polynomial":
        coefficients = np.asarray(calibration["coefficients_kev"], dtype=float)
        if coefficients.size != 3:
            raise ValueError("Only linear/quadratic polynomial calibration is supported")
        a, b, c = coefficients
        if abs(a) < 1e-15:
            return (energy - c) / b
        discriminant = b**2 - 4 * a * (c - energy)
        if np.any(discriminant < 0):
            raise ValueError("Energy is outside the calibration domain")
        return (-b + np.sqrt(discriminant)) / (2 * a)
    return (
        energy * 1000.0 - float(calibration.get("offset_ev", 0.0))
    ) / float(calibration.get("ev_per_bin", 10.0))


def integrate_material_rois(selection, me7_dir, definitions, progress=None):
    """Reintegrate selected material ROIs from registered raw ME7 files."""
    from . import xrf

    data = validate(selection)
    me7_by_name = {path.name: path for path in xrf.me7_files(me7_dir)}
    if not me7_by_name:
        raise FileNotFoundError(f"No scan_*.h5 ME7 files in {me7_dir}")
    calibration = data["attrs"].get("energy_calibration") or {}
    channels = data["attrs"].get("channels") or list(range(xrf.N_CHANNELS))
    deadtime = bool(data["attrs"].get("deadtime_correction", False))
    size = data["frames"]["global_frame_index"].size
    materials = {}
    for name, definition in definitions.items():
        attrs = dict(definition)
        if "pixel_range" in attrs:
            lo, hi = map(int, attrs["pixel_range"])
        elif "energy_range_kev" in attrs:
            lo = max(0, int(np.floor(kev_to_pixel(attrs["energy_range_kev"][0], calibration))))
            hi = min(xrf.N_BINS, int(np.ceil(kev_to_pixel(attrs["energy_range_kev"][1], calibration))))
        else:
            raise ValueError(f"Material {name!r} needs pixel_range or energy_range_kev")
        if not 0 <= lo < hi <= xrf.N_BINS:
            raise ValueError(f"Invalid ROI for {name!r}: {lo}:{hi}")
        attrs["pixel_range"] = [lo, hi]
        attrs["energy_range_kev"] = [
            float(pixel_to_kev(lo, calibration)), float(pixel_to_kev(hi, calibration))
        ]
        attrs.setdefault("display_name", name)
        attrs.setdefault("minimum_counts", None)
        materials[name] = {
            "intensity": np.full(size, np.nan, dtype=float),
            "keep": np.ones(size, dtype=bool),
            "attrs": attrs,
        }

    frame_files = data["frames"]["source_file_index"]
    frame_locals = data["frames"]["source_frame_index"]
    processed = 0
    for file_index, source_file in enumerate(data["source_files"]):
        path = me7_by_name.get(Path(source_file).name)
        if path is None:
            raise FileNotFoundError(f"No matching ME7 file for {Path(source_file).name}")
        globals_for_file = np.flatnonzero(frame_files == file_index)
        valid = globals_for_file[frame_locals[globals_for_file] >= 0]
        if not valid.size:
            continue
        with h5py.File(path, "r") as handle:
            spectra = xrf._summed_spectra(handle, channels, deadtime)
        local_indices = frame_locals[valid]
        if np.any(local_indices >= spectra.shape[0]):
            raise ValueError(f"ME7 file {path.name} has fewer frames than registration")
        for material in materials.values():
            lo, hi = material["attrs"]["pixel_range"]
            material["intensity"][valid] = spectra[local_indices, lo:hi].sum(axis=1)
        processed += valid.size
        if progress is not None:
            progress(processed, size)

    for name, material in materials.items():
        minimum = material["attrs"].get("minimum_counts")
        values = material["intensity"]
        material["keep"] = (
            np.isfinite(values) if minimum is None
            else np.isfinite(values) & (values >= float(minimum))
        )
    data["materials"] = materials
    return validate(data)


def xrd_cut_matches(project_root, scan, material, selection=None):
    """Return whether the active xrd-app frame cut matches a saved XRF selection."""
    from ..config import DataManager

    project_root = Path(project_root).resolve()
    data = selection or load(
        project_root / "XRF" / "Processed" / f"{scan}_xrf_selection.h5"
    )
    if not data["attrs"].get("linked_dataset") or material not in data["materials"]:
        return False
    cut_path = DataManager(project_root, scan=scan).metadata_scan_dir(scan) / "xrf_frame_cut.npz"
    if not cut_path.exists():
        return False
    keep = data["materials"][material]["keep"]
    file_indices = data["frames"]["source_file_index"][keep]
    local_indices = data["frames"]["source_frame_index"][keep]
    expected = {
        (Path(data["source_files"][int(index)]).name, int(local_index))
        for index, local_index in zip(file_indices, local_indices)
    }
    try:
        with np.load(cut_path) as cut:
            if str(cut["material"]) != material:
                return False
            actual = set(zip(
                (str(value) for value in cut["source_file"]),
                (int(value) for value in cut["source_frame_index"]),
            ))
    except (KeyError, OSError, ValueError):
        return False
    return actual == expected


def activate_xrd_roi_project(project_root, scan, material):
    """Expose a linked XRF cut to xrd-app without reading detector frames."""
    from ..config import DataManager
    from . import io

    project_root = Path(project_root).resolve()
    dm = DataManager(project_root, scan=scan)
    selection_path = project_root / "XRF" / "Processed" / f"{scan}_xrf_selection.h5"
    data = load(selection_path)
    if not data["attrs"].get("linked_dataset"):
        raise ValueError("XRF/XRD frame registration has not been built")
    if material not in data["materials"]:
        raise KeyError(material)

    xrd_files = [Path(path) for path in data["source_files"]]
    scan_dir = xrd_files[0].parent.parent
    info = io.scan_info(scan_dir, deep=False)
    registry = dm.scans_registry()
    registry[scan] = {key: info[key] for key in
                      ("dir", "frames_dir", "n_files", "n_frames", "shape")}
    dm.write_scans_registry(registry)
    dm.config.data["scans"] = registry
    dm.config.data.setdefault("detector", {})["shape"] = info.get("shape")
    if not (dm.config.data.get("scan") or {}).get("name"):
        dm.config.data["scan"] = {
            "name": scan, "number": DataManager.scan_number_of(scan),
        }
    raw_root = scan_dir.parent.parent
    notebook_positions = raw_root / "processed" / "SOCKETSERVER" / f"{scan}_position.h5"
    if notebook_positions.exists():
        dm.config.data.setdefault("data_sources", {})["position_csv"] = str(
            notebook_positions
        )
    dm.config.save()

    keep = data["materials"][material]["keep"]
    file_indices = data["frames"]["source_file_index"][keep]
    local_indices = data["frames"]["source_frame_index"][keep]
    source_names = np.asarray([
        Path(data["source_files"][int(index)]).name for index in file_indices
    ])
    cut_path = dm.metadata_scan_dir(scan) / "xrf_frame_cut.npz"
    cut_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cut_path.with_name(".xrf_frame_cut.tmp.npz")
    np.savez_compressed(
        temporary, source_file=source_names, source_frame_index=local_indices,
        material=np.asarray(material),
    )
    temporary.replace(cut_path)
    return {
        "scan": scan, "material": material, "selected_frames": int(keep.sum()),
        "selection_path": selection_path, "cut_path": cut_path,
    }


def sum_selected_xrd(selection, material, progress=None):
    """Sum raw XRD frames retained by one material's XRF threshold mask."""
    data = validate(selection)
    if material not in data["materials"]:
        raise KeyError(material)
    keep = data["materials"][material]["keep"]
    file_indices = data["frames"]["source_file_index"]
    local_indices = data["frames"]["source_frame_index"]
    selected = np.flatnonzero(keep & (local_indices >= 0))
    if not selected.size:
        raise ValueError(f"Material {material!r} retains no registered XRD frames")

    image = None
    processed = 0
    for file_index, source_file in enumerate(data["source_files"]):
        rows = selected[file_indices[selected] == file_index]
        if not rows.size:
            continue
        locals_for_file = np.sort(local_indices[rows])
        with h5py.File(source_file, "r") as handle:
            dataset = handle["entry/data/data"]
            if np.any(locals_for_file >= dataset.shape[0]):
                raise ValueError(f"XRD file {Path(source_file).name} has fewer frames than registration")
            for start in range(0, locals_for_file.size, 8):
                batch_indices = locals_for_file[start:start + 8]
                batch = np.asarray(dataset[batch_indices], dtype=np.float64)
                batch_sum = batch.sum(axis=0)
                image = batch_sum if image is None else image + batch_sum
                processed += batch_indices.size
                if progress is not None:
                    progress(processed, int(selected.size))
    if image is None:
        raise ValueError(f"No readable XRD frames retained for {material!r}")
    np.clip(image, 0, 1e9, out=image)
    return image, int(selected.size)


def apply_threshold(selection, material, minimum_counts):
    """Return a validated selection with one material threshold updated."""
    data = validate(selection)
    if material not in data["materials"]:
        raise KeyError(material)
    values = data["materials"][material]["intensity"]
    if not np.isfinite(values).all():
        raise ValueError(
            f"{material} lacks full per-frame intensities; reload the full intensity cache"
        )
    minimum = None if minimum_counts is None else float(minimum_counts)
    data["materials"][material]["keep"] = (
        np.ones(values.size, dtype=bool) if minimum is None else values >= minimum
    )
    data["materials"][material]["attrs"]["minimum_counts"] = minimum
    return validate(data)


def _decode(values):
    return np.asarray([
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attr_value(value):
    value = _json_value(value)
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def selection_hash(global_indices, materials):
    """Return a deterministic identity for frame registration and material masks."""
    digest = hashlib.sha256()
    digest.update(np.asarray(global_indices, dtype="<i8").tobytes())
    for name in sorted(materials):
        material = materials[name]
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(material["keep"], dtype=np.uint8).tobytes())
        digest.update(json.dumps(
            {key: _json_value(value) for key, value in material.get("attrs", {}).items()},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
    return digest.hexdigest()


def validate(selection):
    """Validate and normalize one in-memory canonical selection."""
    frames = selection.get("frames") or {}
    required = ("global_frame_index", "source_file_index", "source_frame_index", "x", "y")
    missing = [name for name in required if name not in frames]
    if missing:
        raise ValueError(f"Selection frames are missing: {', '.join(missing)}")

    normalized_frames = {
        "global_frame_index": np.asarray(frames["global_frame_index"], dtype=np.int64),
        "source_file_index": np.asarray(frames["source_file_index"], dtype=np.int32),
        "source_frame_index": np.asarray(frames["source_frame_index"], dtype=np.int64),
        "x": np.asarray(frames["x"], dtype=float),
        "y": np.asarray(frames["y"], dtype=float),
    }
    size = normalized_frames["global_frame_index"].size
    if any(values.ndim != 1 or values.size != size for values in normalized_frames.values()):
        raise ValueError("All selection frame arrays must be one-dimensional and equal length")
    global_indices = normalized_frames["global_frame_index"]
    if size and (np.any(global_indices < 0) or np.unique(global_indices).size != size):
        raise ValueError("Global frame indices must be unique and non-negative")

    source_files = [str(path) for path in selection.get("source_files", [])]
    file_indices = normalized_frames["source_file_index"]
    if size and (not source_files or np.any(file_indices < 0) or np.any(file_indices >= len(source_files))):
        raise ValueError("Source file indices do not resolve against source_files")

    normalized_materials = {}
    for name, material in (selection.get("materials") or {}).items():
        safe_name = str(name).strip()
        if not safe_name or "/" in safe_name:
            raise ValueError(f"Invalid material name {name!r}")
        intensity = np.asarray(material.get("intensity"), dtype=float)
        keep = np.asarray(material.get("keep"), dtype=bool)
        if intensity.ndim != 1 or keep.ndim != 1 or intensity.size != size or keep.size != size:
            raise ValueError(f"Material {safe_name!r} arrays must match {size} frames")
        normalized_materials[safe_name] = {
            "intensity": intensity,
            "keep": keep,
            "attrs": dict(material.get("attrs") or {}),
        }
    if not normalized_materials:
        raise ValueError("Selection must contain at least one material")

    attrs = dict(selection.get("attrs") or {})
    attrs["format"] = FORMAT
    attrs["format_version"] = FORMAT_VERSION
    attrs["n_total_frames"] = int(attrs.get("n_total_frames", size))
    if attrs["n_total_frames"] < size:
        raise ValueError("n_total_frames cannot be smaller than the frame table")
    attrs["selection_hash"] = selection_hash(global_indices, normalized_materials)

    spectrum = selection.get("spectrum")
    normalized_spectrum = None
    if spectrum is not None:
        energy = np.asarray(spectrum["energy_kev"], dtype=float)
        counts = np.asarray(spectrum["summed_counts"], dtype=float)
        if energy.ndim != 1 or counts.shape != energy.shape:
            raise ValueError("Spectrum energy and counts must be equal one-dimensional arrays")
        normalized_spectrum = {"energy_kev": energy, "summed_counts": counts}

    return {
        "attrs": attrs,
        "source_files": source_files,
        "frames": normalized_frames,
        "materials": normalized_materials,
        "spectrum": normalized_spectrum,
    }


def save(path, selection):
    """Atomically save a canonical XRF selection HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = validate(selection)
    temporary = path.with_name(f".{path.name}.tmp")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w") as handle:
            for key, value in data["attrs"].items():
                if value is None:
                    continue
                if isinstance(value, (dict, list, tuple)):
                    handle.attrs[key] = json.dumps(value, sort_keys=True)
                else:
                    handle.attrs[key] = value
            handle.create_dataset(
                "source_files", data=np.asarray(data["source_files"], dtype=object),
                dtype=string_dtype,
            )
            frame_group = handle.create_group("frames")
            for name, values in data["frames"].items():
                frame_group.create_dataset(name, data=values, compression="gzip")
            if data["spectrum"] is not None:
                spectrum_group = handle.create_group("spectrum")
                for name, values in data["spectrum"].items():
                    spectrum_group.create_dataset(name, data=values, compression="gzip")
            materials_group = handle.create_group("materials")
            for name, material in data["materials"].items():
                group = materials_group.create_group(name)
                group.create_dataset("intensity", data=material["intensity"], compression="gzip")
                group.create_dataset("keep", data=material["keep"], compression="gzip")
                for key, value in material["attrs"].items():
                    if value is None:
                        continue
                    if isinstance(value, (dict, list, tuple, np.ndarray)):
                        group.attrs[key] = json.dumps(_json_value(value), sort_keys=True)
                    else:
                        group.attrs[key] = value
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load(path):
    """Load and validate a canonical XRF selection HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != FORMAT:
            raise ValueError(f"Not a {FORMAT} file: {path}")
        if int(handle.attrs.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError(f"Unsupported XRF selection version in {path}")
        attrs = {key: _attr_value(value) for key, value in handle.attrs.items()}
        frames = {name: np.asarray(handle[f"frames/{name}"][:]) for name in handle["frames"]}
        materials = {}
        for name, group in handle["materials"].items():
            materials[name] = {
                "intensity": np.asarray(group["intensity"][:], dtype=float),
                "keep": np.asarray(group["keep"][:], dtype=bool),
                "attrs": {key: _attr_value(value) for key, value in group.attrs.items()},
            }
        spectrum = None
        if "spectrum" in handle:
            spectrum = {
                "energy_kev": np.asarray(handle["spectrum/energy_kev"][:], dtype=float),
                "summed_counts": np.asarray(handle["spectrum/summed_counts"][:], dtype=float),
            }
        selection = {
            "attrs": attrs,
            "source_files": _decode(handle["source_files"][:]).tolist(),
            "frames": frames,
            "materials": materials,
            "spectrum": spectrum,
        }
    return validate(selection)


def summary(selection):
    """Return a JSON-safe summary for CLI and GUI status displays."""
    data = validate(selection)
    total = data["frames"]["global_frame_index"].size
    return {
        "scan": data["attrs"].get("scan"),
        "n_total_frames": data["attrs"]["n_total_frames"],
        "n_registered_frames": total,
        "selection_hash": data["attrs"]["selection_hash"],
        "materials": {
            name: {
                "minimum_counts": material["attrs"].get("minimum_counts"),
                "retained_frames": int(material["keep"].sum()),
                "cut_frames": int(total - material["keep"].sum()),
                "retained_percent": float(100.0 * material["keep"].mean()) if total else 0.0,
            }
            for name, material in data["materials"].items()
        },
    }
