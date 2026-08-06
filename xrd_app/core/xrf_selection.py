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


def integrate_material_rois(selection, me7_dir, definitions):
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

    for name, material in materials.items():
        minimum = material["attrs"].get("minimum_counts")
        values = material["intensity"]
        material["keep"] = (
            np.isfinite(values) if minimum is None
            else np.isfinite(values) & (values >= float(minimum))
        )
    data["materials"] = materials
    return validate(data)


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


def import_legacy_linker(link_path, mask_path=None, roi_config_path=None,
                         spectrum_path=None, registration_path=None,
                         intensity_path=None, scan=None):
    """Convert the current notebook linker and sidecars to a canonical selection.

    The linker contains retained rows only. A mask sidecar is therefore required
    to preserve rejected positions and the full acquisition-index registration.
    """
    link_path = Path(link_path)
    mask_path = Path(mask_path) if mask_path else link_path.with_name(
        link_path.name.replace("xrf_xrd_links.h5", "xrf_threshold_masks.npz")
    )
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Legacy mask sidecar is required to reconstruct rejected frames: {mask_path}"
        )

    with h5py.File(link_path, "r") as handle:
        group = handle["links"]
        link_material = _decode(group["Material"][:])
        link_global = np.asarray(group["Global Frame Index"][:], dtype=np.int64)
        link_intensity = np.asarray(group["XRF Intensity"][:], dtype=float)
        link_x = np.asarray(group["X"][:], dtype=float)
        link_y = np.asarray(group["Y"][:], dtype=float)
        link_file = _decode(group["XRD File Link"][:])
        link_local = np.asarray(group["XRD Frame Index"][:], dtype=np.int64)
        link_scan = handle.attrs.get("scan")

    with np.load(mask_path, allow_pickle=False) as masks:
        names = _decode(masks["names"][:]).tolist()
        global_indices = np.asarray(masks["global_frame_indices"][:], dtype=np.int64)
        keep_masks = np.asarray(masks["keep_masks"][:], dtype=bool)
        minimum_counts = np.asarray(masks["minimum_counts"][:], dtype=float)
    if keep_masks.shape != (len(names), len(global_indices)):
        raise ValueError("Legacy keep_masks shape does not match names/global indices")
    if np.unique(global_indices).size != global_indices.size:
        raise ValueError("Legacy global frame indices are not unique")

    index_of = {int(value): index for index, value in enumerate(global_indices)}
    source_files = sorted(set(link_file.tolist()))
    source_index = {path: index for index, path in enumerate(source_files)}
    frame_file = np.full(global_indices.size, -1, dtype=np.int32)
    frame_local = np.full(global_indices.size, -1, dtype=np.int64)
    frame_x = np.full(global_indices.size, np.nan, dtype=float)
    frame_y = np.full(global_indices.size, np.nan, dtype=float)
    materials = {
        name: {
            "intensity": np.full(global_indices.size, np.nan, dtype=float),
            "keep": keep_masks[index].copy(),
            "attrs": {
                "display_name": name,
                "minimum_counts": None if np.isnan(minimum_counts[index])
                else float(minimum_counts[index]),
            },
        }
        for index, name in enumerate(names)
    }
    for material, global_index, intensity, x, y, path, local in zip(
        link_material, link_global, link_intensity, link_x, link_y, link_file, link_local
    ):
        if int(global_index) not in index_of:
            raise ValueError(f"Link global index {global_index} is absent from mask sidecar")
        index = index_of[int(global_index)]
        file_index = source_index[str(path)]
        if frame_file[index] not in (-1, file_index) or frame_local[index] not in (-1, int(local)):
            raise ValueError(f"Conflicting XRD identity for global frame {global_index}")
        frame_file[index] = file_index
        frame_local[index] = int(local)
        frame_x[index] = float(x)
        frame_y[index] = float(y)
        if str(material) in materials:
            materials[str(material)]["intensity"][index] = float(intensity)

    # The full ROI intensity cache restores values for rejected as well as retained
    # positions, allowing thresholds to be changed without rereading raw ME7.
    intensity_path = Path(intensity_path) if intensity_path else link_path.with_name(
        link_path.name.replace("xrf_xrd_links.h5", "xrf_roi_intensities.npz")
    )
    if intensity_path.exists():
        with np.load(intensity_path, allow_pickle=False) as intensities:
            cached_materials = _decode(intensities["materials"][:]).tolist()
            cached_values = np.asarray(intensities["intensities"], dtype=float)
        if cached_values.shape != (len(cached_materials), len(global_indices)):
            raise ValueError("Legacy ROI intensity cache shape does not match registration")
        for row, name in enumerate(cached_materials):
            if name in materials:
                materials[name]["intensity"] = cached_values[row].copy()

    # Complete identities for frames cut by every material from the registration
    # cache when available. The cache preserves acquisition ordering and positions.
    registration_path = Path(registration_path) if registration_path else link_path.with_name(
        link_path.name.replace("xrf_xrd_links.h5", "xrf_xrd_registration.npz")
    )
    if registration_path.exists():
        with np.load(registration_path, allow_pickle=False) as registration:
            signature = json.loads(str(registration["signature"]))
            file_numbers = np.asarray(registration["file_numbers"], dtype=int)
            xrd_counts = np.asarray(registration["xrd_counts"], dtype=int)
            x_values = np.asarray(registration["x_position"], dtype=float)
            y_values = np.asarray(registration["y_position_raw"], dtype=float)
            y_values += float(registration["y_offset"])
        registered_names = signature.get("xrd_files", [])
        if len(registered_names) != len(file_numbers) or len(xrd_counts) != len(file_numbers):
            raise ValueError("Legacy registration file arrays are inconsistent")
        linked_by_name = {Path(path).name: path for path in source_files}
        for name in registered_names:
            if name not in linked_by_name:
                candidate = next((path for path in source_files if Path(path).name == name), None)
                if candidate is None:
                    # Derive the raw directory from any retained link.
                    candidate = str(Path(source_files[0]).with_name(name)) if source_files else name
                linked_by_name[name] = candidate
        source_files = [linked_by_name[name] for name in registered_names]
        source_index = {path: index for index, path in enumerate(source_files)}
        global_start = 0
        for file_index, count in enumerate(xrd_counts):
            for local_index in range(int(count)):
                global_index = global_start + local_index
                index = index_of.get(global_index)
                if index is not None:
                    frame_file[index] = file_index
                    frame_local[index] = local_index
            global_start += int(count)
        if x_values.size <= int(global_indices.max(initial=-1)) or y_values.size <= int(global_indices.max(initial=-1)):
            raise ValueError("Legacy registration positions do not cover selected global indices")
        frame_x = x_values[global_indices]
        frame_y = y_values[global_indices]

    # Without the registration cache, the retained-only linker cannot identify
    # frames cut by every material. Preserve global identity and mark them unresolved.
    unresolved = frame_file < 0
    frame_file[unresolved] = 0
    frame_local[unresolved] = -1

    roi_config_path = Path(roi_config_path) if roi_config_path else link_path.with_name(
        link_path.name.replace("xrf_xrd_links.h5", "xrf_rois.json")
    )
    config = {}
    if roi_config_path.exists():
        with roi_config_path.open() as stream:
            config = json.load(stream)
        for name, material in materials.items():
            roi = (config.get("rois") or {}).get(name, {})
            for key in ("energy_range_kev", "pixel_range"):
                if key in roi:
                    material["attrs"][key] = roi[key]

    spectrum = None
    spectrum_path = Path(spectrum_path) if spectrum_path else link_path.with_name(
        link_path.name.replace("xrf_xrd_links.h5", "me7_spectrum.npz")
    )
    if spectrum_path.exists():
        with np.load(spectrum_path, allow_pickle=False) as saved:
            spectrum = {
                "energy_kev": saved["energy_kev"],
                "summed_counts": saved["spectrum"],
            }

    return validate({
        "attrs": {
            "scan": int(scan if scan is not None else link_scan),
            "n_total_frames": int(global_indices.max() + 1) if global_indices.size else 0,
            "source_kind": "legacy_xrf_prefilter",
            "channels": config.get("channels", []),
            "deadtime_correction": config.get("deadtime_correction"),
            "energy_calibration": config.get("energy_calibration"),
            "unresolved_frame_identities": int(unresolved.sum()),
        },
        "source_files": source_files or ["unresolved"],
        "frames": {
            "global_frame_index": global_indices,
            "source_file_index": frame_file,
            "source_frame_index": frame_local,
            "x": frame_x,
            "y": frame_y,
        },
        "materials": materials,
        "spectrum": spectrum,
    })
