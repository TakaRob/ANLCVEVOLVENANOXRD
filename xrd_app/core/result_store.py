"""Typed HDF5 persistence for numerical analysis catalogs.

The scientific code continues to exchange ordinary dictionaries. This module
flattens their large peak and per-bin feature payloads into columnar datasets and
reconstructs the dictionaries at load boundaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

FORMAT = "xrd-app-results"
VERSION = 1
_STR = h5py.string_dtype("utf-8")
_PEAK_NUMERIC = (
    "x", "y", "npix", "compactness", "snr", "peak_val",
    "integrated_intensity", "cleaned_intensity",
)
_PROFILE_NUMERIC = ("intensity", "integrated", "mean", "det_x", "det_y", "tth", "chi")
_HD_NUMERIC = ("intensity", "integrated", "x", "y")
_TRACK_MEMBER_NUMERIC = (
    "theta", "center_row", "center_col", "chi_deg", "tth_com",
    "peak_intensity", "sum_integrated", "intensity", "detector_x",
    "detector_y", "tth_fwhm", "chi_fwhm", "feature_id",
)


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=True)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _dataset(group, name, values, *, dtype=None):
    array = np.asarray(values, dtype=dtype)
    kwargs = {"compression": "gzip", "compression_opts": 1, "shuffle": True} if array.size else {}
    return group.create_dataset(name, data=array, **kwargs)


def _string_dataset(group, name, values):
    values = np.asarray([str(value) for value in values], dtype=object)
    kwargs = {"compression": "gzip", "compression_opts": 1} if values.size else {}
    return group.create_dataset(name, data=values, dtype=_STR, **kwargs)


def _write_peaks(root, peaks_by_bin):
    group = root.create_group("peaks")
    rows = [(key, peak) for key, peaks in peaks_by_bin.items() for peak in peaks]
    _string_dataset(group, "bin_key", [key for key, _ in rows])
    _string_dataset(group, "label", [peak.get("label", "unknown") for _, peak in rows])
    for field in _PEAK_NUMERIC:
        _dataset(group, field, [float(peak.get(field, np.nan)) for _, peak in rows], dtype=np.float64)
    known = set(_PEAK_NUMERIC) | {"label"}
    extras = [{key: value for key, value in peak.items() if key not in known}
              for _, peak in rows]
    if any(extras):
        _string_dataset(group, "extra_json", [_json(extra) for extra in extras])


def _read_peaks(root):
    group = root["peaks"]
    fields = {name: group[name][:] for name in _PEAK_NUMERIC}
    bins = [_text(value) for value in group["bin_key"][:]]
    labels = [_text(value) for value in group["label"][:]]
    extras = ([_text(value) for value in group["extra_json"][:]]
              if "extra_json" in group else None)
    result = {}
    integer_fields = {"x", "y", "npix"}
    for index, bin_key in enumerate(bins):
        peak = json.loads(extras[index]) if extras is not None else {}
        peak["label"] = labels[index]
        for field, values in fields.items():
            value = float(values[index])
            if np.isfinite(value):
                peak[field] = int(value) if field in integer_fields else value
        result.setdefault(bin_key, []).append(peak)
    return result


def _write_feature_list(root, name, features):
    group = root.create_group(name)
    bases = []
    extent_feature = []
    extent_key = []
    profile_feature = []
    profile_key = []
    profile_values = {field: [] for field in _PROFILE_NUMERIC}
    profile_extras = []
    for index, feature in enumerate(features):
        base = {key: value for key, value in feature.items()
                if key not in ("intensity_profile", "spatial_extent")}
        bases.append(_json(base))
        for key in feature.get("spatial_extent", []) or []:
            extent_feature.append(index)
            extent_key.append(key)
        for key, entry in (feature.get("intensity_profile") or {}).items():
            entry = entry if isinstance(entry, dict) else {}
            profile_feature.append(index)
            profile_key.append(key)
            for field in _PROFILE_NUMERIC:
                profile_values[field].append(float(entry.get(field, np.nan)))
            profile_extras.append(_json({
                field: value for field, value in entry.items()
                if field not in _PROFILE_NUMERIC
            }))
    _string_dataset(group, "feature_json", bases)
    _dataset(group, "extent_feature", extent_feature, dtype=np.int64)
    _string_dataset(group, "extent_key", extent_key)
    _dataset(group, "profile_feature", profile_feature, dtype=np.int64)
    _string_dataset(group, "profile_key", profile_key)
    for field, values in profile_values.items():
        _dataset(group, f"profile_{field}", values, dtype=np.float64)
    if any(value != "{}" for value in profile_extras):
        _string_dataset(group, "profile_extra_json", profile_extras)


def _write_combined_points(root, by_bin):
    group = root.create_group("combined_points")
    bins = []
    labels = []
    xs = []
    ys = []
    for bin_key, by_label in by_bin.items():
        for label, points in by_label.items():
            for x, y in points:
                bins.append(bin_key)
                labels.append(label)
                xs.append(float(x))
                ys.append(float(y))
    _string_dataset(group, "bin_key", bins)
    _string_dataset(group, "label", labels)
    _dataset(group, "x", xs, dtype=np.float64)
    _dataset(group, "y", ys, dtype=np.float64)


def _read_combined_points(root):
    group = root["combined_points"]
    result = {}
    for bin_key, label, x, y in zip(
            group["bin_key"][:], group["label"][:], group["x"][:], group["y"][:]):
        result.setdefault(_text(bin_key), {}).setdefault(_text(label), []).append(
            [float(x), float(y)])
    return result


def _write_tracks(root, tracks):
    group = root.create_group("tracks")
    bases = []
    indices = []
    scans = []
    values = {field: [] for field in _TRACK_MEMBER_NUMERIC}
    extras = []
    for index, track in enumerate(tracks):
        bases.append(_json({key: value for key, value in track.items() if key != "members"}))
        for member in track.get("members", []) or []:
            indices.append(index)
            scans.append(member.get("scan", ""))
            for field in _TRACK_MEMBER_NUMERIC:
                value = member.get(field)
                values[field].append(float(value) if value is not None else np.nan)
            extras.append(_json({field: value for field, value in member.items()
                                 if field not in _TRACK_MEMBER_NUMERIC and field != "scan"}))
    _string_dataset(group, "track_json", bases)
    _dataset(group, "member_track", indices, dtype=np.int64)
    _string_dataset(group, "member_scan", scans)
    for field, field_values in values.items():
        _dataset(group, f"member_{field}", field_values, dtype=np.float64)
    _string_dataset(group, "member_extra_json", extras)


def _read_tracks(root):
    group = root["tracks"]
    tracks = [json.loads(_text(value)) for value in group["track_json"][:]]
    for track in tracks:
        track["members"] = []
    values = {field: group[f"member_{field}"][:] for field in _TRACK_MEMBER_NUMERIC}
    integer_fields = {"detector_x", "detector_y", "feature_id"}
    for row, (index, scan) in enumerate(zip(group["member_track"][:], group["member_scan"][:])):
        member = json.loads(_text(group["member_extra_json"][row]))
        member["scan"] = _text(scan)
        for field, field_values in values.items():
            value = float(field_values[row])
            if np.isfinite(value):
                member[field] = int(value) if field in integer_fields else value
            else:
                member[field] = None
        tracks[int(index)]["members"].append(member)
    return tracks


def _write_hd_features(root, features):
    group = root.create_group("hd_features")
    bases = []
    indices = []
    keys = []
    values = {field: [] for field in _HD_NUMERIC}
    extras = []
    for index, feature in enumerate(features):
        bases.append(_json({key: value for key, value in feature.items()
                            if key != "hd_profile"}))
        for key, entry in (feature.get("hd_profile") or {}).items():
            indices.append(index)
            keys.append(key)
            for field in _HD_NUMERIC:
                values[field].append(float(entry.get(field, np.nan)))
            extras.append(_json({field: value for field, value in entry.items()
                                 if field not in _HD_NUMERIC}))
    _string_dataset(group, "feature_json", bases)
    _dataset(group, "cell_feature", indices, dtype=np.int64)
    _string_dataset(group, "cell_key", keys)
    for field, field_values in values.items():
        _dataset(group, f"cell_{field}", field_values, dtype=np.float64)
    _string_dataset(group, "cell_extra_json", extras)


def _read_hd_features(root):
    group = root["hd_features"]
    features = [json.loads(_text(value)) for value in group["feature_json"][:]]
    for feature in features:
        feature["hd_profile"] = {}
    values = {field: group[f"cell_{field}"][:] for field in _HD_NUMERIC}
    extras = group["cell_extra_json"][:]
    for row, (index, key) in enumerate(zip(group["cell_feature"][:], group["cell_key"][:])):
        entry = json.loads(_text(extras[row]))
        for field, field_values in values.items():
            value = float(field_values[row])
            if np.isfinite(value):
                entry[field] = value
        features[int(index)]["hd_profile"][_text(key)] = entry
    return features


def _read_feature_list(root, name):
    if name not in root:
        return []
    group = root[name]
    features = [json.loads(_text(value)) for value in group["feature_json"][:]]
    for feature in features:
        feature["spatial_extent"] = []
        feature["intensity_profile"] = {}
    for index, key in zip(group["extent_feature"][:], group["extent_key"][:]):
        features[int(index)]["spatial_extent"].append(_text(key))
    profile_indices = group["profile_feature"][:]
    profile_keys = group["profile_key"][:]
    profile_values = {field: group[f"profile_{field}"][:] for field in _PROFILE_NUMERIC}
    extras = group["profile_extra_json"][:] if "profile_extra_json" in group else None
    integer_fields = {"det_x", "det_y"}
    for row, (index, key) in enumerate(zip(profile_indices, profile_keys)):
        entry = json.loads(_text(extras[row])) if extras is not None else {}
        for field, values in profile_values.items():
            value = float(values[row])
            if np.isfinite(value):
                entry[field] = int(value) if field in integer_fields else value
        features[int(index)]["intensity_profile"][_text(key)] = entry
    return features


def metadata(path):
    """Read only small catalog metadata, without materializing numerical payloads."""
    path = Path(path)
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        if _text(handle.attrs.get("format", "")) != FORMAT:
            raise ValueError(f"Unsupported HDF5 result format: {path}")
        return json.loads(_text(handle.attrs.get("metadata_json", "{}")))


def save(path, data) -> Path:
    """Atomically write a numerical result dictionary as typed HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    metadata = dict(data)
    payloads = {}
    for key in ("peaks_by_bin", "kept", "filtered", "features", "by_bin", "tracks"):
        if key in metadata:
            payloads[key] = metadata.pop(key)
    try:
        with h5py.File(tmp, "w") as handle:
            handle.attrs["format"] = FORMAT
            handle.attrs["schema_version"] = VERSION
            handle.attrs["metadata_json"] = _json(metadata)
            payload = np.frombuffer(_json(data).encode("utf-8"), dtype=np.uint8)
            handle.create_dataset("payload_json", data=payload, compression="gzip",
                                  compression_opts=1, shuffle=True)
            if "peaks_by_bin" in payloads:
                _write_peaks(handle, payloads["peaks_by_bin"])
            if "by_bin" in payloads:
                _write_combined_points(handle, payloads["by_bin"])
            if "tracks" in payloads:
                _write_tracks(handle, payloads["tracks"])
            for key in ("kept", "filtered", "features"):
                if key not in payloads:
                    continue
                if key == "features" and any(
                        "hd_profile" in feature for feature in payloads[key]):
                    _write_hd_features(handle, payloads[key])
                else:
                    _write_feature_list(handle, key, payloads[key])
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return path


def load(path):
    """Load a canonical HDF5 result."""
    path = Path(path)
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        if _text(handle.attrs.get("format", "")) != FORMAT:
            raise ValueError(f"Unsupported HDF5 result format: {path}")
        if "payload_json" in handle:
            return json.loads(handle["payload_json"][:].tobytes().decode("utf-8"))
        data = json.loads(_text(handle.attrs.get("metadata_json", "{}")))
        if "peaks" in handle:
            data["peaks_by_bin"] = _read_peaks(handle)
        if "combined_points" in handle:
            data["by_bin"] = _read_combined_points(handle)
        for key in ("kept", "filtered", "features"):
            if key in handle:
                data[key] = _read_feature_list(handle, key)
        if "hd_features" in handle:
            data["features"] = _read_hd_features(handle)
        if "tracks" in handle:
            data["tracks"] = _read_tracks(handle)
        return data
