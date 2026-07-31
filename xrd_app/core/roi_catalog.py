"""Persistence for ROI > Shape catalogs, separate from Shape/Verify."""

from __future__ import annotations

from pathlib import Path

from . import io


def load(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    import json
    with open(path) as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def save_previews(path, preview_features, *, scan, bin_size, name) -> dict:
    """Merge completed previews by ROI and write one dedicated manual catalog."""
    existing = load(path)
    features = list(existing.get("features") or [])
    index_by_roi = {
        tuple(sorted((feature.get("manual_roi") or {}).items())): index
        for index, feature in enumerate(features)
    }
    for feature in preview_features:
        roi_key = tuple(sorted((feature.get("manual_roi") or {}).items()))
        if roi_key and roi_key in index_by_roi:
            features[index_by_roi[roi_key]] = feature
        else:
            index_by_roi[roi_key] = len(features)
            features.append(feature)
    for index, feature in enumerate(features, 1):
        feature["feature_id"] = index
    result = {
        "kind": "manual_roi_catalog",
        "name": name,
        "scan": scan,
        "bin_size": int(bin_size),
        "intensity_definition": "total detector counts inside ROI per spatial bin",
        "n_features": len(features),
        "features": features,
    }
    io.atomic_write_json(path, result)
    return result


def remove_feature(path, roi) -> dict:
    """Remove one feature by its exact detector ROI and renumber the remainder."""
    data = load(path)
    features = [feature for feature in data.get("features", [])
                if feature.get("manual_roi") != roi]
    for index, feature in enumerate(features, 1):
        feature["feature_id"] = index
    data["features"] = features
    data["n_features"] = len(features)
    io.atomic_write_json(path, data)
    return data
