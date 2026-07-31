"""Dedicated ROI > Shape catalog persistence and discovery isolation."""
from __future__ import annotations

from xrd_app.core import catalogs, roi_catalog


def _feature(x0, feature_id=1):
    return {
        "feature_id": feature_id,
        "reflection": "(001)",
        "manual_roi": {"x0": x0, "y0": 2, "x1": x0 + 5, "y1": 8},
        "intensity_profile": {"0_0": {"integrated": float(x0)}},
    }


def test_save_all_uses_dedicated_catalog_invisible_to_shape_verify(tmp_path):
    path = tmp_path / "manual_roimap_3x3.json"
    result = roi_catalog.save_previews(
        path, [_feature(10), _feature(20)], scan="Scan_0037", bin_size=3,
        name="manual")

    assert result["kind"] == "manual_roi_catalog"
    assert result["n_features"] == 2
    assert [f["feature_id"] for f in result["features"]] == [1, 2]
    assert catalogs.parse_name(path.name) is None
    assert path not in catalogs.feature_sources(tmp_path, 3)


def test_save_all_merges_by_roi_and_remove_updates_catalog(tmp_path):
    path = tmp_path / "manual_roimap_3x3.json"
    roi_catalog.save_previews(path, [_feature(10)], scan="Scan_0037",
                              bin_size=3, name="manual")
    updated = _feature(10)
    updated["reflection"] = "(002)"
    result = roi_catalog.save_previews(path, [updated, _feature(30)],
                                       scan="Scan_0037", bin_size=3, name="manual")

    assert result["n_features"] == 2
    assert result["features"][0]["reflection"] == "(002)"
    remaining = roi_catalog.remove_feature(path, _feature(10)["manual_roi"])
    assert remaining["n_features"] == 1
    assert remaining["features"][0]["feature_id"] == 1
