"""ROI summed-image CVEvolve session creation."""
from __future__ import annotations

import numpy as np
import pytest

from xrd_app.config import DataManager, ProjectConfig, default_config
from xrd_app.core import reflection_sum, roi_catalog, roi_cvevolve


def _project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    cfg = ProjectConfig(root, data=default_config("test", root))
    cfg.create_tree(); cfg.save()
    return DataManager(root)


def test_session_requires_saved_manual_roi_catalogs(tmp_path):
    dm = _project(tmp_path)
    with pytest.raises(ValueError, match="No training examples"):
        roi_cvevolve.create_session(dm)


@pytest.mark.parametrize("holdout_pct", [-1, 100, np.nan, np.inf])
def test_session_rejects_invalid_holdout_percentage(tmp_path, holdout_pct):
    dm = _project(tmp_path)

    with pytest.raises(ValueError, match="holdout_pct"):
        roi_cvevolve.create_session(dm, holdout_pct=holdout_pct)


def test_session_exports_sums_labels_config_and_evaluator(tmp_path):
    dm = _project(tmp_path)
    scan = "Scan_0037"
    feature = {"feature_id": 1, "manual_roi": {"x0": 10, "y0": 20, "x1": 18, "y1": 30}}
    roi_catalog.save_previews(dm.roi_map_json("manual", 3, scan), [feature],
                              scan=scan, bin_size=3, name="manual")
    reflection_sum.save(dm, scan, np.ones((40, 50)), is_raw=True)

    result = roi_cvevolve.create_session(dm, dest=tmp_path / "session")

    assert result["examples"] == 1
    data = result["dest"] / "test_data"
    assert (data / "Scan_0037_reflection_sum.npz").exists()
    assert (data / "Scan_0037_rois.json").exists()
    assert (data / "baseline.py").exists()
    assert (data / "evaluate.py").exists()
    assert (result["dest"] / "config.yaml").exists()
    assert (result["dest"] / "prompt.md").exists()


def test_session_removes_only_stale_generated_split_files(tmp_path):
    dm = _project(tmp_path)
    scan = "Scan_0037"
    feature = {"feature_id": 1, "manual_roi": {"x0": 1, "y0": 2, "x1": 3, "y1": 4}}
    roi_catalog.save_previews(dm.roi_map_json("manual", 3, scan), [feature],
                              scan=scan, bin_size=3, name="manual")
    reflection_sum.save(dm, scan, np.ones((5, 6)), is_raw=True)
    dest = tmp_path / "session"
    roi_cvevolve.create_session(dm, dest=dest)
    stale_sum = dest / "holdout_data" / "Scan_9999_reflection_sum.npz"
    stale_labels = dest / "holdout_data" / "Scan_9999_rois.json"
    np.savez(stale_sum, image=np.zeros((1, 1)))
    stale_labels.write_text("{}")
    keep = dest / "holdout_data" / "notes.txt"
    keep.write_text("keep")

    roi_cvevolve.create_session(dm, dest=dest)

    assert not stale_sum.exists()
    assert not stale_labels.exists()
    assert keep.read_text() == "keep"
