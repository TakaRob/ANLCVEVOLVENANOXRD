from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QMessageBox

from xrd_app import workspace
from xrd_app.config import ProjectConfig, default_config
from xrd_app.xrf_gui import XRFAnalysisWindow
from xrd_app.xrf_material_popup import XRFMaterialPopup
from xrd_app.xrf_project import XRFProject


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def isolated_workspace(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr(workspace, "SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(workspace, "SETTINGS_PATH", settings)
    monkeypatch.setattr(workspace, "LAUNCH_DIRECTORY", projects)
    workspace.set_workspace(projects)
    return projects


def _xrd_project(root, name="test"):
    config = ProjectConfig(root, data=default_config(name, root, 24))
    config.create_tree()
    config.save()
    return root


def test_xrf_gui_launches_without_project(application, isolated_workspace):
    window = XRFAnalysisWindow()
    try:
        assert window.project is None
        assert window.main_tabs.tabText(0) == "Setup"
        assert window.main_tabs.isTabEnabled(1) is False
        assert "No project open" in window.project_summary.text()
    finally:
        window.close()


def test_xrf_gui_opens_project_with_existing_addon(application, isolated_workspace):
    root = _xrd_project(isolated_workspace / "sample")
    XRFProject.load(root).create_addon()

    window = XRFAnalysisWindow(root)
    try:
        assert window.project.root == Path(root)
        assert window.main_tabs.isTabEnabled(1) is True
        assert str(root) in window.project_summary.text()
    finally:
        window.close()


def test_xrf_gui_prompts_to_create_missing_addon(
    application, isolated_workspace, monkeypatch,
):
    root = _xrd_project(isolated_workspace / "needs-addon")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    window = XRFAnalysisWindow(root)
    try:
        assert (root / "XRF" / "xrf_config.yaml").exists()
        assert window.project.exists()
    finally:
        window.close()


def test_xrf_gui_discovers_project_local_processed_files(
    application, isolated_workspace,
):
    root = _xrd_project(isolated_workspace / "processed")
    addon = XRFProject.load(root).create_addon()
    from xrd_app.core import xrf_selection
    selection = {
        "attrs": {"scan": 24, "n_total_frames": 1},
        "source_files": ["a.h5"],
        "frames": {
            "global_frame_index": [0], "source_file_index": [0],
            "source_frame_index": [0], "x": [0.0], "y": [0.0],
        },
        "materials": {
            "Br": {"intensity": [1.0], "keep": [True], "attrs": {}},
        },
    }
    xrf_selection.save(addon.selection_path(24), selection)

    window = XRFAnalysisWindow(root)
    try:
        assert window.scan_combo.currentText() == "Scan_0024"
        assert window.roi_table.rowCount() == 1
        assert window.cut_material_combo.count() == 1
        assert "Scan_0024" in window.data_status.text()
    finally:
        window.close()


def test_predictions_do_not_modify_manual_roi_table(application, isolated_workspace):
    root = _xrd_project(isolated_workspace / "predictions")
    addon = XRFProject.load(root).create_addon()
    from xrd_app.core import xrf_selection
    spectrum = np.ones(4096)
    spectrum[1192] = 10000
    selection = {
        "attrs": {
            "scan": 24, "n_total_frames": 1,
            "energy_calibration": {"ev_per_bin": 10.0, "offset_ev": 0.0},
        },
        "source_files": ["a.h5"],
        "frames": {
            "global_frame_index": [0], "source_file_index": [0],
            "source_frame_index": [0], "x": [0.0], "y": [0.0],
        },
        "materials": {
            "Manual": {
                "intensity": [1.0], "keep": [True],
                "attrs": {"energy_range_kev": [11.7, 12.0]},
            },
        },
        "spectrum": {
            "energy_kev": np.arange(4096) * 0.01, "summed_counts": spectrum,
        },
    }
    xrf_selection.save(addon.selection_path(24), selection)
    window = XRFAnalysisWindow(root)
    try:
        before = window.roi_table.rowCount()
        window._predict_materials()
        assert window.prediction_list.count() >= 1
        assert window.roi_table.rowCount() == before
        assert window.roi_table.item(0, 0).text() == "Manual"
    finally:
        window.close()


def test_intensity_cut_updates_selected_material_only(application, isolated_workspace):
    root = _xrd_project(isolated_workspace / "cut")
    addon = XRFProject.load(root).create_addon()
    from xrd_app.core import xrf_selection
    selection = {
        "attrs": {"scan": 24, "n_total_frames": 3},
        "source_files": ["a.h5"],
        "frames": {
            "global_frame_index": [0, 1, 2], "source_file_index": [0, 0, 0],
            "source_frame_index": [0, 1, 2], "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
        },
        "materials": {
            "Br": {"intensity": [1.0, 5.0, 10.0], "keep": [True] * 3, "attrs": {}},
            "Pb": {"intensity": [2.0, 3.0, 4.0], "keep": [True] * 3, "attrs": {}},
        },
    }
    xrf_selection.save(addon.selection_path(24), selection)
    window = XRFAnalysisWindow(root)
    try:
        window.cut_material_combo.setCurrentText("Br")
        window.cut_minimum.setValue(6.0)
        np.testing.assert_array_equal(window.selection["materials"]["Br"]["keep"], [False, False, True])
        np.testing.assert_array_equal(window.selection["materials"]["Pb"]["keep"], [True, True, True])
    finally:
        window.close()


def test_material_popup_converts_units_without_changing_roi(application):
    calibration = {"ev_per_bin": 10.0, "offset_ev": 0.0}
    selection = {
        "attrs": {"scan": 1, "n_total_frames": 1, "energy_calibration": calibration},
        "source_files": ["a.h5"],
        "frames": {
            "global_frame_index": [0], "source_file_index": [0],
            "source_frame_index": [0], "x": [0.0], "y": [0.0],
        },
        "materials": {
            "Br": {
                "intensity": [10.0], "keep": [True],
                "attrs": {"energy_range_kev": [11.7, 12.0], "minimum_counts": None},
            },
        },
        "spectrum": {
            "energy_kev": np.arange(4096) * 0.01,
            "summed_counts": np.ones(4096),
        },
    }
    from xrd_app.core import xrf_selection
    popup = XRFMaterialPopup(xrf_selection.validate(selection))
    try:
        before = popup.definitions()["Br"]["energy_range_kev"]
        popup.units.setCurrentText("pixel")
        pixels = popup.definitions()["Br"]["pixel_range"]
        popup.units.setCurrentText("keV")
        after = popup.definitions()["Br"]["energy_range_kev"]
        assert pixels == [1170.0, 1200.0]
        np.testing.assert_allclose(after, before)
    finally:
        popup.close()
