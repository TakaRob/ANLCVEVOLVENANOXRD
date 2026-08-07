from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication, QMessageBox

from xrd_app import workspace, xrf_gui
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
        assert window.analysis_tabs.tabText(2) == "XRD ROI > Shape Check"
        assert window.create_linked_xrd_button.text() == "Create Linked .h5 File"
        assert window.analysis_tabs.widget(1).isAncestorOf(window.create_linked_xrd_button)
        assert not window.analysis_tabs.widget(2).isAncestorOf(window.create_linked_xrd_button)
        assert window.roi_shape_load.isEnabled() is False
        assert window.spectrum_progress.isVisible() is False
        assert window.spectrum_toolbar.actions()
        assert "Zoom" in {action.text() for action in window.spectrum_toolbar.actions()}
        axis = window.spectrum_canvas.figure.add_subplot(111)
        axis.set(xlim=(0, 10), ylim=(1, 100), yscale="log")
        window.spectrum_canvas._zoom_at_cursor(SimpleNamespace(
            inaxes=axis, xdata=5.0, ydata=10.0, button="up"
        ))
        assert np.diff(axis.get_xlim())[0] < 10
        assert np.diff(np.log10(axis.get_ylim()))[0] < 2
        assert "No project open" in window.project_summary.text()
    finally:
        window.close()


def test_xrf_gui_close_stops_processes(
    application, isolated_workspace, monkeypatch,
):
    window = XRFAnalysisWindow()
    spectrum = object()
    link = object()
    stopped_processes = []
    monkeypatch.setattr(xrf_gui, "stop_process", stopped_processes.append)
    window._spectrum_process = spectrum
    window._link_process = link

    window.close()

    assert stopped_processes == [spectrum, link]
    assert window._spectrum_process is None
    assert window._link_process is None


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
        assert window.cut_material_combo.itemText(0) == "Br"
        assert window.spectrum_save_button.text() == "Selection saved"
        assert "#2e7d32" in window.spectrum_save_button.styleSheet()
        window.roi_table.item(0, 3).setText("11.8")
        assert window.spectrum_save_button.text() == "Save selection"
        assert window.spectrum_save_button.styleSheet() == ""
        assert "Scan_0024" in window.data_status.text()
    finally:
        window.close()


def test_xrf_gui_restores_last_session(application, isolated_workspace):
    root = _xrd_project(isolated_workspace / "session")
    addon = XRFProject.load(root).create_addon()
    from xrd_app.core import xrf_selection
    selection = {
        "attrs": {"scan": 24, "n_total_frames": 2},
        "source_files": ["a.h5"],
        "frames": {
            "global_frame_index": [0, 1], "source_file_index": [0, 0],
            "source_frame_index": [0, 1], "x": [0.0, 1.0], "y": [0.0, 0.0],
        },
        "materials": {
            "Br": {"intensity": [1.0, 8.0], "keep": [True, True], "attrs": {}},
            "Pb": {"intensity": [2.0, 9.0], "keep": [True, True], "attrs": {}},
        },
    }
    path = addon.selection_path(24)
    xrf_selection.save(path, selection)
    info = xrf_selection.summary(xrf_selection.load(path))
    addon.register_selection("Scan_0024", path, info["materials"], info["selection_hash"])

    first = XRFAnalysisWindow(root)
    first.resize(1111, 777)
    first.cut_material_combo.setCurrentText("Pb")
    first.cut_minimum.setValue(7.0)
    first.analysis_tabs.setCurrentIndex(1)
    first.close()

    restored = XRFAnalysisWindow()
    try:
        assert restored.project.root == root
        assert restored.scan_combo.currentText() == "Scan_0024"
        assert restored.main_tabs.currentIndex() == 1
        assert restored.analysis_tabs.currentIndex() == 1
        assert restored.cut_material_combo.currentText() == "Pb"
        assert restored.cut_minimum.value() == 0.0
        assert restored.size().width() == 1111
        assert restored.size().height() == 777
        assert (root / "XRF" / "Metadata" / "gui_state.json").exists()
    finally:
        restored.close()


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
        assert window.selection["materials"]["Br"]["attrs"]["minimum_counts"] == 6.0
        assert window.cut_canvas.figure.axes[0].patches
        assert window.cut_canvas.figure.axes[1].collections
        assert "integrated XRF counts in real space" in window.cut_canvas.figure.axes[1].get_title()
        histogram = window.cut_canvas.figure.axes[0]
        window._set_cut_from_histogram(SimpleNamespace(
            button=1, xdata=8.0, inaxes=histogram,
        ))
        assert window.cut_minimum.value() == 8.0
        np.testing.assert_array_equal(window.selection["materials"]["Br"]["keep"], [False, False, True])
        window._step_cut_material(1)
        assert window.cut_material_combo.currentText() == "Pb"
        window._step_cut_material(1)
        assert window.cut_material_combo.currentText() == "Br"
        np.testing.assert_array_equal(window.selection["materials"]["Pb"]["keep"], [True, True, True])
    finally:
        window.close()


def test_create_linked_file_does_not_load_xrd_images(
    application, isolated_workspace, monkeypatch,
):
    root = _xrd_project(isolated_workspace / "linked-only")
    addon = XRFProject.load(root).create_addon()
    from xrd_app.core import xrf_selection
    scan_dir = isolated_workspace / "raw" / "Scan_0024"
    xrd_dir = scan_dir / "XRD"
    xrd_dir.mkdir(parents=True)
    xrd_path = xrd_dir / "scan_0024_00001.h5"
    import h5py
    with h5py.File(xrd_path, "w") as handle:
        handle.create_dataset("entry/data/data", data=np.ones((1, 2, 2), dtype=np.uint16))
    selection = {
        "attrs": {"scan": 24, "n_total_frames": 1, "linked_dataset": True},
        "source_files": [str(xrd_path)],
        "frames": {
            "global_frame_index": [0], "source_file_index": [0],
            "source_frame_index": [0], "x": [0.0], "y": [0.0],
        },
        "materials": {
            "Br": {"intensity": [10.0], "keep": [True], "attrs": {}},
        },
    }
    xrf_selection.save(addon.selection_path(24), selection)
    sum_path = root / "Metadata" / "Scan_0024" / "reflection_sum.npz"
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sum_path, image=np.full((2, 2), 7, dtype=np.float32))
    window = XRFAnalysisWindow(root)
    try:
        monkeypatch.setattr(
            xrf_selection, "sum_selected_xrd",
            lambda *args, **kwargs: pytest.fail("Create Linked .h5 File must not load XRD images"),
        )
        window._create_linked_xrd()
        assert "no XRD images were read" in window.link_dataset_status.text()
        assert window.create_linked_xrd_button.text() == "Created Linked .h5"
        assert window.roi_shape_load.text() == "Open ROI > Shape Check"
        assert window.roi_shape_load.isEnabled()
        window.cut_minimum.setValue(11.0)
        assert window.create_linked_xrd_button.text() == "Created Linked .h5"
        assert window.roi_shape_load.isEnabled()
        with np.load(sum_path) as saved:
            np.testing.assert_array_equal(saved["image"], np.full((2, 2), 7))
    finally:
        window.close()

    reopened = XRFAnalysisWindow(root)
    try:
        assert reopened.cut_minimum.value() == 0.0
        assert reopened.create_linked_xrd_button.text() == "Created Linked .h5"
        assert reopened.roi_shape_load.isEnabled()
        reopened.cut_minimum.setValue(11.0)
        assert reopened.create_linked_xrd_button.text() == "Created Linked .h5"
        reopened._save_selection()
        assert reopened.create_linked_xrd_button.text() == "Create Linked .h5 File"
        assert not reopened.roi_shape_load.isEnabled()
        assert str(xrd_path.parent.parent) in reopened.project.root.joinpath(
            "Raw", "scans.json"
        ).read_text()
    finally:
        reopened.close()


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
