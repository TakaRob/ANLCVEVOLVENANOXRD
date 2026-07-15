"""Setup tab — project management, data loading, calibration (MVP)."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget, QSizePolicy,
)

from .. import workspace
from ..config import DataManager
from ._console import JobConsole

# The host shows this tab even when no project is open (so the user can make
# one); every other tab gets a placeholder instead.
WORKS_WITHOUT_PROJECT = True

TAB_META = {
    "title": "Setup",
    "order": 10,
    "takes_bin_size": False,
    "scan_dependent": False,
    "general": (
        "Choose a workspace (the XRD-APP Directory holding all projects), create "
        "or open a named project inside it, load scan data (a single .hdf5 → its "
        "scan dir, or a Scans/ parent → all scans), and set calibration "
        "(tth.tiff now; a .poni→tth conversion button is reserved). Each project "
        "keeps its own Raw/Binned/Metadata/Labels; the active scan drives every "
        "other tab."
    ),
}


# The main window scales the *global* application font with its width
# (app.py ``resizeEvent`` → ``app.setFont``), which can push it toward 26pt.
# Qt propagates that font into child dialogs, leaving the non-native
# QFileDialog oversized and truncated ("7...M", "Typ", "/h...ji"). These
# helpers pin popups back to the app's base point size so they render normally.
_DIALOG_FONT_PT = 10


def _reset_dialog_font(dlg):
    f = dlg.font()
    f.setPointSize(_DIALOG_FONT_PT)
    dlg.setFont(f)


def _parse_scan_ranges(text):
    """Parse a quick-select string like ``"215-226, 230"`` into a set of scan
    numbers ``{215, …, 226, 230}``. Tolerant of spaces, ``;`` separators, and a
    leading ``Scan_`` prefix; silently skips unparseable tokens."""
    out = set()
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            a = "".join(ch for ch in a if ch.isdigit())
            b = "".join(ch for ch in b if ch.isdigit())
            if not (a and b):
                continue
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            digits = "".join(ch for ch in part if ch.isdigit())
            if digits:
                out.add(int(digits))
    return out


def _pick_directory(parent, title, start=""):
    """``getExistingDirectory`` with a normal (un-scaled) font."""
    dlg = QFileDialog(parent, title, start)
    dlg.setFileMode(QFileDialog.Directory)
    dlg.setOption(QFileDialog.ShowDirsOnly, True)
    _reset_dialog_font(dlg)
    dlg.resize(900, 600)
    if dlg.exec_() == QDialog.Accepted:
        sel = dlg.selectedFiles()
        return sel[0] if sel else ""
    return ""


def _pick_file(parent, title, start="", name_filter=""):
    """``getOpenFileName`` with a normal (un-scaled) font."""
    dlg = QFileDialog(parent, title, start, name_filter)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    _reset_dialog_font(dlg)
    dlg.resize(900, 600)
    if dlg.exec_() == QDialog.Accepted:
        sel = dlg.selectedFiles()
        return sel[0] if sel else ""
    return ""


class SetupTab(QWidget):
    def __init__(self, project_root, scan=None, bin_size=3):
        super().__init__()
        self.project_root = str(Path(project_root).resolve()) if project_root else None
        self.scan = scan
        self._host = None  # set by MainWindow.set_host for project switching

        lay = QVBoxLayout(self)

        # ---- project management -----------------------------------------
        proj_box = QGroupBox("Project")
        pl = QVBoxLayout(proj_box)

        ws_row = QHBoxLayout()
        ws_row.addWidget(QLabel("Workspace:"))
        self.ws_label = QLabel()
        self.ws_label.setStyleSheet("font-family: monospace;")
        ws_row.addWidget(self.ws_label, 1)
        b_ws = QPushButton("Change…")
        b_ws.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_ws.setMinimumHeight(40)
        b_ws.clicked.connect(self._choose_workspace)
        ws_row.addWidget(b_ws)
        pl.addLayout(ws_row)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Project:"))
        self.proj_combo = QComboBox()
        self.proj_combo.setMinimumWidth(220)
        sel_row.addWidget(self.proj_combo, 1)
        b_open = QPushButton("Open")
        b_open.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_open.setMinimumHeight(40)
        b_open.clicked.connect(self._open_selected)
        b_new = QPushButton("New project…")
        b_new.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_new.setMinimumHeight(40)
        b_new.clicked.connect(self._new_project)
        b_browse = QPushButton("Open other…")
        b_browse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_browse.setMinimumHeight(40)
        b_browse.clicked.connect(self._browse_project)
        sel_row.addWidget(b_open)
        sel_row.addWidget(b_new)
        sel_row.addWidget(b_browse)
        pl.addLayout(sel_row)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-family: monospace;")
        pl.addWidget(self.summary)
        lay.addWidget(proj_box)

        # ---- data loading -----------------------------------------------
        self.load_box = QGroupBox("Load Data")
        ll = QHBoxLayout(self.load_box)
        b_file = QPushButton("Select scan folder…")
        b_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_file.setMinimumHeight(40)
        b_file.setToolTip(
            "Pick one scan folder (e.g. Scan_0203); the XRD/ frames inside are "
            "found automatically — no need to drill down to a single .h5.")
        b_file.clicked.connect(self._pick_scan_folder)
        b_dir = QPushButton("Select scan set…")
        b_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_dir.setMinimumHeight(40)
        b_dir.setToolTip("Pick a folder containing Scan_*/ directories, then choose which scans to register.")
        b_dir.clicked.connect(self._pick_scan_set)
        b_show = QPushButton("Choose scans to show…")
        b_show.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_show.setMinimumHeight(40)
        b_show.setToolTip(
            "Curate which of the registered scans appear in the top Scan "
            "selector (e.g. keep only 215-226). Hidden scans stay registered; "
            "reopen this to change the selection or show all again.")
        b_show.clicked.connect(self._choose_visible_scans)
        b_pos = QPushButton("Load positions…")
        b_pos.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_pos.setMinimumHeight(40)
        b_pos.setToolTip("Load a single position CSV for the active scan, or a "
                         "folder of per-scan scan_NNNN_position.csv files.")
        b_pos.clicked.connect(self._load_positions)
        ll.addWidget(b_file)
        ll.addWidget(b_dir)
        ll.addWidget(b_show)
        ll.addWidget(b_pos)
        ll.addStretch()
        lay.addWidget(self.load_box)

        # ---- calibration ------------------------------------------------
        self.cal_box = QGroupBox("Calibration & Reflections")
        cl = QHBoxLayout(self.cal_box)
        b_tth = QPushButton("Load tth.tiff…")
        b_tth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_tth.setMinimumHeight(40)
        b_tth.clicked.connect(self._load_tth)
        b_poni = QPushButton("Convert .poni → tth…")
        b_poni.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_poni.setMinimumHeight(40)
        b_poni.clicked.connect(self._convert_poni)
        b_refl = QPushButton("Manual reflections…")
        b_refl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_refl.setMinimumHeight(40)
        b_refl.clicked.connect(self._open_reflections)
        b_load_refl = QPushButton("Load reflections…")
        b_load_refl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_load_refl.setMinimumHeight(40)
        b_load_refl.setToolTip(
            "Choose the reflection set the GUIs use for the active scan (the "
            "auto-created one or a set made in Manual reflections).")
        b_load_refl.clicked.connect(self._load_reflections_file)
        b_xrf = QPushButton("XRF calibration…")
        b_xrf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        b_xrf.setMinimumHeight(40)
        b_xrf.setToolTip(
            "Edit the ME7 fluorescence elements + energy calibration: view the "
            "grand-sum spectrum, pick emission lines, detect peaks, set eV/bin.")
        b_xrf.clicked.connect(self._open_xrf_calibration)
        cl.addWidget(b_tth)
        cl.addWidget(b_poni)
        cl.addWidget(b_refl)
        cl.addWidget(b_load_refl)
        cl.addWidget(b_xrf)
        cl.addStretch()
        lay.addWidget(self.cal_box)

        self.console = JobConsole()
        lay.addWidget(self.console, 1)

        self._refresh_projects()
        self.update_context(scan, bin_size)

    # ----- host wiring -------------------------------------------------
    def set_host(self, host):
        """Receive the MainWindow so project actions can rebuild every tab."""
        self._host = host

    # ----- context ----------------------------------------------------
    def update_context(self, scan, bin_size=3):
        self.scan = scan
        self._refresh_summary()

    def _refresh_projects(self):
        """Populate the workspace label and the project picker."""
        ws = workspace.get_workspace()
        self.ws_label.setText(str(ws) if ws else "(not set — click Change…)")
        names = workspace.list_projects(ws)
        self.proj_combo.blockSignals(True)
        self.proj_combo.clear()
        self.proj_combo.addItems(names or ["(no projects yet)"])
        # Select the currently open project, if it lives in the workspace.
        if self.project_root:
            cur = Path(self.project_root).name
            if cur in names:
                self.proj_combo.setCurrentText(cur)
        self.proj_combo.blockSignals(False)

    def _refresh_summary(self):
        has_project = self.project_root is not None
        self.load_box.setEnabled(has_project)
        self.cal_box.setEnabled(has_project)
        if not has_project:
            self.summary.setText(
                "No project open.\n"
                "Set a workspace, then create or open a project above.")
            return
        dm = DataManager(self.project_root, scan=self.scan)
        cfg = dm.config
        shape = cfg.get("detector", "shape")
        all_scans = dm.discover_scans()
        scans = dm.discover_scans(selected_only=True)
        hidden = len(all_scans) - len(scans)
        scans_line = f"scans:  {len(scans)}  {', '.join(scans) if scans else ''}"
        if hidden > 0:
            scans_line += f"   ({hidden} hidden — Choose scans to show…)"
        lines = [
            f"name:   {cfg.get('name')}",
            f"root:   {dm.root}",
            f"active: {dm.scan_name or '—'}",
            f"frame:  {tuple(shape) if shape else '— (load data)'}",
            scans_line,
            f"bins:   {self._bins_summary(dm)}",
            f"tth:    {dm.tth_map()}",
        ]
        from ..core import io
        warn = io.slow_mount_warning(dm.binned_dir_root)
        if warn:
            lines.append(f"\n⚠ {warn}")
        self.summary.setText("\n".join(lines))

    def _bins_summary(self, dm) -> str:
        """Which binned HDF5 files already exist for the active scan."""
        if not dm.scan_name:
            return "— (no active scan)"
        import re
        bins = {}
        bdir = dm.binned_dir(self.scan)
        if bdir.is_dir():
            for p in sorted(bdir.glob("xrd_*x*_bins.h5")):
                m = re.search(r"xrd_(\d+)x(\d+)_bins", p.name)
                if m:
                    bins[f"{m.group(1)}x{m.group(2)}"] = True
        # Also honor the standard sizes via the resolver (config overrides).
        for b in (1, 3, 4, 5):
            if dm.binned_h5(b, scan=self.scan).exists():
                bins[f"{b}x{b}"] = True
        return ", ".join(f"{k} ✓" for k in sorted(bins)) if bins \
            else "none (use Programs → Create bins)"

    # ----- project actions --------------------------------------------
    def _choose_workspace(self):
        start = str(workspace.get_workspace() or Path.home())
        path = _pick_directory(
            self, "Choose the XRD-APP Directory (workspace for all projects)", start)
        if path:
            workspace.set_workspace(path)
            self._refresh_projects()

    def _new_project(self):
        ws = workspace.get_workspace()
        if not ws:
            QMessageBox.information(
                self, "No workspace",
                "Choose a workspace first (Change… next to Workspace).")
            return
        name, ok = QInputDialog.getText(self, "New project", "Project name:")
        name = name.strip()
        if not (ok and name):
            return
        root = workspace.project_root(name, ws)
        if workspace.is_project(root):
            QMessageBox.warning(self, "Already exists",
                                f"A project named “{name}” already exists.")
            return
        try:
            root = workspace.create_project(name, ws)
        except Exception as e:
            QMessageBox.critical(self, "Could not create project",
                                 f"{type(e).__name__}: {e}")
            return
        self._switch_to(root)

    def _open_selected(self):
        name = self.proj_combo.currentText()
        ws = workspace.get_workspace()
        if not ws or name.startswith("("):
            return
        self._switch_to(workspace.project_root(name, ws))

    def _browse_project(self):
        start = str(workspace.get_workspace() or Path.home())
        path = _pick_directory(
            self, "Open a project folder (must contain config.yaml)", start)
        if not path:
            return
        if not workspace.is_project(path):
            QMessageBox.warning(
                self, "Not a project",
                "That folder has no config.yaml. Use “New project…” to create one.")
            return
        self._switch_to(Path(path))

    def _switch_to(self, root):
        if self._host is not None:
            self._host.switch_project(str(root))
        else:  # standalone (no host) — just re-point this tab
            self.project_root = str(Path(root).resolve())
            self._refresh_projects()
            self._refresh_summary()

    # ----- data / calibration actions ---------------------------------
    def _pick_scan_folder(self):
        path = _pick_directory(
            self, "Select a scan folder (its XRD/ frames are found automatically)",
            self.project_root or "")
        if path:
            # discover_scans treats a folder that contains an XRD/ subdir (or
            # loose .h5 frames) as a single scan, so no need to pick a file.
            self._run_setup_job(["scan-detect", "--root", self.project_root,
                                 "--scans-dir", path])

    def _pick_scan_set(self):
        path = _pick_directory(
            self, "Select a folder containing Scan_*/ directories",
            self.project_root or "")
        if not path:
            return
        try:
            from ..core import io
            scans = io.discover_scans(path, deep=False)
        except Exception as e:
            QMessageBox.critical(self, "Could not inspect scans",
                                 f"{type(e).__name__}: {e}")
            return
        if not scans:
            QMessageBox.information(self, "No scans found",
                                    "No scan folders with XRD frames were found there.")
            return
        selected = self._choose_scans(scans)
        if selected is None:
            return
        if not selected:
            QMessageBox.information(self, "No scans selected",
                                    "Select one or more scans to register.")
            return
        args = ["scan-detect", "--root", self.project_root, "--scans-dir", path]
        if len(selected) < len(scans):
            args += ["--scans", ",".join(selected)]
        self._run_setup_job(args)

    def _choose_scans(self, scans):
        dlg = QDialog(self)
        dlg.setWindowTitle("Choose scans")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select the scans to register in this project:"))
        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.MultiSelection)
        for scan in scans:
            approx = "~" if scan.get("frames_estimated") else ""
            text = (f"{scan['name']}  ({scan.get('n_files', 0)} files / "
                    f"{approx}{scan.get('n_frames', 0)} frames)")
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, scan["name"])
            lst.addItem(item)
        lay.addWidget(lst, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
                if lst.item(i).isSelected()]

    def _choose_visible_scans(self):
        """Curate which registered scans appear in the top Scan selector.

        Reads Raw/scans.json (all detected scans), lets the user pick a subset,
        and stores it as ``visible_scans`` in the project config. The rest stay
        registered but hidden — the top selector honors the subset via
        ``discover_scans(selected_only=True)``."""
        if not self.project_root:
            return
        dm = DataManager(self.project_root, scan=self.scan)
        reg = dm.scans_registry()
        if not reg:
            QMessageBox.information(
                self, "No scans registered",
                "Register scans first with “Select scan set…”, then choose "
                "which to show here.")
            return
        names = sorted(reg.keys(), key=lambda n: DataManager.scan_number_of(n) or 0)
        preselected = set(dm.visible_scans() or names)
        selected = self._choose_visible_dialog(names, reg, preselected)
        if selected is None:
            return
        # Empty or "everything" selection → clear the filter (show all), so the
        # top selector can never end up empty.
        if not selected or len(selected) == len(names):
            dm.set_visible_scans(None)
            shown = len(names)
        else:
            dm.set_visible_scans(selected)
            shown = len(selected)
        if self._host is not None and hasattr(self._host, "refresh_project_context"):
            self._host.refresh_project_context()
        self._refresh_summary()
        self.console._append(
            f"[Scan selector now shows {shown}/{len(names)} registered scans]\n")

    def _choose_visible_dialog(self, names, reg, preselected):
        dlg = QDialog(self)
        dlg.setWindowTitle("Choose scans to show")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            f"{len(names)} scans registered. Pick which appear in the top Scan "
            "selector — Ctrl/Shift-click for multiple, or use quick-select. "
            "Unpicked scans stay registered but hidden."))

        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Quick select:"))
        edit = QLineEdit()
        edit.setPlaceholderText("e.g. 215-226, 230")
        qrow.addWidget(edit, 1)
        b_apply = QPushButton("Select matching")
        qrow.addWidget(b_apply)
        lay.addLayout(qrow)

        lst = QListWidget()
        lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for name in names:
            info = reg.get(name, {})
            approx = "~" if info.get("frames_estimated") else ""
            text = (f"{name}  ({info.get('n_files', 0)} files / "
                    f"{approx}{info.get('n_frames', 0)} frames)")
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            lst.addItem(item)
            if name in preselected:
                item.setSelected(True)
        lay.addWidget(lst, 1)

        def _apply_range():
            wanted = _parse_scan_ranges(edit.text())
            if not wanted:
                return
            first = None
            for i in range(lst.count()):
                it = lst.item(i)
                if DataManager.scan_number_of(it.data(Qt.UserRole)) in wanted:
                    it.setSelected(True)
                    if first is None:
                        first = it
            if first is not None:
                lst.scrollToItem(first)
        b_apply.clicked.connect(_apply_range)
        edit.returnPressed.connect(_apply_range)

        brow = QHBoxLayout()
        b_all = QPushButton("Select all")
        b_none = QPushButton("Select none")
        b_all.clicked.connect(lst.selectAll)
        b_none.clicked.connect(lst.clearSelection)
        brow.addWidget(b_all)
        brow.addWidget(b_none)
        brow.addStretch()
        lay.addLayout(brow)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        _reset_dialog_font(dlg)
        dlg.resize(520, 640)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return [lst.item(i).data(Qt.UserRole) for i in range(lst.count())
                if lst.item(i).isSelected()]

    def _run_setup_job(self, args):
        self.console.run(args, on_finished=lambda _code: self._refresh_after_job())

    def _refresh_after_job(self):
        self._refresh_summary()
        if self._host is not None and hasattr(self._host, "refresh_project_context"):
            self._host.refresh_project_context()

    def _load_positions(self):
        """Load scan positions as either a single CSV (this scan) or a folder
        of per-scan ``scan_NNNN_position.csv`` files (multi-scan)."""
        box = QMessageBox(self)
        box.setWindowTitle("Load positions")
        box.setText(
            "Load a single position CSV for the active scan, or a folder "
            "of per-scan scan_NNNN_position.csv files?")
        b_file = box.addButton("Single CSV…", QMessageBox.AcceptRole)
        b_dir = box.addButton("Folder of CSVs…", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_dir:
            # → link --position-root: records the dir; per-scan CSVs are
            #   resolved later as scan_NNNN_position.csv.
            path = _pick_directory(
                self, "Select a folder of scan_NNNN_position.csv files",
                self.project_root or "")
            if path:
                self._run_setup_job(["link", "--root", self.project_root,
                                     "--position-root", path])
        elif clicked is b_file:
            path = _pick_file(
                self, "Select a scan position CSV", self.project_root or "",
                "CSV (*.csv)")
            if not path:
                return
            args = ["link", "--root", self.project_root, "--position-csv", path]
            if self.scan:
                args += ["--scan", str(self.scan)]
            self._run_setup_job(args)

    def _load_tth(self):
        path = _pick_file(
            self, "Select tth.tiff", self.project_root or "", "TIFF (*.tif *.tiff)")
        if path:
            self._run_setup_job(["link", "--root", self.project_root, "--tth", path])

    def _convert_poni(self):
        path = _pick_file(
            self, "Select a pyFAI .poni file", self.project_root or "", "PONI (*.poni)")
        if path:
            args = ["convert-poni", "--root", self.project_root, "--poni", path]
            if self.scan:
                args += ["--scan", str(self.scan)]
            self._run_setup_job(args)

    def _open_reflections(self):
        from .reflection_popup import open_reflection_dialog
        dlg = open_reflection_dialog(self.project_root, scan=self.scan, parent=self)
        dlg.exec_()
        self._refresh_summary()

    def _open_xrf_calibration(self):
        if not self.project_root:
            return
        from .xrf_calibration_popup import open_xrf_calibration_dialog
        dlg = open_xrf_calibration_dialog(self.project_root, scan=self.scan, parent=self)
        dlg.exec_()
        self._refresh_summary()

    def _load_reflections_file(self):
        """Pick the reflection set used by every GUI for the active scan."""
        if not self.project_root:
            return
        dm = DataManager(self.project_root, scan=self.scan)
        if not dm._scan():
            QMessageBox.information(
                self, "No active scan",
                "Select a scan first (top-left), then load its reflections.")
            return
        start = str(dm.metadata_scan_dir(self.scan))
        path = _pick_file(
            self, "Select a reflections file", start,
            "Reflections (reflections.py reflections.json *.py *.json)")
        if not path:
            return
        if path.endswith(".json"):
            path = str(Path(path).with_suffix(".py"))
        dm.set_reflection_source(path, self.scan)
        # Refresh the host header selector + rebuild scan-dependent tabs.
        if self._host is not None:
            if hasattr(self._host, "_populate_reflections"):
                self._host._populate_reflections()
            if hasattr(self._host, "_refresh_context"):
                self._host._refresh_context()
        self.console._append(f"[reflections set for {dm._scan()} → {path}]\n")
        self._refresh_summary()


def make_tab(project_root=".", scan=None, bin_size=3):
    return SetupTab(project_root, scan=scan, bin_size=bin_size)


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
