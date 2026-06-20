"""Single-window host for xrd-app.

A tabbed window that combines Setup, Programs, and the four interactive GUIs
(View/Label, Shape/Verify, Device, Orientation). The header carries the active
scan + bin-size selectors that drive every scan-dependent tab. Tabs are
discovered from :mod:`xrd_app.tabs` (and any registered entry points) and built
lazily so a missing-data tab shows a friendly placeholder instead of crashing.
"""

from __future__ import annotations

import importlib
import json
import traceback
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QTabWidget,
    QVBoxLayout, QWidget,
)

from . import workspace
from .config import DataManager
from .tabs._embed import placeholder

# Built-in tab modules (module path under xrd_app.tabs).
_BUILTIN_TABS = ["setup", "programs", "view_label", "shape_verify", "device", "orientation"]
_BIN_SIZES = [1, 3, 4, 5]


def _discover_tabs(only=None):
    """Return [(module, meta), ...] sorted by meta['order'].

    Built-ins plus any entry points in group ``xrd_app.tabs`` (plugin tabs).
    When ``only`` is a list of module short-names, build just those built-in
    tabs and skip plugin tabs (used by the focused standalone windows).
    """
    defs = []
    for name in (only if only is not None else _BUILTIN_TABS):
        try:
            mod = importlib.import_module(f"xrd_app.tabs.{name}")
            if hasattr(mod, "make_tab") and hasattr(mod, "TAB_META"):
                defs.append((mod, mod.TAB_META))
        except Exception:
            traceback.print_exc()
    if only is not None:
        return sorted(defs, key=lambda d: d[1].get("order", 100))
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="xrd_app.tabs") if hasattr(eps, "select") \
            else eps.get("xrd_app.tabs", [])
        for ep in group:
            try:
                mod = ep.load()
                if hasattr(mod, "make_tab") and hasattr(mod, "TAB_META"):
                    defs.append((mod, mod.TAB_META))
            except Exception:
                traceback.print_exc()
    except Exception:
        pass
    return sorted(defs, key=lambda d: d[1].get("order", 100))


class MainWindow(QMainWindow):
    def __init__(self, project_root=None, scan=None, bin_size=3, tabs=None):
        super().__init__()
        self._only_tabs = tabs
        self._init_scan = scan
        self._init_bin = bin_size
        # Resolve the project: explicit root, else the last-opened one, else
        # none (the Setup tab will prompt to create/open one).
        self._load_project(project_root, scan=scan, bin_size=bin_size)
        self.resize(1500, 950)
        self.base_width = 1500.0
        self.base_font_size = 10.0

        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(self._build_header())

        self.general = QLabel()
        self.general.setWordWrap(True)
        self.general.setStyleSheet("color:#666; padding:4px; background:#f3f3f3;")
        self.general.setVisible(False)
        root.addWidget(self.general)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self._defs = _discover_tabs(only=self._only_tabs)
        self._hosts = []
        self._content = {}
        self._built = {}
        for mod, meta in self._defs:
            host = QWidget()
            hl = QVBoxLayout(host)
            hl.setContentsMargins(0, 0, 0, 0)
            self.tabs.addTab(host, meta.get("title", mod.__name__))
            self._hosts.append(host)

        # Connect only after all tabs exist, so the signal can't fire mid-build.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._populate_scans()
        # Restore last tab.
        last = self._state.get("current_tab", 0)
        if 0 <= last < self.tabs.count():
            self.tabs.setCurrentIndex(last)
        self._ensure_built(self.tabs.currentIndex())

    # ----- project loading / switching --------------------------------
    def _load_project(self, project_root, scan=None, bin_size=None):
        """Point the window at a project (or no project if it can't resolve).

        Precedence: explicit ``project_root`` → last-opened project (settings)
        → none. With no project, ``self.dm`` is None and the Setup tab shows the
        create/open controls while other tabs show a friendly placeholder.
        """
        if project_root is None:
            last = workspace.get_last_project()
            project_root = str(last) if last else None

        if project_root is not None:
            self.project_root = str(Path(project_root).resolve())
            self.dm = DataManager(self.project_root)
            if workspace.is_project(self.project_root):
                workspace.set_last_project(self.project_root)
            self._state = self._load_state()
            self.scan = scan or self._state.get("active_scan") or self.dm.scan_name
            self.bin_size = bin_size or self._state.get("bin_size") or 3
            title = self.dm.config.get("name") or self.project_root
        else:
            self.project_root = None
            self.dm = None
            self._state = {}
            self.scan = None
            self.bin_size = bin_size or 3
            title = "no project — create or open one in Setup"

        self.setWindowTitle(f"xrd-app — {title}")

    def switch_project(self, project_root):
        """Open a different project at runtime and rebuild every tab.

        Deferred to the next event-loop turn so it is safe to call from inside a
        Setup-tab button handler (the Setup widget itself is rebuilt).
        """
        def _do():
            self._load_project(project_root, bin_size=self.bin_size)
            for idx in range(len(self._defs)):
                self._built[idx] = False
            self._populate_scans()
            self._ensure_built(self.tabs.currentIndex())
            self._update_general()
        QTimer.singleShot(0, _do)

    # ----- header -----------------------------------------------------
    def _build_header(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Scan:</b>"))
        self.scan_combo = QComboBox()
        self.scan_combo.setMinimumWidth(160)
        self.scan_combo.currentTextChanged.connect(self._on_scan_changed)
        row.addWidget(self.scan_combo)
        # Bin size is chosen per-tab (each bin-dependent tab has its own Bin
        # selector); there is intentionally no global bin selector here.

        row.addStretch()
        self.help_toggle = QCheckBox("Show General (math & visualizations)")
        self.help_toggle.toggled.connect(self._update_general)
        row.addWidget(self.help_toggle)
        return row

    def _populate_scans(self):
        scans = self.dm.discover_scans() if self.dm else []
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        if self.dm is None:
            self.scan_combo.addItems(["(no project — create one in Setup)"])
            self.scan_combo.blockSignals(False)
            return
        self.scan_combo.addItems(scans or ["(no scans — load in Setup)"])
        if self.scan and self.scan in scans:
            self.scan_combo.setCurrentText(self.scan)
        elif scans:
            self.scan = scans[0]
        self.scan_combo.blockSignals(False)

    # ----- tab lifecycle ----------------------------------------------
    def _ensure_built(self, idx):
        if idx < 0 or idx >= len(self._hosts) or self._built.get(idx):
            return
        mod, meta = self._defs[idx]
        host = self._hosts[idx]
        lay = host.layout()
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        # Without a project, only Setup is usable; others explain why.
        if self.project_root is None and not getattr(mod, "WORKS_WITHOUT_PROJECT", False):
            content = placeholder(
                f"No project open.",
                "Create or open a project in the Setup tab to use this view.")
        else:
            try:
                content = mod.make_tab(self.project_root, scan=self.scan,
                                       bin_size=self.bin_size)
            except Exception as e:
                content = placeholder(f"Could not load “{meta.get('title')}”.",
                                      f"{type(e).__name__}: {e}")
        # Persistent tabs (e.g. Setup) can drive project switching.
        if hasattr(content, "set_host"):
            content.set_host(self)
        lay.addWidget(content)
        self._content[idx] = content
        self._built[idx] = True

    def _on_tab_changed(self, idx):
        self._ensure_built(idx)
        self._update_general()
        self._save_state()

    def _update_general(self, *_):
        idx = self.tabs.currentIndex()
        if self.help_toggle.isChecked() and 0 <= idx < len(self._defs):
            self.general.setText(self._defs[idx][1].get("general", ""))
            self.general.setVisible(True)
        else:
            self.general.setVisible(False)

    # ----- context changes --------------------------------------------
    def _on_scan_changed(self, text):
        if not text or text.startswith("("):
            return
        self.scan = text
        self._refresh_context()

    def _refresh_context(self):
        """Rebuild scan-dependent tabs; push context to persistent ones."""
        for idx, (mod, meta) in enumerate(self._defs):
            if meta.get("scan_dependent", True):
                self._built[idx] = False  # lazy rebuild on next view
            else:
                content = self._content.get(idx)
                if content is not None and hasattr(content, "update_context"):
                    content.update_context(self.scan, self.bin_size)
        self._ensure_built(self.tabs.currentIndex())
        self._save_state()

    # ----- state persistence ------------------------------------------
    def _state_path(self) -> Path:
        return self.dm.metadata_dir / "gui_state.json"

    def _load_state(self) -> dict:
        if self.dm is None:
            return {}
        p = self.dm.metadata_dir / "gui_state.json"
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f) or {}
            except Exception:
                return {}
        return {}

    def _save_state(self):
        if self.dm is None:
            return
        self._state.update({
            "active_scan": self.scan,
            "bin_size": self.bin_size,
            "current_tab": self.tabs.currentIndex(),
        })
        try:
            p = self._state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            pass

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        self._save_state()
        super().closeEvent(event)

    def resizeEvent(self, event):  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        scale_factor = self.width() / self.base_width
        new_size = int(self.base_font_size * scale_factor)
        new_size = max(9, min(new_size, 26))
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            font = app.font()
            font.setPointSize(new_size)
            app.setFont(font)


def launch_app(project_root=None, scan=None, bin_size=3):
    """Create the QApplication and run the single-window app."""
    import sys
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(project_root, scan=scan, bin_size=bin_size)
    win.show()
    return app.exec_()
