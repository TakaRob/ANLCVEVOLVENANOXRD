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

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow, QTabWidget,
    QVBoxLayout, QWidget,
)

from .config import DataManager
from .tabs._embed import placeholder

# Built-in tab modules (module path under xrd_app.tabs).
_BUILTIN_TABS = ["setup", "programs", "view_label", "shape_verify", "device", "orientation"]
_BIN_SIZES = [1, 3, 4, 5]


def _discover_tabs():
    """Return [(module, meta), ...] sorted by meta['order'].

    Built-ins plus any entry points in group ``xrd_app.tabs`` (plugin tabs).
    """
    defs = []
    for name in _BUILTIN_TABS:
        try:
            mod = importlib.import_module(f"xrd_app.tabs.{name}")
            if hasattr(mod, "make_tab") and hasattr(mod, "TAB_META"):
                defs.append((mod, mod.TAB_META))
        except Exception:
            traceback.print_exc()
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
    def __init__(self, project_root=".", scan=None, bin_size=3):
        super().__init__()
        self.project_root = str(Path(project_root).resolve())
        self.dm = DataManager(self.project_root)
        self._state = self._load_state()

        self.scan = scan or self._state.get("active_scan") or self.dm.scan_name
        self.bin_size = bin_size or self._state.get("bin_size") or 3

        self.setWindowTitle(f"xrd-app — {self.dm.config.get('name') or self.project_root}")
        self.resize(1500, 950)

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

        self._defs = _discover_tabs()
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

    # ----- header -----------------------------------------------------
    def _build_header(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Scan:</b>"))
        self.scan_combo = QComboBox()
        self.scan_combo.setMinimumWidth(160)
        self.scan_combo.currentTextChanged.connect(self._on_scan_changed)
        row.addWidget(self.scan_combo)

        row.addWidget(QLabel("<b>Bin:</b>"))
        self.bin_combo = QComboBox()
        self.bin_combo.addItems([f"{b}x{b}" for b in _BIN_SIZES])
        self.bin_combo.setCurrentText(f"{self.bin_size}x{self.bin_size}")
        self.bin_combo.currentTextChanged.connect(self._on_bin_changed)
        row.addWidget(self.bin_combo)

        row.addStretch()
        self.help_toggle = QCheckBox("Show General (math & visualizations)")
        self.help_toggle.toggled.connect(self._update_general)
        row.addWidget(self.help_toggle)
        return row

    def _populate_scans(self):
        scans = self.dm.discover_scans()
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
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
        try:
            content = mod.make_tab(self.project_root, scan=self.scan, bin_size=self.bin_size)
        except Exception as e:
            content = placeholder(f"Could not load “{meta.get('title')}”.",
                                  f"{type(e).__name__}: {e}")
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

    def _on_bin_changed(self, text):
        self.bin_size = int(text.split("x")[0])
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
        p = self.dm.metadata_dir / "gui_state.json"
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f) or {}
            except Exception:
                return {}
        return {}

    def _save_state(self):
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


def launch_app(project_root=".", scan=None, bin_size=3):
    """Create the QApplication and run the single-window app."""
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(project_root, scan=scan, bin_size=bin_size)
    win.show()
    return app.exec_()
