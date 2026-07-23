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
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QMainWindow, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from . import workspace
from .config import DataManager
from .tabs._embed import placeholder

# Built-in tab modules (module path under xrd_app.tabs).
# The skew-free territorial reference is its own "territory" tab: it shows the
# cell-model map, or an in-tab "Build territorial reference" button when the scan
# has no territorial artifacts yet (see tabs/territory.py). The Device View tab
# still hides territorial catalogs from its grid dropdown (they can't render on a
# fixed grid) but no longer opens the map as a popup.
_BUILTIN_TABS = ["setup", "programs", "view_label", "shape_verify", "device",
                 "hd_device", "territory", "orientation", "scan_summary",
                 "rocking_study", "rsm"]
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
    def __init__(self, project_root=None, scan=None, bin_size=3, tabs=None, fresh=False):
        super().__init__()
        self._only_tabs = tabs
        self._init_scan = scan
        self._init_bin = bin_size
        # Fresh session: ignore the remembered last project and per-project
        # gui_state (active tab/scan/bin) for this initial load. Reset to False
        # afterwards so switching projects at runtime restores their state.
        self._fresh = fresh
        # Resolve the project: explicit root, else the last-opened one, else
        # none (the Setup tab will prompt to create/open one).
        self._load_project(project_root, scan=scan, bin_size=bin_size)
        self.base_width = 1500.0
        self.base_font_size = 10.0
        self._resize_to_screen()
        self._build_window_menu()

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
        # Restore last tab (skipped in a fresh session — _state is empty → tab 0).
        last = self._state.get("current_tab", 0)
        if 0 <= last < self.tabs.count():
            self.tabs.setCurrentIndex(last)
        self._ensure_built(self.tabs.currentIndex())
        self._sync_header_extra()
        # Fresh only governs the initial load; later project switches restore
        # their own saved state normally.
        self._fresh = False

    def _resize_to_screen(self):
        """Start large, but never larger than the usable desktop area."""
        # A small hard minimum so the window is always shrinkable regardless of
        # any tab's size hint (tab content lives in scroll areas — see
        # _ensure_built), which is what makes it resizable below the screen on
        # GNOME Wayland.
        self.setMinimumSize(640, 480)
        screen_at = getattr(QApplication, "screenAt", None)
        screen = (screen_at(self.pos()) if screen_at else None) or QApplication.primaryScreen()
        if screen is None:
            self.resize(1500, 950)
            return
        rect = screen.availableGeometry()
        width = min(1500, max(480, int(rect.width() * 0.92)), rect.width())
        height = min(950, max(360, int(rect.height() * 0.90)), rect.height())
        self.resize(width, height)
        self.move(
            rect.x() + max(0, (rect.width() - width) // 2),
            rect.y() + max(0, (rect.height() - height) // 2),
        )
        # Wayland may honor the initial geometry only after the surface is
        # mapped; re-apply the clamp once the event loop is running. move() is a
        # no-op under Wayland (clients can't position themselves) but resize is
        # respected, so this still pulls an over-tall window back onto the screen.
        QTimer.singleShot(0, self._clamp_to_screen)

    def _clamp_to_screen(self):
        """Shrink the window back within the usable desktop if it overflows."""
        screen_at = getattr(QApplication, "screenAt", None)
        screen = (screen_at(self.pos()) if screen_at else None) or QApplication.primaryScreen()
        if screen is None:
            return
        rect = screen.availableGeometry()
        w = min(self.width(), rect.width())
        h = min(self.height(), rect.height())
        if w != self.width() or h != self.height():
            self.resize(w, h)

    def _build_window_menu(self):
        view = self.menuBar().addMenu("View")

        maximize = QAction("Maximize", self)
        maximize.setShortcut("Ctrl+M")
        maximize.triggered.connect(self.showMaximized)
        view.addAction(maximize)

        full_screen = QAction("Full Screen", self)
        full_screen.setShortcut("F11")
        full_screen.triggered.connect(self._toggle_full_screen)
        view.addAction(full_screen)

    def _toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ----- project loading / switching --------------------------------
    def _load_project(self, project_root, scan=None, bin_size=None):
        """Point the window at a project (or no project if it can't resolve).

        Precedence: explicit ``project_root`` → last-opened project (settings)
        → none. With no project, ``self.dm`` is None and the Setup tab shows the
        create/open controls while other tabs show a friendly placeholder.
        """
        if project_root is None and not getattr(self, "_fresh", False):
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
            self._sync_header_extra()
            self._update_general()
        QTimer.singleShot(0, _do)

    def refresh_project_context(self):
        """Reload project metadata after Setup changes scans or linked inputs."""
        if self.project_root is None:
            return
        self.dm = DataManager(self.project_root)
        current = self.scan
        self._populate_scans()
        scans = self.dm.discover_scans(selected_only=True)
        if current in scans:
            self.scan = current
            self.scan_combo.setCurrentText(current)
        elif scans:
            self.scan = scans[0]
        self._populate_reflections()
        self._refresh_context()

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

        # Reflection-set selector (per scan): which reflections.py every tab
        # overlays. Defaults to Auto (per-scan/project/bundled); the user can
        # point it at a set made in Manual reflections, or Browse for one.
        row.addSpacing(12)
        row.addWidget(QLabel("<b>Reflections:</b>"))
        self.refl_combo = QComboBox()
        self.refl_combo.setMinimumWidth(200)
        self.refl_combo.activated.connect(self._on_reflection_changed)
        row.addWidget(self.refl_combo)

        # Slot for the active tab's own header controls (e.g. Shape/Verify lifts
        # its Bin + Scan/Feature Catalog + Load bar up here, so the whole top row
        # is a single bar with no duplicated Scan selector).
        row.addSpacing(12)
        self.header_extra = QHBoxLayout()
        self.header_extra.setContentsMargins(0, 0, 0, 0)
        self._cur_extra = None
        row.addLayout(self.header_extra)

        row.addStretch()
        self.help_toggle = QCheckBox("Show General (math & visualizations)")
        self.help_toggle.toggled.connect(self._update_general)
        row.addWidget(self.help_toggle)
        return row

    def _populate_scans(self):
        scans = self.dm.discover_scans(selected_only=True) if self.dm else []
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
        self._populate_reflections()

    # ----- reflection-set selector ------------------------------------
    _REFL_BROWSE = "__browse__"

    def _populate_reflections(self):
        """Fill the Reflections combo for the active scan."""
        combo = getattr(self, "refl_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        if self.dm is None:
            combo.addItem("Auto (default)", None)
            combo.blockSignals(False)
            return
        combo.addItem("Auto (default)", None)
        seen = set()
        per_scan = self.dm.metadata_scan_dir(self.scan) / "reflections.py"
        if per_scan.exists():
            combo.addItem(f"Per-scan ({per_scan.name})", str(per_scan))
            seen.add(str(per_scan))
        proj = self.dm.metadata_dir / "reflections.py"
        if proj.exists() and str(proj) not in seen:
            combo.addItem("Project default", str(proj))
            seen.add(str(proj))
        current = self.dm.reflection_source(self.scan)
        if current is not None and str(current) not in seen:
            combo.addItem(f"Selected ({Path(current).name})", str(current))
            seen.add(str(current))
        combo.addItem("Browse…", self._REFL_BROWSE)
        # Reflect the saved per-scan choice.
        if current is not None:
            i = combo.findData(str(current))
            if i >= 0:
                combo.setCurrentIndex(i)
        combo.blockSignals(False)

    def _on_reflection_changed(self, _idx):
        if self.dm is None:
            return
        data = self.refl_combo.currentData()
        if data == self._REFL_BROWSE:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select a reflections file", str(self.dm.metadata_dir),
                "Reflections (reflections.py reflections.json *.py *.json)")
            if not path:
                self._populate_reflections()  # revert selection
                return
            if path.endswith(".json"):
                path = str(Path(path).with_suffix(".py"))
            self.dm.set_reflection_source(path, self.scan)
        elif data is None:
            self.dm.clear_reflection_source(self.scan)
        else:
            self.dm.set_reflection_source(data, self.scan)
        self._populate_reflections()
        self._refresh_context()

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
        # Host the tab in a resizable scroll area so a tall tab (e.g. Setup with
        # many stacked controls) can never force the whole window taller than the
        # display. With setWidgetResizable(True) the content still fills the
        # viewport when the window is large; scrollbars only appear when it isn't
        # — this is what keeps the window shrinkable below the screen on Wayland,
        # where the minimum size hint would otherwise pin it larger than 1080.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        lay.addWidget(scroll)
        # Keep _content[idx] pointing at the real tab widget (header-extra /
        # get-bar lookups depend on it), not the scroll wrapper.
        self._content[idx] = content
        self._built[idx] = True

    def _sync_header_extra(self):
        """Show the active tab's own header controls (if any) in the top row.

        A tab whose embedded window exposes ``header_bar()`` (the Shape/Verify
        viewer) gets that bar lifted into the global header, so Scan / Bin /
        Scan Catalog / Feature Catalog share one row with no duplication.
        """
        idx = self.tabs.currentIndex()
        content = self._content.get(idx)
        win = getattr(content, "_embedded_window", None)
        bar = win.header_bar() if (win is not None and hasattr(win, "header_bar")) else None
        if bar is self._cur_extra:
            return
        if self._cur_extra is not None:
            try:  # the old bar may already be gone after a tab rebuild
                self.header_extra.removeWidget(self._cur_extra)
                self._cur_extra.setParent(None)
            except RuntimeError:
                pass
        self._cur_extra = bar
        if bar is not None:
            self.header_extra.addWidget(bar)

    def _on_tab_changed(self, idx):
        self._ensure_built(idx)
        self._sync_header_extra()
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
        self._populate_reflections()
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
        self._sync_header_extra()
        self._save_state()

    # ----- state persistence ------------------------------------------
    def _state_path(self) -> Path:
        return self.dm.metadata_dir / "gui_state.json"

    def _load_state(self) -> dict:
        if self.dm is None or getattr(self, "_fresh", False):
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
        app = QApplication.instance()
        if app:
            font = app.font()
            font.setPointSize(new_size)
            app.setFont(font)


def _descendant_pids(root_pid):
    """Return all descendant PIDs of ``root_pid`` (Linux/WSL, via ``/proc``).

    Used to sweep up jobs the GUI spawned — the ``xrd-app`` CLI subprocesses
    behind the Programs console, hd-device-map, and territory-map QProcesses —
    plus anything they in turn spawned. Returns ``[]`` where ``/proc`` is
    unavailable (non-Linux), so callers degrade gracefully.
    """
    import os
    ppid_of = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        ppid_of[int(entry)] = int(line.split()[1])
                        break
        except (OSError, ValueError):
            continue
    kids_of = {}
    for pid, ppid in ppid_of.items():
        kids_of.setdefault(ppid, []).append(pid)
    out, stack = [], list(kids_of.get(root_pid, []))
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(kids_of.get(pid, []))
    return out


def _terminate_child_jobs():
    """Kill every subprocess the GUI spawned so nothing lingers after exit.

    SIGTERM first (lets the CLI jobs unwind), then SIGKILL any that ignore it.
    Only touches our own descendants — never the launching shell.
    """
    import os
    import signal
    import time
    pids = _descendant_pids(os.getpid())
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    # Brief grace period, then hard-kill stragglers.
    for _ in range(20):
        pids = [p for p in pids if _pid_alive(p)]
        if not pids:
            return
        time.sleep(0.05)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _pid_alive(pid):
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # e.g. EPERM — it exists but we can't signal it
    return True


def launch_app(project_root=None, scan=None, bin_size=3, fresh=False):
    """Create the QApplication and run the single-window app."""
    import signal
    import sys
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(project_root, scan=scan, bin_size=bin_size, fresh=fresh)
    win.show()

    # Ctrl-C support. Qt's C++ event loop otherwise swallows SIGINT, so Ctrl-C
    # in the launching terminal does nothing. Install a handler that closes the
    # window (saving gui_state via closeEvent) and stops the loop; a periodic
    # no-op timer wakes the Python interpreter often enough to actually run it.
    def _handle_sigint(*_a):
        print("\n[xrd-app] Ctrl-C — closing GUI and stopping all jobs…",
              file=sys.stderr)
        app.closeAllWindows()
        app.quit()

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except (ValueError, OSError):
        pass  # not on the main thread / unsupported — skip gracefully
    _sigint_wakeup = QTimer()
    _sigint_wakeup.timeout.connect(lambda: None)
    _sigint_wakeup.start(200)

    try:
        rc = app.exec_()
    finally:
        # Sweep up any jobs the GUI spawned (Programs console, hd/territory
        # maps) so closing the window — by Ctrl-C or the X button — never
        # leaves heavy CLI subprocesses running in the background.
        _terminate_child_jobs()
    return rc
