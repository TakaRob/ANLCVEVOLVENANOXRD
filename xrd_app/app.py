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
    QAction, QApplication, QCheckBox, QComboBox, QHBoxLayout,
    QLabel, QMainWindow, QPushButton, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from . import __version__, workspace
from .config import DataManager
from .gui.lifecycle import dispose_widget
from .tabs._embed import placeholder

# Built-in tab modules (module path under xrd_app.tabs).
# The skew-free territorial reference is its own "territory" tab: it shows the
# cell-model map, or an in-tab "Build territorial reference" button when the scan
# has no territorial artifacts yet (see tabs/territory.py). The Device View tab
# still hides territorial catalogs from its grid dropdown (they can't render on a
# fixed grid) but no longer opens the map as a popup.
_BUILTIN_TABS = ["setup", "programs", "view_label", "shape_verify", "roi_shape",
                 "device", "hd_device", "territory", "orientation", "scan_summary"]

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
    def __init__(self, project_root=None, scan=None, bin_size=None, tabs=None, fresh=False):
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

        self.setWindowTitle(f"xrd-app v{__version__} — {title}")

    def switch_project(self, project_root):
        """Open a different project at runtime and rebuild every tab.

        Deferred to the next event-loop turn so it is safe to call from inside a
        Setup-tab button handler (the Setup widget itself is rebuilt).
        """
        def _do():
            self._dispose_tabs()
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
        self._refresh_context()

    # ----- header -----------------------------------------------------
    def _build_header(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("<b>Scan:</b>"))
        self.scan_combo = QComboBox()
        self.scan_combo.setMinimumWidth(160)
        self.scan_combo.currentTextChanged.connect(self._on_scan_changed)
        row.addWidget(self.scan_combo)
        scan_steps = QVBoxLayout()
        scan_steps.setContentsMargins(0, 0, 0, 0)
        scan_steps.setSpacing(0)
        self.scan_prev_btn = QPushButton("▲")
        self.scan_next_btn = QPushButton("▼")
        for button in (self.scan_prev_btn, self.scan_next_btn):
            button.setFixedSize(20, 13)
            button.setStyleSheet("QPushButton { padding:0; font-size:8px; }")
        self.scan_prev_btn.setToolTip("Previous scan")
        self.scan_next_btn.setToolTip("Next scan")
        self.scan_prev_btn.clicked.connect(lambda: self._step_scan(-1))
        self.scan_next_btn.clicked.connect(lambda: self._step_scan(1))
        scan_steps.addWidget(self.scan_prev_btn)
        scan_steps.addWidget(self.scan_next_btn)
        row.addLayout(scan_steps)
        # Bin size is chosen per-tab (each bin-dependent tab has its own Bin
        # selector); there is intentionally no global bin selector here.

        # No reflection-set selector here: reflections always resolve per-scan →
        # per-project → bundled default (see DataManager.reflections). Author a
        # set in Setup → Manual reflections; it writes the per-scan/project file.

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
            self._update_scan_step_buttons(0)
            return
        self.scan_combo.addItems(scans or ["(no scans — load in Setup)"])
        if self.scan and self.scan in scans:
            self.scan_combo.setCurrentText(self.scan)
        elif scans:
            self.scan = scans[0]
        self.scan_combo.blockSignals(False)
        self._update_scan_step_buttons(len(scans))

    def _update_scan_step_buttons(self, n_scans):
        enabled = n_scans > 1
        for button in (getattr(self, "scan_prev_btn", None),
                       getattr(self, "scan_next_btn", None)):
            if button is not None:
                button.setEnabled(enabled)

    def _step_scan(self, amount):
        """Select the previous/next valid scan, wrapping at either end."""
        valid = [i for i in range(self.scan_combo.count())
                 if not self.scan_combo.itemText(i).startswith("(")]
        if len(valid) < 2:
            return
        current = self.scan_combo.currentIndex()
        try:
            position = valid.index(current)
        except ValueError:
            position = 0
        self.scan_combo.setCurrentIndex(valid[(position + amount) % len(valid)])

    # ----- tab lifecycle ----------------------------------------------
    def _dispose_tab(self, idx):
        """Close tab resources now, then schedule their Qt objects for deletion."""
        if idx == self.tabs.currentIndex() and self._cur_extra is not None:
            try:
                self.header_extra.removeWidget(self._cur_extra)
                self._cur_extra.deleteLater()
            except RuntimeError:
                pass
            self._cur_extra = None
        content = self._content.pop(idx, None)
        if content is not None:
            dispose_widget(content)
        host = self._hosts[idx]
        lay = host.layout()
        while lay.count():
            item = lay.takeAt(0)
            wrapper = item.widget()
            if wrapper is not None:
                dispose_widget(wrapper)
        self._built[idx] = False

    def _dispose_tabs(self, scan_dependent_only=False):
        for idx, (_mod, meta) in enumerate(self._defs):
            if not scan_dependent_only or meta.get("scan_dependent", True):
                self._dispose_tab(idx)

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
                dispose_widget(w)
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
        self._connect_bin_context(content)
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
    def _connect_bin_context(self, content):
        """Subscribe to the generic per-tab bin context contract, if exposed."""
        source = content
        if not hasattr(source, "bin_size_changed"):
            source = getattr(content, "_embedded_window", None)
        signal = getattr(source, "bin_size_changed", None)
        if signal is not None:
            signal.connect(self._on_bin_size_changed)
        current = getattr(source, "current_bin_size", None)
        if callable(current):
            self._on_bin_size_changed(current())

    def _on_bin_size_changed(self, bin_size):
        try:
            bin_size = int(bin_size)
        except (TypeError, ValueError):
            return
        if bin_size <= 0 or bin_size == self.bin_size:
            return
        self.bin_size = bin_size
        self._save_state()

    def _on_scan_changed(self, text):
        if not text or text.startswith("("):
            return
        self.scan = text
        self._refresh_context()

    def _refresh_context(self):
        """Rebuild scan-dependent tabs; push context to persistent ones."""
        self._dispose_tabs(scan_dependent_only=True)
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
        self._dispose_tabs()
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


def _remote_x_display():
    """True on a forwarded/remote X display (e.g. ``ssh -X``).

    ``SSH_CONNECTION``/``SSH_CLIENT`` mark an SSH session; ``DISPLAY`` that isn't
    a local socket (``:0`` / ``unix:0``) is networked — ssh forwarding sets
    ``DISPLAY=localhost:NN``, which counts as remote. Overridable with
    ``XRD_FORCE_LOCAL_X=1`` (treat as local) — mirrors rsm_view's GL guard.
    """
    import os
    if os.environ.get("XRD_FORCE_LOCAL_X") == "1":
        return False
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"):
        return True
    disp = os.environ.get("DISPLAY", "")
    if disp and not (disp.startswith(":") or disp.startswith("unix:")):
        return True
    return False


def _harden_env_for_remote_x():
    """Make Qt survive a forwarded X connection. Must run before QApplication.

    Over ``ssh -X`` two Qt defaults break the app:
      * MIT-SHM pixmaps don't exist across the network — blitting a large raw
        frame triggers ``XIO: fatal IO error`` and kills the app. Disable SHM.
      * The xcb GL integration spawns a separate native GL window (a second
        taskbar icon) that crashes the X server / compositor on interaction.
        Turn it off (the 3D view already degrades to a hint in this case).
    Both are set with ``setdefault`` so an explicit override still wins; GL is
    left alone when the user forces it on with ``XRD_FORCE_GL=1``.
    """
    import os
    if not _remote_x_display():
        return
    os.environ.setdefault("QT_X11_NO_MITSHM", "1")
    if os.environ.get("XRD_FORCE_GL") != "1":
        os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")


def launch_app(project_root=None, scan=None, bin_size=None, fresh=False):
    """Create the QApplication and run the single-window app."""
    import signal
    import sys
    # Harden the environment for forwarded/remote X *before* QApplication reads
    # it (see _harden_env_for_remote_x). No-op on a local display.
    _harden_env_for_remote_x()
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
