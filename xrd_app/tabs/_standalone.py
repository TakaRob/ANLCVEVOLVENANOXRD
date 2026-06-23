"""Run a single tab as its own window (``python -m xrd_app.tabs.<name>``).

Each tab module calls :func:`run_standalone` from its ``__main__`` block. The
window is a focused two-tab :class:`~xrd_app.app.MainWindow` — a fully functional
Setup tab plus the requested GUI — so the standalone tools get the same
project/scan switching as the combined app.
"""

from __future__ import annotations

import argparse
import sys

from PyQt5.QtWidgets import QApplication


def run_standalone(make_tab, title):
    parser = argparse.ArgumentParser(description=f"xrd-app tab: {title}")
    parser.add_argument("--project", default=None,
                        help="Project root (default: last-opened project)")
    parser.add_argument("--scan", default=None, help="Scan number/name")
    parser.add_argument("--bin-size", type=int, default=3, help="Bin size")
    args = parser.parse_args()

    # Derive the tab's module short-name, e.g. "xrd_app.tabs.device" -> "device".
    # Under ``python -m xrd_app.tabs.device`` the defining module is "__main__",
    # so fall back to its import spec to recover the real dotted name.
    mod_name = make_tab.__module__
    if mod_name == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        if spec is not None:
            mod_name = spec.name
    name = mod_name.rsplit(".", 1)[-1]
    tabs = ["setup"] if name == "setup" else ["setup", name]

    from PyQt5.QtCore import Qt
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    from ..app import MainWindow  # local import avoids an import cycle
    win = MainWindow(args.project, scan=args.scan, bin_size=args.bin_size, tabs=tabs)
    win.show()
    sys.exit(app.exec_())


def run_dialog_standalone(open_dialog, title):
    """Run a popup tool (a ``QDialog``) as its own window.

    Mirrors :func:`run_standalone` for tools that are dialogs rather than tabs:
    same ``--project/--scan/--bin-size`` arguments and last-opened-project
    resolution, so e.g. ``python -m xrd_app.tabs.reflection_popup`` opens the
    Manual Reflections window directly. ``open_dialog(project_root, scan,
    bin_size)`` must return a ``QDialog``.
    """
    parser = argparse.ArgumentParser(description=f"xrd-app tool: {title}")
    parser.add_argument("--project", default=None,
                        help="Project root (default: last-opened project)")
    parser.add_argument("--scan", default=None, help="Scan number/name")
    parser.add_argument("--bin-size", type=int, default=3, help="Bin size")
    args = parser.parse_args()

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QMessageBox
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    from .. import workspace  # local import avoids an import cycle
    project = args.project
    if project is None:
        last = workspace.get_last_project()
        project = str(last) if last else None
    if project is None:
        QMessageBox.critical(
            None, title,
            "No project to open.\n\nPass --project <root>, or open/create a "
            "project in the main app first (xrd-app gui).")
        sys.exit(1)

    dlg = open_dialog(project, scan=args.scan, bin_size=args.bin_size)
    sys.exit(dlg.exec_())
