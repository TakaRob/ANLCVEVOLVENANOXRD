"""Run a single tab as its own window (``python -m xrd_app.tabs.<name>``).

Each tab module calls :func:`run_standalone` from its ``__main__`` block so the
individual tools still work alone, with a minimal Setup header that shows the
loaded project/scan, then the identical UI.
"""

from __future__ import annotations

import argparse
import sys

from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget,
)

from ..config import DataManager


def run_standalone(make_tab, title):
    parser = argparse.ArgumentParser(description=f"xrd-app tab: {title}")
    parser.add_argument("--project", default=".", help="Project root")
    parser.add_argument("--scan", default=None, help="Scan number/name")
    parser.add_argument("--bin-size", type=int, default=3, help="Bin size")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    dm = DataManager(args.project, scan=args.scan)
    container = QWidget()
    lay = QVBoxLayout(container)
    header = QLabel(
        f"<b>{title}</b> — project: {dm.config.get('name') or args.project}"
        f"  ·  scan: {dm.scan_name or '—'}  ·  bin: {args.bin_size}x{args.bin_size}")
    header.setStyleSheet("padding:4px; background:#eee;")
    lay.addWidget(header)
    lay.addWidget(make_tab(args.project, scan=args.scan, bin_size=args.bin_size), 1)

    win = QMainWindow()
    win.setWindowTitle(f"xrd-app — {title}")
    win.setCentralWidget(container)
    win.resize(1400, 900)
    win.show()
    sys.exit(app.exec_())
