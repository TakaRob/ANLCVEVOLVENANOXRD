"""Helpers for embedding legacy QMainWindow GUIs as tab content."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


def embed_window(win) -> QWidget:
    """Wrap a constructed QMainWindow in a container suitable for a tab.

    A QMainWindow embeds fine as a child widget; all four legacy GUIs keep their
    content in the central widget, so nothing is lost. We retain a reference on
    the container so the window is not garbage-collected.
    """
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(win)
    container._embedded_window = win
    return container


def embed_with_toolbar(win, buttons) -> QWidget:
    """Embed a window with a thin toolbar of buttons above it.

    ``buttons`` is a list of ``(label, callback)`` tuples. Keeps the window from
    being garbage-collected.
    """
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    bar = QHBoxLayout()
    for label, cb in buttons:
        b = QPushButton(label)
        b.clicked.connect(cb)
        bar.addWidget(b)
    bar.addStretch()
    lay.addLayout(bar)
    lay.addWidget(win)
    container._embedded_window = win
    return container


def placeholder(message: str, detail: str = "") -> QWidget:
    """A simple centered message widget shown when a tab can't be built yet."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.addStretch()
    lbl = QLabel(message)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("font-size: 14px;")
    lay.addWidget(lbl)
    if detail:
        d = QLabel(detail)
        d.setAlignment(Qt.AlignCenter)
        d.setWordWrap(True)
        d.setStyleSheet("color: #999; font-size: 11px;")
        lay.addWidget(d)
    lay.addStretch()
    return w
