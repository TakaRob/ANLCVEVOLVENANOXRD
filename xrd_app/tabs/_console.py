"""Embedded job console: run a CLI subprocess, stream output, parse progress.

Used by the Programs tab so every "Run" button shells out to the CLI engine
(``python -m xrd_app.cli ...``) while showing live output, a progress bar, and a
Cancel button. Closing the app or pressing Cancel kills the process.
"""

from __future__ import annotations

import re
import sys

from PyQt5.QtCore import QProcess
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
    QWidget,
)

_PROGRESS_RE = re.compile(r"PROGRESS\s+(\d+)\s*/\s*(\d+)")


class JobConsole(QWidget):
    """A read-only console + progress bar + cancel button driving one QProcess."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None

        lay = QVBoxLayout(self)
        bar_row = QHBoxLayout()
        self.status = QLabel("idle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        bar_row.addWidget(self.status, 1)
        bar_row.addWidget(self.progress, 2)
        bar_row.addWidget(self.cancel_btn)
        lay.addLayout(bar_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("monospace", 9))
        lay.addWidget(self.log)

    # ----- lifecycle ---------------------------------------------------
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def run(self, args, cwd=None):
        """Run ``[python, -m, xrd_app.cli, *args]`` (args is the CLI arg list)."""
        if self.is_running():
            self._append("\n[a job is already running — cancel it first]\n")
            return
        self.progress.setValue(0)
        self.log.clear()
        cmd = [sys.executable, "-m", "xrd_app.cli", *[str(a) for a in args]]
        self._append("$ " + " ".join(cmd) + "\n")

        self._proc = QProcess(self)
        if cwd:
            self._proc.setWorkingDirectory(str(cwd))
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(cmd[0], cmd[1:])
        self.status.setText("running")
        self.cancel_btn.setEnabled(True)

    def cancel(self):
        if self.is_running():
            self._proc.kill()
            self._append("\n[cancelled]\n")

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        self.cancel()
        super().closeEvent(event)

    # ----- internals ---------------------------------------------------
    def _on_output(self):
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            m = _PROGRESS_RE.search(line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n:
                    self.progress.setValue(int(100 * i / n))
                self.status.setText(f"{i}/{n}")
                continue  # don't echo raw PROGRESS markers
            self._append(line + "\n")

    def _on_finished(self, code, _status):
        self.progress.setValue(100 if code == 0 else self.progress.value())
        self.status.setText("done" if code == 0 else f"failed (exit {code})")
        self.cancel_btn.setEnabled(False)
        self._append(f"\n[exit {code}]\n")

    def _append(self, text):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().End)
