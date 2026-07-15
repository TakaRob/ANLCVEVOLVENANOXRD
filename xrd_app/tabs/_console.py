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
        self._queue = None  # None = single-job mode; a list = batch queue

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

    def run(self, args, cwd=None, on_finished=None, header=None):
        """Run ``[python, -m, xrd_app.cli, *args]`` (args is the CLI arg list).

        ``on_finished(exit_code)`` is invoked once when the process exits (e.g.
        to refresh a status label after the job completes). ``header`` is an
        optional note printed above the command (it survives the log clear).
        """
        if self.is_running():
            self._append("\n[a job is already running — cancel it first]\n")
            return
        self._queue = None
        self._on_finished_cb = on_finished
        self.progress.setValue(0)
        self.log.clear()
        if header:
            self._append(header if header.endswith("\n") else header + "\n")
        self._start(args, cwd)

    def run_many(self, jobs, cwd=None, on_all_finished=None, header=None):
        """Queue several CLI invocations, run them back-to-back in one console.

        ``jobs`` is a list of arg-lists (each like a single :meth:`run` call).
        Output from all jobs streams into the same log with a per-job header;
        one failing job does not abort the queue. ``on_all_finished(n_failed)``
        fires once when the whole queue drains (or is cancelled). Used by the
        Programs tab to fan a run out over many scans × algorithms via the CLI.
        """
        if self.is_running():
            self._append("\n[a job is already running — cancel it first]\n")
            return
        jobs = [list(j) for j in jobs]
        if not jobs:
            return
        self._queue = jobs
        self._queue_total = len(jobs)
        self._queue_cwd = cwd
        self._queue_failures = 0
        self._on_all_finished_cb = on_all_finished
        self.progress.setValue(0)
        self.log.clear()
        if header:
            self._append(header if header.endswith("\n") else header + "\n")
        self._run_next_in_queue()

    def _run_next_in_queue(self):
        if not self._queue:
            n_failed = self._queue_failures
            total = self._queue_total
            self._queue = None
            self.status.setText(
                f"done ({total - n_failed}/{total})" if not n_failed
                else f"finished with {n_failed}/{total} failed")
            self.cancel_btn.setEnabled(False)
            self._append(f"\n[batch complete: {total - n_failed}/{total} "
                         f"succeeded{f', {n_failed} failed' if n_failed else ''}]\n")
            cb = getattr(self, "_on_all_finished_cb", None)
            self._on_all_finished_cb = None
            if cb is not None:
                try:
                    cb(n_failed)
                except Exception:
                    pass
            return
        args = self._queue.pop(0)
        idx = self._queue_total - len(self._queue)
        self._append(f"\n{'='*60}\n[job {idx}/{self._queue_total}]\n{'='*60}\n")
        self.status.setText(f"job {idx}/{self._queue_total}")
        self._start(args, self._queue_cwd)

    def _start(self, args, cwd=None):
        """Launch one QProcess for ``args`` (does not clear the log)."""
        cmd = [sys.executable, "-m", "xrd_app.cli", *[str(a) for a in args]]
        self._append("$ " + " ".join(cmd) + "\n")
        self._proc = QProcess(self)
        if cwd:
            self._proc.setWorkingDirectory(str(cwd))
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._on_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.start(cmd[0], cmd[1:])
        if self._queue is None:
            self.status.setText("running")
        self.cancel_btn.setEnabled(True)

    def cancel(self):
        if self.is_running():
            self._queue = None  # drop any remaining queued jobs
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
        self._append(f"\n[exit {code}]\n")
        # Batch mode: tally, then advance the queue (a cancel sets _queue=None).
        if self._queue is not None:
            if code != 0:
                self._queue_failures += 1
            self._run_next_in_queue()
            return
        self.status.setText("done" if code == 0 else f"failed (exit {code})")
        self.cancel_btn.setEnabled(False)
        cb = getattr(self, "_on_finished_cb", None)
        self._on_finished_cb = None
        if cb is not None:
            try:
                cb(code)
            except Exception:
                pass

    def _append(self, text):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().End)
