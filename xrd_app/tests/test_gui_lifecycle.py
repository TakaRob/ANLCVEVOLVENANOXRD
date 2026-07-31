"""Headless tests for GUI resource lifecycle helpers."""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication, QWidget

from xrd_app.gui.lifecycle import dispose_widget, stop_thread
from xrd_app.tabs._console import JobConsole


def _app():
    return QApplication.instance() or QApplication([])


def test_dispose_widget_closes_before_deferred_delete():
    _app()
    closed = []

    class Window(QWidget):
        def closeEvent(self, event):  # noqa: N802
            closed.append(True)
            super().closeEvent(event)

    dispose_widget(Window())
    assert closed == [True]


def test_stop_thread_requests_interruption_and_waits():
    class Worker(QThread):
        def run(self):
            while not self.isInterruptionRequested():
                self.msleep(1)

    worker = Worker()
    worker.start()
    assert stop_thread(worker)
    assert not worker.isRunning()


def test_job_console_handles_process_start_failure(monkeypatch):
    app = _app()
    console = JobConsole()
    monkeypatch.setattr(sys, "executable", "/definitely/missing/python")

    console.run(["--help"])
    deadline = time.monotonic() + 2
    while console._proc is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert console._proc is None
    assert "failed to start" in console.log.toPlainText()
    assert console.status.text().startswith("failed")
