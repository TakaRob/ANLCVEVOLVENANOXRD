"""Small Qt lifecycle helpers shared by embedded GUI components."""

import os
import shutil
import signal
import time

from PyQt5.QtCore import QObject, QPoint, QProcess, Qt, QTimer
from PyQt5.QtGui import QColor, QCursor, QPainter, QPen, QPixmap, QPolygon
from PyQt5.QtWidgets import QWidget


_PROCESS_GROUP_PROPERTY = "_xrd_owned_process_group"


class _CursorOverlay(QWidget):
    """Draw an arrow inside a window without relying on the native cursor plane."""

    def __init__(self, window):
        super().__init__(window)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.resize(24, 32)

    def paintEvent(self, event):  # noqa: N802 (Qt signature)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        points = QPolygon([
            QPoint(2, 1), QPoint(2, 25), QPoint(8, 19), QPoint(13, 30),
            QPoint(18, 28), QPoint(13, 17), QPoint(22, 17),
        ])
        painter.setPen(QPen(QColor("white"), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(QColor("black"))
        painter.drawPolygon(points)
        painter.setPen(QPen(QColor("black"), 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPolygon(points)


class _CursorOverlayManager(QObject):
    def __init__(self, application):
        super().__init__(application)
        self.application = application
        self.overlays = {}
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(16)

    def _update(self):
        position = QCursor.pos()
        active = None
        for window in self.application.topLevelWidgets():
            if not window.isWindow() or not window.isVisible():
                continue
            local = window.mapFromGlobal(position)
            if window.rect().contains(local):
                active = window
                overlay = self.overlays.get(window)
                if overlay is None:
                    overlay = _CursorOverlay(window)
                    self.overlays[window] = overlay
                overlay.move(local.x() - 2, local.y() - 1)
                overlay.raise_()
                overlay.show()
                break
        for window, overlay in list(self.overlays.items()):
            if not window.isVisible():
                overlay.deleteLater()
                del self.overlays[window]
            elif window is not active:
                overlay.hide()


def install_visible_cursor(application):
    """Install a cursor rendered as normal window content on all app windows."""
    manager = getattr(application, "_xrd_cursor_overlay_manager", None)
    if manager is None:
        manager = _CursorOverlayManager(application)
        application._xrd_cursor_overlay_manager = manager
    return manager


def start_process(process, program, arguments=()):
    """Start a QProcess, isolated in its own session on Linux/WSL when possible."""
    program = str(program)
    arguments = [str(arg) for arg in arguments]
    executable = shutil.which(program)
    setsid = shutil.which("setsid") if os.name == "posix" and executable else None
    if setsid:
        process.setProperty(_PROCESS_GROUP_PROPERTY, True)
        process.start(setsid, [program, *arguments])
    else:
        process.setProperty(_PROCESS_GROUP_PROPERTY, False)
        process.start(program, arguments)


def _owned_process_group(process):
    if not process.property(_PROCESS_GROUP_PROPERTY) or not hasattr(os, "killpg"):
        return None
    pid = int(process.processId())
    return pid if pid > 0 else None


def _signal_process_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError:
        return False
    return True


def _process_group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def dispose_widget(widget):
    """Close a widget and its exposed embedded window before deferred deletion."""
    if widget is None:
        return
    embedded = getattr(widget, "_embedded_window", None)
    if embedded is not None and embedded is not widget:
        dispose_widget(embedded)
        try:
            widget._embedded_window = None
        except RuntimeError:
            pass
    try:
        widget.close()
        widget.deleteLater()
    except RuntimeError:
        pass


def stop_thread(thread):
    """Request cooperative interruption and wait for a worker to finish."""
    if thread is None or not thread.isRunning():
        return True
    thread.requestInterruption()
    return thread.wait()


def stop_process_async(process, timeout_ms=3000):
    """Request TERM now and schedule KILL without blocking the Qt event loop."""
    if process is None or process.state() == QProcess.NotRunning:
        return
    pgid = _owned_process_group(process)
    if pgid is not None:
        _signal_process_group(pgid, signal.SIGTERM)
    else:
        process.terminate()

    def kill_if_running():
        if pgid is not None:
            if _process_group_alive(pgid):
                _signal_process_group(pgid, signal.SIGKILL)
        elif process.state() != QProcess.NotRunning:
            process.kill()

    QTimer.singleShot(timeout_ms, kill_if_running)


def stop_process(process, timeout_ms=3000):
    """TERM an owned process group, then KILL it after a bounded grace period."""
    if process is None:
        return True
    pgid = _owned_process_group(process)
    if process.state() == QProcess.NotRunning and pgid is None:
        return True
    if pgid is not None:
        _signal_process_group(pgid, signal.SIGTERM)
    else:
        process.terminate()
    deadline = time.monotonic() + timeout_ms / 1000
    while pgid is not None and _process_group_alive(pgid) and time.monotonic() < deadline:
        if process.state() == QProcess.NotRunning:
            time.sleep(0.01)
        else:
            process.waitForFinished(min(50, timeout_ms))
    if pgid is None and process.waitForFinished(timeout_ms):
        return True
    if pgid is not None and not _process_group_alive(pgid):
        process.waitForFinished(100)
        return True
    if pgid is not None:
        _signal_process_group(pgid, signal.SIGKILL)
    else:
        process.kill()
    process.waitForFinished(timeout_ms)
    return pgid is None or not _process_group_alive(pgid)
