"""Small Qt lifecycle helpers shared by embedded GUI components."""

from PyQt5.QtCore import QProcess


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


def stop_process(process, timeout_ms=3000):
    """Kill a running QProcess and wait briefly for OS resource cleanup."""
    if process is None or process.state() == QProcess.NotRunning:
        return True
    process.kill()
    return process.waitForFinished(timeout_ms)
