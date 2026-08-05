"""PyQt5 launchers for the packaged labeling and analysis GUIs."""

from __future__ import annotations

import importlib
from pathlib import Path

_GUI_TOOLS = {
    'label': 'xrd_app.gui.labeling',
    'view': 'xrd_app.gui.viewer',
    'device-map': 'xrd_app.gui.device_map',
    'orientation': 'xrd_app.gui.orientation',
}


def launch(tool: str, root: str | Path = '.', scan=None, bin_size: int = 3) -> int:
    """Launch a packaged GUI tool and return its process exit code."""
    root = Path(root).resolve()
    if tool not in _GUI_TOOLS:
        raise ValueError(f"Unknown GUI tool: {tool!r}. Choose from {sorted(_GUI_TOOLS)}.")

    module_name = _GUI_TOOLS[tool]
    mod = importlib.import_module(module_name)
    if not hasattr(mod, 'launch_gui'):
        raise RuntimeError(f"{module_name} does not expose launch_gui()")

    print(f"[gui] launching {tool} ({module_name})")
    import inspect
    kwargs = {'project_root': str(root), 'scan': scan, 'bin_size': bin_size}
    sig = inspect.signature(mod.launch_gui)
    kwargs = {key: value for key, value in kwargs.items() if key in sig.parameters}
    try:
        mod.launch_gui(**kwargs)
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
