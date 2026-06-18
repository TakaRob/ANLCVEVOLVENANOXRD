"""PyQt5 GUI launchers for labeling, feature viewing, and detector/orientation maps.

Ported GUIs live in this package as modules exposing ``launch_gui(project_root)``
and resolve every path through :class:`xrd_tools.config.DataManager`. For any GUI
not yet ported, :func:`launch` falls back to running the original script from the
project's ``analysis/`` directory in a subprocess.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# tool name -> (in-package module, original script filename)
_GUI_TOOLS = {
    'label': ('xrd_tools.gui.labeling', 'labeling_tool.py'),
    'view': ('xrd_tools.gui.viewer', 'feature_viewer.py'),
    'device-map': ('xrd_tools.gui.device_map', 'device_map.py'),
    'orientation': ('xrd_tools.gui.orientation', 'orientation_map.py'),
}


def _find_script(filename: str, root: Path) -> Path | None:
    for c in (root / 'analysis' / filename, root / filename):
        if c.exists():
            return c
    return None


def launch(tool: str, root: str | Path = '.', scan=None, bin_size: int = 3,
           extra_args: Iterable[str] = ()) -> int:
    """Launch a GUI tool. Returns the process exit code."""
    root = Path(root).resolve()
    if tool not in _GUI_TOOLS:
        raise ValueError(f"Unknown GUI tool: {tool!r}. Choose from {sorted(_GUI_TOOLS)}.")

    module_name, script_name = _GUI_TOOLS[tool]

    # Prefer the in-package, DataManager-aware implementation.
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        mod = None

    if mod is not None and hasattr(mod, 'launch_gui'):
        print(f"[gui] launching {tool} ({module_name})")
        # labeling.launch_gui has no bin_size param; pass what each accepts.
        import inspect
        kwargs = {'project_root': str(root), 'scan': scan, 'bin_size': bin_size}
        sig = inspect.signature(mod.launch_gui)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        try:
            mod.launch_gui(**kwargs)
            return 0
        except SystemExit as e:  # GUIs call sys.exit(app.exec_())
            return int(e.code or 0)

    # Fallback: run the original analysis/ script in a subprocess.
    script = _find_script(script_name, root)
    if script is None:
        print(f"Error: '{tool}' GUI is not packaged yet and {script_name} "
              f"was not found under {root}.")
        return 1
    cmd = [sys.executable, str(script)]
    try:
        if '--project-root' in script.read_text(errors='ignore'):
            cmd += ['--project-root', str(root)]
    except OSError:
        pass
    cmd += list(extra_args)
    print(f"[gui] launching {tool} (subprocess): {' '.join(cmd)}")
    return subprocess.call(cmd)
