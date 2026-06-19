"""App-level settings: the workspace dir holding all projects, and the last
project opened.

A *project* is a directory with a ``config.yaml`` (see :mod:`xrd_app.config`).
The *workspace* is the parent "XRD-APP Directory" that holds many such projects
side by side::

    <workspace>/
      Luo Scan 203/      config.yaml + Raw/ Binned/ Metadata/ ...
      Another Sample/    config.yaml + ...

These settings live outside any single project — in ``~/.xrd-app/settings.json``
— so the GUI can find and list projects before one is opened. The workspace is
chosen once (first launch) and remembered; ``last_project`` lets the app reopen
where you left off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import CONFIG_FILENAME, ProjectConfig, default_config

SETTINGS_DIR = Path.home() / ".xrd-app"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


# ----- raw settings I/O ----------------------------------------------------
def load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text()) or {}
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


# ----- workspace -----------------------------------------------------------
def get_workspace() -> Optional[Path]:
    w = load_settings().get("workspace")
    return Path(w) if w else None


def set_workspace(path) -> Path:
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    s = load_settings()
    s["workspace"] = str(p)
    save_settings(s)
    return p


# ----- last project --------------------------------------------------------
def get_last_project() -> Optional[Path]:
    """Path to the last-opened project, or None if it no longer exists."""
    last = load_settings().get("last_project")
    if not last:
        return None
    p = Path(last)
    return p if (p / CONFIG_FILENAME).exists() else None


def set_last_project(root) -> None:
    s = load_settings()
    s["last_project"] = str(Path(root).resolve())
    save_settings(s)


# ----- project discovery / creation ---------------------------------------
def is_project(path) -> bool:
    return (Path(path) / CONFIG_FILENAME).exists()


def list_projects(workspace: Optional[Path] = None) -> list[str]:
    """Names of project sub-directories (those containing config.yaml)."""
    ws = workspace or get_workspace()
    if not ws or not Path(ws).is_dir():
        return []
    return sorted(p.name for p in Path(ws).iterdir()
                  if p.is_dir() and is_project(p))


def project_root(name: str, workspace: Optional[Path] = None) -> Path:
    ws = workspace or get_workspace()
    return Path(ws) / name


def create_project(name: str, workspace: Optional[Path] = None,
                   scan_number: Optional[int] = None) -> Path:
    """Create ``<workspace>/<name>/`` with config.yaml + the standard tree.

    Records it as the last-opened project and returns its root path.
    """
    ws = workspace or get_workspace()
    if not ws:
        raise ValueError("No workspace set — choose an XRD-APP Directory first.")
    root = Path(ws) / name
    root.mkdir(parents=True, exist_ok=True)
    cfg = ProjectConfig(root, data=default_config(name, root, scan_number))
    cfg.create_tree()
    cfg.save()
    set_last_project(root)
    return root
