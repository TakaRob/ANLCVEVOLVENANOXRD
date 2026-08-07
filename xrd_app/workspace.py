"""App-level settings: the workspace dir holding all projects, and the last
project opened.

A *project* is a directory with a ``config.yaml`` (see :mod:`xrd_app.config`).
The *workspace* is the parent "XRD-APP Directory" that holds many such projects
side by side::

    <workspace>/
      Luo Scan 203/      config.yaml + Raw/ Binned/ Metadata/ ...
      Another Sample/    config.yaml + ...

These settings live outside any single project — in ``~/.xrd-app/settings.json``
— so the GUI can find and list projects before one is opened. The current and
recent workspaces are remembered, while the launch directory is searched each
session; ``last_project`` lets the app reopen where you left off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import (CONFIG_FILENAME, DataManager, ProjectConfig, default_config,
                     safe_component)

SETTINGS_DIR = Path.home() / ".xrd-app"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"
# Captured before any file dialog or tab can affect the process directory.
LAUNCH_DIRECTORY = Path.cwd().resolve()


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
    previous = [s.get("workspace"), *s.get("recent_workspaces", [])]
    s["workspace"] = str(p)
    s["recent_workspaces"] = [
        str(candidate) for candidate in dict.fromkeys(previous)
        if candidate and Path(candidate).resolve() != p
    ][:9]
    save_settings(s)
    return p


def get_launch_directory() -> Path:
    """Directory from which the GUI process was launched."""
    return LAUNCH_DIRECTORY


def list_workspaces() -> list[Path]:
    """Current, launch, and recently selected workspaces that still exist."""
    s = load_settings()
    candidates = [s.get("workspace"), str(LAUNCH_DIRECTORY),
                  *s.get("recent_workspaces", [])]
    out = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if path.is_dir() and path not in out:
            out.append(path)
    return out


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


def get_last_xrf_project() -> Optional[Path]:
    """Path to the last XRF project, or None if its add-on no longer exists."""
    last = load_settings().get("last_xrf_project")
    if not last:
        return None
    path = Path(last)
    return path if (path / "XRF" / "xrf_config.yaml").exists() else None


def set_last_xrf_project(root) -> None:
    settings = load_settings()
    settings["last_xrf_project"] = str(Path(root).resolve())
    save_settings(settings)


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


def discover_projects() -> list[Path]:
    """Projects in the launch, current, and recently selected workspaces."""
    projects = []
    for ws in list_workspaces():
        if is_project(ws) and ws not in projects:
            projects.append(ws)
        for name in list_projects(ws):
            root = (ws / name).resolve()
            if root not in projects:
                projects.append(root)
    return projects


def project_root(name: str, workspace: Optional[Path] = None) -> Path:
    ws = workspace or get_workspace()
    if not ws:
        raise ValueError("No workspace set - choose an XRD-APP Directory first.")
    ws = Path(ws).resolve()
    root = (ws / safe_component(name, label="project name")).resolve()
    try:
        root.relative_to(ws)
    except ValueError:
        raise ValueError("Invalid project name: path escapes workspace") from None
    return root


def create_project(name: str, workspace: Optional[Path] = None,
                   scan_number: Optional[int] = None) -> Path:
    """Create ``<workspace>/<name>/`` with config.yaml + the standard tree.

    Records it as the last-opened project and returns its root path.
    """
    ws = workspace or get_workspace()
    if not ws:
        raise ValueError("No workspace set - choose an XRD-APP Directory first.")
    name = safe_component(name, label="project name")
    root = project_root(name, ws)
    config_path = root / CONFIG_FILENAME
    if config_path.exists():
        raise FileExistsError(f"Project already exists; refusing to overwrite {config_path}")
    root.mkdir(parents=True, exist_ok=True)
    cfg = ProjectConfig(root, data=default_config(name, root, scan_number))
    cfg.create_tree()
    cfg.save()

    set_last_project(root)
    return root
