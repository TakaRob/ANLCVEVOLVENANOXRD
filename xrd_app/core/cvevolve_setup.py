"""Scaffold a CVEvolve project from bundled defaults and toggle Hutch tracking.

Beamline users don't arrive with a ``config.yaml`` / prompt files, so
:func:`scaffold_project` copies the bundled templates in ``xrd_app/assets/`` into
a project's ``CVEvolve/`` directory, filling in the workspace paths and session
name. :func:`set_hutch` flips the Hutch (SQLite) tracking backend on/off in an
existing config, and :func:`default_hutch_db` is the single source of truth for
where that DB lives — shared by the CLI and the GUI's live SQL view so both agree
without parsing output.

Pure logic (no PyQt, no click): the CLI commands are thin wrappers over this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

# Bundled template -> filename written into the project CVEvolve/ dir.
_TEMPLATES = {
    "cvevolve_config.yaml": "config.yaml",
    "cvevolve_prompt.md": "prompt.md",
    "cvevolve_holdout_prompt.md": "holdout_test_prompt.md",
}


def _assets_dir() -> Path:
    """Bundled ``xrd_app/assets/`` directory (same asset root as DataManager)."""
    return Path(__file__).resolve().parent.parent / "assets"


def default_hutch_db(config_path, name: Optional[str] = None) -> Path:
    """Canonical Hutch SQLite path for a config: ``<cfg_dir>/hutch/<name>.db``.

    ``name`` defaults to the config's ``name`` field, then the config stem. Kept
    in one place so the CLI (which writes it) and the GUI (which reads it for the
    live table view) resolve the same path.
    """
    config_path = Path(config_path)
    if name is None:
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            name = data.get("name")
        except (OSError, yaml.YAMLError):
            name = None
        name = name or config_path.stem
    return config_path.parent / "hutch" / f"{name}.db"


def scaffold_project(dest_dir, name: str, force: bool = False) -> Dict[str, list]:
    """Copy the bundled CVEvolve templates into ``dest_dir``, filling placeholders.

    Writes ``config.yaml`` / ``prompt.md`` / ``holdout_test_prompt.md``. The
    config's ``{{ROOT}}`` (this CVEvolve dir) and ``{{NAME}}`` (session name)
    placeholders are expanded so ``workspace`` and ``hutch`` paths point into
    the project. Existing files are left untouched unless ``force``.

    Returns ``{"written": [...], "skipped": [...]}`` (absolute paths as str).
    """
    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "hutch").mkdir(exist_ok=True)  # gitignored; Hutch DB lands here
    assets = _assets_dir()

    written, skipped = [], []
    for template, out_name in _TEMPLATES.items():
        out_path = dest / out_name
        if out_path.exists() and not force:
            skipped.append(str(out_path))
            continue
        text = (assets / template).read_text()
        if out_name.endswith(".yaml"):
            text = text.replace("{{ROOT}}", str(dest)).replace("{{NAME}}", name)
        out_path.write_text(text)
        written.append(str(out_path))
    return {"written": written, "skipped": skipped}


def set_hutch(config_path, enabled: bool, db_path=None) -> Path:
    """Enable/disable Hutch in an existing config; return the resolved DB path.

    Sets ``hutch.enabled`` and, when enabling, ensures ``hutch.db_path`` is set
    (using the existing value, the explicit ``db_path``, or
    :func:`default_hutch_db`). The parent ``hutch/`` directory is created so the
    daemon can write there immediately.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    hutch = data.get("hutch")
    if not isinstance(hutch, dict):
        hutch = {"project": "cvevolve", "run_id": None,
                 "daemon_url": None, "strict": False}
    hutch["enabled"] = bool(enabled)

    resolved = db_path or hutch.get("db_path") or default_hutch_db(config_path)
    resolved = Path(resolved)
    hutch["db_path"] = str(resolved)
    data["hutch"] = hutch

    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
