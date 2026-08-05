"""Import completed CVEvolve winners into a project's algorithm library."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import yaml

from .processing import load_detector
from .save_algorithm import register_in_catalog


def session_root(config_path) -> Path:
    """Resolve the session root produced by a CVEvolve config."""
    config_path = Path(config_path).resolve()
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}
    workspace = config.get("workspace") or {}
    root = workspace.get("root_dir")
    name = config.get("name")
    if not root or not name:
        raise ValueError("CVEvolve config requires name and workspace.root_dir")
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve() / str(name)


def _safe_name(value: str) -> str:
    import re
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return name or "cvevolve_winner"


def _holdout_metric(session: Path, candidate_id: str | None):
    database = session / "history" / "search_history.sqlite"
    if not database.is_file() or not candidate_id:
        return None, None
    uri = f"file:{database}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            row = connection.execute(
                "SELECT metric_name, value FROM holdout_test_metrics "
                "WHERE candidate_id = ? AND value IS NOT NULL "
                "ORDER BY id DESC LIMIT 1", (candidate_id,),
            ).fetchone()
    except sqlite3.Error:
        return None, None
    return row if row else (None, None)


def register_winner(config_path, project_root, *, name=None, bin_size=None) -> dict:
    """Validate and register a completed peak-detector winner.

    Returns metadata for the cataloged project-owned detector. Re-registering the
    same name replaces its copied files and catalog entry.
    """
    from ..config import DataManager

    session = session_root(config_path)
    reports = session / "reports"
    summary_path = reports / "final_summary.json"
    winner = reports / "best_candidate.py"
    if not summary_path.is_file() or not winner.is_file():
        raise FileNotFoundError(
            f"Completed CVEvolve winner not found under {reports}; let the run finish first")
    summary = json.loads(summary_path.read_text())
    candidate = summary.get("best_candidate") or {}
    load_detector(winner)

    detector_name = _safe_name(name or candidate.get("candidate_name") or session.name)
    dm = DataManager(project_root)
    output_dir = dm.project_algorithms_dir("peak") / detector_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    source_path = Path(candidate.get("code_file_path") or candidate.get("code_path") or "")
    if source_path.is_file():
        source_dir = source_path.parent
        for child in source_dir.iterdir():
            target = output_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            elif child.suffix != ".pyc":
                shutil.copy2(child, target)
    shutil.copy2(winner, output_dir / "detector.py")
    load_detector(output_dir / "detector.py")

    metric_name, holdout_value = _holdout_metric(session, candidate.get("candidate_id"))
    primary_value = candidate.get("primary_metric_value")
    holdout_is_f1 = bool(metric_name and "f1" in metric_name.lower() and "f2" not in metric_name.lower())
    entry = {
        "name": detector_name,
        "bin_size": f"{int(bin_size)}x{int(bin_size)}" if bin_size else None,
        "file": f"{detector_name}/detector.py",
        "role": "detector",
        "kind": "peak",
        "holdout_f1": holdout_value if holdout_is_f1 else None,
        "holdout_f2": None if holdout_is_f1 else holdout_value,
        "development_f2": primary_value,
        "source": "CVEvolve",
        "session": str(session),
        "candidate_id": candidate.get("candidate_id"),
    }
    register_in_catalog(dm.project_algorithms_dir("peak") / "catalog.json", entry)
    return {"path": output_dir / "detector.py", "entry": entry}
