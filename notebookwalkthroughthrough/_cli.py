"""Small helpers shared by the percent-format tutorial notebooks."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def command_text(module: str, *args: object) -> str:
    """Return the exact Python-module command used by a notebook cell."""
    return shlex.join([sys.executable, "-m", module, *(str(arg) for arg in args)])


def run_command(
    module: str,
    *args: object,
    execute: bool = True,
    heavy: bool = False,
    allow_heavy: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Print a command, optionally execute it, and print captured output."""
    command = [sys.executable, "-m", module, *(str(arg) for arg in args)]
    print("$", shlex.join(command))
    if not execute:
        print("  Not run. Set the notebook's execution toggle to True when ready.")
        return None
    if heavy and not allow_heavy:
        print("  Heavy command blocked. Review it, then set RUN_HEAVY = True.")
        return None

    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    print(f"[exit status {result.returncode}]")
    return result


def project_ready(project_root: Path) -> bool:
    """Explain a missing project without raising during an introductory run."""
    config = project_root.expanduser() / "config.yaml"
    if config.is_file():
        return True
    print(f"No xrd-app project found at {project_root.expanduser()}")
    print("Set XRD_PROJECT or edit PROJECT_ROOT before running project commands.")
    return False
