"""Detector geometry: convert a pyFAI ``.poni`` calibration to a 2θ-per-pixel map.

The whole pipeline consumes a ``tth.tiff`` (2θ in degrees at every pixel). Many
users arrive with a pyFAI ``.poni`` instead; this module turns one into the other
so everything downstream stays uniform. pyFAI is imported lazily so the core app
installs without it (``pip install 'xrd-app[poni]'`` to enable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _require_pyfai():
    try:
        import pyFAI  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "pyFAI is required for .poni → tth conversion. Install it with "
            "`pip install 'xrd-app[poni]'` or `pip install pyFAI`."
        ) from e
    return pyFAI


def poni_to_tth(poni_path, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Return a 2θ-per-pixel map (degrees) from a pyFAI ``.poni`` file.

    ``shape`` is ``(rows, cols)``; if omitted, pyFAI uses the detector shape
    recorded in the ``.poni``. Raises ``ImportError`` if pyFAI is missing and
    ``FileNotFoundError`` if the poni file does not exist.
    """
    pyFAI = _require_pyfai()
    poni_path = Path(poni_path)
    if not poni_path.exists():
        raise FileNotFoundError(f".poni not found: {poni_path}")

    ai = pyFAI.load(str(poni_path))
    # Prefer the modern, more precise API (returns 2θ in degrees directly);
    # fall back to twoThetaArray (radians) on older pyFAI.
    try:
        arr = ai.center_array(shape, unit="2th_deg")
        return np.asarray(arr, dtype=np.float64)
    except (AttributeError, TypeError):
        tth_rad = ai.twoThetaArray(shape) if shape is not None else ai.twoThetaArray()
        return np.degrees(tth_rad).astype(np.float64)


def save_tth_tiff(tth: np.ndarray, output) -> Path:
    """Write a 2θ map to a TIFF (float32) and return the path."""
    import tifffile
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output), tth.astype(np.float32))
    return output


def convert_poni_file(poni_path, output, shape: Optional[Tuple[int, int]] = None) -> Path:
    """Convert a ``.poni`` to a ``tth.tiff`` on disk. Returns the output path."""
    return save_tth_tiff(poni_to_tth(poni_path, shape), output)
