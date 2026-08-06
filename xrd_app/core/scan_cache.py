"""Process-local cache for immutable scan metadata used by multiple GUI tabs."""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np


_LOCK = threading.RLock()
_TTH_CACHE = OrderedDict()
_GRID_CACHE = OrderedDict()
_CATALOG_CACHE = OrderedDict()
_MAX_TTH = 2
_MAX_METADATA = 4


def _signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _put(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def clear():
    """Release all shared scan data, normally after changing project or scan."""
    with _LOCK:
        _TTH_CACHE.clear()
        _GRID_CACHE.clear()
        _CATALOG_CACHE.clear()


def load_tth_data(path):
    """Load a 2-theta map and its reusable radial-bin lookup arrays once."""
    key = _signature(path)
    with _LOCK:
        cached = _TTH_CACHE.get(key)
        if cached is not None:
            _TTH_CACHE.move_to_end(key)
            return cached

    import tifffile

    from .algorithms import compute_tth_binning

    tth = tifffile.imread(key[0]).astype(np.float64)
    edges, centers, n_bins, indices, counts = compute_tth_binning(tth)
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order]
    boundaries = np.searchsorted(sorted_indices, np.arange(n_bins + 1))
    value = {
        "map": tth,
        "edges": edges,
        "centers": centers,
        "n_bins": n_bins,
        "indices": indices,
        "counts": counts,
        "valid_mask": counts > 50,
        "order": order,
        "boundaries": boundaries,
    }
    with _LOCK:
        return _put(_TTH_CACHE, key, value, _MAX_TTH)


def load_grid_mapping(path):
    """Load and cache a grid mapping, returning a private mutable copy."""
    from . import io

    key = _signature(path)
    with _LOCK:
        cached = _GRID_CACHE.get(key)
        if cached is None:
            cached = _put(_GRID_CACHE, key, io.load_grid_mapping(path), _MAX_METADATA)
        else:
            _GRID_CACHE.move_to_end(key)
        return copy.deepcopy(cached)


def load_features_any(path):
    """Load and cache a feature catalog, returning private mutable feature data."""
    from . import catalogs

    key = _signature(path)
    with _LOCK:
        cached = _CATALOG_CACHE.get(key)
        if cached is None:
            cached = _put(_CATALOG_CACHE, key, catalogs.load_features_any(path),
                          _MAX_METADATA)
        else:
            _CATALOG_CACHE.move_to_end(key)
        return copy.deepcopy(cached)
