"""Territorial shape finder — Phase-2 linking over a *physical* neighbor graph.

Companion to ``gaussian.py`` for the territorial / cell-model reference binning
(see ``core/territory.py``). The only difference from the baseline is **how
candidate detections are paired for linking**: instead of the N×N grid's fixed
8-neighborhood, it links peaks across the territories that are physically
adjacent in true (X, Y) — the ``neighbors`` lists carried in the territorial
grid mapping. Detection, the gaussian-profile filter, and characterization are
reused verbatim from ``gaussian.py`` (a territory is just an irregular bin).

``core.processing.run_shapes`` calls :func:`link_peaks` with the neighbor map
when the grid mapping carries a ``territories`` block; otherwise the baseline
gaussian linker runs. Registered in ``catalog.json`` as ``territory`` — select
with ``xrd-app shapes --algorithm territory --variant territory``.
"""

from collections import defaultdict

import numpy as np

# Reuse the baseline characterization wholesale — it reads only member tuples /
# intensity_profile and makes no assumption about bin geometry. Dual import:
# `gaussian` works under core.io.load_module (the sibling dir is on sys.path);
# `.gaussian` works when imported as a package module (py_compile / tests).
try:  # noqa: F401  (names re-exported for run_shapes)
    from gaussian import (
        DEFAULT_LINK_TOLERANCE, characterize_features, check_gaussian_profile,
        estimate_beam_center, _best_per_bin, _coerce,
    )
except ImportError:  # pragma: no cover
    from .gaussian import (
        DEFAULT_LINK_TOLERANCE, characterize_features, check_gaussian_profile,
        estimate_beam_center, _best_per_bin, _coerce,
    )


# ── Phase 2: spatial linking over the territory neighbor graph ──────
def link_peaks(all_detections, neighbors, link_tolerance=DEFAULT_LINK_TOLERANCE):
    """Link same-peak detections across physically-adjacent territories.

    ``all_detections`` maps ``territory_key -> [peak dicts]``; ``neighbors`` maps
    ``territory_key -> [adjacent territory_key, ...]`` (from the territorial grid
    mapping). Two detections in adjacent territories are merged when they fall
    within ``link_tolerance`` pixels at the same detector position — identical
    Union-Find to ``gaussian.link_peaks``, only the adjacency source differs.

    Returns a list of features, each a list of member nodes
    ``(territory_key, peak_index, row, col, x, y, peak_dict)``. ``row``/``col``
    are the territory centroid in grid-like units (``centroid_rc``) when present,
    so the downstream gaussian distance check behaves as on the N×N grid; they
    fall back to the ``"<tid>_0"`` key when no centroid is supplied.
    """
    # Per-territory centroid (row, col) for the gaussian distance check. The
    # neighbors map may be a bare {key: [keys]} or carry centroids; accept both.
    nodes = []
    node_idx_by_key = defaultdict(list)
    for bk, peaks in all_detections.items():
        r, c = _key_rc(bk)
        for pi, p in enumerate(peaks):
            node_idx_by_key[bk].append(len(nodes))
            nodes.append((bk, pi, r, c, p['x'], p['y'], p))

    if not nodes:
        return []

    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Link within each territory and across each adjacent territory. Iterating
    # neighbor *pairs* (not a 3×3 window) is what makes this work for irregular
    # cells; the link test itself is the same Euclidean detector-pixel check.
    for bk, idxs in node_idx_by_key.items():
        # within-territory
        for a in range(len(idxs)):
            ia = idxs[a]
            xa, ya = nodes[ia][4], nodes[ia][5]
            for b in range(a + 1, len(idxs)):
                ib = idxs[b]
                if (xa - nodes[ib][4]) ** 2 + (ya - nodes[ib][5]) ** 2 <= link_tolerance ** 2:
                    union(ia, ib)
        # across neighbors (each unordered pair handled once via bk < nb)
        for nb in neighbors.get(bk, []):
            if nb <= bk or nb not in node_idx_by_key:
                continue
            for ia in idxs:
                xa, ya = nodes[ia][4], nodes[ia][5]
                for ib in node_idx_by_key[nb]:
                    if (xa - nodes[ib][4]) ** 2 + (ya - nodes[ib][5]) ** 2 <= link_tolerance ** 2:
                        union(ia, ib)

    components = defaultdict(list)
    for idx in range(len(nodes)):
        components[find(idx)].append(idx)

    return [[nodes[i] for i in member_indices] for member_indices in components.values()]


# Module-level centroid table, populated by run_shapes before linking so that
# node (row, col) reflects each territory's physical centroid (grid units).
_CENTROID_RC: dict = {}


def set_centroids(centroid_rc: dict):
    """Register ``{territory_key: [row, col]}`` centroids for the next link run.

    ``core.processing.run_shapes`` calls this with the ``centroid_rc`` values
    from the territorial grid mapping so the gaussian profile check measures
    distances in true sample space rather than the synthetic key index.
    """
    global _CENTROID_RC
    _CENTROID_RC = centroid_rc or {}


def _key_rc(bk):
    """(row, col) for a territory: its physical centroid if known, else the key."""
    rc = _CENTROID_RC.get(bk)
    if rc is not None:
        return float(rc[0]), float(rc[1])
    a, _, b = bk.partition("_")
    try:
        return int(a), int(b or 0)
    except ValueError:
        return 0, 0
