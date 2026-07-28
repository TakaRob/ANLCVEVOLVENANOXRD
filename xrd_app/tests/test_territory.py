"""Tests for physical-neighbor construction used by territory linking."""

import numpy as np

from xrd_app.core import territory


def test_delaunay_adjacency_does_not_bridge_concave_gap():
    # A U-shaped boundary makes unconstrained Delaunay join the two top tips
    # across empty space. Local scan neighbors are one unit apart.
    points = np.array([
        [0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0],
        [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
        [4.0, 1.0], [4.0, 2.0], [4.0, 3.0], [4.0, 4.0],
    ])

    adjacency = territory._delaunay_adjacency(points)

    assert 12 not in adjacency[4]
    assert 4 not in adjacency[12]
    assert all(adjacency)
    assert 1 in adjacency[0]


def _drifting_line(n=8):
    cells = [str(i) for i in range(n)]
    neighbors = {
        cell: [other for other in cells if abs(int(other) - i) == 1]
        for i, cell in enumerate(cells)
    }
    peaks = {
        cell: [{"x": 100 + 2 * i, "y": 200, "label": "(012)", "snr": 10}]
        for i, cell in enumerate(cells)
    }
    return cells, neighbors, peaks


def test_frontier_growth_follows_smooth_detector_drift():
    cells, neighbors, peaks = _drifting_line()

    grown = territory.grow_peak_feature(
        peaks, neighbors, cells[0], peaks[cells[0]][0],
        link_tolerance=3, anchor="frontier")

    assert set(grown) == set(cells)


def test_seed_growth_truncates_same_detector_drift():
    cells, neighbors, peaks = _drifting_line()

    grown = territory.grow_peak_feature(
        peaks, neighbors, cells[0], peaks[cells[0]][0],
        link_tolerance=3, anchor="seed")

    assert set(grown) == {"0", "1"}


def test_growth_does_not_cross_reflection_or_detector_jump():
    cells, neighbors, peaks = _drifting_line(5)
    peaks["2"][0]["label"] = "(001)"
    peaks["2"].append({"x": 130, "y": 200, "label": "(012)", "snr": 20})

    grown = territory.grow_peak_feature(
        peaks, neighbors, cells[0], peaks[cells[0]][0],
        link_tolerance=3, anchor="frontier")

    assert set(grown) == {"0", "1"}
