#!/usr/bin/env python3
"""Compare fixed-seed and drift-following feature growth on saved 1x1 peaks.

Produces a 2 x 2 figure showing old/new segmentation on the regular grid and on
true-coordinate territorial cells. No raw detector frames are read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from xrd_app.core import io, result_store
from xrd_app.core.territory import grow_peak_feature


DEFAULT_PROJECT = Path("/home/takaji/rocking_203_214")
DEFAULT_PEAK_ALGO = "5x5_tophat_band_adaptive_snr"


def _load(path: Path):
    return io.load_grid_mapping(path) if "grid_mapping" in path.name else result_store.load(path)


def _regular_neighbors(keys):
    keys = set(keys)
    out = {}
    for key in keys:
        row, col = (int(v) for v in key.split("_"))
        out[key] = [
            f"{row + dr}_{col + dc}"
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and f"{row + dr}_{col + dc}" in keys
        ]
    return out


def _closest_peak(peaks, reflection, x, y):
    candidates = [p for p in peaks if p.get("label") == reflection]
    if not candidates:
        raise ValueError(f"No {reflection} peak found in the seed cell")
    return min(candidates, key=lambda p: (p["x"] - x) ** 2 + (p["y"] - y) ** 2)


def _frames(cells, bins):
    return {int(frame) for cell in cells for frame in bins.get(cell, [])}


def _metrics(cells, peaks, bins, reference_frames):
    frames = _frames(cells, bins)
    overlap = len(frames & reference_frames)
    precision = overlap / len(frames) if frames else 0.0
    recall = overlap / len(reference_frames) if reference_frames else 0.0
    points = np.array([[p["x"], p["y"]] for p in peaks.values()], dtype=float)
    drift = np.ptp(points, axis=0) if len(points) else np.zeros(2)
    return {
        "cells": len(cells),
        "frames": len(frames),
        "precision": precision,
        "recall": recall,
        "detector_dx": float(drift[0]),
        "detector_dy": float(drift[1]),
    }


def _plot(ax, result, coordinates, reference_cells, title, norm, limits):
    reference_xy = np.array(
        [coordinates[cell] for cell in reference_cells if cell in coordinates],
        dtype=float)
    if len(reference_xy):
        ax.scatter(reference_xy[:, 0], reference_xy[:, 1], c="#d5d5d5", s=12,
                   marker="s", linewidths=0, label="batch footprint")

    cells = list(result)
    xy = np.array([coordinates[cell] for cell in cells], dtype=float)
    intensity = np.array([
        max(float(result[cell].get("cleaned_intensity", 0.0)), 0.1)
        for cell in cells
    ])
    ax.scatter(xy[:, 0], xy[:, 1], c=intensity, s=10, marker="s",
               linewidths=0, cmap="magma", norm=norm, label="grown footprint")
    ax.set_title(f"{title}\n{len(cells):,} cells")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[3], limits[2])
    ax.set_xlabel("column" if "Regular" in title else "stage X")
    ax.set_ylabel("row" if "Regular" in title else "stage Y")


def _limits(coordinates, cells):
    xy = np.array([coordinates[cell] for cell in cells if cell in coordinates], dtype=float)
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    pad = np.maximum((hi - lo) * 0.04, 0.2)
    return lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]


def run(project: Path, scan: str, feature_id: int, output: Path,
        peak_algo: str = DEFAULT_PEAK_ALGO, tolerance: float = 5.0):
    labels = project / "Labels" / scan
    metadata = project / "Metadata" / scan

    shapes = _load(labels / "territory_shapes_1x1_territory_coord.h5")
    feature = next((f for f in shapes["kept"] if f.get("feature_id") == feature_id), None)
    if feature is None:
        raise ValueError(f"Feature {feature_id} is not in the territorial catalog")

    regular_gm = _load(metadata / "grid_mapping_1x1.h5")
    territory_gm = _load(metadata / "grid_mapping_1x1_territory.h5")
    regular_peaks = _load(labels / f"{peak_algo}_peaks_1x1.h5")["peaks_by_bin"]
    territory_peaks = _load(
        labels / f"{peak_algo}_peaks_1x1_territory.h5")["peaks_by_bin"]

    center_territory = feature["center_bin"]
    profile = feature["intensity_profile"][center_territory]
    reflection = feature["reflection"]
    territory_seed = _closest_peak(
        territory_peaks[center_territory], reflection,
        profile["det_x"], profile["det_y"])

    center_frame = int(territory_gm["bins"][center_territory][0])
    frame_to_regular = {
        int(frame): key
        for key, frames in regular_gm["bins"].items()
        for frame in frames
    }
    center_regular = frame_to_regular[center_frame]
    regular_seed = _closest_peak(
        regular_peaks[center_regular], reflection,
        territory_seed["x"], territory_seed["y"])

    regular_neighbors = _regular_neighbors(regular_gm["bins"])
    territory_neighbors = {
        key: info.get("neighbors", [])
        for key, info in territory_gm["territories"].items()
    }

    rr, rc = (int(v) for v in center_regular.split("_"))
    regular_old_limit = {
        key for key in regular_gm["bins"]
        if max(abs(int(key.split("_")[0]) - rr),
               abs(int(key.split("_")[1]) - rc)) <= 10
    }
    tr, tc = territory_gm["territories"][center_territory]["centroid_rc"]
    territory_old_limit = {
        key for key, info in territory_gm["territories"].items()
        if max(abs(info["centroid_rc"][0] - tr),
               abs(info["centroid_rc"][1] - tc)) <= 10
    }

    results = {
        "regular_old": grow_peak_feature(
            regular_peaks, regular_neighbors, center_regular, regular_seed,
            tolerance, anchor="seed", allowed_cells=regular_old_limit),
        "regular_new": grow_peak_feature(
            regular_peaks, regular_neighbors, center_regular, regular_seed,
            tolerance, anchor="frontier"),
        "territory_old": grow_peak_feature(
            territory_peaks, territory_neighbors, center_territory, territory_seed,
            tolerance, anchor="seed", allowed_cells=territory_old_limit),
        "territory_new": grow_peak_feature(
            territory_peaks, territory_neighbors, center_territory, territory_seed,
            tolerance, anchor="frontier"),
    }

    reference_frames = _frames(feature["spatial_extent"], territory_gm["bins"])
    metrics = {
        "scan": scan,
        "feature_id": feature_id,
        "reflection": reflection,
        "link_tolerance": tolerance,
        "reference_frames": len(reference_frames),
        "methods": {
            "regular_old": _metrics(results["regular_old"], results["regular_old"],
                                    regular_gm["bins"], reference_frames),
            "regular_new": _metrics(results["regular_new"], results["regular_new"],
                                    regular_gm["bins"], reference_frames),
            "territory_old": _metrics(results["territory_old"], results["territory_old"],
                                      territory_gm["bins"], reference_frames),
            "territory_new": _metrics(results["territory_new"], results["territory_new"],
                                      territory_gm["bins"], reference_frames),
        },
    }

    regular_xy = {
        key: (int(key.split("_")[1]), int(key.split("_")[0]))
        for key in regular_gm["bins"]
    }
    territory_xy = {
        key: tuple(info["centroid_xy"])
        for key, info in territory_gm["territories"].items()
    }
    reference_regular = {
        frame_to_regular[frame]
        for frame in reference_frames if frame in frame_to_regular
    }
    reference_territory = set(feature["spatial_extent"])
    regular_limits = _limits(regular_xy, reference_regular)
    territory_limits = _limits(territory_xy, reference_territory)
    all_intensity = [
        max(float(peak.get("cleaned_intensity", 0.0)), 0.1)
        for result in results.values() for peak in result.values()
    ]
    norm = LogNorm(vmin=max(np.percentile(all_intensity, 2), 0.1),
                   vmax=max(all_intensity))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    _plot(axes[0, 0], results["regular_old"], regular_xy, reference_regular,
          "Regular old: fixed seed + square cap", norm, regular_limits)
    _plot(axes[0, 1], results["regular_new"], regular_xy, reference_regular,
          "Regular new: local drift following", norm, regular_limits)
    _plot(axes[1, 0], results["territory_old"], territory_xy, reference_territory,
          "Territory old: fixed seed + square cap", norm, territory_limits)
    _plot(axes[1, 1], results["territory_new"], territory_xy, reference_territory,
          "Territory new: local drift following", norm, territory_limits)
    fig.suptitle(
        f"{scan} feature {feature_id} {reflection}: fixed detector center vs local drift",
        fontsize=15, weight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    fig.colorbar(sm, ax=axes, label="cleaned peak intensity")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, metrics_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--scan", default="Scan_0203")
    parser.add_argument("--feature-id", type=int, default=1297)
    parser.add_argument("--peak-algo", default=DEFAULT_PEAK_ALGO)
    parser.add_argument("--link-tolerance", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(
        f"territory_drift_{args.scan}_{args.feature_id}.png")
    metrics, metrics_path = run(
        args.project, args.scan, args.feature_id, output,
        args.peak_algo, args.link_tolerance)
    print(json.dumps(metrics, indent=2))
    print(f"Saved figure: {output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
