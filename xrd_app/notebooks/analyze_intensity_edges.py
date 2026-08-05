#!/usr/bin/env python3
"""Study intensity-defined edges and centers of one drift-linked 1x1 feature.

Shows regular and true-coordinate territorial maps, measured 10/25/50% intensity
edges, and circular versus rotated-elliptical Gaussian fits. Reads saved HDF5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from xrd_app.core.territory import grow_peak_feature
from xrd_app.notebooks.compare_territory_drift import (
    DEFAULT_PEAK_ALGO, DEFAULT_PROJECT, _closest_peak, _load, _regular_neighbors,
)


def _fit_model(xy, intensity, elliptical):
    background = float(np.percentile(intensity, 10))
    weights = np.clip(intensity - background, 0, None)
    center = np.average(xy, axis=0, weights=weights)
    covariance = np.cov(xy.T, aweights=weights) + np.eye(2) * 1e-6
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    sigmas = np.sqrt(np.maximum(values, 0.25))
    theta = float(np.arctan2(vectors[1, 0], vectors[0, 0]))
    amplitude = float(max(intensity.max() - background, 1.0))

    if elliptical:
        initial = [background, amplitude, center[0], center[1],
                   np.log(sigmas[0]), np.log(sigmas[1]), theta]
        lower = [0, 0, xy[:, 0].min(), xy[:, 1].min(),
                 np.log(0.25), np.log(0.25), -np.pi]
        upper = [intensity.max(), intensity.max() * 3, xy[:, 0].max(),
                 xy[:, 1].max(), np.log(np.ptp(xy[:, 0]) * 2 + 1),
                 np.log(np.ptp(xy[:, 1]) * 2 + 1), np.pi]
    else:
        sigma = float(np.sqrt(sigmas.prod()))
        initial = [background, amplitude, center[0], center[1], np.log(sigma)]
        lower = [0, 0, xy[:, 0].min(), xy[:, 1].min(), np.log(0.25)]
        upper = [intensity.max(), intensity.max() * 3, xy[:, 0].max(),
                 xy[:, 1].max(), np.log(max(np.ptp(xy, axis=0)) * 2 + 1)]

    scale = max(float(np.percentile(intensity, 90)), 1.0)

    def predict(params):
        bg, amp, cx, cy = params[:4]
        delta = xy - [cx, cy]
        if elliptical:
            sx, sy, angle = np.exp(params[4]), np.exp(params[5]), params[6]
            cosine, sine = np.cos(angle), np.sin(angle)
            major = cosine * delta[:, 0] + sine * delta[:, 1]
            minor = -sine * delta[:, 0] + cosine * delta[:, 1]
            radius2 = (major / sx) ** 2 + (minor / sy) ** 2
        else:
            sigma = np.exp(params[4])
            radius2 = np.sum(delta ** 2, axis=1) / sigma ** 2
        return bg + amp * np.exp(-0.5 * radius2)

    fit = least_squares(lambda p: (predict(p) - intensity) / scale,
                        initial, bounds=(lower, upper), max_nfev=3000)
    predicted = predict(fit.x)
    residual = float(np.sum((intensity - predicted) ** 2))
    total = float(np.sum((intensity - intensity.mean()) ** 2))
    params = fit.x
    if elliptical:
        sx, sy = np.exp(params[4]), np.exp(params[5])
        angle = params[6]
    else:
        sx = sy = np.exp(params[4])
        angle = 0.0
    return {
        "background": float(params[0]), "amplitude": float(params[1]),
        "center": [float(params[2]), float(params[3])],
        "sigma_major": float(max(sx, sy)), "sigma_minor": float(min(sx, sy)),
        "theta_deg": float(np.degrees(angle if sx >= sy else angle + np.pi / 2)),
        "aspect": float(max(sx, sy) / min(sx, sy)),
        "r2": 1.0 - residual / total if total else 0.0,
        "predicted": predicted,
    }


def _dataset(project, scan, feature_id, peak_algo, tolerance):
    labels, metadata = project / "Labels" / scan, project / "Metadata" / scan
    shapes = _load(labels / "territory_shapes_1x1_territory_coord.h5")
    feature = next(f for f in shapes["kept"] if f.get("feature_id") == feature_id)
    regular_gm = _load(metadata / "grid_mapping_1x1.h5")
    territory_gm = _load(metadata / "grid_mapping_1x1_territory.h5")
    regular_peaks = _load(labels / f"{peak_algo}_peaks_1x1.h5")["peaks_by_bin"]
    territory_peaks = _load(
        labels / f"{peak_algo}_peaks_1x1_territory.h5")["peaks_by_bin"]

    center_t = feature["center_bin"]
    profile = feature["intensity_profile"][center_t]
    reflection = feature["reflection"]
    seed_t = _closest_peak(territory_peaks[center_t], reflection,
                           profile["det_x"], profile["det_y"])
    frame = territory_gm["bins"][center_t][0]
    frame_to_regular = {
        int(value): key for key, values in regular_gm["bins"].items() for value in values
    }
    center_r = frame_to_regular[int(frame)]
    seed_r = _closest_peak(regular_peaks[center_r], reflection,
                           seed_t["x"], seed_t["y"])

    regular = grow_peak_feature(
        regular_peaks, _regular_neighbors(regular_gm["bins"]), center_r, seed_r,
        tolerance, anchor="frontier")
    territory_neighbors = {
        key: info.get("neighbors", [])
        for key, info in territory_gm["territories"].items()
    }
    territory = grow_peak_feature(
        territory_peaks, territory_neighbors, center_t, seed_t,
        tolerance, anchor="frontier")
    coordinates = {
        "regular": {
            key: (int(key.split("_")[1]), int(key.split("_")[0]))
            for key in regular_gm["bins"]
        },
        "territory": {
            key: tuple(info["centroid_rc"])[::-1]
            for key, info in territory_gm["territories"].items()
        },
    }
    return feature, {"regular": regular, "territory": territory}, coordinates


def _arrays(result, coordinates):
    keys = list(result)
    xy = np.asarray([coordinates[key] for key in keys], dtype=float)
    intensity = np.asarray([
        max(float(result[key].get("cleaned_intensity", 0)), 0.1) for key in keys
    ])
    detector = np.asarray([[result[key]["x"], result[key]["y"]] for key in keys])
    return xy, intensity, detector


def _intensity_edges(xy, intensity):
    background = float(np.percentile(intensity, 10))
    cleaned = np.clip(intensity - background, 0, None)
    nx = min(350, max(100, int(np.ptp(xy[:, 0]) * 4)))
    ny = min(350, max(100, int(np.ptp(xy[:, 1]) * 4)))
    gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), nx)
    gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), ny)
    xx, yy = np.meshgrid(gx, gy)
    zz = griddata(xy, cleaned, (xx, yy), method="linear", fill_value=0.0)
    zz = gaussian_filter(zz, sigma=1.5)
    levels = [0.1, 0.25, 0.5]
    return xx, yy, zz, [fraction * float(zz.max()) for fraction in levels]


def _add_edges(ax, edge_grid, fit):
    levels = [0.1, 0.25, 0.5]
    colors = ["#00e5ff", "#5cff72", "white"]
    xx, yy, zz, contour_levels = edge_grid
    ax.contour(xx, yy, zz, levels=contour_levels, colors=colors, linewidths=1.5)
    for fraction, color in zip(levels, colors):
        radius = np.sqrt(-2 * np.log(fraction))
        ellipse = Ellipse(
            fit["center"], 2 * radius * fit["sigma_major"],
            2 * radius * fit["sigma_minor"], angle=fit["theta_deg"],
            fill=False, edgecolor=color, linewidth=1.0, linestyle="--")
        ax.add_patch(ellipse)


def _plot_row(axes, name, result, coordinates):
    xy, intensity, detector = _arrays(result, coordinates)
    background = float(np.percentile(intensity, 10))
    weights = np.clip(intensity - background, 0, None)
    center_mass = np.average(xy, axis=0, weights=weights)
    peak_center = xy[np.argmax(intensity)]
    circular = _fit_model(xy, intensity, elliptical=False)
    elliptical = _fit_model(xy, intensity, elliptical=True)
    edge_grid = _intensity_edges(xy, intensity)
    norm = LogNorm(vmin=max(float(np.percentile(intensity, 2)), 0.1),
                   vmax=float(intensity.max()))

    ax = axes[0]
    points = ax.scatter(xy[:, 0], xy[:, 1], c=intensity, s=11, marker="s",
                        linewidths=0, cmap="magma", norm=norm)
    _add_edges(ax, edge_grid, elliptical)
    ax.scatter(*peak_center, marker="*", s=130, c="#00e5ff", edgecolor="black",
               label="brightest")
    ax.scatter(*center_mass, marker="+", s=130, c="#5cff72", linewidths=2.5,
               label="center of mass")
    ax.scatter(*elliptical["center"], marker="x", s=90, c="white", linewidths=2,
               label="ellipse center")
    ax.set_title(f"{name}: measured intensity edges")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("spatial column")
    ax.set_ylabel("spatial row")
    plt.colorbar(points, ax=ax, label="cleaned peak intensity")

    ax = axes[1]
    residual = intensity - elliptical["predicted"]
    limit = max(float(np.percentile(np.abs(residual), 98)), 1.0)
    points = ax.scatter(xy[:, 0], xy[:, 1], c=residual, s=11, marker="s",
                        linewidths=0, cmap="coolwarm", vmin=-limit, vmax=limit)
    _add_edges(ax, edge_grid, elliptical)
    ax.set_title(
        f"{name}: ellipse residual\n"
        f"circle R2={circular['r2']:.3f}; ellipse R2={elliptical['r2']:.3f}; "
        f"aspect={elliptical['aspect']:.1f}")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlabel("spatial column")
    ax.set_ylabel("spatial row")
    plt.colorbar(points, ax=ax, label="measured - fitted intensity")

    edge_metrics = {}
    xx, yy, zz, contour_levels = edge_grid
    for fraction, level in zip((0.1, 0.25, 0.5), contour_levels):
        mask = zz >= level
        edge_xy = np.column_stack([xx[mask], yy[mask]])
        edge_weight = zz[mask]
        center = np.average(edge_xy, axis=0, weights=edge_weight)
        covariance = np.cov(edge_xy.T, aweights=edge_weight)
        values = np.maximum(np.linalg.eigvalsh(covariance), 1e-12)
        edge_metrics[str(fraction)] = {
            "center": center.tolist(),
            "width": float(np.ptp(edge_xy[:, 0])),
            "height": float(np.ptp(edge_xy[:, 1])),
            "principal_aspect": float(np.sqrt(values[-1] / values[0])),
        }

    matrix = np.c_[xy, np.ones(len(xy))]
    coefficients = np.linalg.lstsq(matrix, detector, rcond=None)[0]
    predicted_detector = matrix @ coefficients
    drift_r2 = 1.0 - float(np.sum((detector - predicted_detector) ** 2)) / max(
        float(np.sum((detector - detector.mean(axis=0)) ** 2)), 1e-12)
    return {
        "cells": len(xy),
        "brightest_center": peak_center.tolist(),
        "center_of_mass": center_mass.tolist(),
        "circular_fit": {k: v for k, v in circular.items() if k != "predicted"},
        "elliptical_fit": {k: v for k, v in elliptical.items() if k != "predicted"},
        "intensity_edges": edge_metrics,
        "detector_drift_r2": drift_r2,
        "detector_gradient": coefficients[:2].tolist(),
    }


def run(project, scan, feature_id, output, peak_algo, tolerance):
    feature, results, coordinates = _dataset(
        project, scan, feature_id, peak_algo, tolerance)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    metrics = {
        "scan": scan, "feature_id": feature_id,
        "reflection": feature["reflection"], "link_tolerance": tolerance,
    }
    metrics["regular"] = _plot_row(
        axes[0], "Regular", results["regular"], coordinates["regular"])
    metrics["territory"] = _plot_row(
        axes[1], "Territory", results["territory"], coordinates["territory"])
    fig.suptitle(
        f"{scan} feature {feature_id} {feature['reflection']}: intensity edges and centers",
        fontsize=15, weight="bold")
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
    parser.add_argument("--feature-id", type=int, default=1209)
    parser.add_argument("--peak-algo", default=DEFAULT_PEAK_ALGO)
    parser.add_argument("--link-tolerance", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(
        f"intensity_edges_{args.scan}_{args.feature_id}.png")
    metrics, metrics_path = run(
        args.project, args.scan, args.feature_id, output,
        args.peak_algo, args.link_tolerance)
    print(json.dumps(metrics, indent=2))
    print(f"Saved figure: {output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
