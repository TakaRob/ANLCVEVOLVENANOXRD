# %% [markdown]
# # Fixed-ROI thresholding versus xrd-app shape detection
#
# This notebook-style script compares two ways of outlining the same Bragg feature:
#
# 1. **Conventional fixed ROI:** keep the detector aperture fixed at the feature's
#    brightest detector position, calculate an aperture SNR in every spatial bin,
#    impose one user-selected SNR cutoff, and retain the connected component that
#    contains the feature center.
# 2. **xrd-app:** detect each peak after radial-background subtraction and white
#    top-hat filtering, allow the detector position to move locally, link detections
#    across neighboring bins, and retain a shape with a Gaussian-like spatial
#    intensity profile.
#
# The territorial catalog is the true-position reference. Shapes from the 3x3 and
# territorial catalogs are matched by reflection, detector location, and shared raw
# frames. The displayed ideal, median, and poor cases are selected objectively from
# eligible matches by the 90th, 50th, and 10th percentiles of elliptical-Gaussian
# fit R2. The fitted Gaussian curves are visualization/quality measures; xrd-app's
# production Gaussian-like check is a monotonic profile test, not this fit.

# %%
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from scipy.optimize import least_squares


# %% [markdown]
# ## Configuration
#
# `TakaTest/TakaProject` is the fast local fallback. Set `PROJECT` to the
# consolidated SixScan project when that share is available and preferred.

# %%
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT = REPO_ROOT / "TakaTest" / "TakaProject"
SCAN = "Scan_0203"
BIN_SIZE = 3
PEAK_ALGORITHM = "5x5_tophat_band_adaptive_snr"
TERRITORY_SHAPES = "territory_shapes_1x1_territory.json"

DETECTOR_MATCH_PX = 8.0
MIN_BINS = 12
MAX_BINS = 250
MIN_MATCH_IOU = 0.30
MAX_FIT_ASPECT = 6.0
CASE_QUANTILES = {"Ideal": 0.90, "Median": 0.50, "Poor": 0.10}
DISPLAY_THRESHOLD = 4.0
THRESHOLDS = np.arange(2.0, 8.1, 0.5)
ROI_HALF_WIDTH = 3       # 7 x 7 signal aperture
BACKGROUND_HALF_WIDTH = 8
MAP_PADDING = 3

OUTPUT_DIR = REPO_ROOT / "FinalProject" / "Figures" / "roi_shape_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = PROJECT / "Labels" / SCAN
METADATA = PROJECT / "Metadata" / SCAN
BINNED = PROJECT / "Binned" / SCAN
PATHS = {
    "shapes3": LABELS / f"gaussian_shapes_{BIN_SIZE}x{BIN_SIZE}.json",
    "peaks3": LABELS / f"{PEAK_ALGORITHM}_peaks_{BIN_SIZE}x{BIN_SIZE}.json",
    "grid3": METADATA / f"grid_mapping_{BIN_SIZE}x{BIN_SIZE}.json",
    "shapes_t": LABELS / TERRITORY_SHAPES,
    "grid_t": METADATA / "grid_mapping_1x1_territory.json",
    "h5": BINNED / f"xrd_{BIN_SIZE}x{BIN_SIZE}_bins.h5",
}
for name, path in PATHS.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def load_json(path):
    with path.open() as stream:
        return json.load(stream)


shapes3 = load_json(PATHS["shapes3"])
peaks3 = load_json(PATHS["peaks3"])
grid3 = load_json(PATHS["grid3"])
shapes_t = load_json(PATHS["shapes_t"])
grid_t = load_json(PATHS["grid_t"])
print(f"3x3 shapes: {shapes3['n_kept']}; territorial shapes: {shapes_t['n_kept']}")
print(f"Detector: {peaks3['detector']}; detection SNR: {peaks3['snr']}")


# %% [markdown]
# ## Match identical physical shapes
#
# Feature IDs cannot be compared across catalogs. Each footprint is expanded to
# its underlying raw-frame IDs, then candidate pairs must share a reflection,
# overlap in raw frames, and lie within eight detector pixels. Greedy assignment
# prioritizes frame IoU and then detector distance.

# %%
def shape_frames(feature, grid):
    return {
        int(frame)
        for key in feature["spatial_extent"]
        for frame in grid["bins"].get(key, [])
    }


def match_shapes():
    candidates = []
    territory_frames = {
        feature["feature_id"]: shape_frames(feature, grid_t)
        for feature in shapes_t["kept"]
    }
    for feature3 in shapes3["kept"]:
        frames3 = shape_frames(feature3, grid3)
        for feature_t in shapes_t["kept"]:
            if feature3["reflection"] != feature_t["reflection"]:
                continue
            distance = float(np.hypot(
                feature3["detector_x"] - feature_t["detector_x"],
                feature3["detector_y"] - feature_t["detector_y"],
            ))
            if distance > DETECTOR_MATCH_PX:
                continue
            frames_t = territory_frames[feature_t["feature_id"]]
            overlap = len(frames3 & frames_t)
            if not overlap:
                continue
            union = len(frames3 | frames_t)
            candidates.append({
                "feature3": feature3,
                "feature_t": feature_t,
                "frames3": frames3,
                "frames_t": frames_t,
                "overlap": overlap,
                "iou": overlap / union,
                "coverage3": overlap / len(frames3),
                "coverage_t": overlap / len(frames_t),
                "detector_distance": distance,
            })

    candidates.sort(key=lambda item: (-item["iou"], item["detector_distance"]))
    used3, used_t, matches = set(), set(), []
    for candidate in candidates:
        id3 = candidate["feature3"]["feature_id"]
        id_t = candidate["feature_t"]["feature_id"]
        if id3 in used3 or id_t in used_t:
            continue
        used3.add(id3)
        used_t.add(id_t)
        matches.append(candidate)
    return matches


matches = match_shapes()
print(f"One-to-one matched shapes: {len(matches)}")


# %% [markdown]
# ## Rank shape quality without hand-picking examples
#
# An elliptical Gaussian is fit to each 3x3 shape's cleaned peak-intensity map.
# R2 is used only to choose representative examples and describe shape quality.
# Eligibility excludes tiny shapes and very large scan-spanning components so an
# edge and a meaningful spatial profile are visible in the paper figure.

# %%
def profile_arrays(feature):
    keys = list(feature.get("intensity_profile", {}))
    xy = np.asarray(
        [[int(key.split("_")[1]), int(key.split("_")[0])] for key in keys],
        dtype=float,
    )
    intensity = np.asarray(
        [max(float(feature["intensity_profile"][key]["intensity"]), 0.0) for key in keys]
    )
    return keys, xy, intensity


def fit_elliptical_gaussian(xy, values):
    background = max(float(np.percentile(values, 5)), 0.0)
    weights = np.clip(values - background, 0, None)
    if weights.sum() <= 0 or len(values) < 6:
        return None
    center = np.average(xy, axis=0, weights=weights)
    covariance = np.cov(xy.T, aweights=weights) + np.eye(2) * 1e-3
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    sigma = np.sqrt(np.maximum(eigenvalues, 0.25))
    theta = float(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    initial = np.array([
        background, max(values.max() - background, 1.0), center[0], center[1],
        np.log(sigma[0]), np.log(sigma[1]), theta,
    ])
    span = np.maximum(np.ptp(xy, axis=0), 1.0)
    lower = [0.0, 0.0, xy[:, 0].min() - 1, xy[:, 1].min() - 1,
             np.log(0.25), np.log(0.25), -np.pi]
    upper = [values.max(), values.max() * 4, xy[:, 0].max() + 1,
             xy[:, 1].max() + 1, np.log(span.max() * 3),
             np.log(span.max() * 3), np.pi]
    scale = max(float(np.percentile(values, 90)), 1.0)

    def predict(parameters, points=xy):
        bg, amplitude, cx, cy = parameters[:4]
        sx, sy, angle = np.exp(parameters[4]), np.exp(parameters[5]), parameters[6]
        delta = points - [cx, cy]
        cosine, sine = np.cos(angle), np.sin(angle)
        major = cosine * delta[:, 0] + sine * delta[:, 1]
        minor = -sine * delta[:, 0] + cosine * delta[:, 1]
        return bg + amplitude * np.exp(-0.5 * ((major / sx) ** 2 + (minor / sy) ** 2))

    result = least_squares(
        lambda parameters: (predict(parameters) - values) / scale,
        initial, bounds=(lower, upper), max_nfev=2500,
    )
    predicted = predict(result.x)
    residual = float(np.sum((values - predicted) ** 2))
    total = float(np.sum((values - values.mean()) ** 2))
    sx, sy = np.exp(result.x[4]), np.exp(result.x[5])
    angle = result.x[6]
    if sy > sx:
        sx, sy = sy, sx
        angle += np.pi / 2
    return {
        "parameters": result.x,
        "predict": predict,
        "predicted": predicted,
        "r2": 1.0 - residual / total if total else 0.0,
        "center": np.asarray(result.x[2:4]),
        "sigma_major": float(sx),
        "sigma_minor": float(sy),
        "theta": float(angle),
        "aspect": float(sx / sy),
    }


eligible = []
for match in matches:
    feature = match["feature3"]
    if match["iou"] < MIN_MATCH_IOU:
        continue
    if not MIN_BINS <= feature["n_bins"] <= MAX_BINS:
        continue
    keys, xy, intensity = profile_arrays(feature)
    if len(keys) < MIN_BINS or np.ptp(xy[:, 0]) < 2 or np.ptp(xy[:, 1]) < 2:
        continue
    fit = fit_elliptical_gaussian(xy, intensity)
    if (
        fit is None
        or not np.isfinite(fit["r2"])
        or fit["aspect"] > MAX_FIT_ASPECT
    ):
        continue
    match.update({"keys": keys, "xy": xy, "intensity": intensity, "fit": fit})
    eligible.append(match)

eligible.sort(key=lambda item: item["fit"]["r2"])
if len(eligible) < len(CASE_QUANTILES):
    raise RuntimeError(f"Only {len(eligible)} eligible matched shapes")

cases = []
used = set()
for label_name, quantile in CASE_QUANTILES.items():
    target = float(np.quantile([item["fit"]["r2"] for item in eligible], quantile))
    options = [item for item in eligible if item["feature3"]["feature_id"] not in used]
    selected = min(options, key=lambda item: abs(item["fit"]["r2"] - target))
    selected["case"] = label_name
    used.add(selected["feature3"]["feature_id"])
    cases.append(selected)

for item in cases:
    f3, ft = item["feature3"], item["feature_t"]
    print(
        f"{item['case']:>6}: 3x3 feature {f3['feature_id']}, territory feature "
        f"{ft['feature_id']}, {f3['reflection']}, bins={f3['n_bins']}, "
        f"Gaussian R2={item['fit']['r2']:.3f}, catalog IoU={item['iou']:.3f}"
    )


# %% [markdown]
# ## Calculate conventional fixed-ROI SNR
#
# For each local 3x3 spatial bin, the conventional signal is the sum in a fixed
# 7x7 detector aperture after subtracting the median of a surrounding square
# annulus. Its uncertainty is the annulus MAD noise times the square root of the
# aperture pixel count. This is standard aperture photometry, but unlike xrd-app
# it does not recenter as the Bragg peak moves on the detector.

# %%
def local_keys(feature, padding=MAP_PADDING):
    rows = [int(key.split("_")[0]) for key in feature["spatial_extent"]]
    cols = [int(key.split("_")[1]) for key in feature["spatial_extent"]]
    r0, r1 = max(min(rows) - padding, 0), min(max(rows) + padding, grid3["n_bin_rows"] - 1)
    c0, c1 = max(min(cols) - padding, 0), min(max(cols) + padding, grid3["n_bin_cols"] - 1)
    return [
        f"{row}_{col}"
        for row in range(r0, r1 + 1)
        for col in range(c0, c1 + 1)
        if f"{row}_{col}" in grid3["bins"]
    ]


def fixed_roi_snr(image, detector_x, detector_y):
    height, width = image.shape
    x0, y0 = int(round(detector_x)), int(round(detector_y))
    outer = BACKGROUND_HALF_WIDTH
    if x0 - outer < 0 or y0 - outer < 0 or x0 + outer >= width or y0 + outer >= height:
        return np.nan
    patch = np.asarray(
        image[y0 - outer:y0 + outer + 1, x0 - outer:x0 + outer + 1],
        dtype=np.float32,
    )
    yy, xx = np.indices(patch.shape)
    center = outer
    inner_mask = (
        (np.abs(xx - center) <= ROI_HALF_WIDTH)
        & (np.abs(yy - center) <= ROI_HALF_WIDTH)
    )
    annulus_mask = (
        (np.maximum(np.abs(xx - center), np.abs(yy - center)) >= ROI_HALF_WIDTH + 2)
        & (np.maximum(np.abs(xx - center), np.abs(yy - center)) <= outer)
    )
    aperture = patch[inner_mask]
    annulus = patch[annulus_mask]
    background = float(np.median(annulus))
    sigma = 1.4826 * float(np.median(np.abs(annulus - background)))
    if sigma <= 0:
        sigma = float(np.std(annulus))
    if sigma <= 0:
        return np.nan
    signal = float(np.sum(aperture - background))
    return signal / (sigma * np.sqrt(aperture.size))


def xrd_peak_snr(feature, key):
    profile = feature["intensity_profile"].get(key)
    if not profile:
        return np.nan
    candidates = [
        peak for peak in peaks3["peaks_by_bin"].get(key, [])
        if peak.get("label") == feature["reflection"]
    ]
    if not candidates:
        return np.nan
    peak = min(candidates, key=lambda candidate: (
        candidate["x"] - profile["det_x"]
    ) ** 2 + (candidate["y"] - profile["det_y"]) ** 2)
    distance = np.hypot(peak["x"] - profile["det_x"], peak["y"] - profile["det_y"])
    return float(peak["snr"]) if distance <= DETECTOR_MATCH_PX else np.nan


def component_at_threshold(snr_by_key, center_key, threshold):
    parsed = {
        tuple(int(part) for part in bin_key.split("_")): bin_key
        for bin_key in snr_by_key
    }
    rows = [position[0] for position in parsed]
    cols = [position[1] for position in parsed]
    r0, c0 = min(rows), min(cols)
    mask = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for (row, col), key in parsed.items():
        value = snr_by_key[key]
        mask[row - r0, col - c0] = np.isfinite(value) and value >= threshold
    labels, _ = label(mask, structure=np.ones((3, 3), dtype=int))
    center = tuple(int(value) for value in center_key.split("_"))
    center_label = labels[center[0] - r0, center[1] - c0]
    if center_label == 0:
        return set()
    return {
        key for (row, col), key in parsed.items()
        if labels[row - r0, col - c0] == center_label
    }


def frame_iou(keys, reference_frames):
    frames = {
        int(frame) for key in keys for frame in grid3["bins"].get(key, [])
    }
    return len(frames & reference_frames) / len(frames | reference_frames) if frames else 0.0


with h5py.File(PATHS["h5"], "r") as h5:
    for item in cases:
        feature = item["feature3"]
        keys = local_keys(feature)
        item["local_keys"] = keys
        item["roi_snr"] = {
            key: fixed_roi_snr(h5[key], feature["detector_x"], feature["detector_y"])
            for key in keys
        }
        item["xrd_snr"] = {key: xrd_peak_snr(feature, key) for key in keys}
        item["roi_components"] = {
            float(threshold): component_at_threshold(
                item["roi_snr"], feature["center_bin"], threshold
            )
            for threshold in THRESHOLDS
        }
        item["roi_iou"] = {
            float(threshold): frame_iou(keys_at_threshold, item["frames_t"])
            for threshold, keys_at_threshold in item["roi_components"].items()
        }
        item["xrd_iou"] = frame_iou(set(feature["spatial_extent"]), item["frames_t"])


# %% [markdown]
# ## Main paper figure: maps and edge dropoff
#
# Each row is one objectively selected shape. The first two panels show the same
# feature in territorial and regular 3x3 coordinates. The third panel shows the
# conventional fixed-ROI SNR and its connected SNR=4 outline. The final panel is a
# one-dimensional cut along the fitted major axis: adaptive xrd-app peak SNR,
# conventional fixed-ROI SNR, a visualization-only Gaussian fit, and the cutoff.

# %%
def map_arrays(values_by_key):
    keys = list(values_by_key)
    xy = np.asarray([[int(key.split("_")[1]), int(key.split("_")[0])] for key in keys])
    values = np.asarray([values_by_key[key] for key in keys], dtype=float)
    return xy, values


def add_component_outline(ax, keys, color="#0072B2", linewidth=2.0):
    if not keys:
        return
    xy = np.asarray([[int(key.split("_")[1]), int(key.split("_")[0])] for key in keys])
    c0, r0 = xy.min(axis=0)
    grid = np.zeros((np.ptp(xy[:, 1]) + 3, np.ptp(xy[:, 0]) + 3), dtype=float)
    grid[xy[:, 1] - r0 + 1, xy[:, 0] - c0 + 1] = 1.0
    ax.contour(
        np.arange(c0 - 1, c0 + grid.shape[1] - 1),
        np.arange(r0 - 1, r0 + grid.shape[0] - 1),
        grid, levels=[0.5], colors=[color], linewidths=linewidth,
    )


def major_coordinate(xy, fit):
    direction = np.asarray([np.cos(fit["theta"]), np.sin(fit["theta"])])
    perpendicular = np.asarray([-direction[1], direction[0]])
    delta = xy - fit["center"]
    return delta @ direction, delta @ perpendicular


def plot_dropoff(ax, item):
    feature, fit = item["feature3"], item["fit"]
    all_xy, _ = map_arrays({key: 0 for key in item["local_keys"]})
    distance, transverse = major_coordinate(all_xy, fit)
    band = np.abs(transverse) <= max(1.5, fit["sigma_minor"])
    distance = distance[band]
    selected_keys = np.asarray(item["local_keys"], dtype=object)[band]
    order = np.argsort(distance)
    distance, selected_keys = distance[order], selected_keys[order]
    xrd = np.asarray([item["xrd_snr"][key] for key in selected_keys], dtype=float)
    roi = np.asarray([item["roi_snr"][key] for key in selected_keys], dtype=float)

    finite_xrd = np.isfinite(xrd)
    finite_roi = np.isfinite(roi)
    ax.scatter(distance[finite_xrd], xrd[finite_xrd], color="#D55E00", s=24,
               zorder=4, label="xrd-app adaptive peak SNR")
    ax.scatter(distance[finite_roi], roi[finite_roi], facecolors="none",
               edgecolors="#0072B2", s=24, zorder=3, label="fixed-ROI SNR")

    if finite_xrd.sum() >= 4:
        x_line = np.linspace(distance.min(), distance.max(), 300)
        x_obs = distance[finite_xrd]
        y_obs = xrd[finite_xrd]
        baseline = max(min(float(np.percentile(y_obs, 10)), DISPLAY_THRESHOLD), 0.0)
        amplitude = max(float(y_obs.max() - baseline), 1.0)
        initial = [baseline, amplitude, float(x_obs[np.argmax(y_obs)]),
                   max(float(np.std(x_obs)), 1.0)]

        def model(parameters, x):
            bg, amp, center, sigma = parameters
            return bg + amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)

        result = least_squares(
            lambda parameters: model(parameters, x_obs) - y_obs,
            initial, bounds=([0, 0, x_obs.min() - 2, 0.25],
                             [y_obs.max(), y_obs.max() * 4,
                              x_obs.max() + 2, max(np.ptp(x_obs) * 2, 1)]),
        )
        fitted = model(result.x, x_line)
        ax.plot(x_line, fitted, color="#D55E00", linewidth=2,
                label="Gaussian fit to xrd-app SNR")
        fwhm = 2.3548 * result.x[3]
        above = x_line[fitted >= DISPLAY_THRESHOLD]
        threshold_breadth = float(np.ptp(above)) if len(above) > 1 else 0.0
        ax.text(
            0.03, 0.04,
            f"fit FWHM = {fwhm:.1f} bins\nSNR=4 breadth = {threshold_breadth:.1f} bins",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    if finite_roi.sum() >= 3:
        smooth = gaussian_filter1d(np.nan_to_num(roi, nan=0.0), sigma=0.8)
        ax.plot(distance, smooth, color="#0072B2", linewidth=1.5, alpha=0.8,
                label="smoothed fixed-ROI SNR")
    ax.axhline(DISPLAY_THRESHOLD, color="black", linestyle="--", linewidth=1.2,
               label=f"SNR cutoff = {DISPLAY_THRESHOLD:g}")
    ax.set_yscale("symlog", linthresh=2)
    ax.set_xlabel("distance along feature major axis (3x3 bins)")
    ax.set_ylabel("signal-to-noise ratio")
    ax.grid(alpha=0.18)


fig, axes = plt.subplots(3, 4, figsize=(16.5, 11.5), constrained_layout=True)
for row, item in enumerate(cases):
    feature3, feature_t = item["feature3"], item["feature_t"]

    ax = axes[row, 0]
    territory_xy = np.asarray([
        grid_t["territories"][key]["centroid_rc"][::-1]
        for key in feature_t["spatial_extent"]
    ])
    territory_intensity = np.asarray([
        max(float(feature_t["intensity_profile"][key]["intensity"]), 0.1)
        for key in feature_t["spatial_extent"]
    ])
    points = ax.scatter(
        territory_xy[:, 0], territory_xy[:, 1], c=territory_intensity,
        cmap="magma", norm=LogNorm(vmin=max(np.percentile(territory_intensity, 5), 0.1),
                                    vmax=territory_intensity.max()),
        marker="s", s=25, linewidths=0,
    )
    ax.set_title(f"{item['case']}: territorial reference\nfeature {feature_t['feature_id']}")
    ax.set_xlabel("true-position column coordinate")
    ax.set_ylabel("true-position row coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.colorbar(points, ax=ax, label="cleaned peak intensity", shrink=0.82)

    ax = axes[row, 1]
    profile = feature3["intensity_profile"]
    xy3, values3 = map_arrays({key: profile[key]["intensity"] for key in profile})
    points = ax.scatter(
        xy3[:, 0], xy3[:, 1], c=np.maximum(values3, 0.1), cmap="magma",
        norm=LogNorm(vmin=max(np.percentile(np.maximum(values3, 0.1), 5), 0.1),
                     vmax=max(values3.max(), 0.1)), marker="s", s=34, linewidths=0,
    )
    add_component_outline(ax, set(feature3["spatial_extent"]))
    ax.set_title(
        f"xrd-app 3x3 linked shape\nfeature {feature3['feature_id']}; "
        f"Gaussian R2={item['fit']['r2']:.2f}"
    )
    ax.set_xlabel("3x3 scan column")
    ax.set_ylabel("3x3 scan row")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.colorbar(points, ax=ax, label="cleaned peak intensity", shrink=0.82)

    ax = axes[row, 2]
    roi_xy, roi_values = map_arrays(item["roi_snr"])
    finite = np.isfinite(roi_values)
    limit = max(float(np.nanpercentile(np.abs(roi_values), 95)), DISPLAY_THRESHOLD)
    points = ax.scatter(
        roi_xy[finite, 0], roi_xy[finite, 1], c=roi_values[finite],
        cmap="coolwarm", vmin=-limit, vmax=limit, marker="s", s=34, linewidths=0,
    )
    conventional = item["roi_components"][DISPLAY_THRESHOLD]
    add_component_outline(ax, conventional)
    ax.scatter(feature3["center_col"], feature3["center_row"], marker="*", s=90,
               color="#F0E442", edgecolor="black", zorder=5)
    ax.set_title(
        f"fixed detector ROI, SNR >= {DISPLAY_THRESHOLD:g}\n"
        f"{len(conventional)} bins; territory IoU={item['roi_iou'][DISPLAY_THRESHOLD]:.2f}"
    )
    ax.set_xlabel("3x3 scan column")
    ax.set_ylabel("3x3 scan row")
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    fig.colorbar(points, ax=ax, label="fixed-aperture SNR", shrink=0.82)

    plot_dropoff(axes[row, 3], item)
    axes[row, 3].set_title(
        f"edge dropoff: {feature3['reflection']}\n"
        f"xrd-app/territory IoU={item['xrd_iou']:.2f}"
    )

handles = [
    Line2D([], [], marker="o", linestyle="none", color="#D55E00",
           label="xrd-app adaptive peak SNR"),
    Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
           markeredgecolor="#0072B2", label="fixed-ROI SNR"),
    Line2D([], [], color="#D55E00", linewidth=2, label="Gaussian fit (visualization)"),
    Line2D([], [], color="black", linestyle="--", label="SNR cutoff"),
]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
           ncol=4, frameon=False)
fig.suptitle(
    "Fixed detector ROI versus adaptive peak tracking and linked shape outlines",
    fontsize=15, weight="bold",
)
for suffix in ("png", "pdf"):
    path = OUTPUT_DIR / f"roi_vs_xrdapp_examples.{suffix}"
    fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
    print(f"Saved {path}")
plt.close(fig)


# %% [markdown]
# ## Threshold sensitivity
#
# A conventional outline changes whenever its manually selected SNR limit changes.
# The curves below compare each conventional footprint with the independently
# binned territorial footprint using shared-frame IoU. The horizontal line is the
# saved xrd-app 3x3 shape's agreement with that same territorial reference.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True, constrained_layout=True)
for ax, item in zip(axes, cases):
    threshold_values = np.asarray(sorted(item["roi_iou"]))
    iou_values = np.asarray([item["roi_iou"][value] for value in threshold_values])
    areas = np.asarray([len(item["roi_components"][value]) for value in threshold_values])
    ax.plot(threshold_values, iou_values, marker="o", color="#0072B2",
            label="fixed-ROI footprint")
    ax.axhline(item["xrd_iou"], color="#D55E00", linewidth=2,
               label="xrd-app saved shape")
    best = int(np.argmax(iou_values))
    ax.scatter(threshold_values[best], iou_values[best], marker="*", s=130,
               color="#F0E442", edgecolor="black", zorder=4)
    ax.set_title(
        f"{item['case']} case: feature {item['feature3']['feature_id']}\n"
        f"ROI area {areas.min()}-{areas.max()} bins"
    )
    ax.set_xlabel("chosen fixed-ROI SNR cutoff")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
axes[0].set_ylabel("frame-footprint IoU with territorial reference")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Conventional outlines depend on a feature-specific SNR cutoff", weight="bold")
for suffix in ("png", "pdf"):
    path = OUTPUT_DIR / f"roi_threshold_sensitivity.{suffix}"
    fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
    print(f"Saved {path}")
plt.close(fig)


# %% [markdown]
# ## Save reproducible metrics
#
# The JSON records selection quantiles, catalog lineage, fit quality, matched-shape
# overlap, and every threshold result. It can be used to write the figure caption
# without reading values from the plot.

# %%
metrics = {
    "project": str(PROJECT),
    "scan": SCAN,
    "bin_size": BIN_SIZE,
    "peak_algorithm": PEAK_ALGORITHM,
    "peak_detection_snr": peaks3["snr"],
    "conventional_roi": {
        "signal_aperture": f"{2 * ROI_HALF_WIDTH + 1}x{2 * ROI_HALF_WIDTH + 1}",
        "background_window": f"{2 * BACKGROUND_HALF_WIDTH + 1}x{2 * BACKGROUND_HALF_WIDTH + 1}",
        "display_threshold": DISPLAY_THRESHOLD,
        "threshold_sweep": THRESHOLDS.tolist(),
    },
    "selection": {
        "eligible_matches": len(eligible),
        "quantiles": CASE_QUANTILES,
        "min_bins": MIN_BINS,
        "max_bins": MAX_BINS,
        "min_match_iou": MIN_MATCH_IOU,
        "max_fit_aspect": MAX_FIT_ASPECT,
        "detector_match_px": DETECTOR_MATCH_PX,
    },
    "cases": [],
}
for item in cases:
    feature3, feature_t = item["feature3"], item["feature_t"]
    metrics["cases"].append({
        "case": item["case"],
        "feature_3x3": feature3["feature_id"],
        "feature_territory": feature_t["feature_id"],
        "reflection": feature3["reflection"],
        "n_bins_3x3": feature3["n_bins"],
        "n_cells_territory": feature_t["n_bins"],
        "gaussian_fit_r2": item["fit"]["r2"],
        "gaussian_aspect": item["fit"]["aspect"],
        "catalog_match_iou": item["iou"],
        "catalog_match_coverage_3x3": item["coverage3"],
        "catalog_match_coverage_territory": item["coverage_t"],
        "detector_distance_px": item["detector_distance"],
        "xrdapp_territory_iou": item["xrd_iou"],
        "fixed_roi_iou_by_snr": {
            str(threshold): value for threshold, value in item["roi_iou"].items()
        },
        "fixed_roi_bins_by_snr": {
            str(threshold): len(keys)
            for threshold, keys in item["roi_components"].items()
        },
        "shape_reason": feature3["reason"],
        "mean_peak_snr": feature3["mean_snr"],
        "chi_fwhm_deg": feature3.get("chi_fwhm"),
        "tth_fwhm_deg": feature3.get("tth_fwhm"),
    })

metrics_path = OUTPUT_DIR / "roi_shape_comparison_metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(f"Saved {metrics_path}")
