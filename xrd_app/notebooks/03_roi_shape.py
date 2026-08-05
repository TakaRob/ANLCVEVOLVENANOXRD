"""Reproduce ROI > Shape with explicit detector rectangles and matplotlib."""

# %% [markdown]
# # ROI > Shape
# Enter detector rectangles as `(x0, y0, x1, y1)`, where upper bounds are
# exclusive. The notebook overlays them on the scan detector sum, reduces each
# rectangle exactly across all spatial bins, displays its map, and optionally
# merges it into the dedicated manual ROI catalog.

# %% Configuration
from pathlib import Path

PROJECT_ROOT = Path("/home/takaji/rocking_203_214")
SCAN = "Scan_0203"
BIN_SIZE = 3
ROIS = [
    (450, 450, 475, 475),
]
ROI_LABELS = ["manual ROI 1"]
CATALOG_NAME = "notebook_roi"
METRIC = "integrated"  # "integrated", "intensity", or "mean"
NORMALIZE_FRAMES = False

# Detector summation can be heavy if no cached reflection_sum.npz exists.
COMPUTE_DETECTOR_SUM = False
OVERWRITE_DETECTOR_SUM = False
RUN_EXACT_ROI_MAP = False
SAVE_CATALOG = False

# Optional existing ROI catalog used when RUN_EXACT_ROI_MAP is False.
ROI_CATALOG_PATH = None

# %% Imports and paths
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from xrd_app.config import DataManager
from xrd_app.core import io, processing, reflection_sum, roi_catalog, roi_map

# Resolve the selected scan and ROI catalog path through DataManager.
dm = DataManager(PROJECT_ROOT, scan=SCAN)
if not dm.config.exists():
    raise FileNotFoundError(f"Not an xrd-app project: {PROJECT_ROOT}")
catalog_path = Path(ROI_CATALOG_PATH) if ROI_CATALOG_PATH else dm.roi_map_path(
    CATALOG_NAME, BIN_SIZE
)
grid_path = dm.grid_mapping(bin_size=BIN_SIZE)

# Normalize and validate each detector rectangle before reading data.
detector_rois = [roi_map.normalize_roi(roi) for roi in ROIS]
if len(ROI_LABELS) != len(detector_rois):
    raise ValueError("ROI_LABELS must contain one label for every ROI.")
print(f"Project/scan: {dm.root} / {dm.scan_name}")
print(f"ROIs: {detector_rois}")

# %% Load or compute the detector grand sum
sum_path = reflection_sum.sum_path(dm, dm.scan_name)
if COMPUTE_DETECTOR_SUM:
    # Sum every detector image once and cache the result in scan metadata.
    status = reflection_sum.compute_and_save(
        dm,
        scan=dm.scan_name,
        overwrite=OVERWRITE_DETECTOR_SUM,
    )
    print(status)
if not sum_path.exists():
    raise FileNotFoundError(
        f"Detector sum not found: {sum_path}. Set COMPUTE_DETECTOR_SUM=True."
    )

# Load only the cached two-dimensional sum image from the compressed archive.
with np.load(sum_path) as saved:
    detector_image = saved["image"].astype(np.float32)

# %% Preview detector rectangles
positive = detector_image[np.isfinite(detector_image) & (detector_image > 0)]
vmin = float(np.percentile(positive, 5)) if positive.size else 0.0
vmax = float(np.percentile(positive, 99.8)) if positive.size else 1.0

fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
# Display weak and strong detector features together on a logarithmic scale.
image = axis.imshow(
    np.log1p(np.clip(detector_image, 0, None)),
    origin="upper",
    cmap="inferno",
    vmin=np.log1p(vmin),
    vmax=np.log1p(vmax),
)
for index, ((x0, y0, x1, y1), label) in enumerate(
    zip(detector_rois, ROI_LABELS), 1
):
    # Overlay the exact half-open detector rectangle used by ROI sampling.
    axis.add_patch(
        Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="cyan", linewidth=1.5
        )
    )
    axis.text(x0, y0, f" {index}: {label}", color="cyan", va="bottom", fontsize=8)
axis.set_title(f"{dm.scan_name}: detector sum with ROI > Shape rectangles")
axis.set_xlabel("Detector x / column (pixels)")
axis.set_ylabel("Detector y / row (pixels)")
# Add a scale for the logarithmically displayed summed detector counts.
fig.colorbar(image, ax=axis, label="log(1 + summed detector counts)")

preview_path = dm.figures_dir / dm.scan_name / f"{CATALOG_NAME}_detector_rois.png"
# Create the per-scan figure directory before saving.
preview_path.parent.mkdir(parents=True, exist_ok=True)
# Save the detector preview while retaining its inline display.
fig.savefig(preview_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved detector preview: {preview_path}")

# %% Build exact ROI spatial maps or load a saved catalog
if RUN_EXACT_ROI_MAP:
    # Load and validate the grid used to interpret each spatial-bin key.
    grid = io.validate_grid_mapping_bin_size(grid_path, BIN_SIZE)
    # Open the most efficient available HDF5/archive-backed bin source.
    source = io.open_bin_source(
        dm, BIN_SIZE, scan=dm.scan_name, grid_mapping=grid_path
    )
    try:
        # Reduce all detector rectangles in one exact pass over spatial bins.
        sampled = roi_map.sample_rois(
            source,
            detector_rois,
            grid_mapping=grid,
            metric=METRIC,
            normalize_frames=NORMALIZE_FRAMES,
            fast=False,
            log=print,
        )
    finally:
        # Release the HDF5 source even if ROI sampling raises an error.
        source.close()

    # Load calibration once to annotate every ROI feature consistently.
    tth_map = io.load_tth_map(dm.tth_map())
    # Estimate the beam center used to convert detector coordinates to chi.
    beam_center = processing.estimate_beam_center(tth_map)
    features = []
    for index, (result, label) in enumerate(zip(sampled, ROI_LABELS), 1):
        # Convert the exact map to the dedicated manual-ROI feature schema.
        feature = roi_map.to_shape_feature(
            result,
            reflection=label,
            feature_id=index,
            tth_map=tth_map,
            beam_center=beam_center,
        )
        features.append(feature)

    if SAVE_CATALOG:
        if NORMALIZE_FRAMES:
            raise ValueError(
                "Dedicated ROI catalogs describe total counts; set NORMALIZE_FRAMES=False to save."
            )
        # Merge exact features by detector rectangle into the dedicated catalog.
        catalog = roi_catalog.save_previews(
            catalog_path,
            features,
            scan=dm.scan_name,
            bin_size=BIN_SIZE,
            name=CATALOG_NAME,
        )
        print(f"Saved {catalog['n_features']} ROI feature(s): {catalog_path}")
else:
    # Load previously saved exact ROI features without rereading detector data.
    catalog = roi_catalog.load(catalog_path)
    if not catalog:
        raise FileNotFoundError(
            f"ROI catalog not found: {catalog_path}. Set RUN_EXACT_ROI_MAP=True."
        )
    if catalog.get("scan") != dm.scan_name or int(catalog.get("bin_size", -1)) != BIN_SIZE:
        raise ValueError(f"ROI catalog does not match {dm.scan_name} at {BIN_SIZE}x{BIN_SIZE}.")
    features = catalog.get("features", [])
    sampled = [
        {
            "profile": feature.get("intensity_profile") or {},
            "n_bin_rows": feature.get("n_bin_rows", 0),
            "n_bin_cols": feature.get("n_bin_cols", 0),
            "metric": METRIC,
        }
        for feature in features
    ]
print(f"ROI spatial maps ready: {len(sampled)}")

# %% Plot and save ROI spatial maps
if not sampled:
    raise ValueError("No ROI spatial maps are available to plot.")
fig, axes = plt.subplots(
    1, len(sampled), figsize=(5.4 * len(sampled), 4.8),
    squeeze=False, constrained_layout=True
)
for index, (axis, result) in enumerate(zip(axes.flat, sampled), 1):
    # Convert sparse spatial-bin reductions to a NaN-masked regular grid.
    values = roi_map.grid_array(result, metric=METRIC)
    # Display the selected ROI reduction over the sample-space grid.
    map_image = axis.imshow(
        values, origin="upper", cmap="magma", interpolation="nearest", aspect="equal"
    )
    label = features[index - 1].get("reflection", f"ROI {index}")
    center = features[index - 1].get("center_bin")
    if center:
        row, col = (int(value) for value in center.split("_", 1))
        axis.plot(col, row, marker="+", color="cyan", markersize=10, markeredgewidth=1.5)
    axis.set_title(f"{index}: {label}")
    axis.set_xlabel("Spatial bin column")
    axis.set_ylabel("Spatial bin row")
    # Add a separate intensity scale because each ROI can differ substantially.
    fig.colorbar(map_image, ax=axis, label=f"ROI {METRIC} counts")

fig.suptitle(f"{dm.scan_name}: ROI > Shape spatial maps, {BIN_SIZE}x{BIN_SIZE}")
maps_path = dm.figures_dir / dm.scan_name / f"{CATALOG_NAME}_spatial_maps.png"
# Save exact/reloaded ROI maps alongside the detector preview.
fig.savefig(maps_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved ROI spatial maps: {maps_path}")
