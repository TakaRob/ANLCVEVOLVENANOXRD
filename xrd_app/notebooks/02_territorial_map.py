"""Build and plot the true-position territorial reference without Qt."""

# %% [markdown]
# # Territorial map
# This workflow groups measured stage positions into irregular territories,
# builds one detector image per territory, detects peaks, links shapes through
# physical neighbors, and colors the true `(X, Y)` territory polygons.

# %% Configuration
from pathlib import Path

PROJECT_ROOT = Path("/home/takaji/rocking_203_214")
SCAN = "Scan_0203"
TARGET_SIZE = 1
VARIANT = "territory"
PEAK_ALGORITHM = None
SNR_THRESHOLD = 4.0
LINK_TOLERANCE = 5
N_WORKERS = 1
COMPRESSION = "zstd"
REFLECTION = None  # Example: "(001)"; None includes all reflections.
MAP_METRIC = "intensity"  # "intensity", "chi", or "shape_count"

RUN_TERRITORY_GRID = False
RUN_BINNING = False
RUN_PEAKS = False
RUN_SHAPES = False
OVERWRITE = False

PEAKS_JSON = None
SHAPES_JSON = None

# %% Imports and paths
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from xrd_app.config import DataManager
from xrd_app.core import catalogs, io, lineage, processing, territory

# Resolve all data sources and result paths through DataManager.
dm = DataManager(PROJECT_ROOT, scan=SCAN)
if not dm.config.exists():
    raise FileNotFoundError(f"Not an xrd-app project: {PROJECT_ROOT}")
if dm.scan_number() is None:
    raise ValueError(f"Could not resolve scan number from {SCAN!r}")

grid_path = dm.grid_mapping(bin_size=1, variant=VARIANT)
bins_path = dm.binned_h5(1, variant=VARIANT)
archive_path = dm.unbinned_archive_h5()
detector_path = dm.detector_script(PEAK_ALGORITHM, bin_size=1)
shape_algorithm_path = dm.shape_script("territory")
peak_name = detector_path.stem
shape_name = shape_algorithm_path.stem
peak_path = Path(PEAKS_JSON) if PEAKS_JSON else dm.peaks_json(
    peak_name, 1, variant=VARIANT
)
shape_path = Path(SHAPES_JSON) if SHAPES_JSON else dm.shapes_json(
    shape_name, 1, variant=VARIANT
)

# %% Build or load the territorial grid
if RUN_TERRITORY_GRID:
    if grid_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Grid exists; set OVERWRITE=True to replace: {grid_path}")
    # Grow disjoint cells over measured X/Y positions and save their polygons.
    grid = territory.build_territory_mapping(
        xrd_dir=dm.xrd_frames_dir(),
        pos_csv=dm.position_csv(),
        target_size=TARGET_SIZE,
        scan_number=dm.scan_number(),
        output=grid_path,
        log=print,
        archive=archive_path if archive_path.exists() else None,
    )
else:
    # Load and validate the nominal 1x1 territorial mapping.
    grid = io.validate_grid_mapping_bin_size(grid_path, 1)

if not grid.get("positions_real", False) or not grid.get("territories"):
    raise ValueError("Selected mapping is not a true-position territorial grid.")
print(f"Territories: {len(grid['territories'])} -> {grid_path}")

# %% Build detector images for each territory
if RUN_BINNING:
    if bins_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Bins exist; set OVERWRITE=True to replace: {bins_path}")
    # Sum each territory's assigned frames into a detector image.
    io.build_bins(
        grid,
        bins_path,
        bin_size=1,
        compression=COMPRESSION,
        log=print,
        archive=archive_path if archive_path.exists() else None,
    )
if not bins_path.exists():
    raise FileNotFoundError(f"Territorial bins not found: {bins_path}")

# %% Run or load territorial peak detection
if RUN_PEAKS:
    if peak_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Peaks exist; set OVERWRITE=True to replace: {peak_path}")
    # Detect Bragg peaks independently in each irregular cell image.
    peaks = processing.run_peaks(
        bins_h5=bins_path,
        tth_path=dm.tth_map(),
        detector_path=detector_path,
        reflections_path=dm.reflections(),
        bin_size=1,
        snr_threshold=SNR_THRESHOLD,
        n_workers=N_WORKERS,
        log=print,
    )
    peaks["scan"] = dm.scan_name
    peaks["algorithm"] = peak_name
    # Record the territorial source variant in peak provenance.
    peaks["lineage"] = lineage.peak_lineage(
        scan=dm.scan_name,
        bin_size=1,
        algorithm=peak_name,
        detector_file=detector_path,
        snr=SNR_THRESHOLD,
        variant=VARIANT,
    )
    # Save and register the peak catalog atomically.
    io.atomic_write_json(peak_path, peaks)
    catalogs.record_catalog(dm.labels_dir(), peak_path.name, peaks["lineage"])
else:
    # Reject a peak catalog from another scan, bin size, or source variant.
    catalogs.validate_result_identity(
        peak_path,
        expected_scan=dm.scan_name,
        expected_bin_size=1,
        expected_variant=VARIANT,
    )
    with open(peak_path) as handle:
        peaks = json.load(handle)
print(f"Peaks: {peaks.get('n_peaks', 0)} -> {peak_path}")

# %% Run or load physical-neighbor shape linking
if RUN_SHAPES:
    if shape_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Shapes exist; set OVERWRITE=True to replace: {shape_path}")
    # Link detections only through the territorial physical-neighbor graph.
    shapes = processing.run_shapes(
        peaks=peaks,
        tth_path=dm.tth_map(),
        grid_mapping=grid,
        reflections_path=dm.reflections(),
        bin_size=1,
        link_tolerance=LINK_TOLERANCE,
        shape_path=shape_algorithm_path,
        log=print,
    )
    shapes["scan"] = dm.scan_name
    shapes["shape_algo"] = shape_name
    shapes["peak_source"] = peaks.get("algorithm", peak_path.name)
    # Chain the shape result to the exact territorial peak catalog.
    shapes["lineage"] = lineage.shape_lineage(
        scan=dm.scan_name,
        bin_size=1,
        shape_algorithm=shape_name,
        link_tolerance=LINK_TOLERANCE,
        peak_source=lineage.from_peaks_data(peaks, fallback_file=peak_path.name),
        peak_source_file=peak_path.name,
    )
    # Save and register the territorial shape catalog atomically.
    io.atomic_write_json(shape_path, shapes)
    catalogs.record_catalog(dm.labels_dir(), shape_path.name, shapes["lineage"])
else:
    # Reject a shape catalog that does not match this territorial source.
    catalogs.validate_result_identity(
        shape_path,
        expected_scan=dm.scan_name,
        expected_bin_size=1,
        expected_variant=VARIANT,
    )
    with open(shape_path) as handle:
        shapes = json.load(handle)
print(f"Shapes: {shapes.get('n_kept', len(shapes.get('kept', [])))} -> {shape_path}")

# %% Reduce shape features to one value per territory
values_by_key = {}
counts_by_key = {}
strength_by_key = {}
for feature in shapes.get("kept", []):
    if REFLECTION and feature.get("reflection") != REFLECTION:
        continue
    for key, profile in (feature.get("intensity_profile") or {}).items():
        counts_by_key[key] = counts_by_key.get(key, 0) + 1
        intensity = float(profile.get("integrated", profile.get("intensity", 0.0)))
        if MAP_METRIC == "shape_count":
            values_by_key[key] = float(counts_by_key[key])
        elif MAP_METRIC == "chi":
            chi = profile.get("chi", feature.get("chi_deg"))
            if chi is not None and intensity >= strength_by_key.get(key, -np.inf):
                values_by_key[key] = float(chi)
                strength_by_key[key] = intensity
        elif MAP_METRIC == "intensity":
            values_by_key[key] = max(values_by_key.get(key, -np.inf), intensity)
        else:
            raise ValueError("MAP_METRIC must be 'intensity', 'chi', or 'shape_count'.")

# %% Plot true-position polygons and save the map
keys = []
polygons = []
for key, info in grid["territories"].items():
    polygon = info.get("polygon") or []
    if len(polygon) >= 3:
        keys.append(key)
        polygons.append(polygon)

plot_values = np.array([values_by_key.get(key, np.nan) for key in keys], dtype=float)
cmap = "twilight" if MAP_METRIC == "chi" else "viridis"
vmin, vmax = (-180, 180) if MAP_METRIC == "chi" else (None, None)

fig, axis = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
# Add all physical-space territory polygons as one efficient collection.
collection = PolyCollection(
    polygons, array=plot_values, cmap=cmap, edgecolors="0.25", linewidths=0.15
)
collection.set_clim(vmin=vmin, vmax=vmax)
axis.add_collection(collection)
axis.autoscale_view()
axis.set_aspect("equal", adjustable="box")
axis.set_xlabel("Stage X position")
axis.set_ylabel("Stage Y position")
reflection_label = REFLECTION or "all reflections"
axis.set_title(f"{dm.scan_name}: territorial {MAP_METRIC} ({reflection_label})")
# Add a metric-labeled scale to the territorial map.
fig.colorbar(collection, ax=axis, label=MAP_METRIC.replace("_", " "))

figure_path = (
    dm.figures_dir / dm.scan_name /
    f"territorial_{MAP_METRIC}_{REFLECTION or 'all'}.png"
)
# Create the per-scan figure directory before saving.
figure_path.parent.mkdir(parents=True, exist_ok=True)
# Save a publication-ready copy and retain the inline VS Code display.
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Finite mapped territories: {np.isfinite(plot_values).sum()} / {len(plot_values)}")
print(f"Saved territorial map: {figure_path}")
