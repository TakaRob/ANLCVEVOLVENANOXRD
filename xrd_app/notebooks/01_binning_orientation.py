"""Build regular bins, run peak/shape algorithms, and plot shape orientation."""

# %% [markdown]
# # Binning, algorithms, and shape orientation
# Heavy stages are opt-in. Leave their switches off to load existing artifacts.
# The output is a shape-level chi map: each linked shape paints its spatial
# extent with its detector azimuth. Chi is cyclic, so the color map wraps at
# -180/180 degrees.

# %% Configuration
from pathlib import Path

PROJECT_ROOT = Path("/home/takaji/rocking_203_214")
SCAN = "Scan_0203"
BIN_SIZE = 3
VARIANT = None
DESKEW_METHOD = "auto"
PEAK_ALGORITHM = None  # None selects the best compatible detector.
SHAPE_ALGORITHM = "gaussian"
SNR_THRESHOLD = 4.0
LINK_TOLERANCE = 5
N_WORKERS = 1
COMPRESSION = "zstd"

RUN_GRID = False
RUN_BINNING = False
RUN_PEAKS = False
RUN_SHAPES = False
OVERWRITE = False

# Optional existing artifact overrides when the RUN switch is False.
PEAKS_PATH = None
SHAPES_PATH = None

# %% Imports and paths

import matplotlib.pyplot as plt
import numpy as np

from xrd_app.config import DataManager
from xrd_app.core import catalogs, io, lineage, processing

# Resolve all scan-specific inputs and outputs through DataManager.
dm = DataManager(PROJECT_ROOT, scan=SCAN)
if not dm.config.exists():
    raise FileNotFoundError(f"Not an xrd-app project: {PROJECT_ROOT}")
if dm.scan_number() is None:
    raise ValueError(f"Could not resolve scan number from {SCAN!r}")

grid_path = dm.grid_mapping(bin_size=BIN_SIZE, variant=VARIANT)
bins_path = dm.binned_h5(BIN_SIZE, variant=VARIANT)
archive_path = dm.unbinned_archive_h5()
tth_path = dm.tth_map()
reflections_path = dm.reflections()

# Resolve the selected peak and shape algorithm implementations.
detector_path = dm.detector_script(PEAK_ALGORITHM, bin_size=BIN_SIZE)
shape_algorithm_path = dm.shape_script(SHAPE_ALGORITHM)
peak_name = detector_path.stem
shape_name = shape_algorithm_path.stem
peak_path = Path(PEAKS_PATH) if PEAKS_PATH else dm.peaks_path(
    peak_name, BIN_SIZE, variant=VARIANT
)
shape_path = Path(SHAPES_PATH) if SHAPES_PATH else dm.shapes_path(
    shape_name, BIN_SIZE, variant=VARIANT
)

print(f"Project/scan: {dm.root} / {dm.scan_name}")
print(f"Detector: {detector_path.name}")
print(f"Shape algorithm: {shape_algorithm_path.name}")

# %% Build or load the true-position grid
if RUN_GRID:
    if grid_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Grid exists; set OVERWRITE=True to replace: {grid_path}")
    # Generate the regular spatial grid from measured X/Y positions.
    grid = io.generate_grid_mapping(
        xrd_dir=dm.xrd_frames_dir(),
        pos_csv=dm.position_csv(),
        bin_size=BIN_SIZE,
        scan_number=dm.scan_number(),
        output=grid_path,
        deskew=True,
        deskew_method=DESKEW_METHOD,
        log=print,
        archive=archive_path if archive_path.exists() else None,
    )
else:
    # Load and validate the existing grid's bin size.
    grid = io.validate_grid_mapping_bin_size(grid_path, BIN_SIZE)

if not grid.get("positions_real", False):
    raise ValueError("Refusing to bin: grid was not built from measured X/Y positions.")
print(f"Grid: {grid['n_bin_rows']} x {grid['n_bin_cols']} bins -> {grid_path}")

# %% Build detector-image bins
if RUN_BINNING:
    if bins_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Bins exist; set OVERWRITE=True to replace: {bins_path}")
    # Sum raw/archive frames into one detector image per spatial bin.
    io.build_bins(
        grid,
        bins_path,
        bin_size=BIN_SIZE,
        compression=COMPRESSION,
        log=print,
        archive=archive_path if archive_path.exists() else None,
    )
if not bins_path.exists():
    raise FileNotFoundError(f"Binned detector images not found: {bins_path}")
print(f"Bins: {bins_path}")

# %% Run or load per-bin peak detection
if RUN_PEAKS:
    if peak_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Peaks exist; set OVERWRITE=True to replace: {peak_path}")
    # Detect Bragg peaks independently in every spatial-bin image.
    peaks = processing.run_peaks(
        bins_h5=bins_path,
        tth_path=tth_path,
        detector_path=detector_path,
        reflections_path=reflections_path,
        bin_size=BIN_SIZE,
        snr_threshold=SNR_THRESHOLD,
        n_workers=N_WORKERS,
        log=print,
    )
    peaks["scan"] = dm.scan_name
    peaks["algorithm"] = peak_name
    # Record reproducible peak-detection provenance.
    peaks["lineage"] = lineage.peak_lineage(
        scan=dm.scan_name,
        bin_size=BIN_SIZE,
        algorithm=peak_name,
        detector_file=detector_path,
        snr=SNR_THRESHOLD,
        variant=VARIANT,
    )
    # Save the typed result before registering its catalog lineage.
    catalogs.save_result(peak_path, peaks)
    catalogs.record_catalog(dm.labels_dir(), peak_path.name, peaks["lineage"])
else:
    # Validate that the chosen artifact belongs to this scan/bin/variant.
    catalogs.validate_result_identity(
        peak_path,
        expected_scan=dm.scan_name,
        expected_bin_size=BIN_SIZE,
        expected_variant=VARIANT,
    )
    peaks = catalogs.load_result(peak_path)
print(f"Peaks: {peaks.get('n_peaks', 0)} -> {peak_path}")

# %% Run or load cross-bin shape linking
if RUN_SHAPES:
    if shape_path.exists() and not OVERWRITE:
        raise FileExistsError(f"Shapes exist; set OVERWRITE=True to replace: {shape_path}")
    # Link neighboring detections and characterize physically persistent shapes.
    shapes = processing.run_shapes(
        peaks=peaks,
        tth_path=tth_path,
        grid_mapping=grid,
        reflections_path=reflections_path,
        bin_size=BIN_SIZE,
        link_tolerance=LINK_TOLERANCE,
        shape_path=shape_algorithm_path,
        log=print,
    )
    shapes["scan"] = dm.scan_name
    shapes["shape_algo"] = shape_name
    shapes["peak_source"] = peaks.get("algorithm", peak_path.name)
    # Chain shape provenance to the exact upstream peak result.
    shapes["lineage"] = lineage.shape_lineage(
        scan=dm.scan_name,
        bin_size=BIN_SIZE,
        shape_algorithm=shape_name,
        link_tolerance=LINK_TOLERANCE,
        peak_source=lineage.from_peaks_data(peaks),
        peak_source_file=peak_path.name,
    )
    # Save and register the typed shape catalog atomically.
    catalogs.save_result(shape_path, shapes)
    catalogs.record_catalog(dm.labels_dir(), shape_path.name, shapes["lineage"])
else:
    # Validate the existing shape artifact before plotting it.
    catalogs.validate_result_identity(
        shape_path,
        expected_scan=dm.scan_name,
        expected_bin_size=BIN_SIZE,
        expected_variant=VARIANT,
    )
    shapes = catalogs.load_result(shape_path)
print(f"Shapes: {shapes.get('n_kept', len(shapes.get('kept', [])))} -> {shape_path}")

# %% Build shape-orientation grids by reflection
features = shapes.get("kept", [])
orientation_grids = {}
strength_grids = {}
for feature in features:
    reflection = feature.get("reflection", "unknown")
    chi = feature.get("chi_deg")
    if chi is None:
        continue
    orientation_grid = orientation_grids.setdefault(
        reflection,
        np.full((grid["n_bin_rows"], grid["n_bin_cols"]), np.nan),
    )
    strength_grid = strength_grids.setdefault(
        reflection,
        np.full((grid["n_bin_rows"], grid["n_bin_cols"]), -np.inf),
    )
    for key, profile in (feature.get("intensity_profile") or {}).items():
        row, col = (int(value) for value in key.split("_", 1))
        if 0 <= row < orientation_grid.shape[0] and 0 <= col < orientation_grid.shape[1]:
            strength = float(profile.get("integrated", profile.get("intensity", 0.0)))
            if strength >= strength_grid[row, col]:
                orientation_grid[row, col] = float(chi)
                strength_grid[row, col] = strength

if not orientation_grids:
    raise ValueError("No kept shapes with chi_deg were found in the selected catalog.")

# %% Plot and save the shape orientation image
n_panels = len(orientation_grids)
fig, axes = plt.subplots(
    1, n_panels, figsize=(5.2 * n_panels, 4.8), squeeze=False, constrained_layout=True
)
image = None
for axis, (reflection, values) in zip(axes.flat, sorted(orientation_grids.items())):
    # Render orientation with a cyclic color scale.
    image = axis.imshow(
        values, origin="upper", cmap="twilight", vmin=-180, vmax=180,
        interpolation="nearest", aspect="equal"
    )
    axis.set_title(f"{reflection} ({np.isfinite(values).sum()} bins)")
    axis.set_xlabel("Spatial bin column")
    axis.set_ylabel("Spatial bin row")

# Add one shared physical-orientation color bar.
fig.colorbar(image, ax=axes.ravel().tolist(), label="Shape orientation chi (degrees)")
fig.suptitle(f"{dm.scan_name}: linked-shape orientation, {BIN_SIZE}x{BIN_SIZE}")
figure_path = dm.figures_dir / dm.scan_name / f"shape_orientation_{BIN_SIZE}x{BIN_SIZE}.png"
# Create the scan figure directory before saving.
figure_path.parent.mkdir(parents=True, exist_ok=True)
# Save a publication-ready copy while retaining the inline display.
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved orientation image: {figure_path}")
