"""Scan 24 first pass: filter raw XRD links using calibrated ME7 XRF ROIs."""

# %% [markdown]
# # Scan 24 XRF-guided XRD prefilter
#
# This workflow follows the conventions in the 2026-2 Luo `xrd_scripts`:
#
# - raw data: `Raw/Scan_0024/{ME7,XRD}/scan_0024_NNNNN.h5`
# - positions: `processed/SOCKETSERVER/Scan_0024_position.h5`
# - Y correction: interpolate `position_offset.json` at the recorded sample theta
# - one numbered ME7 file corresponds to the same-numbered XRD file
# - points within each file correspond by local frame index
#
# Only narrow XRF slices are read when building the table. XRD detector images
# are not read or copied; each selected point stores an XRD file/frame link.

# %% Configuration
from pathlib import Path

DATA_ROOT = Path("/mnt/z/isn/2026-2/2026-2-Luo/xrd_scripts")
SCAN = 24
POSITION_OFFSET_FILE = DATA_ROOT / "position_offset.json"
SCAN_MASTER_FILE = DATA_ROOT / f"Scan_{SCAN:04d}.h5"

# Match the existing mictools ROI `Roi(0, 6, ..., name=...)`: detector channels
# 0 through 5 are summed. Set True only if deadtime-corrected counts are desired;
# the existing processed Br ROI appears to use uncorrected ROI sums.
CHANNELS = list(range(6))
DEADTIME_CORRECTION = False

# Supplied CCD calibration, returning keV for an MCA pixel.
ENERGY_CALIBRATION = {
    "quadratic_kev": 5.263744e-7,
    "linear_kev": 8.41967e-3,
    "offset_kev": 1.136032,
}

# Use every scan line and every spatial point for the exact whole-scan spectrum.
SPECTRUM_FILE_STRIDE = 1
SPECTRUM_POINT_STRIDE = 1
OUTPUT_DIR = DATA_ROOT / "processed" / "xrf_prefilter"
SPECTRUM_CACHE = OUTPUT_DIR / f"Scan_{SCAN:04d}_me7_spectrum.npz"
INTENSITY_CACHE = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_roi_intensities.npz"
ROI_CONFIG_FILE = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_rois.json"
PEAK_TABLE_CSV = OUTPUT_DIR / f"Scan_{SCAN:04d}_me7_detected_peaks.csv"
CUT_SUMMARY_CSV = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_cut_summary.csv"
OUTPUT_CSV = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_xrd_links.csv"

# %% Imports
import json
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.widgets import SpanSelector
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from xrd_app.core.xrf import EMISSION_LINES

H5_DATASET = "entry/data/data"
DT_FACTOR = "entry/instrument/NDAttributes/CHAN{channel1}DTFactor"
FILE_PATTERN = re.compile(r"scan_(\d+)_(\d+)\.h5$")


def pixel_to_kev(pixel):
    """Apply the supplied quadratic CCD calibration."""
    pixel = np.asarray(pixel, dtype=float)
    cal = ENERGY_CALIBRATION
    return (
        cal["quadratic_kev"] * pixel**2
        + cal["linear_kev"] * pixel
        + cal["offset_kev"]
    )


def kev_to_pixel(energy_kev):
    """Invert the monotonic positive branch of the quadratic calibration."""
    cal = ENERGY_CALIBRATION
    a = cal["quadratic_kev"]
    b = cal["linear_kev"]
    c = cal["offset_kev"] - np.asarray(energy_kev, dtype=float)
    discriminant = b**2 - 4 * a * c
    if np.any(discriminant < 0):
        raise ValueError("Requested energy is outside the calibration domain")
    return (-b + np.sqrt(discriminant)) / (2 * a)


def energy_range_to_pixels(energy_range_kev):
    """Convert a calibrated energy interval to an inclusive-exclusive slice."""
    lo_kev, hi_kev = sorted(map(float, energy_range_kev))
    lo = max(0, int(np.floor(kev_to_pixel(lo_kev))))
    hi = min(4096, int(np.ceil(kev_to_pixel(hi_kev))))
    if hi <= lo:
        raise ValueError(f"Empty energy ROI {energy_range_kev}")
    return lo, hi


def numbered_files(folder):
    """Return `{file_number: path}` and reject ambiguous duplicate numbers."""
    found = {}
    for path in sorted(folder.glob(f"scan_{SCAN:04d}_*.h5")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        number = int(match.group(2))
        if number in found:
            raise ValueError(f"Duplicate file number {number} in {folder}")
        found[number] = path
    return found


# %% Discover and validate scan 24 inputs
scan_dir = DATA_ROOT / "Raw" / f"Scan_{SCAN:04d}"
me7_files = numbered_files(scan_dir / "ME7")
xrd_files = numbered_files(scan_dir / "XRD")
position_file = DATA_ROOT / "processed" / "SOCKETSERVER" / f"Scan_{SCAN:04d}_position.h5"

if not me7_files:
    raise FileNotFoundError(f"No ME7 files in {scan_dir / 'ME7'}")
if not xrd_files:
    raise FileNotFoundError(f"No XRD files in {scan_dir / 'XRD'}")
if not position_file.exists():
    raise FileNotFoundError(f"No processed position file: {position_file}")
if not POSITION_OFFSET_FILE.exists():
    raise FileNotFoundError(f"No position offset file: {POSITION_OFFSET_FILE}")
if not SCAN_MASTER_FILE.exists():
    raise FileNotFoundError(f"No scan master file: {SCAN_MASTER_FILE}")

common_numbers = sorted(me7_files.keys() & xrd_files.keys())
if not common_numbers:
    raise ValueError("ME7 and XRD have no matching numbered files")

print(f"Data root:           {DATA_ROOT}")
print(f"Matched ME7/XRD:     {len(common_numbers)} files")
print(f"ME7-only numbers:    {sorted(me7_files.keys() - xrd_files.keys())}")
print(f"XRD-only numbers:    {sorted(xrd_files.keys() - me7_files.keys())}")

# %% Build the exact sequential ME7/XRD/position registration
with h5py.File(position_file, "r") as handle:
    position_group = handle["entry/data"]
    x_position = np.asarray(position_group["X_Position"][:], dtype=float)
    y_position_raw = np.asarray(position_group["Y_Position"][:], dtype=float)

with h5py.File(SCAN_MASTER_FILE, "r") as handle:
    theta_values = np.asarray(
        handle["entry/instrument/bluesky/streams/baseline/sample_theta/value"][:],
        dtype=float,
    )
scan_theta_deg = float(np.nanmean(theta_values))

with POSITION_OFFSET_FILE.open() as stream:
    position_offset = json.load(stream)
offset_theta = np.asarray(position_offset["theta"], dtype=float)
offset_y = np.asarray(position_offset["y_offset"], dtype=float)
if offset_theta.shape != offset_y.shape or offset_theta.size == 0:
    raise ValueError("position_offset.json theta and y_offset must be non-empty equal arrays")
order = np.argsort(offset_theta)
offset_theta = offset_theta[order]
offset_y = offset_y[order]
nearest = int(np.argmin(np.abs(offset_theta - scan_theta_deg)))
if abs(offset_theta[nearest] - scan_theta_deg) <= 0.01:
    y_offset = float(offset_y[nearest])
else:
    y_offset = float(np.interp(scan_theta_deg, offset_theta, offset_y))
y_position = y_position_raw + y_offset
print(f"Sample theta:        {scan_theta_deg:.6f} deg")
print(f"Applied Y offset:    {y_offset:.3f} (corrected Y = raw Y + offset)")

records = []
global_start = 0
for file_number in common_numbers:
    me7_file = me7_files[file_number]
    xrd_file = xrd_files[file_number]
    with h5py.File(me7_file, "r") as handle:
        me7_count = int(handle[H5_DATASET].shape[0])
    with h5py.File(xrd_file, "r") as handle:
        xrd_data = handle[H5_DATASET]
        xrd_count = int(xrd_data.shape[0]) if xrd_data.ndim == 3 else 1
    mapped_count = min(me7_count, xrd_count)
    records.append({
        "file_number": file_number,
        "me7_file": me7_file,
        "xrd_file": xrd_file,
        "me7_count": me7_count,
        "xrd_count": xrd_count,
        "mapped_count": mapped_count,
        "global_start": global_start,
    })
    global_start += xrd_count

total_xrd_frames = sum(record["xrd_count"] for record in records)
total_mapped_frames = sum(record["mapped_count"] for record in records)
if len(x_position) < total_xrd_frames or len(y_position_raw) < total_xrd_frames:
    raise ValueError(
        f"Position array has {len(x_position)} points but XRD has {total_xrd_frames} frames"
    )

unmatched_me7 = sum(record["me7_count"] - record["mapped_count"] for record in records)
unmatched_xrd = sum(record["xrd_count"] - record["mapped_count"] for record in records)
print(f"XRD frames:          {total_xrd_frames:,}")
print(f"Position points:     {len(x_position):,}")
print(f"Mapped XRF/XRD:      {total_mapped_frames:,}")
print(f"Unmatched ME7/XRD:   {unmatched_me7}/{unmatched_xrd}")

# %% Calibrated overview spectrum
# Cache the exact sum so subsequent runs do not reread the complete ME7 scan.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
spectrum_signature = json.dumps({
    "scan": SCAN,
    "files": [record["me7_file"].name for record in records],
    "me7_counts": [record["me7_count"] for record in records],
    "channels": CHANNELS,
    "deadtime_correction": DEADTIME_CORRECTION,
    "energy_calibration": ENERGY_CALIBRATION,
    "file_stride": SPECTRUM_FILE_STRIDE,
    "point_stride": SPECTRUM_POINT_STRIDE,
}, sort_keys=True)

overview = None
if SPECTRUM_CACHE.exists():
    with np.load(SPECTRUM_CACHE, allow_pickle=False) as cached:
        if str(cached["signature"]) == spectrum_signature:
            overview = cached["spectrum"].astype(np.float64)
            print(f"Loaded cached ME7 spectrum: {SPECTRUM_CACHE}")

if overview is None:
    overview = np.zeros(4096, dtype=np.float64)
    sampled_records = records[::max(1, int(SPECTRUM_FILE_STRIDE))]
    for record in sampled_records:
        with h5py.File(record["me7_file"], "r") as handle:
            data = handle[H5_DATASET]
            point_slice = slice(None, None, max(1, int(SPECTRUM_POINT_STRIDE)))
            for channel in CHANNELS:
                spectra = data[point_slice, channel, :].astype(np.float64)
                if DEADTIME_CORRECTION:
                    factor_path = DT_FACTOR.format(channel1=channel + 1)
                    if factor_path in handle:
                        factors = np.asarray(handle[factor_path][point_slice], dtype=float)
                        spectra *= factors[:, None]
                overview += spectra.sum(axis=0)
    np.savez_compressed(
        SPECTRUM_CACHE,
        spectrum=overview,
        energy_kev=pixel_to_kev(np.arange(overview.size)),
        signature=spectrum_signature,
    )
    print(f"Saved ME7 spectrum cache: {SPECTRUM_CACHE}")

energy_axis_kev = pixel_to_kev(np.arange(overview.size))

# %% 1. Full calibrated spectrum, before labels or selected ranges
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(energy_axis_kev, overview, color="black", linewidth=0.8)
ax.set(
    title=f"Scan {SCAN}: complete ME7 spectrum",
    xlabel="calibrated energy (keV)",
    ylabel="summed counts",
    xlim=(1.2, 15.0),
    yscale="log",
)
ax.grid(alpha=0.2)
plt.show()

# %% 2. Detect the ten clearest peaks and show xrd-app library candidates
# Detection is performed on log counts so low-energy peaks are not hidden by Br/Pb.
# Candidate labels within +/-200 eV come from xrd-app's public line dictionary.
smoothed_log_counts = np.log10(np.maximum(gaussian_filter1d(overview, 2), 1.0))
search_mask = (energy_axis_kev >= 1.4) & (energy_axis_kev <= 13.6)
search_bins = np.flatnonzero(search_mask)
detected_local, properties = find_peaks(
    smoothed_log_counts[search_mask], prominence=0.05, distance=15, width=1,
)
detected_bins = search_bins[detected_local]
ranked = np.argsort(properties["prominences"])[::-1]
# Ignore tiny high-energy-baseline ripples; keep the ten strongest visible features.
ranked = [index for index in ranked if overview[detected_bins[index]] >= 100][:10]

detected_peak_rows = []
for rank, index in enumerate(ranked, start=1):
    spectrum_bin = int(detected_bins[index])
    observed_kev = float(energy_axis_kev[spectrum_bin])
    candidates = sorted(
        (
            (abs(line["energy_ev"] / 1000.0 - observed_kev), line)
            for line in EMISSION_LINES
            if abs(line["energy_ev"] / 1000.0 - observed_kev) <= 0.200
        ),
        key=lambda item: item[0],
    )
    # Only Br/Pb are composition-supported here. Other peaks retain their measured
    # energy as the display name; nearby library entries remain informational.
    confirmed_name = {
        1191: "Br_Ka",
        1328: "Br_Kb",
        1056: "Pb_La",
        1262: "Pb_Lb",
    }.get(spectrum_bin)
    display_name = confirmed_name or f"Observed {observed_kev:.3f} keV"
    candidate_names = "; ".join(
        f"{line['element']}_{line['line']} ({line['energy_ev'] / 1000:.3f} keV)"
        for _, line in candidates
    ) or "none"
    detected_peak_rows.append({
        "prominence_rank": rank,
        "name": display_name,
        "spectrum_bin": spectrum_bin,
        "observed_kev": observed_kev,
        "counts": float(overview[spectrum_bin]),
        "log_prominence": float(properties["prominences"][index]),
        "nearby_xrd_app_library_lines": candidate_names,
    })

detected_peaks = pd.DataFrame(detected_peak_rows).sort_values("observed_kev")
detected_peaks.to_csv(PEAK_TABLE_CSV, index=False)
display(detected_peaks)
print(f"Wrote detected peak table to {PEAK_TABLE_CSV}")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(energy_axis_kev, overview, color="black", linewidth=0.8)
ax.set(
    title="Detected peaks and nearby xrd-app library lines",
    xlabel="calibrated energy (keV)", ylabel="summed counts",
    xlim=(1.2, 15.0), yscale="log",
)
for _, peak in detected_peaks.iterrows():
    ax.axvline(peak["observed_kev"], color="0.55", linewidth=0.7, linestyle=":")
    ax.annotate(
        peak["name"], (peak["observed_kev"], peak["counts"]),
        xytext=(2, 5), textcoords="offset points", rotation=55,
        ha="left", va="bottom", fontsize=8,
    )
ax.grid(alpha=0.2)
plt.show()

# %% 3. Define the named peaks and ranges to analyze
# Give each peak exactly one keV or MCA pixel range. Set minimum_counts later,
# after inspecting the distributions. Add observed peaks by their measured energy.
XRF_ROIS = {
    "Br": {"energy_range_kev": (11.761, 12.061), "minimum_counts": None},
    "Pb": {"energy_range_kev": (10.464, 10.764), "minimum_counts": None},
    # "Observed 9.318 keV": {
    #     "energy_range_kev": (9.15, 9.45), "minimum_counts": None,
    # },
    # "My pixel ROI": {"pixel_range": (800, 830), "minimum_counts": None},
}
FOCUS_PEAK = "Br"

# Set True to reuse the last saved names, ranges, and minimum thresholds instead.
LOAD_SAVED_ROI_CONFIG = False
if LOAD_SAVED_ROI_CONFIG and ROI_CONFIG_FILE.exists():
    with ROI_CONFIG_FILE.open() as stream:
        saved_config = json.load(stream)
    if int(saved_config.get("scan", -1)) == SCAN:
        XRF_ROIS = saved_config["rois"]
        FOCUS_PEAK = saved_config.get("focus_peak", FOCUS_PEAK)
        print(f"Loaded saved ROI configuration: {ROI_CONFIG_FILE}")

for name, roi in XRF_ROIS.items():
    if "pixel_range" not in roi and "energy_range_kev" not in roi:
        raise ValueError(f"{name!r} needs pixel_range or energy_range_kev")
    if "pixel_range" not in roi:
        roi["pixel_range"] = energy_range_to_pixels(roi["energy_range_kev"])
    lo, hi = map(int, roi["pixel_range"])
    if not 0 <= lo < hi <= overview.size:
        raise ValueError(f"Invalid pixel range for {name!r}: {(lo, hi)}")
    roi["pixel_range"] = (lo, hi)
    roi["energy_range_kev"] = (float(pixel_to_kev(lo)), float(pixel_to_kev(hi)))
    roi.setdefault("minimum_counts", None)
if FOCUS_PEAK not in XRF_ROIS:
    raise KeyError(f"FOCUS_PEAK {FOCUS_PEAK!r} is not present in XRF_ROIS")

display(pd.DataFrame([
    {
        "name": name,
        "pixel_range": f"{roi['pixel_range'][0]}:{roi['pixel_range'][1]}",
        "energy_lo_kev": roi["energy_range_kev"][0],
        "energy_hi_kev": roi["energy_range_kev"][1],
        "minimum_counts": roi["minimum_counts"],
    }
    for name, roi in XRF_ROIS.items()
]))

# %% 4. Overlay the ranges you selected
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(energy_axis_kev, overview, color="black", linewidth=0.8)
ax.set(
    title="Selected XRF integration ranges",
    xlabel="calibrated energy (keV)", ylabel="summed counts",
    xlim=(1.2, 15.0), yscale="log",
)
for name, roi in XRF_ROIS.items():
    lo, hi = roi["energy_range_kev"]
    ax.axvspan(lo, hi, alpha=0.25, label=name)
ax.legend()
ax.grid(alpha=0.2)
plt.show()

# %% 5. Enlarge the selected focus region
focus_roi = XRF_ROIS[FOCUS_PEAK]
focus_lo, focus_hi = focus_roi["pixel_range"]
padding = max(10, focus_hi - focus_lo)
region_lo = max(0, focus_lo - padding)
region_hi = min(overview.size, focus_hi + padding)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(
    energy_axis_kev[region_lo:region_hi], overview[region_lo:region_hi],
    color="black", linewidth=1,
)
ax.axvspan(*focus_roi["energy_range_kev"], color="tab:orange", alpha=0.25)
ax.set(
    title=f"{FOCUS_PEAK}: selected spectral region",
    xlabel="calibrated energy (keV)", ylabel="summed counts", yscale="log",
)
ax.grid(alpha=0.2)
plt.show()

# %% 6. Integrate each named range at every scan position (cached)
def read_roi_intensities(record):
    """Read only the selected MCA slices for one ME7 scan line."""
    count = record["mapped_count"]
    values = {material: np.zeros(count, dtype=np.float64) for material in XRF_ROIS}
    with h5py.File(record["me7_file"], "r") as handle:
        data = handle[H5_DATASET]
        for channel in CHANNELS:
            factors = None
            if DEADTIME_CORRECTION:
                factor_path = DT_FACTOR.format(channel1=channel + 1)
                if factor_path in handle:
                    factors = np.asarray(handle[factor_path][:count], dtype=float)
            for material, roi in XRF_ROIS.items():
                lo, hi = roi["pixel_range"]
                roi_counts = data[:count, channel, lo:hi].sum(axis=1, dtype=np.float64)
                if factors is not None:
                    roi_counts *= factors
                values[material] += roi_counts
    return values


intensity_signature = json.dumps({
    "scan": SCAN,
    "files": [record["me7_file"].name for record in records],
    "mapped_counts": [record["mapped_count"] for record in records],
    "channels": CHANNELS,
    "deadtime_correction": DEADTIME_CORRECTION,
    "pixel_ranges": {
        material: list(roi["pixel_range"]) for material, roi in XRF_ROIS.items()
    },
}, sort_keys=True)

intensity_chunks = None
if INTENSITY_CACHE.exists():
    with np.load(INTENSITY_CACHE, allow_pickle=False) as cached:
        if str(cached["signature"]) == intensity_signature:
            materials = [str(value) for value in cached["materials"]]
            offsets = cached["record_offsets"].astype(int)
            intensity_chunks = {
                material: [
                    cached["intensities"][index, offsets[i]:offsets[i + 1]]
                    for i in range(len(records))
                ]
                for index, material in enumerate(materials)
            }
            print(f"Loaded cached ROI intensities: {INTENSITY_CACHE}")

if intensity_chunks is None:
    intensity_chunks = {material: [] for material in XRF_ROIS}
    for record in records:
        for material, values in read_roi_intensities(record).items():
            intensity_chunks[material].append(values)
    materials = list(XRF_ROIS)
    record_offsets = np.concatenate((
        [0], np.cumsum([record["mapped_count"] for record in records])
    ))
    np.savez_compressed(
        INTENSITY_CACHE,
        intensities=np.stack([
            np.concatenate(intensity_chunks[material]) for material in materials
        ]),
        materials=np.asarray(materials),
        record_offsets=record_offsets,
        signature=intensity_signature,
    )
    print(f"Saved ROI intensity cache: {INTENSITY_CACHE}")

# %% 7. Inspect the focus peak's count distribution and 1D spatial heatmap
focus_values = np.concatenate(intensity_chunks[FOCUS_PEAK])
fig, axes = plt.subplots(
    2, 1, figsize=(13, 6), constrained_layout=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
axes[0].hist(focus_values, bins=150, log=True, color="tab:blue", alpha=0.85)
axes[0].set(
    title=f"{FOCUS_PEAK}: integrated-count distribution",
    xlabel="integrated XRF counts per scan position", ylabel="scan positions",
)
heat = axes[1].imshow(
    focus_values[np.newaxis, :], aspect="auto", interpolation="nearest",
    cmap="viridis", extent=(0, len(focus_values), 0, 1),
)
axes[1].set(
    title="1D acquisition-order heatmap", xlabel="scan position in acquisition order",
    yticks=[],
)
fig.colorbar(heat, ax=axes[1], label="integrated counts", orientation="horizontal", pad=0.35)
plt.show()
print(
    f"{FOCUS_PEAK}: min={focus_values.min():.3g}, "
    f"median={np.median(focus_values):.3g}, "
    f"p95={np.percentile(focus_values, 95):.3g}, max={focus_values.max():.3g}"
)

# %% 8. Define minimum counts, then report how much will be removed
# Set thresholds by name after inspecting the plots above. None keeps all data.
MINIMUM_COUNTS = {
    "Br": None,  # Example: 5000
    "Pb": None,
}
for name, minimum in MINIMUM_COUNTS.items():
    if name not in XRF_ROIS:
        raise KeyError(f"Threshold provided for unknown peak {name!r}")
    XRF_ROIS[name]["minimum_counts"] = minimum

cut_rows = []
for name, chunks in intensity_chunks.items():
    values = np.concatenate(chunks)
    minimum = XRF_ROIS[name]["minimum_counts"]
    keep = np.ones(values.size, dtype=bool) if minimum is None else values >= minimum
    cut_rows.append({
        "name": name,
        "minimum_counts": minimum,
        "total_positions": int(values.size),
        "retained_positions": int(keep.sum()),
        "cut_positions": int((~keep).sum()),
        "cut_percent": float(100.0 * (~keep).mean()),
    })
cut_summary = pd.DataFrame(cut_rows)
cut_summary.to_csv(CUT_SUMMARY_CSV, index=False)
display(cut_summary)
print(f"Wrote cut summary to {CUT_SUMMARY_CSV}")

focus_minimum = XRF_ROIS[FOCUS_PEAK]["minimum_counts"]
focus_keep = (
    np.ones(focus_values.size, dtype=bool)
    if focus_minimum is None else focus_values >= focus_minimum
)
fig, axes = plt.subplots(
    2, 1, figsize=(13, 6), constrained_layout=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
axes[0].hist(focus_values, bins=150, log=True, color="tab:blue", alpha=0.85)
if focus_minimum is not None:
    axes[0].axvspan(focus_values.min(), focus_minimum, color="white", alpha=0.8)
    axes[0].axvline(focus_minimum, color="tab:red", label="minimum")
    axes[0].legend()
axes[0].set(
    title=f"{FOCUS_PEAK}: threshold removes {100 * (~focus_keep).mean():.2f}%",
    xlabel="integrated XRF counts per scan position", ylabel="scan positions",
)
masked_heat = np.ma.masked_where(~focus_keep, focus_values)[np.newaxis, :]
white_viridis = plt.get_cmap("viridis").copy()
white_viridis.set_bad("white")
heat = axes[1].imshow(
    masked_heat, aspect="auto", interpolation="nearest", cmap=white_viridis,
    extent=(0, len(focus_values), 0, 1),
)
axes[1].set(
    title="Thresholded 1D heatmap; removed positions are white",
    xlabel="scan position in acquisition order", yticks=[],
)
fig.colorbar(heat, ax=axes[1], label="integrated counts", orientation="horizontal", pad=0.35)
plt.show()

# %% 9. Build filtered XRD links from the minimum-count settings
rows = []
point_number = {material: 0 for material in XRF_ROIS}
for record_index, record in enumerate(records):
    start = record["global_start"]
    for material, chunks in intensity_chunks.items():
        intensities = chunks[record_index]
        minimum = XRF_ROIS[material]["minimum_counts"]
        keep = np.ones(len(intensities), dtype=bool)
        if minimum is not None:
            keep &= intensities >= minimum
        for local_index in np.flatnonzero(keep):
            global_index = start + int(local_index)
            rows.append({
                "material": material,
                "point_index": point_number[material],
                "x_position": float(x_position[global_index]),
                "y_position": float(y_position[global_index]),
                "y_position_raw": float(y_position_raw[global_index]),
                "y_position_offset": y_offset,
                "sample_theta_deg": scan_theta_deg,
                "xrf_intensity": float(intensities[local_index]),
                "xrf_roi_lo_kev": XRF_ROIS[material]["energy_range_kev"][0],
                "xrf_roi_hi_kev": XRF_ROIS[material]["energy_range_kev"][1],
                "xrf_file": str(record["me7_file"]),
                "xrf_frame_index": int(local_index),
                "xrd_file": str(record["xrd_file"]),
                "xrd_frame_index": int(local_index),
                "global_frame_index": global_index,
            })
            point_number[material] += 1

columns = [
    "material", "point_index", "x_position", "y_position", "y_position_raw",
    "y_position_offset", "sample_theta_deg", "xrf_intensity",
    "xrf_roi_lo_kev", "xrf_roi_hi_kev", "xrf_file", "xrf_frame_index",
    "xrd_file", "xrd_frame_index", "global_frame_index",
]
filtered_points = pd.DataFrame(rows, columns=columns).set_index(
    ["material", "point_index"]
).sort_index()

print(filtered_points.groupby(level="material").size().rename("retained_points"))
display(filtered_points.head(20))

# %% Save configuration and links only; no XRD image data is copied
with ROI_CONFIG_FILE.open("w") as stream:
    json.dump({
        "scan": SCAN,
        "channels": CHANNELS,
        "deadtime_correction": DEADTIME_CORRECTION,
        "energy_calibration": ENERGY_CALIBRATION,
        "focus_peak": FOCUS_PEAK,
        "rois": XRF_ROIS,
    }, stream, indent=2)
filtered_points.to_csv(OUTPUT_CSV)
print(f"Wrote ROI configuration to {ROI_CONFIG_FILE}")
print(f"Wrote {len(filtered_points):,} links to {OUTPUT_CSV}")

# %% 10. Spatial maps after filtering; removed/no-data positions are white
fig, axes = plt.subplots(1, len(XRF_ROIS), figsize=(7 * len(XRF_ROIS), 5),
                         constrained_layout=True)
for ax, material in zip(np.atleast_1d(axes), XRF_ROIS):
    all_values = np.concatenate(intensity_chunks[material])
    global_indices = np.concatenate([
        record["global_start"] + np.arange(record["mapped_count"])
        for record in records
    ]).astype(int)
    minimum = XRF_ROIS[material]["minimum_counts"]
    keep = np.ones(all_values.size, dtype=bool)
    if minimum is not None:
        keep &= all_values >= minimum

    # White underlay preserves the full scanned footprint as explicit no-data.
    ax.scatter(
        x_position[global_indices], y_position[global_indices],
        c="white", edgecolors="none", s=3,
    )
    image = ax.scatter(
        x_position[global_indices[keep]], y_position[global_indices[keep]],
        c=all_values[keep], s=3, cmap="viridis", edgecolors="none",
    )
    ax.set_facecolor("white")
    ax.set(
        title=f"{material}: retained {keep.mean() * 100:.2f}%",
        xlabel="X position (um)", ylabel="corrected Y position (um)",
    )
    ax.set_aspect("equal")
    if keep.any():
        fig.colorbar(image, ax=ax, label="integrated XRF counts")
plt.show()
