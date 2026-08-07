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
# Only XRF spectra are read here. Threshold decisions are saved as Boolean masks
# aligned to global frame indices; the larger XRD detector images remain untouched.

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
REGISTRATION_CACHE = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_xrd_registration.npz"
SPECTRUM_CACHE = OUTPUT_DIR / f"Scan_{SCAN:04d}_me7_spectrum.npz"
INTENSITY_CACHE = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_roi_intensities.npz"
ROI_CONFIG_FILE = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_rois.json"
PEAK_TABLE_CSV = OUTPUT_DIR / f"Scan_{SCAN:04d}_me7_detected_peaks.csv"
CUT_SUMMARY_CSV = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_cut_summary.csv"
XRF_MASK_CACHE = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_threshold_masks.npz"
LINK_TABLE_H5 = OUTPUT_DIR / f"Scan_{SCAN:04d}_xrf_xrd_links.h5"

# %% Imports
import json
import re
import time

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

# %% Build or reload the exact sequential ME7/XRD/position registration
# The raw scan-master and position reads are small. Opening every ME7/XRD file to
# inspect its frame count is slow over /mnt/z, so persist that result after run one.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
registration_signature = json.dumps({
    "scan": SCAN,
    "me7_files": [me7_files[number].name for number in common_numbers],
    "xrd_files": [xrd_files[number].name for number in common_numbers],
}, sort_keys=True)

registration = None
if REGISTRATION_CACHE.exists():
    with np.load(REGISTRATION_CACHE, allow_pickle=False) as cached:
        if str(cached["signature"]) == registration_signature:
            registration = {key: cached[key] for key in cached.files if key != "signature"}
            print(f"Loaded cached registration: {REGISTRATION_CACHE}")

if registration is None:
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
        raise ValueError("position_offset theta and y_offset must be non-empty equal arrays")
    order = np.argsort(offset_theta)
    offset_theta = offset_theta[order]
    offset_y = offset_y[order]
    nearest = int(np.argmin(np.abs(offset_theta - scan_theta_deg)))
    if abs(offset_theta[nearest] - scan_theta_deg) <= 0.01:
        y_offset = float(offset_y[nearest])
    else:
        y_offset = float(np.interp(scan_theta_deg, offset_theta, offset_y))

    me7_counts = []
    xrd_counts = []
    for file_number in common_numbers:
        with h5py.File(me7_files[file_number], "r") as handle:
            me7_counts.append(int(handle[H5_DATASET].shape[0]))
        with h5py.File(xrd_files[file_number], "r") as handle:
            xrd_data = handle[H5_DATASET]
            xrd_counts.append(int(xrd_data.shape[0]) if xrd_data.ndim == 3 else 1)

    registration = {
        "x_position": x_position,
        "y_position_raw": y_position_raw,
        "scan_theta_deg": np.asarray(scan_theta_deg),
        "y_offset": np.asarray(y_offset),
        "file_numbers": np.asarray(common_numbers),
        "me7_counts": np.asarray(me7_counts),
        "xrd_counts": np.asarray(xrd_counts),
    }
    np.savez_compressed(REGISTRATION_CACHE, signature=registration_signature, **registration)
    print(f"Saved registration cache: {REGISTRATION_CACHE}")

x_position = registration["x_position"].astype(float)
y_position_raw = registration["y_position_raw"].astype(float)
scan_theta_deg = float(registration["scan_theta_deg"])
y_offset = float(registration["y_offset"])
y_position = y_position_raw + y_offset

records = []
global_start = 0
for index, file_number in enumerate(registration["file_numbers"].astype(int)):
    me7_count = int(registration["me7_counts"][index])
    xrd_count = int(registration["xrd_counts"][index])
    records.append({
        "file_number": file_number,
        "me7_file": me7_files[file_number],
        "xrd_file": xrd_files[file_number],
        "me7_count": me7_count,
        "xrd_count": xrd_count,
        "mapped_count": min(me7_count, xrd_count),
        "global_start": global_start,
    })
    global_start += xrd_count

print(f"Sample theta:        {scan_theta_deg:.6f} deg")
print(f"Applied Y offset:    {y_offset:.3f} (corrected Y = raw Y + offset)")

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
# Use "all", one name such as "Br", or multiple names such as ["Br", "Pb"].
FOCUS_PEAK = "all"

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

if FOCUS_PEAK == "all":
    focus_peaks = list(XRF_ROIS)
elif isinstance(FOCUS_PEAK, str):
    focus_peaks = [FOCUS_PEAK]
else:
    focus_peaks = list(FOCUS_PEAK)
unknown_focus_peaks = [name for name in focus_peaks if name not in XRF_ROIS]
if unknown_focus_peaks:
    raise KeyError(f"Unknown FOCUS_PEAK entries: {unknown_focus_peaks}")
if not focus_peaks:
    raise ValueError("FOCUS_PEAK must select at least one ROI")

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

# %% 5. Enlarge the selected focus regions
fig, axes = plt.subplots(
    len(focus_peaks), 1, figsize=(11, 4 * len(focus_peaks)),
    constrained_layout=True, squeeze=False,
)
for ax, name in zip(axes[:, 0], focus_peaks):
    focus_roi = XRF_ROIS[name]
    focus_lo, focus_hi = focus_roi["pixel_range"]
    padding = max(10, focus_hi - focus_lo)
    region_lo = max(0, focus_lo - padding)
    region_hi = min(overview.size, focus_hi + padding)
    ax.plot(
        energy_axis_kev[region_lo:region_hi], overview[region_lo:region_hi],
        color="black", linewidth=1,
    )
    ax.axvspan(*focus_roi["energy_range_kev"], color="tab:orange", alpha=0.25)
    ax.set(
        title=f"{name}: selected spectral region",
        xlabel="calibrated energy (keV)", ylabel="summed counts", yscale="log",
    )
    ax.grid(alpha=0.2)
plt.show()

# %% 6. Integrate each named range at every scan position (cached)
def read_roi_intensities(record):
    """Read one ME7 file once, then integrate every selected ROI in memory."""
    count = record["mapped_count"]
    with h5py.File(record["me7_file"], "r") as handle:
        # Files are gzip-chunked as one full 7x4096 spectrum per point. A narrow
        # HDF5 slice still decompresses that full chunk, so one bulk read avoids
        # repeating decompression for every channel and selected range.
        spectra = handle[H5_DATASET][:count, CHANNELS, :].astype(np.float64)
        if DEADTIME_CORRECTION:
            for channel_index, channel in enumerate(CHANNELS):
                factor_path = DT_FACTOR.format(channel1=channel + 1)
                if factor_path in handle:
                    factors = np.asarray(handle[factor_path][:count], dtype=float)
                    spectra[:, channel_index, :] *= factors[:, None]

    return {
        material: spectra[:, :, lo:hi].sum(axis=(1, 2), dtype=np.float64)
        for material, roi in XRF_ROIS.items()
        for lo, hi in [roi["pixel_range"]]
    }


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
    started = time.perf_counter()
    intensity_chunks = {material: [] for material in XRF_ROIS}
    for record_index, record in enumerate(records, start=1):
        for material, values in read_roi_intensities(record).items():
            intensity_chunks[material].append(values)
        if record_index == 1 or record_index % 10 == 0 or record_index == len(records):
            print(
                f"Integrated ME7 file {record_index}/{len(records)} "
                f"({time.perf_counter() - started:.1f} s)"
            )
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
    print(
        f"Saved ROI intensity cache in {time.perf_counter() - started:.1f} s: "
        f"{INTENSITY_CACHE}"
    )

# %% 7. Define minimum counts and compare distribution with real space
# Edit these values and rerun this cell. The dotted line moves right as the minimum
# increases, and the corresponding below-threshold real-space positions turn white.
MINIMUM_COUNTS = {
    "Br": 10,  # Example: 5000
    "Pb": 5,
}
for name, minimum in MINIMUM_COUNTS.items():
    if name not in XRF_ROIS:
        raise KeyError(f"Threshold provided for unknown peak {name!r}")
    XRF_ROIS[name]["minimum_counts"] = minimum

global_indices = np.concatenate([
    record["global_start"] + np.arange(record["mapped_count"])
    for record in records
]).astype(int)

fig, axes = plt.subplots(
    len(focus_peaks), 2, figsize=(15, 5.5 * len(focus_peaks)),
    constrained_layout=True, squeeze=False,
)
for row, name in enumerate(focus_peaks):
    focus_values = np.concatenate(intensity_chunks[name])
    focus_minimum = XRF_ROIS[name]["minimum_counts"]
    focus_keep = (
        np.ones(focus_values.size, dtype=bool)
        if focus_minimum is None else focus_values >= focus_minimum
    )

    axes[row, 0].hist(
        focus_values, bins=150, log=True, color="tab:blue", alpha=0.85,
    )
    if focus_minimum is not None:
        axes[row, 0].axvline(
            focus_minimum, color="tab:red", linestyle=":", linewidth=2,
            label=f"minimum = {focus_minimum:g}",
        )
        axes[row, 0].legend()
    axes[row, 0].set(
        title=(f"{name}: scan positions vs integrated counts\n"
               f"cut = {100 * (~focus_keep).mean():.2f}%"),
        xlabel="integrated XRF counts per scan position", ylabel="scan positions",
    )

    # Draw every measured position white first, then overlay retained intensities.
    axes[row, 1].scatter(
        x_position[global_indices], y_position[global_indices],
        c="white", edgecolors="none", s=3,
    )
    image = axes[row, 1].scatter(
        x_position[global_indices[focus_keep]], y_position[global_indices[focus_keep]],
        c=focus_values[focus_keep], cmap="viridis", edgecolors="none", s=3,
    )
    axes[row, 1].set_facecolor("white")
    axes[row, 1].set(
        title=f"{name}: integrated XRF counts in real space",
        xlabel="X position (um)", ylabel="corrected Y position (um)",
    )
    axes[row, 1].set_aspect("equal")
    if focus_keep.any():
        fig.colorbar(image, ax=axes[row, 1], label="integrated XRF counts")
plt.show()

print("Focused XRF ROIs:")
for name in focus_peaks:
    values = np.concatenate(intensity_chunks[name])
    minimum = XRF_ROIS[name]["minimum_counts"]
    keep = np.ones(values.size, dtype=bool) if minimum is None else values >= minimum
    print(
        f"{name}: min={values.min():.3g}, median={np.median(values):.3g}, "
        f"p95={np.percentile(values, 95):.3g}, max={values.max():.3g}; "
        f"minimum_counts={minimum}; removed={100 * (~keep).mean():.2f}%"
    )

# %% 8. Save the cropped XRF-to-XRD link table
# No detector images are copied. Each retained row points to one frame in a raw
# XRD file, so downstream processing can load only positions that pass the cut.
mask_names = list(XRF_ROIS)
keep_masks = []
cut_rows = []
link_rows = []
point_number = {name: 0 for name in mask_names}
for name in mask_names:
    minimum = XRF_ROIS[name]["minimum_counts"]
    all_values = np.concatenate(intensity_chunks[name])
    all_keep = (
        np.ones(all_values.size, dtype=bool)
        if minimum is None else all_values >= minimum
    )
    keep_masks.append(all_keep)
    cut_rows.append({
        "name": name,
        "minimum_counts": minimum,
        "total_positions": int(all_values.size),
        "retained_positions": int(all_keep.sum()),
        "cut_positions": int((~all_keep).sum()),
        "cut_percent": float(100.0 * (~all_keep).mean()),
    })

    for record_index, record in enumerate(records):
        values = intensity_chunks[name][record_index]
        keep = np.ones(values.size, dtype=bool) if minimum is None else values >= minimum
        for local_index in np.flatnonzero(keep):
            global_index = record["global_start"] + int(local_index)
            link_rows.append({
                "Material": name,
                "Point": point_number[name],
                "X": float(x_position[global_index]),
                "Y": float(y_position[global_index]),
                "XRF Intensity": float(values[local_index]),
                "XRD File Link": str(record["xrd_file"]),
                "XRD Frame Index": int(local_index),
                "Global Frame Index": global_index,
            })
            point_number[name] += 1

cut_summary = pd.DataFrame(cut_rows)
link_table = pd.DataFrame(link_rows, columns=[
    "Material", "Point", "X", "Y", "XRF Intensity", "XRD File Link",
    "XRD Frame Index", "Global Frame Index",
])
cut_summary.to_csv(CUT_SUMMARY_CSV, index=False)
np.savez_compressed(
    XRF_MASK_CACHE,
    names=np.asarray(mask_names),
    keep_masks=np.stack(keep_masks),
    global_frame_indices=global_indices,
    minimum_counts=np.asarray([
        np.nan if XRF_ROIS[name]["minimum_counts"] is None
        else float(XRF_ROIS[name]["minimum_counts"])
        for name in mask_names
    ]),
)
string_dtype = h5py.string_dtype(encoding="utf-8")
with h5py.File(LINK_TABLE_H5, "w") as handle:
    table = handle.create_group("links")
    table.create_dataset(
        "Material", data=link_table["Material"].to_numpy(dtype=object),
        dtype=string_dtype,
    )
    table.create_dataset("X", data=link_table["X"].to_numpy(dtype=float))
    table.create_dataset("Y", data=link_table["Y"].to_numpy(dtype=float))
    table.create_dataset(
        "XRD File Link", data=link_table["XRD File Link"].to_numpy(dtype=object),
        dtype=string_dtype,
    )
    table.create_dataset(
        "XRD Frame Index", data=link_table["XRD Frame Index"].to_numpy(dtype=np.int64),
    )
    table.create_dataset(
        "Global Frame Index",
        data=link_table["Global Frame Index"].to_numpy(dtype=np.int64),
    )
    table.create_dataset(
        "XRF Intensity", data=link_table["XRF Intensity"].to_numpy(dtype=float),
    )
    table.create_dataset("Point", data=link_table["Point"].to_numpy(dtype=np.int64))
    handle.attrs["scan"] = SCAN
    handle.attrs["sample_theta_deg"] = scan_theta_deg
    handle.attrs["y_position_offset"] = y_offset
with ROI_CONFIG_FILE.open("w") as stream:
    json.dump({
        "scan": SCAN,
        "channels": CHANNELS,
        "deadtime_correction": DEADTIME_CORRECTION,
        "energy_calibration": ENERGY_CALIBRATION,
        "focus_peak": FOCUS_PEAK,
        "rois": XRF_ROIS,
    }, stream, indent=2)


display(cut_summary)
print(link_table.groupby("Material").size().rename("linked_xrd_frames"))
display(link_table.head(20))
print(f"Wrote {len(link_table):,} cropped XRF-to-XRD links to {LINK_TABLE_H5}")
print("Raw XRD detector images were not copied.")
