"""Spatial binning and detector-peak tracking for XRF-cropped XRD links."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

H5_DATASET = "entry/data/data"
LINK_COLUMNS = (
    "Material", "Point", "X", "Y", "XRF Intensity", "XRD File Link",
    "XRD Frame Index", "Global Frame Index",
)
TRACK_COLUMNS = (
    "peak_id", "seed_x", "seed_y", "bin_key", "bin_row", "bin_col",
    "sample_x", "sample_y", "n_frames", "intensity", "maximum", "com_x",
    "com_y", "shift_x", "shift_y",
)


def _decode(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def load_links(path):
    """Load the column-oriented linker written by the XRF prefilter notebook."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        group = handle["links"]
        links = pd.DataFrame({
            "Material": _decode(group["Material"][:]),
            "Point": np.asarray(group["Point"][:], dtype=np.int64),
            "X": np.asarray(group["X"][:], dtype=float),
            "Y": np.asarray(group["Y"][:], dtype=float),
            "XRF Intensity": np.asarray(group["XRF Intensity"][:], dtype=float),
            "XRD File Link": _decode(group["XRD File Link"][:]),
            "XRD Frame Index": np.asarray(group["XRD Frame Index"][:], dtype=np.int64),
            "Global Frame Index": np.asarray(group["Global Frame Index"][:], dtype=np.int64),
        })
        attrs = dict(handle.attrs)
    return links, attrs


def select_links(links, material, sample_roi=None):
    """Select one material and optional ``(x_min, x_max, y_min, y_max)`` ROI."""
    selected = links.loc[links["Material"] == material].copy()
    if selected.empty:
        available = sorted(links["Material"].unique())
        raise ValueError(f"Material {material!r} is unavailable; choose from {available}")
    if sample_roi is not None:
        x_min, x_max, y_min, y_max = map(float, sample_roi)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("sample ROI must satisfy x_max > x_min and y_max > y_min")
        selected = selected.loc[
            selected["X"].between(x_min, x_max)
            & selected["Y"].between(y_min, y_max)
        ].copy()
    if selected.empty:
        raise ValueError("The selected sample ROI contains no linked frames")
    return selected.reset_index(drop=True)


def read_frame(row, region=None):
    """Read one linked raw XRD frame or detector-space region."""
    with h5py.File(row["XRD File Link"], "r") as handle:
        dataset = handle[H5_DATASET]
        index = int(row["XRD Frame Index"])
        if region is None:
            return dataset[index].astype(np.float32)
        y0, y1, x0, x1 = map(int, region)
        return dataset[index, y0:y1, x0:x1].astype(np.float32)


def assign_bins(links, bin_width, origin=None):
    """Assign physical sample coordinates to square bins of ``bin_width``."""
    bin_width = float(bin_width)
    if bin_width <= 0:
        raise ValueError("bin width must be positive")
    x0, y0 = origin or (float(links["X"].min()), float(links["Y"].min()))
    assigned = links.copy()
    assigned["bin_col"] = np.floor((assigned["X"] - x0) / bin_width).astype(int)
    assigned["bin_row"] = np.floor((assigned["Y"] - y0) / bin_width).astype(int)
    assigned["bin_key"] = (
        assigned["bin_row"].astype(str) + "_" + assigned["bin_col"].astype(str)
    )
    return assigned, (x0, y0)


def bin_summary(links, bin_widths):
    """Summarize occupied-bin and frame-count consequences of candidate widths."""
    origin = (float(links["X"].min()), float(links["Y"].min()))
    rows = []
    for width in bin_widths:
        assigned, _ = assign_bins(links, width, origin=origin)
        occupancy = assigned.groupby("bin_key", sort=False).size()
        rows.append({
            "bin_width": float(width),
            "occupied_bins": int(occupancy.size),
            "median_frames_per_bin": float(occupancy.median()),
            "maximum_frames_per_bin": int(occupancy.max()),
        })
    return pd.DataFrame(rows)


def sum_rows(rows):
    """Sum linked frames while opening each source HDF5 file only once."""
    summed = None
    for path, file_rows in rows.groupby("XRD File Link", sort=False):
        with h5py.File(path, "r") as handle:
            dataset = handle[H5_DATASET]
            for frame_index in file_rows["XRD Frame Index"].astype(int):
                frame = dataset[frame_index].astype(np.float64)
                summed = frame if summed is None else summed + frame
    return summed


def sampled_sum(links, sample_count):
    """Sum evenly spaced linked frames for reproducible detector peak selection."""
    count = min(max(1, int(sample_count)), len(links))
    indices = np.unique(np.linspace(0, len(links) - 1, count, dtype=int))
    return sum_rows(links.iloc[indices]), indices


def detect_peaks(image, sensitivity=6.0, min_distance=12, max_peaks=12):
    """Detect separated detector maxima after broad-background subtraction."""
    image = np.asarray(image, dtype=float)
    background = ndimage.gaussian_filter(image, sigma=10.0)
    cleaned = np.clip(image - background, 0, None)
    finite = cleaned[np.isfinite(cleaned)]
    if not finite.size:
        return []
    median = float(np.median(finite))
    mad = 1.4826 * float(np.median(np.abs(finite - median)))
    threshold = median + float(sensitivity) * max(mad, 1e-9)
    radius = max(1, int(min_distance))
    maxima = cleaned == ndimage.maximum_filter(cleaned, size=2 * radius + 1)
    ys, xs = np.nonzero(maxima & (cleaned > threshold))
    order = np.argsort(cleaned[ys, xs])[::-1][:max(0, int(max_peaks))]
    return [(int(xs[index]), int(ys[index])) for index in order]


def measure_peak(image, seed_x, seed_y, track_radius=12, com_radius=4):
    """Locate a peak near its seed and measure background-subtracted COM."""
    height, width = image.shape
    x0, x1 = max(0, seed_x - track_radius), min(width, seed_x + track_radius + 1)
    y0, y1 = max(0, seed_y - track_radius), min(height, seed_y + track_radius + 1)
    search = image[y0:y1, x0:x1].astype(float)
    if not search.size or not np.isfinite(search).any():
        return np.nan, np.nan, np.nan, np.nan
    local_y, local_x = np.unravel_index(np.nanargmax(search), search.shape)
    peak_x, peak_y = x0 + int(local_x), y0 + int(local_y)
    cx0, cx1 = max(0, peak_x - com_radius), min(width, peak_x + com_radius + 1)
    cy0, cy1 = max(0, peak_y - com_radius), min(height, peak_y + com_radius + 1)
    patch = image[cy0:cy1, cx0:cx1].astype(float)
    weights = np.clip(patch - float(np.nanpercentile(search, 25)), 0, None)
    intensity = float(np.nansum(weights))
    if intensity <= 0:
        return float(peak_x), float(peak_y), 0.0, float(image[peak_y, peak_x])
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1]
    com_x = float(np.nansum(weights * xx) / intensity)
    com_y = float(np.nansum(weights * yy) / intensity)
    return com_x, com_y, intensity, float(image[peak_y, peak_x])


def track(link_path, material, bin_width, sample_roi=None, peak_centers=None,
          detector_sum_sample=100, auto_sensitivity=6.0, auto_min_distance=12,
          max_auto_peaks=12, track_radius=12, com_radius=4, max_frames=5000,
          log=print):
    """Spatially bin linked frames and track detector peaks across occupied bins."""
    links, attrs = load_links(link_path)
    selected = select_links(links, material, sample_roi=sample_roi)
    if len(selected) > max_frames:
        raise ValueError(
            f"Selection has {len(selected):,} frames, above max_frames={max_frames:,}; "
            "choose a smaller sample ROI or raise the limit deliberately"
        )
    detector_sum, sample_indices = sampled_sum(selected, detector_sum_sample)
    centers = [tuple(map(int, center)) for center in (peak_centers or [])]
    if not centers:
        centers = detect_peaks(
            detector_sum, sensitivity=auto_sensitivity,
            min_distance=auto_min_distance, max_peaks=max_auto_peaks,
        )
    if not centers:
        raise ValueError("No detector peaks were selected or detected")

    assigned, origin = assign_bins(selected, bin_width)
    groups = list(assigned.groupby("bin_key", sort=False))
    rows = []
    log(f"Tracking {len(centers)} peaks in {len(groups)} bins from {len(selected)} frames")
    for index, (bin_key, bin_links) in enumerate(groups, 1):
        image = sum_rows(bin_links)
        for peak_id, (seed_x, seed_y) in enumerate(centers, 1):
            com_x, com_y, intensity, maximum = measure_peak(
                image, seed_x, seed_y, track_radius=track_radius,
                com_radius=com_radius,
            )
            rows.append({
                "peak_id": peak_id,
                "seed_x": seed_x,
                "seed_y": seed_y,
                "bin_key": bin_key,
                "bin_row": int(bin_links["bin_row"].iloc[0]),
                "bin_col": int(bin_links["bin_col"].iloc[0]),
                "sample_x": float(bin_links["X"].mean()),
                "sample_y": float(bin_links["Y"].mean()),
                "n_frames": len(bin_links),
                "intensity": intensity,
                "maximum": maximum,
                "com_x": com_x,
                "com_y": com_y,
                "shift_x": com_x - seed_x,
                "shift_y": com_y - seed_y,
            })
        if index == 1 or index % 25 == 0 or index == len(groups):
            log(f"  binned {index}/{len(groups)}")
    return {
        "tracking": pd.DataFrame(rows, columns=TRACK_COLUMNS),
        "detector_sum": detector_sum,
        "sample_indices": sample_indices,
        "peak_centers": np.asarray(centers, dtype=np.int64),
        "material": material,
        "bin_width": float(bin_width),
        "bin_origin": origin,
        "sample_roi": sample_roi,
        "n_selected_frames": len(selected),
        "source_attrs": attrs,
    }


def save_result(path, result):
    """Save a tracking result as a compact column-oriented HDF5 product."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracking = result["tracking"]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("detector_sum", data=result["detector_sum"], compression="gzip")
        handle.create_dataset("sample_indices", data=result["sample_indices"])
        handle.create_dataset("peak_centers", data=result["peak_centers"])
        group = handle.create_group("tracking")
        for column in TRACK_COLUMNS:
            values = tracking[column].to_numpy()
            if values.dtype.kind in "OUS":
                group.create_dataset(column, data=values.astype(object), dtype=string_dtype)
            else:
                group.create_dataset(column, data=values)
        handle.attrs["material"] = result["material"]
        handle.attrs["bin_width"] = result["bin_width"]
        handle.attrs["bin_origin"] = result["bin_origin"]
        handle.attrs["n_selected_frames"] = result["n_selected_frames"]
        if result["sample_roi"] is not None:
            handle.attrs["sample_roi"] = result["sample_roi"]
    return path


def load_result(path):
    """Load a saved linked-XRD tracking result."""
    with h5py.File(path, "r") as handle:
        group = handle["tracking"]
        data = {}
        for column in TRACK_COLUMNS:
            values = group[column][:]
            data[column] = _decode(values) if values.dtype.kind in "OUS" else values
        return {
            "tracking": pd.DataFrame(data),
            "detector_sum": np.asarray(handle["detector_sum"][:]),
            "sample_indices": np.asarray(handle["sample_indices"][:], dtype=np.int64),
            "peak_centers": np.asarray(handle["peak_centers"][:], dtype=np.int64),
            "attrs": dict(handle.attrs),
        }
