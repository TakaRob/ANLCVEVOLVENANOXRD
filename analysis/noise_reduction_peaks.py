# %%  [markdown]
# # Radial Noise Reduction & Peak Discovery
# 
# **Scan 203** | 26-ID-C nanoprobe | 15 keV
# 
# The detector background follows a radial gradient tied to the 2-theta
# geometry (donut/bell-curve shape along Debye-Scherrer arcs). This notebook:
# 
# 1. Characterizes the radial noise profile directly from 5x5 binned images
# 2. Fits four noise models: Gaussian, split Gaussian, skewed Gaussian, Fourier-smoothed
# 3. Shows side-by-side before/after on both the summed image and individual 5x5 bins
# 4. Runs a tunable peak-finding algorithm on noise-reduced images
# 5. Compares discovered peaks against Ming's 31 annotations
# 6. Provides an interactive viewer to browse 5x5 bins and tune parameters

# %%
import os
import json
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import h5py
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.stats import skewnorm

%matplotlib inline
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (12, 8)

# ===== PATHS =====
PROJECT_ROOT = Path(os.path.abspath('')).parent
SCAN_NUMBER = 203
SCAN_DIR = PROJECT_ROOT / 'raw_scans' / f'Scan_{SCAN_NUMBER:04d}'
XRD_DIR = SCAN_DIR / 'XRD'
POSITION_CSV = PROJECT_ROOT / '203 other' / 'Scan_0203_position.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'scan203' / 'ground_truth'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MING_ANNOTATIONS = PROJECT_ROOT / 'results' / 'scan203' / 'scan_203_xrd_annotations.json'
INTEGRATED_TIFF = PROJECT_ROOT / 'raw_scans' / 'scan_203_sum.tiff'
TTH_TIFF = PROJECT_ROOT / 'raw_scans' / 'tth.tiff'

H5_DATASET = 'entry/data/data'

# ===== REFLECTIONS =====
DEGS = [6.81319, 7.51422, 10.61748, 13.00831, 15.01266,
        16.07224, 16.79944, 18.42549, 21.29655, 22.59817, 26.16205]
DEG_LABELS = ['PbI2', '(001)', '(011)', '(111)', '(002)',
              'ITO', '(012)', '(112)']

# ===== TUNABLE PARAMETERS =====
BIN_SIZE = 5
NOISE_REDUCTION_STRENGTH = 1.0   # 0.0 = no reduction, 1.0 = full subtraction
NOISE_VERTICAL_SHIFT = 0.0       # shift the noise model up/down before subtracting
PEAK_PERCENTILE = 97.0           # peak detection sensitivity (lower = more sensitive)
PEAK_MIN_PIXELS = 3              # minimum connected-component size
PEAK_PAD = 10                    # bounding box padding
MATCH_TOLERANCE = 40             # pixels, for comparing against Ming's boxes

print(f'Project root: {PROJECT_ROOT}')
print(f'Integrated image: {INTEGRATED_TIFF.name} (exists: {INTEGRATED_TIFF.exists()})')
print(f'2-theta map: {TTH_TIFF.name} (exists: {TTH_TIFF.exists()})')
print(f'Ming annotations: {MING_ANNOTATIONS.name} (exists: {MING_ANNOTATIONS.exists()})')

# %%  [markdown]
# ---
# ## 1. Load Data & Grid Mapping
# 
# Load the fully summed detector image, 2-theta geometry, Ming's annotations,
# and build the spatial scan grid for 5x5 binning.

# %%
# Load detector data
xrd_sum = tifffile.imread(INTEGRATED_TIFF).astype(np.float64)
tth = tifffile.imread(TTH_TIFF).astype(np.float64)
DET_H, DET_W = xrd_sum.shape
print(f'Detector shape: {DET_H} x {DET_W}')
print(f'Summed image intensity range: [{xrd_sum.min():.0f}, {xrd_sum.max():.0f}]')
print(f'2-theta range: [{tth.min():.3f}, {tth.max():.3f}] deg')

# Load Ming's annotations
with open(MING_ANNOTATIONS) as f:
    ming_data = json.load(f)

ming_boxes = []
for box in ming_data['annotations']['0']['boxes']:
    ming_boxes.append({
        'reflection': box.get('reflection', 'unknown'),
        'family': box.get('reflection', 'unknown').rsplit('-', 1)[0],
        'x0': int(box['x0']), 'y0': int(box['y0']),
        'x1': int(box['x1']), 'y1': int(box['y1']),
        'cx': int(round((box['x0'] + box['x1']) / 2.0)),
        'cy': int(round((box['y0'] + box['y1']) / 2.0)),
    })

print(f'Ming annotations: {len(ming_boxes)} boxes')
ming_families = sorted(set(b['family'] for b in ming_boxes))
print(f'Families: {ming_families}')

# %%
# Grid mapping functions (reused from ground_truth_builder)

def load_xrd_metadata(xrd_dir, scan_number):
    t0 = time.time()
    xrd_files = sorted(xrd_dir.glob(f'scan_{scan_number:04d}_*.h5'))
    xrd_files = [f for f in xrd_files if f.stat().st_size > 0]
    print(f'[load_xrd_metadata] Found {len(xrd_files)} XRD H5 files')
    xrd_file_map = []
    for fi, fp in enumerate(xrd_files):
        with h5py.File(fp, 'r') as f:
            n_frames = f[H5_DATASET].shape[0]
        for j in range(n_frames):
            xrd_file_map.append((fi, j))
        if (fi + 1) % 50 == 0 or fi == len(xrd_files) - 1:
            print(f'  File metadata: {fi + 1}/{len(xrd_files)}  ({time.time()-t0:.1f}s)')
    n_total = len(xrd_file_map)
    print(f'  Total frames: {n_total}')
    return xrd_files, xrd_file_map, n_total


def load_positions_from_csv(csv_path, n_total_frames):
    print(f'[load_positions] Reading {csv_path.name}...')
    csv_x, csv_y = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_x.append(float(row['X_Position']))
            csv_y.append(float(row['Y_Position']))
    n_csv = len(csv_x)
    frame_x = np.full(n_total_frames, np.nan)
    frame_y = np.full(n_total_frames, np.nan)
    n_mapped = min(n_csv, n_total_frames)
    frame_x[:n_mapped] = csv_x[:n_mapped]
    frame_y[:n_mapped] = csv_y[:n_mapped]
    print(f'  {n_csv} CSV rows mapped to {n_total_frames} frames')
    return frame_x, frame_y


def build_scan_grid(frame_x, n_total_frames, kernel=20, order=50):
    valid_mask = ~np.isnan(frame_x)
    x_for_grid = frame_x.copy()
    if np.any(~valid_mask):
        x_for_grid[~valid_mask] = np.interp(
            np.where(~valid_mask)[0], np.where(valid_mask)[0], frame_x[valid_mask])
    x_smooth = np.convolve(x_for_grid, np.ones(kernel)/kernel, mode='same')
    x_max_idx = argrelextrema(x_smooth, np.greater, order=order)[0]
    x_min_idx = argrelextrema(x_smooth, np.less, order=order)[0]
    turn_idx = np.sort(np.concatenate([x_max_idx, x_min_idx]))
    seg_starts = np.concatenate([[0], turn_idx])
    seg_ends = np.concatenate([turn_idx, [n_total_frames]])
    grid_row = np.zeros(n_total_frames, dtype=int)
    grid_col = np.zeros(n_total_frames, dtype=int)
    for i in range(len(seg_starts)):
        s, e = seg_starts[i], seg_ends[i]
        grid_row[s:e] = i
        cols = np.arange(e - s)
        if i % 2 == 1:
            cols = cols[::-1]
        grid_col[s:e] = cols
    n_rows = grid_row.max() + 1
    n_cols = grid_col.max() + 1
    print(f'[build_scan_grid] {n_rows} rows x {n_cols} cols')
    return grid_row, grid_col, n_rows, n_cols


def load_and_sum_frames(frame_indices, xrd_files, xrd_file_map):
    summed = None
    by_file = {}
    for gi in frame_indices:
        fi, fj = xrd_file_map[gi]
        by_file.setdefault(fi, []).append(fj)
    for fi, frame_list in by_file.items():
        with h5py.File(xrd_files[fi], 'r') as f:
            ds = f[H5_DATASET]
            for fj in frame_list:
                frame = ds[fj].astype(np.float64)
                if summed is None:
                    summed = frame
                else:
                    summed += frame
    return summed


# Run grid mapping
print('=' * 60)
print('Loading XRD metadata and building scan grid')
print('=' * 60)

xrd_files, xrd_file_map, n_total_frames = load_xrd_metadata(XRD_DIR, SCAN_NUMBER)

if POSITION_CSV.exists():
    frame_x, frame_y = load_positions_from_csv(POSITION_CSV, n_total_frames)
else:
    raise FileNotFoundError(f'Position CSV not found: {POSITION_CSV}')

grid_row, grid_col, n_rows, n_cols = build_scan_grid(frame_x, n_total_frames)

# Build grid-to-frames lookup
grid_to_frames = {}
for gi in range(n_total_frames):
    key = (grid_row[gi], grid_col[gi])
    grid_to_frames.setdefault(key, []).append(gi)

# Build 5x5 bin mapping
def build_bin_mapping(n_rows, n_cols, bin_size, grid_to_frames):
    n_bin_rows = (n_rows + bin_size - 1) // bin_size
    n_bin_cols = (n_cols + bin_size - 1) // bin_size
    bin_to_pixels = {}
    for br in range(n_bin_rows):
        for bc in range(n_bin_cols):
            pixels, frames = [], []
            for dr in range(bin_size):
                for dc in range(bin_size):
                    r, c = br * bin_size + dr, bc * bin_size + dc
                    if r < n_rows and c < n_cols:
                        cell_frames = grid_to_frames.get((r, c), [])
                        if cell_frames:
                            pixels.append((r, c))
                            frames.extend(cell_frames)
            if frames:
                bin_to_pixels[(br, bc)] = {'pixels': pixels, 'frames': frames}
    return bin_to_pixels, n_bin_rows, n_bin_cols

bin_mapping, n_bin_rows, n_bin_cols = build_bin_mapping(
    n_rows, n_cols, BIN_SIZE, grid_to_frames)
print(f'\n{BIN_SIZE}x{BIN_SIZE} bins: {len(bin_mapping)} bins ({n_bin_rows}x{n_bin_cols} grid)')
avg_frames = np.mean([len(info['frames']) for info in bin_mapping.values()])
print(f'Average frames per bin: {avg_frames:.1f}')

# %%  [markdown]
# ---
# ## 2. Characterize Radial Noise Profile from 5x5 Bins
# 
# The detector background follows the 2-theta geometry. We sample a
# representative set of 5x5 bins, compute the radial profile (median
# intensity vs 2-theta) for each, then average across bins. This gives
# us the noise shape at the actual per-bin intensity level — no scaling
# needed when subtracting later.
# 
# We also compute the profile from the fully summed image for reference.

# %%
# Compute radial profile from 5x5 bins
TTH_BIN_WIDTH = 0.05  # degrees per bin
N_SAMPLE_BINS = 50    # number of 5x5 bins to average over

tth_min, tth_max = tth.min(), tth.max()
tth_edges = np.arange(tth_min, tth_max + TTH_BIN_WIDTH, TTH_BIN_WIDTH)
tth_centers = 0.5 * (tth_edges[:-1] + tth_edges[1:])
n_tth_bins = len(tth_centers)

print(f'2-theta range: [{tth_min:.3f}, {tth_max:.3f}] deg')
print(f'Bin width: {TTH_BIN_WIDTH} deg -> {n_tth_bins} radial bins')

# Pre-compute per-pixel tth bin assignment (shared everywhere)
tth_flat = tth.ravel()
bin_indices = np.digitize(tth_flat, tth_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_tth_bins - 1)

# Count pixels per radial bin (for filtering low-count bins)
radial_count = np.zeros(n_tth_bins, dtype=int)
for i in range(n_tth_bins):
    radial_count[i] = np.count_nonzero(bin_indices == i)

# --- Sample 5x5 bins and compute per-bin radial profiles ---
bin_keys_all = sorted(bin_mapping.keys())
rng = np.random.default_rng(42)
n_sample = min(N_SAMPLE_BINS, len(bin_keys_all))
sample_indices = rng.choice(len(bin_keys_all), size=n_sample, replace=False)
sample_bins = [bin_keys_all[i] for i in sorted(sample_indices)]

print(f'\nSampling {n_sample} of {len(bin_keys_all)} bins for radial profile...')

per_bin_profiles = []
t0 = time.time()

for si, bk in enumerate(sample_bins):
    info = bin_mapping[bk]
    n_frames = len(info['frames'])
    summed = load_and_sum_frames(info['frames'], xrd_files, xrd_file_map)
    summed[summed < 0] = 0
    summed[summed > 1e9] = 0

    # Radial profile for this bin
    img_flat = summed.ravel()
    profile = np.zeros(n_tth_bins)
    for i in range(n_tth_bins):
        mask = bin_indices == i
        vals = img_flat[mask]
        if len(vals) > 0:
            profile[i] = np.median(vals)
    per_bin_profiles.append(profile)

    if (si + 1) % 10 == 0 or si == n_sample - 1:
        elapsed = time.time() - t0
        rate = (si + 1) / elapsed
        eta = (n_sample - si - 1) / rate
        print(f'  [{100*(si+1)/n_sample:5.1f}%] {si+1}/{n_sample} bins | '
              f'{rate:.1f} bins/s | ETA {eta:.0f}s')

per_bin_profiles = np.array(per_bin_profiles)  # (n_sample, n_tth_bins)

# Average across sampled bins -> representative 5x5-bin radial profile
radial_median = np.median(per_bin_profiles, axis=0)
radial_mean = np.mean(per_bin_profiles, axis=0)
radial_std = np.std(per_bin_profiles, axis=0)

elapsed = time.time() - t0
print(f'\nRadial profile from {n_sample} bins computed in {elapsed:.1f}s')
print(f'Profile intensity range: [{radial_median.min():.1f}, {radial_median.max():.1f}]')

# Also compute from the fully summed image for reference
radial_median_sum = np.zeros(n_tth_bins)
img_flat_sum = xrd_sum.ravel()
for i in range(n_tth_bins):
    mask = bin_indices == i
    vals = img_flat_sum[mask]
    if len(vals) > 0:
        radial_median_sum[i] = np.median(vals)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(tth_centers, radial_median, 'b-', lw=1.2, label='5x5 bin median')
axes[0].fill_between(tth_centers,
                      radial_median - radial_std,
                      radial_median + radial_std,
                      alpha=0.2, color='b', label='bin-to-bin std')
axes[0].set_xlabel('2-theta (deg)')
axes[0].set_ylabel('Intensity (per 5x5 bin)')
axes[0].set_title(f'Radial profile (median of {n_sample} bins)')
axes[0].legend(fontsize=8)

for d, lab in zip(DEGS, DEG_LABELS):
    if tth_min <= d <= tth_max:
        axes[0].axvline(d, color='green', ls='--', alpha=0.4, lw=0.8)
        axes[0].text(d, axes[0].get_ylim()[1]*0.95, lab, fontsize=6,
                     rotation=90, va='top', ha='right', color='green')

axes[1].plot(tth_centers, radial_std, 'orange', lw=0.8)
axes[1].set_xlabel('2-theta (deg)')
axes[1].set_ylabel('Std deviation across bins')
axes[1].set_title('Bin-to-bin variation per annulus')

# Overlay: individual bin profiles (thin lines)
axes[2].set_xlabel('2-theta (deg)')
axes[2].set_ylabel('Intensity')
axes[2].set_title('Individual bin profiles (subset)')
for prof in per_bin_profiles[::max(1, n_sample//10)]:
    axes[2].plot(tth_centers, prof, 'gray', lw=0.3, alpha=0.5)
axes[2].plot(tth_centers, radial_median, 'b-', lw=1.5, label='Median')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

# %%  [markdown]
# ---
# ## 3. Fit Noise Models to 5x5 Radial Profile
# 
# Four approaches to model the radial background measured from 5x5 bins:
# 
# 1. **Gaussian** — symmetric bell curve I(2$\theta$) = A exp(-(2$\theta$-$\mu$)$^2$ / 2$\sigma^2$) + C
# 2. **Split Gaussian** — different widths left/right of the peak ($\sigma_L \neq \sigma_R$)
# 3. **Skewed Gaussian** — asymmetric via scipy.stats.skewnorm
# 4. **Fourier low-pass** — smooth the empirical profile without parametric assumptions
# 
# Since the profile is already at the 5x5-bin intensity level, the fitted
# models can be subtracted directly from any 5x5 bin image without rescaling.

# %%
# ===== Model definitions =====

def gaussian_model(x, amplitude, mu, sigma, offset):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2) + offset


def split_gaussian_model(x, amplitude, mu, sigma_left, sigma_right, offset):
    result = np.empty_like(x)
    left = x <= mu
    right = ~left
    result[left] = amplitude * np.exp(-0.5 * ((x[left] - mu) / sigma_left)**2)
    result[right] = amplitude * np.exp(-0.5 * ((x[right] - mu) / sigma_right)**2)
    return result + offset


def skewed_gaussian_model(x, amplitude, mu, sigma, skew, offset):
    z = (x - mu) / sigma
    pdf_vals = skewnorm.pdf(z, skew) / skewnorm.pdf(skewnorm.mean(skew), skew)
    return amplitude * pdf_vals + offset


def fourier_lowpass(profile, cutoff_fraction=0.05):
    """Low-pass filter the radial profile using FFT."""
    n = len(profile)
    fft_vals = np.fft.rfft(profile)
    freqs = np.fft.rfftfreq(n)
    fft_vals[freqs > cutoff_fraction] = 0
    return np.fft.irfft(fft_vals, n=n)


# ===== Fit each model to the 5x5-bin radial median profile =====
x_data = tth_centers
y_data = radial_median  # median across sampled 5x5 bins

# Mask out radial bins with too few detector pixels
valid = radial_count > 50
x_fit = x_data[valid]
y_fit = y_data[valid]

# Initial guesses from the data
peak_idx = np.argmax(y_fit)
mu_init = x_fit[peak_idx]
amp_init = y_fit[peak_idx] - np.percentile(y_fit, 5)
sigma_init = 3.0
offset_init = np.percentile(y_fit, 5)

print(f'Fitting to 5x5-bin radial profile (median of {n_sample} bins)')
print(f'Initial guesses: amplitude={amp_init:.1f}, mu={mu_init:.2f}, '
      f'sigma={sigma_init:.1f}, offset={offset_init:.1f}')
print()

fits = {}

# 1. Gaussian
try:
    popt, pcov = curve_fit(gaussian_model, x_fit, y_fit,
                           p0=[amp_init, mu_init, sigma_init, offset_init],
                           maxfev=10000)
    fitted_gauss = gaussian_model(x_data, *popt)
    residual = np.sqrt(np.mean((y_data[valid] - fitted_gauss[valid])**2))
    fits['gaussian'] = {'profile': fitted_gauss, 'params': popt, 'rmse': residual}
    print(f'Gaussian: A={popt[0]:.1f}, mu={popt[1]:.2f}, sigma={popt[2]:.2f}, '
          f'offset={popt[3]:.1f}  RMSE={residual:.2f}')
except Exception as e:
    print(f'Gaussian fit failed: {e}')

# 2. Split Gaussian
try:
    popt2, pcov2 = curve_fit(split_gaussian_model, x_fit, y_fit,
                              p0=[amp_init, mu_init, sigma_init, sigma_init, offset_init],
                              maxfev=10000)
    fitted_split = split_gaussian_model(x_data, *popt2)
    residual2 = np.sqrt(np.mean((y_data[valid] - fitted_split[valid])**2))
    fits['split_gaussian'] = {'profile': fitted_split, 'params': popt2, 'rmse': residual2}
    print(f'Split Gaussian: A={popt2[0]:.1f}, mu={popt2[1]:.2f}, '
          f'sigma_L={popt2[2]:.2f}, sigma_R={popt2[3]:.2f}, '
          f'offset={popt2[4]:.1f}  RMSE={residual2:.2f}')
except Exception as e:
    print(f'Split Gaussian fit failed: {e}')

# 3. Skewed Gaussian
try:
    popt3, pcov3 = curve_fit(skewed_gaussian_model, x_fit, y_fit,
                              p0=[amp_init, mu_init, sigma_init, 0.0, offset_init],
                              maxfev=10000,
                              bounds=([0, tth_min, 0.1, -20, 0],
                                      [np.inf, tth_max, 30, 20, np.inf]))
    fitted_skew = skewed_gaussian_model(x_data, *popt3)
    residual3 = np.sqrt(np.mean((y_data[valid] - fitted_skew[valid])**2))
    fits['skewed_gaussian'] = {'profile': fitted_skew, 'params': popt3, 'rmse': residual3}
    print(f'Skewed Gaussian: A={popt3[0]:.1f}, mu={popt3[1]:.2f}, '
          f'sigma={popt3[2]:.2f}, skew={popt3[3]:.2f}, '
          f'offset={popt3[4]:.1f}  RMSE={residual3:.2f}')
except Exception as e:
    print(f'Skewed Gaussian fit failed: {e}')

# 4. Fourier low-pass (multiple cutoff levels)
for cutoff in [0.02, 0.05, 0.10]:
    fitted_fourier = fourier_lowpass(y_data, cutoff_fraction=cutoff)
    residual_f = np.sqrt(np.mean((y_data[valid] - fitted_fourier[valid])**2))
    key = f'fourier_{cutoff}'
    fits[key] = {'profile': fitted_fourier, 'params': {'cutoff': cutoff}, 'rmse': residual_f}
    print(f'Fourier (cutoff={cutoff}): RMSE={residual_f:.2f}')

print(f'\nBest fit by RMSE: {min(fits.keys(), key=lambda k: fits[k]["rmse"])}')

# %%
# Plot all fits overlaid on the 5x5-bin radial data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

model_names = {
    'gaussian': 'Gaussian',
    'split_gaussian': 'Split Gaussian',
    'skewed_gaussian': 'Skewed Gaussian',
    'fourier_0.05': 'Fourier LP (0.05)',
}

model_colors = {
    'gaussian': '#e74c3c',
    'split_gaussian': '#2ecc71',
    'skewed_gaussian': '#9b59b6',
    'fourier_0.05': '#f39c12',
}

for ax, (key, label) in zip(axes.flat, model_names.items()):
    if key not in fits:
        ax.text(0.5, 0.5, f'{label}\nfit failed', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        continue

    ax.plot(x_data, y_data, 'k-', lw=0.5, alpha=0.6, label='5x5 bin median')
    ax.plot(x_data, fits[key]['profile'], color=model_colors[key], lw=2,
            label=f'{label}\nRMSE={fits[key]["rmse"]:.2f}')

    # Residual on secondary axis
    ax2 = ax.twinx()
    residuals = y_data - fits[key]['profile']
    ax2.plot(x_data, residuals, color=model_colors[key], lw=0.5, alpha=0.3)
    ax2.axhline(0, color='gray', ls=':', lw=0.5)
    ax2.set_ylabel('Residual', fontsize=8, color='gray')
    ax2.tick_params(labelsize=7, colors='gray')

    ax.set_xlabel('2-theta (deg)')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(label)

plt.suptitle('Noise model fits to 5x5-bin radial profile', fontsize=13)
plt.tight_layout()
plt.show()

# %%  [markdown]
# ---
# ## 4. Build Background Images & Apply Noise Reduction
# 
# For each noise model, map the fitted 5x5-bin radial profile back to the
# full detector to create a 2D background image, then subtract it.
# 
# Since the profile was fitted at the 5x5-bin intensity level, it can be
# subtracted directly from any 5x5 bin image. For the full summed image,
# we scale up by `n_total_frames / avg_frames_per_bin`.
# 
# Two tunable parameters:
# - **NOISE_REDUCTION_STRENGTH** (0-1): fraction of background to subtract
# - **NOISE_VERTICAL_SHIFT**: additive shift to the noise model before subtraction

# %%
def build_background_image(tth_map, tth_centers, radial_profile, bin_indices_2d=None):
    """Map a 1D radial profile back to the 2D detector.
    Returns background image of same shape as tth_map."""
    if bin_indices_2d is None:
        tth_flat = tth_map.ravel()
        idx = np.digitize(tth_flat, tth_edges) - 1
        idx = np.clip(idx, 0, len(radial_profile) - 1)
    else:
        idx = bin_indices_2d.ravel()
    bg = radial_profile[idx].reshape(tth_map.shape)
    return bg


def subtract_background(image, bg_image, strength=1.0, shift=0.0):
    """Subtract background from image.
    strength: 0=none, 1=full subtraction.
    shift: vertical offset added to bg before subtraction."""
    corrected = image - strength * (bg_image + shift)
    return corrected


# Pre-compute 2D bin indices (shared across all background images)
bin_indices_2d = np.digitize(tth.ravel(), tth_edges) - 1
bin_indices_2d = np.clip(bin_indices_2d, 0, n_tth_bins - 1)

# Select the primary models to compare
PRIMARY_MODELS = ['gaussian', 'split_gaussian', 'skewed_gaussian', 'fourier_0.05']
PRIMARY_MODELS = [m for m in PRIMARY_MODELS if m in fits]

# Build background images for each model (at 5x5-bin level)
bg_images = {}
corrected_sum = {}

# For the summed image, scale the 5x5-bin profile up
avg_frames_per_bin = np.mean([len(info['frames']) for info in bin_mapping.values()])
sum_scale = n_total_frames / avg_frames_per_bin

for model_key in PRIMARY_MODELS:
    profile = fits[model_key]['profile']
    bg = build_background_image(tth, tth_centers, profile, bin_indices_2d)
    bg_images[model_key] = bg

    # Summed image: scale up the 5x5-level background
    corrected_sum[model_key] = subtract_background(
        xrd_sum, bg * sum_scale,
        strength=NOISE_REDUCTION_STRENGTH,
        shift=NOISE_VERTICAL_SHIFT * sum_scale)

print(f'Built background images for: {PRIMARY_MODELS}')
print(f'Profile level: 5x5 bin (~{avg_frames_per_bin:.0f} frames)')
print(f'Scale factor for summed image: {sum_scale:.1f}x')
print(f'Noise reduction strength: {NOISE_REDUCTION_STRENGTH}')
print(f'Vertical shift: {NOISE_VERTICAL_SHIFT}')

# %%
# Side-by-side: original vs each noise-reduced version (full summed image)
n_models = len(PRIMARY_MODELS)
fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))

# Original
vmin_orig = np.percentile(xrd_sum, 50)
vmax_orig = np.percentile(xrd_sum, 99.5)

axes[0, 0].imshow(xrd_sum, cmap='inferno', vmin=vmin_orig, vmax=vmax_orig)
axes[0, 0].set_title('Original (summed)', fontsize=9)

# Background model in bottom row
axes[1, 0].imshow(xrd_sum, cmap='inferno', vmin=vmin_orig, vmax=vmax_orig)
axes[1, 0].set_title('Original (reference)', fontsize=9)

# Draw Ming's boxes on the original
for b in ming_boxes:
    rect = patches.Rectangle((b['x0'], b['y0']), b['x1']-b['x0'], b['y1']-b['y0'],
                             lw=0.8, edgecolor='cyan', facecolor='none')
    axes[0, 0].add_patch(rect)

for i, model_key in enumerate(PRIMARY_MODELS):
    corrected = corrected_sum[model_key]
    pos = corrected.copy()
    pos[pos < 0] = 0
    vmax_c = np.percentile(pos[pos > 0], 99.5) if (pos > 0).any() else 1

    axes[0, i+1].imshow(pos, cmap='inferno', vmin=0, vmax=vmax_c)
    axes[0, i+1].set_title(f'{model_key}\ncorrected', fontsize=9)

    # Show the background itself
    bg_vis = bg_images[model_key]
    axes[1, i+1].imshow(bg_vis, cmap='viridis')
    axes[1, i+1].set_title(f'{model_key}\nbackground model', fontsize=9)

    for b in ming_boxes:
        rect = patches.Rectangle((b['x0'], b['y0']), b['x1']-b['x0'], b['y1']-b['y0'],
                                 lw=0.8, edgecolor='cyan', facecolor='none')
        axes[0, i+1].add_patch(rect)

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f'Noise reduction comparison (strength={NOISE_REDUCTION_STRENGTH}, '
             f'shift={NOISE_VERTICAL_SHIFT})', fontsize=12)
plt.tight_layout()
plt.show()

# %%  [markdown]
# ---
# ## 5. Apply to Individual 5x5 Bins
# 
# Load a few sample 5x5 bin images and show before/after noise reduction.
# Since the noise model was fitted directly at the 5x5-bin intensity level,
# no rescaling is needed — subtract the background image directly.

# %%
# Pick sample bins spread across the scan (corners + center + random)
bin_keys_sorted = sorted(bin_mapping.keys())
n_bins_total = len(bin_keys_sorted)

sample_indices = [0, n_bins_total//4, n_bins_total//2, 3*n_bins_total//4, n_bins_total-1]
# Add a few more from the middle
rng = np.random.default_rng(42)
extra = rng.choice(range(n_bins_total), size=3, replace=False)
sample_indices = sorted(set(sample_indices + list(extra)))[:8]
sample_bins = [bin_keys_sorted[i] for i in sample_indices]

print(f'Loading {len(sample_bins)} sample bins for comparison...')

# Choose the best model for demonstration
best_model = min(PRIMARY_MODELS, key=lambda k: fits[k]['rmse'])
print(f'Using model: {best_model} (lowest RMSE)')

fig, axes = plt.subplots(len(sample_bins), 3, figsize=(12, 3.5 * len(sample_bins)))
if len(sample_bins) == 1:
    axes = axes[np.newaxis, :]

bg_profile = fits[best_model]['profile']

for row_idx, bk in enumerate(sample_bins):
    info = bin_mapping[bk]
    n_frames = len(info['frames'])

    t0 = time.time()
    summed = load_and_sum_frames(info['frames'], xrd_files, xrd_file_map)
    summed[summed < 0] = 0
    summed[summed > 1e9] = 0
    load_time = time.time() - t0

    # Background at 5x5-bin level — subtract directly, no scaling needed
    bg = build_background_image(tth, tth_centers, bg_profile, bin_indices_2d)
    corrected = subtract_background(summed, bg,
                                     strength=NOISE_REDUCTION_STRENGTH,
                                     shift=NOISE_VERTICAL_SHIFT)
    corrected_pos = np.clip(corrected, 0, None)

    # Plot: original | background | corrected
    finite_orig = summed[summed > 0]
    vmax_o = np.percentile(finite_orig, 99.5) if len(finite_orig) > 0 else 1

    axes[row_idx, 0].imshow(summed, cmap='inferno', vmin=0, vmax=vmax_o)
    axes[row_idx, 0].set_title(f'Bin ({bk[0]},{bk[1]}) original\n{n_frames} frames',
                                fontsize=8)

    axes[row_idx, 1].imshow(bg, cmap='viridis')
    axes[row_idx, 1].set_title(f'Background model\n(5x5-bin level)', fontsize=8)

    vmax_c = np.percentile(corrected_pos[corrected_pos > 0], 99.5) if (corrected_pos > 0).any() else 1
    axes[row_idx, 2].imshow(corrected_pos, cmap='inferno', vmin=0, vmax=vmax_c)
    axes[row_idx, 2].set_title(f'Noise-reduced\n({best_model})', fontsize=8)

    # Overlay Ming's boxes on original and corrected
    for b in ming_boxes:
        for col in [0, 2]:
            rect = patches.Rectangle((b['x0'], b['y0']),
                                     b['x1']-b['x0'], b['y1']-b['y0'],
                                     lw=0.6, edgecolor='cyan', facecolor='none')
            axes[row_idx, col].add_patch(rect)

    print(f'  Bin ({bk[0]},{bk[1]}): {n_frames} frames, loaded in {load_time:.1f}s')

for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f'5x5 bin noise reduction ({best_model}, strength={NOISE_REDUCTION_STRENGTH})',
             fontsize=12)
plt.tight_layout()
plt.show()

# %%  [markdown]
# ---
# ## 6. Peak Finding Algorithm
# 
# Detect hotspots on the noise-reduced images. The algorithm:
# 1. For each 2-theta arc (reflection family), mask the image to that arc
# 2. Threshold at a tunable percentile within the arc
# 3. Find connected components above threshold
# 4. Filter by minimum size and return bounding boxes
# 
# This runs on both the summed image and individual bins, then
# compares discoveries against Ming's 31.

# %%
def find_peaks_on_image(image, tth_map, degs, deg_labels, target_labels=None,
                         line_tol=0.3, percentile=97.0, min_pixels=3, pad=10,
                         ignore_edge=2):
    """Detect peaks along 2-theta arcs on a (possibly noise-reduced) image.
    Returns dict of {label: [(y0, y1, x0, x1, cx, cy, peak_intensity), ...]}"""
    if target_labels is None:
        target_labels = deg_labels

    label_to_deg = {lab: deg for lab, deg in zip(deg_labels, degs)}
    all_peaks = {}

    for lab in target_labels:
        if lab not in label_to_deg:
            continue

        d = label_to_deg[lab]
        line_mask = np.abs(tth_map - d) < line_tol

        if np.count_nonzero(line_mask) == 0:
            all_peaks[lab] = []
            continue

        # Threshold within the arc
        line_vals = image[line_mask]
        thr = np.percentile(line_vals, percentile)
        hotspot = line_mask & (image >= thr)

        # Edge cleanup
        if ignore_edge > 0:
            hotspot[:ignore_edge, :] = False
            hotspot[-ignore_edge:, :] = False
            hotspot[:, :ignore_edge] = False
            hotspot[:, -ignore_edge:] = False

        cc, n_comp = ndi.label(hotspot)
        peaks = []
        for comp_id in range(1, n_comp + 1):
            ys, xs = np.where(cc == comp_id)
            if len(ys) < min_pixels:
                continue
            y0 = max(int(ys.min()) - pad, 0)
            y1 = min(int(ys.max()) + pad + 1, image.shape[0])
            x0 = max(int(xs.min()) - pad, 0)
            x1 = min(int(xs.max()) + pad + 1, image.shape[1])
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            peak_val = float(image[ys, xs].max())
            peaks.append((y0, y1, x0, x1, cx, cy, peak_val))

        # Deduplicate
        peaks = sorted(peaks, key=lambda r: (r[0], r[2]))
        dedup = []
        for p in peaks:
            keep = True
            for q in dedup:
                if p[0] >= q[0] and p[1] <= q[1] and p[2] >= q[2] and p[3] <= q[3]:
                    keep = False
                    break
            if keep:
                dedup.append(p)

        all_peaks[lab] = dedup

    return all_peaks


def match_peaks_to_ming(detected_peaks, ming_boxes, tolerance=40):
    """Compare detected peaks against Ming's annotations.
    Returns (matched, new, missed) lists."""
    ming_centers = [(b['cx'], b['cy'], b['reflection']) for b in ming_boxes]

    all_detected = []
    for lab, peaks in detected_peaks.items():
        for p in peaks:
            all_detected.append({'label': lab, 'cx': p[4], 'cy': p[5],
                                  'box': p[:4], 'intensity': p[6]})

    matched = []
    new_peaks = []
    ming_matched = set()

    for det in all_detected:
        best_dist = float('inf')
        best_ming_idx = None
        for mi, (mx, my, mref) in enumerate(ming_centers):
            dist = np.sqrt((det['cx'] - mx)**2 + (det['cy'] - my)**2)
            if dist < best_dist:
                best_dist = dist
                best_ming_idx = mi

        if best_dist <= tolerance and best_ming_idx not in ming_matched:
            matched.append({'detected': det, 'ming': ming_boxes[best_ming_idx],
                            'distance': best_dist})
            ming_matched.add(best_ming_idx)
        else:
            new_peaks.append(det)

    missed = [ming_boxes[i] for i in range(len(ming_boxes)) if i not in ming_matched]

    return matched, new_peaks, missed


print('Peak finding functions defined')

# %%
# Run peak finding on the summed image: original vs noise-reduced
print('=' * 60)
print('Peak finding on summed image')
print('=' * 60)

# On original
peaks_original = find_peaks_on_image(
    xrd_sum, tth, DEGS, DEG_LABELS,
    percentile=PEAK_PERCENTILE, min_pixels=PEAK_MIN_PIXELS, pad=PEAK_PAD)

n_orig = sum(len(v) for v in peaks_original.values())
matched_orig, new_orig, missed_orig = match_peaks_to_ming(peaks_original, ming_boxes,
                                                            MATCH_TOLERANCE)
print(f'\nOriginal image: {n_orig} peaks detected')
print(f'  Matched to Ming: {len(matched_orig)}/{len(ming_boxes)}')
print(f'  New (not in Ming): {len(new_orig)}')
print(f'  Ming missed: {len(missed_orig)}')

# On each noise-reduced model
results_by_model = {}
for model_key in PRIMARY_MODELS:
    corrected = corrected_sum[model_key]
    corrected_pos = np.clip(corrected, 0, None)

    peaks = find_peaks_on_image(
        corrected_pos, tth, DEGS, DEG_LABELS,
        percentile=PEAK_PERCENTILE, min_pixels=PEAK_MIN_PIXELS, pad=PEAK_PAD)

    n_det = sum(len(v) for v in peaks.values())
    matched, new_p, missed = match_peaks_to_ming(peaks, ming_boxes, MATCH_TOLERANCE)

    results_by_model[model_key] = {
        'peaks': peaks, 'matched': matched,
        'new': new_p, 'missed': missed
    }

    print(f'\n{model_key}: {n_det} peaks detected')
    print(f'  Matched to Ming: {len(matched)}/{len(ming_boxes)}')
    print(f'  New (not in Ming): {len(new_p)}')
    print(f'  Ming missed: {len(missed)}')
    if new_p:
        print(f'  New peak locations:')
        for p in new_p[:10]:
            print(f'    {p["label"]:>8s} at ({p["cx"]}, {p["cy"]}) intensity={p["intensity"]:.0f}')

# %%
# Visualize: original vs best model, with Ming's boxes (cyan) and new peaks (yellow)
best_model_key = min(PRIMARY_MODELS,
                      key=lambda k: len(results_by_model[k]['missed']))
print(f'Best model (fewest Ming misses): {best_model_key}')

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Original with Ming's boxes
axes[0].imshow(xrd_sum, cmap='inferno',
               vmin=np.percentile(xrd_sum, 50),
               vmax=np.percentile(xrd_sum, 99.5))
for b in ming_boxes:
    rect = patches.Rectangle((b['x0'], b['y0']), b['x1']-b['x0'], b['y1']-b['y0'],
                             lw=1.2, edgecolor='cyan', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(b['x0'], b['y0']-3, b['reflection'], color='cyan',
                 fontsize=5, va='bottom')
axes[0].set_title(f'Original + Ming\'s {len(ming_boxes)} boxes', fontsize=10)

# Noise-reduced with detected peaks
corrected_vis = np.clip(corrected_sum[best_model_key], 0, None)
vmax_c = np.percentile(corrected_vis[corrected_vis > 0], 99.5) if (corrected_vis > 0).any() else 1
axes[1].imshow(corrected_vis, cmap='inferno', vmin=0, vmax=vmax_c)

result = results_by_model[best_model_key]

# Ming matches in cyan
for m in result['matched']:
    b = m['detected']['box']
    rect = patches.Rectangle((b[2], b[0]), b[3]-b[2], b[1]-b[0],
                             lw=1.2, edgecolor='cyan', facecolor='none')
    axes[1].add_patch(rect)

# New peaks in yellow
for p in result['new']:
    b = p['box']
    rect = patches.Rectangle((b[2], b[0]), b[3]-b[2], b[1]-b[0],
                             lw=1.5, edgecolor='yellow', facecolor='none', ls='--')
    axes[1].add_patch(rect)
    axes[1].text(b[2], b[0]-3, f'{p["label"]}*', color='yellow',
                 fontsize=6, va='bottom', fontweight='bold')

# Missed Ming peaks in red
for b in result['missed']:
    rect = patches.Rectangle((b['x0'], b['y0']), b['x1']-b['x0'], b['y1']-b['y0'],
                             lw=1.5, edgecolor='red', facecolor='none', ls=':')
    axes[1].add_patch(rect)

axes[1].set_title(f'{best_model_key}: {len(result["matched"])} matched (cyan) | '
                   f'{len(result["new"])} new (yellow) | '
                   f'{len(result["missed"])} missed (red)', fontsize=9)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# %%  [markdown]
# ---
# ## 7. Sensitivity Sweep
# 
# Sweep the peak detection percentile and noise reduction strength to
# find the settings that recover the most peaks while keeping false
# positives manageable.

# %%
# Sweep percentile thresholds on the best model's noise-reduced image
percentiles_to_test = [99.0, 98.0, 97.0, 96.0, 95.0, 93.0, 90.0]
strengths_to_test = [0.0, 0.5, 0.8, 1.0, 1.2]

print(f'Sweeping percentile x strength on {best_model_key}...')
print(f'{"Strength":>10s}  {"Pctile":>7s}  {"Detected":>9s}  '
      f'{"Matched":>8s}  {"New":>5s}  {"Missed":>7s}')
print('-' * 55)

sweep_results = []

for strength in strengths_to_test:
    # Summed image: scale bg up from 5x5-bin level
    corrected_sweep = subtract_background(
        xrd_sum, bg_images[best_model_key] * sum_scale,
        strength=strength, shift=NOISE_VERTICAL_SHIFT * sum_scale)
    corrected_sweep = np.clip(corrected_sweep, 0, None)

    for pct in percentiles_to_test:
        peaks = find_peaks_on_image(
            corrected_sweep, tth, DEGS, DEG_LABELS,
            percentile=pct, min_pixels=PEAK_MIN_PIXELS, pad=PEAK_PAD)

        n_det = sum(len(v) for v in peaks.values())
        matched, new_p, missed = match_peaks_to_ming(peaks, ming_boxes, MATCH_TOLERANCE)

        sweep_results.append({
            'strength': strength, 'percentile': pct,
            'detected': n_det, 'matched': len(matched),
            'new': len(new_p), 'missed': len(missed)
        })

        print(f'{strength:10.1f}  {pct:7.1f}  {n_det:9d}  '
              f'{len(matched):8d}  {len(new_p):5d}  {len(missed):7d}')

# Find best combo: maximize matched, minimize missed, keep new reasonable
best_combo = max(sweep_results,
                  key=lambda r: (r['matched'], -r['missed'], -r['new']))
print(f'\nRecommended: strength={best_combo["strength"]}, '
      f'percentile={best_combo["percentile"]} '
      f'({best_combo["matched"]} matched, {best_combo["new"]} new, '
      f'{best_combo["missed"]} missed)')

# %%
# Heatmap of matched count across the sweep grid
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric, title in zip(axes,
                              ['matched', 'new', 'missed'],
                              ['Matched (higher=better)',
                               'New discoveries',
                               'Missed (lower=better)']):
    grid = np.zeros((len(strengths_to_test), len(percentiles_to_test)))
    for r in sweep_results:
        si = strengths_to_test.index(r['strength'])
        pi = percentiles_to_test.index(r['percentile'])
        grid[si, pi] = r[metric]

    im = ax.imshow(grid, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(percentiles_to_test)))
    ax.set_xticklabels([f'{p:.0f}' for p in percentiles_to_test], fontsize=8)
    ax.set_yticks(range(len(strengths_to_test)))
    ax.set_yticklabels([f'{s:.1f}' for s in strengths_to_test], fontsize=8)
    ax.set_xlabel('Percentile threshold')
    ax.set_ylabel('Noise reduction strength')
    ax.set_title(title, fontsize=10)

    for si in range(len(strengths_to_test)):
        for pi in range(len(percentiles_to_test)):
            ax.text(pi, si, f'{int(grid[si, pi])}', ha='center', va='center',
                    fontsize=7, color='white' if grid[si, pi] < grid.max()*0.7 else 'black')

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('Parameter sweep: peak detection on summed image', fontsize=12)
plt.tight_layout()
plt.show()

# %%  [markdown]
# ---
# ## 8. Interactive 5x5 Bin Viewer
# 
# Browse each 5x5 bin image with:
# - Side-by-side original vs noise-reduced
# - Detected peaks overlaid (yellow = new, cyan = matched to Ming)
# - Sliders for noise reduction strength, vertical shift, and peak sensitivity
# 
# Requires `%matplotlib widget` for interactivity.

# %%
%matplotlib widget

import ipywidgets as widgets
from IPython.display import display

# Choose the model to use for the interactive viewer
VIEWER_MODEL = best_model_key
viewer_bg_profile = fits[VIEWER_MODEL]['profile']
bin_keys_viewer = sorted(bin_mapping.keys())
n_viewer_bins = len(bin_keys_viewer)

# State
viewer_pos = [0]

# Figure: 2 columns (original | corrected)
viewer_fig, viewer_axes = plt.subplots(1, 2, figsize=(14, 6))
viewer_fig.canvas.header_visible = False

placeholder = np.zeros((DET_H, DET_W))
im_orig = viewer_axes[0].imshow(placeholder, cmap='inferno', origin='upper')
im_corr = viewer_axes[1].imshow(placeholder, cmap='inferno', origin='upper')
title_orig = viewer_axes[0].set_title('', fontsize=9)
title_corr = viewer_axes[1].set_title('', fontsize=9)


def update_viewer(bin_idx, strength, shift, pctile):
    bk = bin_keys_viewer[bin_idx]
    info = bin_mapping[bk]
    n_frames = len(info['frames'])

    summed = load_and_sum_frames(info['frames'], xrd_files, xrd_file_map)
    summed[summed < 0] = 0
    summed[summed > 1e9] = 0

    # Background at 5x5-bin level — subtract directly
    bg = build_background_image(tth, tth_centers, viewer_bg_profile, bin_indices_2d)
    corrected = subtract_background(summed, bg, strength=strength, shift=shift)
    corrected_pos = np.clip(corrected, 0, None)

    # Update original image
    finite = summed[summed > 0]
    vmax_o = np.percentile(finite, 99.5) if len(finite) > 0 else 1
    im_orig.set_data(summed)
    im_orig.set_clim(0, vmax_o)

    # Update corrected image
    vmax_c = np.percentile(corrected_pos[corrected_pos > 0], 99.5) if (corrected_pos > 0).any() else 1
    im_corr.set_data(corrected_pos)
    im_corr.set_clim(0, vmax_c)

    # Clear old patches
    for ax in viewer_axes:
        for p in list(ax.patches):
            p.remove()
        for t in list(ax.texts):
            t.remove()

    # Run peak detection on corrected image
    peaks = find_peaks_on_image(
        corrected_pos, tth, DEGS, DEG_LABELS,
        percentile=pctile, min_pixels=PEAK_MIN_PIXELS, pad=PEAK_PAD)
    n_peaks = sum(len(v) for v in peaks.values())
    matched, new_p, missed = match_peaks_to_ming(peaks, ming_boxes, MATCH_TOLERANCE)

    # Draw Ming's boxes on original (cyan)
    for b in ming_boxes:
        rect = patches.Rectangle((b['x0'], b['y0']),
                                 b['x1']-b['x0'], b['y1']-b['y0'],
                                 lw=0.8, edgecolor='cyan', facecolor='none')
        viewer_axes[0].add_patch(rect)

    # Draw detected peaks on corrected
    for m in matched:
        b = m['detected']['box']
        rect = patches.Rectangle((b[2], b[0]), b[3]-b[2], b[1]-b[0],
                                 lw=1, edgecolor='cyan', facecolor='none')
        viewer_axes[1].add_patch(rect)

    for p in new_p:
        b = p['box']
        rect = patches.Rectangle((b[2], b[0]), b[3]-b[2], b[1]-b[0],
                                 lw=1.2, edgecolor='yellow', facecolor='none', ls='--')
        viewer_axes[1].add_patch(rect)
        viewer_axes[1].text(b[2], b[0]-2, f'{p["label"]}*', color='yellow',
                            fontsize=6, va='bottom', fontweight='bold')

    title_orig.set_text(f'Original | Bin ({bk[0]},{bk[1]}) | {n_frames} frames')
    title_corr.set_text(f'Noise-reduced | {n_peaks} peaks ({len(matched)} match, '
                         f'{len(new_p)} new)')

    viewer_fig.canvas.draw_idle()


# Widgets
w_strength = widgets.FloatSlider(
    value=1.0, min=0.0, max=2.0, step=0.05,
    description='Strength:', layout=widgets.Layout(width='350px'),
    style={'description_width': '90px'})

w_shift = widgets.FloatSlider(
    value=0.0, min=-500.0, max=500.0, step=10.0,
    description='V-shift:', layout=widgets.Layout(width='350px'),
    style={'description_width': '90px'})

w_pctile = widgets.FloatSlider(
    value=97.0, min=85.0, max=99.9, step=0.5,
    description='Peak pctile:', layout=widgets.Layout(width='350px'),
    style={'description_width': '90px'})

w_bin_slider = widgets.IntSlider(
    value=0, min=0, max=n_viewer_bins - 1, step=1,
    description='Bin #:', layout=widgets.Layout(width='350px'),
    style={'description_width': '90px'})

w_prev = widgets.Button(description='<< Prev', layout=widgets.Layout(width='80px'))
w_next = widgets.Button(description='Next >>', layout=widgets.Layout(width='80px'))
w_apply = widgets.Button(description='Apply', button_style='primary',
                          layout=widgets.Layout(width='80px'))
w_info = widgets.Label(value='')


def on_apply(_):
    update_viewer(w_bin_slider.value, w_strength.value,
                  w_shift.value, w_pctile.value)
    bk = bin_keys_viewer[w_bin_slider.value]
    w_info.value = (f'Bin {w_bin_slider.value+1}/{n_viewer_bins} '
                    f'({bk[0]},{bk[1]}) | '
                    f's={w_strength.value:.2f} shift={w_shift.value:.0f} '
                    f'pct={w_pctile.value:.1f}')


def on_prev(_):
    if w_bin_slider.value > 0:
        w_bin_slider.value -= 1
        on_apply(None)


def on_next(_):
    if w_bin_slider.value < n_viewer_bins - 1:
        w_bin_slider.value += 1
        on_apply(None)


w_prev.on_click(on_prev)
w_next.on_click(on_next)
w_apply.on_click(on_apply)

# Layout
sliders = widgets.VBox([w_strength, w_shift, w_pctile, w_bin_slider])
nav = widgets.HBox([w_prev, w_next, w_apply, w_info])
controls = widgets.VBox([nav, sliders])

display(widgets.VBox([viewer_fig.canvas, controls]))

# Initial display
on_apply(None)

# %%  [markdown]
# ---
# ## 9. Run Peak Finding on All 5x5 Bins
# 
# After tuning the parameters above, apply noise reduction and peak
# finding across all bins. This produces a per-bin peak list that can
# feed into the ground truth builder.

# %%
%matplotlib inline

# Set the tuned parameters here (update after using the interactive viewer)
TUNED_STRENGTH = 1.0
TUNED_SHIFT = 0.0
TUNED_PERCENTILE = 97.0
TUNED_MODEL = best_model_key

tuned_bg_profile = fits[TUNED_MODEL]['profile']

print(f'Running peak finding on all {len(bin_mapping)} bins')
print(f'Model: {TUNED_MODEL}, strength: {TUNED_STRENGTH}, '
      f'shift: {TUNED_SHIFT}, percentile: {TUNED_PERCENTILE}')
print()

all_bin_peaks = {}
peak_counts_per_bin = []
new_peak_counts = []

t0 = time.time()
last_print = t0

for idx, bk in enumerate(bin_keys_viewer):
    info = bin_mapping[bk]
    n_frames = len(info['frames'])

    summed = load_and_sum_frames(info['frames'], xrd_files, xrd_file_map)
    summed[summed < 0] = 0
    summed[summed > 1e9] = 0

    # Background at 5x5-bin level — subtract directly
    bg = build_background_image(tth, tth_centers, tuned_bg_profile, bin_indices_2d)
    corrected = subtract_background(summed, bg, strength=TUNED_STRENGTH,
                                     shift=TUNED_SHIFT)
    corrected_pos = np.clip(corrected, 0, None)

    peaks = find_peaks_on_image(
        corrected_pos, tth, DEGS, DEG_LABELS,
        percentile=TUNED_PERCENTILE, min_pixels=PEAK_MIN_PIXELS, pad=PEAK_PAD)

    n_peaks = sum(len(v) for v in peaks.values())
    _, new_p, _ = match_peaks_to_ming(peaks, ming_boxes, MATCH_TOLERANCE)

    all_bin_peaks[bk] = peaks
    peak_counts_per_bin.append(n_peaks)
    new_peak_counts.append(len(new_p))

    now = time.time()
    if (idx + 1) == len(bin_keys_viewer) or (now - last_print) > 15 or (idx + 1) % max(1, len(bin_keys_viewer)//10) == 0:
        elapsed = now - t0
        rate = (idx + 1) / elapsed
        eta = (len(bin_keys_viewer) - idx - 1) / rate
        pct = 100 * (idx + 1) / len(bin_keys_viewer)
        print(f'  [{pct:5.1f}%] {idx+1}/{len(bin_keys_viewer)} bins | '
              f'{n_peaks} peaks | {rate:.1f} bins/s | ETA {eta:.0f}s')
        last_print = now

total_time = time.time() - t0
print(f'\nDone in {total_time:.0f}s ({total_time/60:.1f} min)')
print(f'Peaks per bin: min={min(peak_counts_per_bin)}, '
      f'max={max(peak_counts_per_bin)}, '
      f'mean={np.mean(peak_counts_per_bin):.1f}, '
      f'median={np.median(peak_counts_per_bin):.0f}')
print(f'New peaks (not in Ming): total={sum(new_peak_counts)}, '
      f'bins with new={sum(1 for c in new_peak_counts if c > 0)}')

# %%
# Spatial map of peak counts and new discoveries
peak_count_map = np.full((n_bin_rows, n_bin_cols), np.nan)
new_count_map = np.full((n_bin_rows, n_bin_cols), np.nan)

for idx, bk in enumerate(bin_keys_viewer):
    br, bc = bk
    peak_count_map[br, bc] = peak_counts_per_bin[idx]
    new_count_map[br, bc] = new_peak_counts[idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(peak_count_map, cmap='viridis', interpolation='nearest', aspect='auto')
axes[0].set_title('Total peaks per 5x5 bin', fontsize=10)
axes[0].set_xlabel('bin col')
axes[0].set_ylabel('bin row')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(new_count_map, cmap='YlOrRd', interpolation='nearest', aspect='auto')
axes[1].set_title('New peaks (not in Ming) per bin', fontsize=10)
axes[1].set_xlabel('bin col')
axes[1].set_ylabel('bin row')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

plt.suptitle(f'Peak finding results ({TUNED_MODEL}, pctile={TUNED_PERCENTILE})', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Aggregate: which new peak locations appear across many bins?
# Cluster new peaks by detector position to find persistent new reflections

all_new_peaks = []
for idx, bk in enumerate(bin_keys_viewer):
    peaks = all_bin_peaks[bk]
    _, new_p, _ = match_peaks_to_ming(peaks, ming_boxes, MATCH_TOLERANCE)
    for p in new_p:
        all_new_peaks.append({
            'bin': bk, 'label': p['label'],
            'cx': p['cx'], 'cy': p['cy'],
            'intensity': p['intensity']
        })

print(f'Total new peak instances across all bins: {len(all_new_peaks)}')

if all_new_peaks:
    # Simple spatial clustering: group by proximity on the detector
    new_coords = np.array([[p['cx'], p['cy']] for p in all_new_peaks])

    # Agglomerative clustering by distance
    from scipy.cluster.hierarchy import fcluster, linkage

    if len(new_coords) > 1:
        Z = linkage(new_coords, method='average')
        cluster_ids = fcluster(Z, t=MATCH_TOLERANCE, criterion='distance')
    else:
        cluster_ids = np.array([1])

    n_clusters = cluster_ids.max()
    print(f'Clustered into {n_clusters} unique detector positions')
    print()

    print(f'{"Cluster":>8s}  {"Count":>6s}  {"Median cx":>10s}  {"Median cy":>10s}  '
          f'{"Label":>8s}  {"Med Intensity":>14s}')
    print('-' * 65)

    cluster_summary = []
    for ci in range(1, n_clusters + 1):
        mask = cluster_ids == ci
        members = [all_new_peaks[j] for j in range(len(all_new_peaks)) if mask[j]]
        cx_med = int(np.median([m['cx'] for m in members]))
        cy_med = int(np.median([m['cy'] for m in members]))
        labels = [m['label'] for m in members]
        most_common = max(set(labels), key=labels.count)
        med_int = np.median([m['intensity'] for m in members])

        cluster_summary.append({
            'cluster': ci, 'count': len(members),
            'cx': cx_med, 'cy': cy_med,
            'label': most_common, 'intensity': med_int
        })

        print(f'{ci:8d}  {len(members):6d}  {cx_med:10d}  {cy_med:10d}  '
              f'{most_common:>8s}  {med_int:14.1f}')

    # Sort by frequency (most common = most likely real)
    cluster_summary.sort(key=lambda x: x['count'], reverse=True)
    print(f'\nMost persistent new peaks (appear in many bins):')
    for cs in cluster_summary[:10]:
        print(f'  {cs["label"]} at ({cs["cx"]}, {cs["cy"]}): '
              f'seen in {cs["count"]}/{len(bin_keys_viewer)} bins '
              f'({100*cs["count"]/len(bin_keys_viewer):.0f}%)')

# %%
# Visualize the persistent new peaks on the summed image
if all_new_peaks and cluster_summary:
    fig, ax = plt.subplots(figsize=(10, 10))

    corrected_best = np.clip(corrected_sum[TUNED_MODEL], 0, None)
    vmax = np.percentile(corrected_best[corrected_best > 0], 99.5)
    ax.imshow(corrected_best, cmap='inferno', vmin=0, vmax=vmax)

    # Ming's boxes in cyan
    for b in ming_boxes:
        rect = patches.Rectangle((b['x0'], b['y0']),
                                 b['x1']-b['x0'], b['y1']-b['y0'],
                                 lw=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

    # Persistent new peaks in yellow, size proportional to frequency
    min_count = 5  # only show peaks that appear in at least this many bins
    persistent = [cs for cs in cluster_summary if cs['count'] >= min_count]

    for cs in persistent:
        size = max(15, min(40, cs['count'] // 2))
        circle = plt.Circle((cs['cx'], cs['cy']), size,
                            fill=False, edgecolor='yellow', lw=2, ls='--')
        ax.add_patch(circle)
        ax.text(cs['cx'] + size + 3, cs['cy'],
                f'{cs["label"]} ({cs["count"]}x)',
                color='yellow', fontsize=7, va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    ax.set_title(f'Persistent new peaks (in {min_count}+ bins): '
                  f'{len(persistent)} found | '
                  f'cyan=Ming, yellow=new', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    print(f'\n{len(persistent)} persistent new peaks shown (threshold: {min_count}+ bins)')
else:
    print('No new peaks found')

# %%
# Export: save the noise model and discovered peaks for use in ground_truth_builder

export_path = OUTPUT_DIR / 'noise_model_and_peaks.json'

export_data = {
    'noise_model': {
        'model_type': TUNED_MODEL,
        'tth_centers': tth_centers.tolist(),
        'radial_profile': fits[TUNED_MODEL]['profile'].tolist(),
        'rmse': float(fits[TUNED_MODEL]['rmse']),
        'strength': TUNED_STRENGTH,
        'shift': TUNED_SHIFT,
        'n_total_frames': n_total_frames,
    },
    'peak_finding': {
        'percentile': TUNED_PERCENTILE,
        'min_pixels': PEAK_MIN_PIXELS,
        'pad': PEAK_PAD,
        'match_tolerance': MATCH_TOLERANCE,
    },
    'persistent_new_peaks': [
        {'label': cs['label'], 'cx': cs['cx'], 'cy': cs['cy'],
         'count': cs['count'], 'intensity': float(cs['intensity'])}
        for cs in cluster_summary
        if cs['count'] >= 5
    ] if all_new_peaks and cluster_summary else [],
}

with open(export_path, 'w') as f:
    json.dump(export_data, f, indent=2)

size_kb = export_path.stat().st_size / 1024
print(f'Saved: {export_path.name} ({size_kb:.1f} KB)')
print(f'  Noise model: {TUNED_MODEL}')
print(f'  Persistent new peaks: {len(export_data["persistent_new_peaks"])}')

# %%  [markdown]
# ---
# ## 10. Noise Model Grading Metrics
# 
# Three metrics to score how well a noise model performs on each 5×5 bin:
# 
# 1. **Negative Residual Penalty (NRP)** — overfitting detector. After subtracting the
#    noise model, negative pixels indicate the model over-subtracted. Score = mean of
#    negative residuals (more negative = worse overfitting). We want to minimize this.
# 
# 2. **Background Flatness (BF)** — median absolute deviation (MAD) of pixels in the
#    cleaned image, excluding detected peaks. A good noise model leaves a flat background.
#    Lower MAD = flatter = better.
# 
# 3. **Peak-to-Background Contrast (PBC)** — ratio of median peak intensity to median
#    background intensity in the cleaned image. Higher contrast = peaks stand out more
#    from the noise floor = better.

# %%
def compute_peak_mask(image, tth_map, degs, deg_labels, line_tol=0.3, percentile=97.0, min_pixels=3):
    """Binary mask of detected peak regions on the image."""
    mask = np.zeros(image.shape, dtype=bool)
    label_to_deg = {lab: deg for lab, deg in zip(deg_labels, degs)}
    for lab in deg_labels:
        if lab not in label_to_deg:
            continue
        d = label_to_deg[lab]
        line_mask = np.abs(tth_map - d) < line_tol
        if np.count_nonzero(line_mask) == 0:
            continue
        line_vals = image[line_mask]
        thr = np.percentile(line_vals, percentile)
        hotspot = line_mask & (image >= thr)
        cc, n_comp = ndi.label(hotspot)
        for comp_id in range(1, n_comp + 1):
            if np.count_nonzero(cc == comp_id) >= min_pixels:
                mask[cc == comp_id] = True
    return mask


def grade_negative_residual_penalty(residual_image):
    """Mean of negative pixels. More negative = worse overfitting."""
    neg = residual_image[residual_image < 0]
    if len(neg) == 0:
        return 0.0
    return float(np.mean(neg))


def grade_background_flatness(cleaned_image, peak_mask):
    """MAD of non-peak pixels. Lower = flatter background."""
    bg_pixels = cleaned_image[~peak_mask]
    if len(bg_pixels) == 0:
        return np.nan
    med = np.median(bg_pixels)
    return float(np.median(np.abs(bg_pixels - med)))


def grade_peak_to_background_contrast(cleaned_image, peak_mask):
    """Median peak intensity / median background intensity. Higher = better."""
    peak_pixels = cleaned_image[peak_mask]
    bg_pixels = cleaned_image[~peak_mask]
    if len(peak_pixels) == 0 or len(bg_pixels) == 0:
        return np.nan
    med_bg = np.median(bg_pixels)
    med_peak = np.median(peak_pixels)
    if med_bg <= 0:
        med_bg = max(np.percentile(bg_pixels, 75), 1e-6)
    return float(med_peak / med_bg)


def grade_noise_model(original, cleaned, peak_mask):
    """Compute all three grades for a noise-reduced image."""
    residual = cleaned  # cleaned = original - noise_model, so negative = overfit
    nrp = grade_negative_residual_penalty(residual)
    bf = grade_background_flatness(np.clip(cleaned, 0, None), peak_mask)
    pbc = grade_peak_to_background_contrast(np.clip(cleaned, 0, None), peak_mask)
    return {'negative_residual_penalty': nrp, 'background_flatness': bf,
            'peak_to_background_contrast': pbc}


print('Grading functions defined')
print('  NRP: negative residual penalty (minimize, closer to 0 = less overfitting)')
print('  BF:  background flatness / MAD (minimize, lower = flatter)')
print('  PBC: peak-to-background contrast (maximize, higher = clearer peaks)')

# %%  [markdown]
# ---
# ## 11. Per-Image Noise Optimization & Grading
# 
# For each 5×5 bin image:
# 1. Compute its own radial profile
# 2. Fit each noise model (Gaussian, split Gaussian, skewed Gaussian, Fourier)
#    with per-image optimized parameters
# 3. Score each fit using the three grading metrics
# 4. Select the best model per image
# 5. Save all fitted parameters to a JSON file for reproducibility

# %%
def compute_radial_profile(image, bin_indices, n_tth_bins):
    """Compute the radial median profile for a single image."""
    img_flat = image.ravel()
    profile = np.zeros(n_tth_bins)
    for i in range(n_tth_bins):
        mask = bin_indices == i
        vals = img_flat[mask]
        if len(vals) > 0:
            profile[i] = np.median(vals)
    return profile


def fit_all_models_to_profile(x_data, y_data, valid_mask, tth_min, tth_max):
    """Fit all four noise models to a radial profile. Returns dict of results."""
    x_fit = x_data[valid_mask]
    y_fit = y_data[valid_mask]

    if len(x_fit) < 5:
        return {}

    peak_idx = np.argmax(y_fit)
    mu_init = x_fit[peak_idx]
    amp_init = y_fit[peak_idx] - np.percentile(y_fit, 5)
    sigma_init = 3.0
    offset_init = np.percentile(y_fit, 5)

    results = {}

    # Gaussian
    try:
        popt, _ = curve_fit(gaussian_model, x_fit, y_fit,
                            p0=[amp_init, mu_init, sigma_init, offset_init],
                            maxfev=10000)
        fitted = gaussian_model(x_data, *popt)
        results['gaussian'] = {
            'profile': fitted,
            'params': {'amplitude': popt[0], 'mu': popt[1],
                       'sigma': popt[2], 'offset': popt[3]},
            'param_array': popt.tolist()
        }
    except Exception:
        pass

    # Split Gaussian
    try:
        popt, _ = curve_fit(split_gaussian_model, x_fit, y_fit,
                            p0=[amp_init, mu_init, sigma_init, sigma_init, offset_init],
                            maxfev=10000)
        fitted = split_gaussian_model(x_data, *popt)
        results['split_gaussian'] = {
            'profile': fitted,
            'params': {'amplitude': popt[0], 'mu': popt[1],
                       'sigma_left': popt[2], 'sigma_right': popt[3], 'offset': popt[4]},
            'param_array': popt.tolist()
        }
    except Exception:
        pass

    # Skewed Gaussian
    try:
        popt, _ = curve_fit(skewed_gaussian_model, x_fit, y_fit,
                            p0=[amp_init, mu_init, sigma_init, 0.0, offset_init],
                            maxfev=10000,
                            bounds=([0, tth_min, 0.1, -20, 0],
                                    [np.inf, tth_max, 30, 20, np.inf]))
        fitted = skewed_gaussian_model(x_data, *popt)
        results['skewed_gaussian'] = {
            'profile': fitted,
            'params': {'amplitude': popt[0], 'mu': popt[1],
                       'sigma': popt[2], 'skew': popt[3], 'offset': popt[4]},
            'param_array': popt.tolist()
        }
    except Exception:
        pass

    # Fourier low-pass (optimize cutoff from a few candidates)
    best_fourier = None
    best_fourier_rmse = np.inf
    for cutoff in [0.02, 0.03, 0.05, 0.07, 0.10]:
        fitted = fourier_lowpass(y_data, cutoff_fraction=cutoff)
        rmse = np.sqrt(np.mean((y_data[valid_mask] - fitted[valid_mask])**2))
        if rmse < best_fourier_rmse:
            best_fourier_rmse = rmse
            best_fourier = {'profile': fitted, 'cutoff': cutoff}
    if best_fourier is not None:
        results['fourier'] = {
            'profile': best_fourier['profile'],
            'params': {'cutoff': best_fourier['cutoff']},
            'param_array': [best_fourier['cutoff']]
        }

    return results


def optimize_strength_for_model(original, bg_image, tth_map, strengths=None):
    """Find the noise reduction strength that minimizes NRP while keeping it near zero."""
    if strengths is None:
        strengths = np.arange(0.5, 1.55, 0.05)
    best_strength = 1.0
    best_score = np.inf
    for s in strengths:
        cleaned = original - s * bg_image
        nrp = grade_negative_residual_penalty(cleaned)
        # We want NRP close to zero (not too negative = overfit, not too positive = underfit)
        # Penalize negative more heavily than positive
        score = abs(nrp) + (2.0 * abs(nrp) if nrp < 0 else 0)
        if score < best_score:
            best_score = score
            best_strength = s
    return float(best_strength)


print('Per-image optimization functions defined')

# %%
# Run per-image optimization across all 5x5 bins
PARAMS_OUTPUT = OUTPUT_DIR / 'per_bin_noise_params.json'

print(f'Optimizing noise models for all {len(bin_mapping)} bins...')
print(f'{"Bin":>12s}  {"Model":>16s}  {"Strength":>9s}  {"NRP":>9s}  {"BF":>9s}  {"PBC":>9s}')
print('-' * 75)

valid_mask = radial_count > 50
all_bin_results = {}
grade_rows = []

t0 = time.time()
last_print = t0

for idx, bk in enumerate(bin_keys_viewer):
    info = bin_mapping[bk]

    # Load image
    summed = load_and_sum_frames(info['frames'], xrd_files, xrd_file_map)
    summed[summed < 0] = 0
    summed[summed > 1e9] = 0

    # Compute per-image radial profile
    profile = compute_radial_profile(summed, bin_indices_2d, n_tth_bins)

    # Fit all models to this image's profile
    model_fits = fit_all_models_to_profile(
        tth_centers, profile, valid_mask, tth_min, tth_max)

    if not model_fits:
        all_bin_results[f'{bk[0]}_{bk[1]}'] = {'error': 'no fits converged'}
        continue

    # For each model: build bg, optimize strength, grade
    best_model_key = None
    best_composite = -np.inf
    bin_result = {}

    for mkey, mfit in model_fits.items():
        bg = build_background_image(tth, tth_centers, mfit['profile'], bin_indices_2d)
        opt_strength = optimize_strength_for_model(summed, bg, tth)
        cleaned = summed - opt_strength * bg
        cleaned_pos = np.clip(cleaned, 0, None)

        peak_mask = compute_peak_mask(cleaned_pos, tth, DEGS, DEG_LABELS,
                                       percentile=TUNED_PERCENTILE, min_pixels=PEAK_MIN_PIXELS)
        grades = grade_noise_model(summed, cleaned, peak_mask)
        grades['strength'] = opt_strength

        bin_result[mkey] = {
            'params': mfit['params'] if isinstance(mfit['params'], dict)
                      else {'values': mfit['params']},
            'param_array': mfit['param_array'],
            'strength': opt_strength,
            'grades': grades
        }

        # Composite score: maximize PBC, minimize BF, minimize |NRP|
        # Normalize: PBC is good positive, BF and NRP are bad
        pbc = grades['peak_to_background_contrast'] if not np.isnan(grades['peak_to_background_contrast']) else 0
        bf = grades['background_flatness'] if not np.isnan(grades['background_flatness']) else 1e6
        nrp = grades['negative_residual_penalty']
        composite = pbc - 0.01 * bf - 0.1 * abs(nrp)

        if composite > best_composite:
            best_composite = composite
            best_model_key = mkey

    bin_result['best_model'] = best_model_key
    all_bin_results[f'{bk[0]}_{bk[1]}'] = bin_result

    best_grades = bin_result[best_model_key]['grades']
    grade_rows.append({
        'bin': bk, 'model': best_model_key,
        'strength': bin_result[best_model_key]['strength'],
        'nrp': best_grades['negative_residual_penalty'],
        'bf': best_grades['background_flatness'],
        'pbc': best_grades['peak_to_background_contrast']
    })

    now = time.time()
    if (idx + 1) == len(bin_keys_viewer) or (now - last_print) > 20 or \
       (idx + 1) % max(1, len(bin_keys_viewer) // 10) == 0:
        elapsed = now - t0
        rate = (idx + 1) / elapsed
        eta = (len(bin_keys_viewer) - idx - 1) / rate
        g = best_grades
        print(f'({bk[0]:2d},{bk[1]:2d}) {idx+1:>4d}  {best_model_key:>16s}  '
              f'{bin_result[best_model_key]["strength"]:9.2f}  '
              f'{g["negative_residual_penalty"]:9.2f}  '
              f'{g["background_flatness"]:9.2f}  '
              f'{g["peak_to_background_contrast"]:9.2f}  '
              f'[{100*(idx+1)/len(bin_keys_viewer):.0f}% ETA {eta:.0f}s]')
        last_print = now

total_time = time.time() - t0
print(f'\nOptimization complete: {total_time:.0f}s ({total_time/60:.1f} min)')

# Save parameters
with open(PARAMS_OUTPUT, 'w') as f:
    json.dump(all_bin_results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

size_kb = PARAMS_OUTPUT.stat().st_size / 1024
print(f'Saved: {PARAMS_OUTPUT.name} ({size_kb:.1f} KB)')
print(f'  Bins processed: {len(grade_rows)}/{len(bin_keys_viewer)}')

# %%
# Grade summary statistics and spatial maps
if grade_rows:
    nrps = [r['nrp'] for r in grade_rows]
    bfs = [r['bf'] for r in grade_rows]
    pbcs = [r['pbc'] for r in grade_rows if not np.isnan(r['pbc'])]
    models_chosen = [r['model'] for r in grade_rows]

    print('=== Grade Summary Across All Bins ===')
    print(f'NRP  (neg residual penalty): mean={np.mean(nrps):.2f}, '
          f'median={np.median(nrps):.2f}, std={np.std(nrps):.2f}')
    print(f'BF   (background flatness):  mean={np.mean(bfs):.2f}, '
          f'median={np.median(bfs):.2f}, std={np.std(bfs):.2f}')
    print(f'PBC  (peak/bg contrast):     mean={np.mean(pbcs):.2f}, '
          f'median={np.median(pbcs):.2f}, std={np.std(pbcs):.2f}')
    print()

    from collections import Counter
    model_counts = Counter(models_chosen)
    print('Best model distribution:')
    for m, c in model_counts.most_common():
        print(f'  {m:>20s}: {c:4d} bins ({100*c/len(models_chosen):.1f}%)')

    # Spatial maps
    nrp_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    bf_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    pbc_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    model_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    model_names_list = sorted(model_counts.keys())
    model_to_idx = {m: i for i, m in enumerate(model_names_list)}

    for r in grade_rows:
        br, bc = r['bin']
        nrp_map[br, bc] = r['nrp']
        bf_map[br, bc] = r['bf']
        pbc_map[br, bc] = r['pbc']
        model_map[br, bc] = model_to_idx[r['model']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im0 = axes[0, 0].imshow(nrp_map, cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axes[0, 0].set_title('Negative Residual Penalty\n(closer to 0 = better)', fontsize=9)
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(bf_map, cmap='viridis_r', interpolation='nearest', aspect='auto')
    axes[0, 1].set_title('Background Flatness (MAD)\n(lower = better)', fontsize=9)
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[1, 0].imshow(pbc_map, cmap='plasma', interpolation='nearest', aspect='auto')
    axes[1, 0].set_title('Peak-to-Background Contrast\n(higher = better)', fontsize=9)
    plt.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    cmap_model = plt.cm.get_cmap('Set1', len(model_names_list))
    im3 = axes[1, 1].imshow(model_map, cmap=cmap_model, interpolation='nearest',
                             aspect='auto', vmin=-0.5, vmax=len(model_names_list)-0.5)
    axes[1, 1].set_title('Best Model Per Bin', fontsize=9)
    cbar = plt.colorbar(im3, ax=axes[1, 1], shrink=0.8,
                         ticks=range(len(model_names_list)))
    cbar.ax.set_yticklabels(model_names_list, fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel('bin col')
        ax.set_ylabel('bin row')

    plt.suptitle('Per-Bin Noise Model Grades', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Histogram of grades
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(nrps, bins=40, color='steelblue', edgecolor='white', lw=0.5)
    axes[0].axvline(0, color='red', ls='--', lw=1)
    axes[0].set_xlabel('Negative Residual Penalty')
    axes[0].set_ylabel('Count')
    axes[0].set_title('NRP distribution')

    axes[1].hist(bfs, bins=40, color='teal', edgecolor='white', lw=0.5)
    axes[1].set_xlabel('Background Flatness (MAD)')
    axes[1].set_title('BF distribution')

    axes[2].hist(pbcs, bins=40, color='darkorange', edgecolor='white', lw=0.5)
    axes[2].set_xlabel('Peak-to-Background Contrast')
    axes[2].set_title('PBC distribution')

    plt.tight_layout()
    plt.show()

# %%  [markdown]
# ---
# ## 12. Shielded Optimization — Peak-Suppressed Noise Fitting
# 
# The noise model fitting above can be biased by real diffraction peaks:
# the optimizer sees peaks as part of the "noise" and tries to fit through them,
# distorting the background shape.
# 
# **Shielded optimization** addresses this by:
# 1. Suppressing peaks before fitting — pixels above a tunable percentile threshold
#    within each 2-theta arc are replaced with the arc median
# 2. Fitting the noise model to this peak-suppressed radial profile
# 3. Jointly optimizing the percentile filter threshold alongside the model parameters
# 4. Grading the result on the *original* (un-suppressed) image
# 
# This produces a cleaner noise model that's not pulled toward peak locations.

# %%
def suppress_peaks_in_image(image, tth_map, tth_edges, bin_indices, n_tth_bins,
                           suppress_percentile=90.0):
    """Replace bright pixels (above percentile per 2-theta arc) with the arc median.
    Returns a peak-suppressed copy of the image."""
    suppressed = image.copy().ravel()
    img_flat = image.ravel()
    for i in range(n_tth_bins):
        mask = bin_indices == i
        vals = img_flat[mask]
        if len(vals) == 0:
            continue
        med = np.median(vals)
        thr = np.percentile(vals, suppress_percentile)
        bright = mask & (img_flat > thr)
        suppressed[bright] = med
    return suppressed.reshape(image.shape)


def compute_suppressed_radial_profile(image, bin_indices, n_tth_bins, suppress_percentile=90.0):
    """Compute radial profile from a peak-suppressed version of the image."""
    img_flat = image.ravel()
    profile = np.zeros(n_tth_bins)
    for i in range(n_tth_bins):
        mask = bin_indices == i
        vals = img_flat[mask]
        if len(vals) == 0:
            continue
        thr = np.percentile(vals, suppress_percentile)
        filtered = vals[vals <= thr]
        if len(filtered) == 0:
            filtered = vals
        profile[i] = np.median(filtered)
    return profile


def shielded_optimize(image, tth_centers, bin_indices_2d, n_tth_bins,
                      valid_mask, tth_min, tth_max, tth_map, tth_edges,
                      suppress_percentiles=None):
    """Jointly optimize suppress_percentile and noise model parameters.
    For each suppress_percentile, fit all models to the suppressed profile,
    then grade on the original image. Returns best combo."""
    if suppress_percentiles is None:
        suppress_percentiles = [80, 85, 88, 90, 92, 95, 97]

    best_result = None
    best_composite = -np.inf

    for sp in suppress_percentiles:
        # Compute radial profile with peaks suppressed
        profile = compute_suppressed_radial_profile(
            image, bin_indices_2d, n_tth_bins, suppress_percentile=sp)

        # Fit all noise models to this suppressed profile
        model_fits = fit_all_models_to_profile(
            tth_centers, profile, valid_mask, tth_min, tth_max)

        if not model_fits:
            continue

        for mkey, mfit in model_fits.items():
            bg = build_background_image(tth_map, tth_centers, mfit['profile'], bin_indices_2d)

            # Optimize strength on the ORIGINAL image
            opt_strength = optimize_strength_for_model(image, bg, tth_map)
            cleaned = image - opt_strength * bg
            cleaned_pos = np.clip(cleaned, 0, None)

            peak_mask = compute_peak_mask(cleaned_pos, tth_map, DEGS, DEG_LABELS,
                                           percentile=TUNED_PERCENTILE,
                                           min_pixels=PEAK_MIN_PIXELS)
            grades = grade_noise_model(image, cleaned, peak_mask)

            pbc = grades['peak_to_background_contrast'] if not np.isnan(grades['peak_to_background_contrast']) else 0
            bf = grades['background_flatness'] if not np.isnan(grades['background_flatness']) else 1e6
            nrp = grades['negative_residual_penalty']
            composite = pbc - 0.01 * bf - 0.1 * abs(nrp)

            if composite > best_composite:
                best_composite = composite
                best_result = {
                    'model': mkey,
                    'suppress_percentile': sp,
                    'strength': opt_strength,
                    'params': mfit['params'] if isinstance(mfit['params'], dict)
                              else {'values': mfit['params']},
                    'param_array': mfit['param_array'],
                    'grades': grades,
                    'composite': composite,
                    'profile': mfit['profile'].tolist()
                }

    return best_result


print('Shielded optimization functions defined')

# %%
# Run shielded optimization on all 5x5 bins
SHIELDED_OUTPUT = OUTPUT_DIR / 'per_bin_shielded_params.json'

print(f'Running shielded optimization on all {len(bin_mapping)} bins...')
print(f'Testing suppress percentiles: [80, 85, 88, 90, 92, 95, 97]')
print()
print(f'{"Bin":>12s}  {"Model":>16s}  {"Sup%":>5s}  {"Str":>6s}  '
      f'{"NRP":>9s}  {"BF":>9s}  {"PBC":>9s}')
print('-' * 80)

shielded_results = {}
shielded_grade_rows = []

t0 = time.time()
last_print = t0

for idx, bk in enumerate(bin_keys_viewer):
    info = bin_mapping[bk]

    summed = load_and_sum_frames(info['frames'], xrd_files, xrd_file_map)
    summed[summed < 0] = 0
    summed[summed > 1e9] = 0

    result = shielded_optimize(
        summed, tth_centers, bin_indices_2d, n_tth_bins,
        valid_mask, tth_min, tth_max, tth, tth_edges)

    key = f'{bk[0]}_{bk[1]}'
    if result is None:
        shielded_results[key] = {'error': 'no fits converged'}
        continue

    # Don't save the full profile array to keep file size reasonable
    save_result = {k: v for k, v in result.items() if k != 'profile'}
    shielded_results[key] = save_result

    shielded_grade_rows.append({
        'bin': bk,
        'model': result['model'],
        'suppress_percentile': result['suppress_percentile'],
        'strength': result['strength'],
        'nrp': result['grades']['negative_residual_penalty'],
        'bf': result['grades']['background_flatness'],
        'pbc': result['grades']['peak_to_background_contrast'],
    })

    now = time.time()
    if (idx + 1) == len(bin_keys_viewer) or (now - last_print) > 20 or \
       (idx + 1) % max(1, len(bin_keys_viewer) // 10) == 0:
        elapsed = now - t0
        rate = (idx + 1) / elapsed
        eta = (len(bin_keys_viewer) - idx - 1) / rate
        g = result['grades']
        print(f'({bk[0]:2d},{bk[1]:2d}) {idx+1:>4d}  {result["model"]:>16s}  '
              f'{result["suppress_percentile"]:5.0f}  {result["strength"]:6.2f}  '
              f'{g["negative_residual_penalty"]:9.2f}  '
              f'{g["background_flatness"]:9.2f}  '
              f'{g["peak_to_background_contrast"]:9.2f}  '
              f'[{100*(idx+1)/len(bin_keys_viewer):.0f}% ETA {eta:.0f}s]')
        last_print = now

total_time = time.time() - t0
print(f'\nShielded optimization complete: {total_time:.0f}s ({total_time/60:.1f} min)')

# Save
with open(SHIELDED_OUTPUT, 'w') as f:
    json.dump(shielded_results, f, indent=2,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)

size_kb = SHIELDED_OUTPUT.stat().st_size / 1024
print(f'Saved: {SHIELDED_OUTPUT.name} ({size_kb:.1f} KB)')

# %%
# Compare standard vs shielded optimization
if shielded_grade_rows and grade_rows:
    print('=== Standard vs Shielded Optimization ===')
    print()

    std_nrps = [r['nrp'] for r in grade_rows]
    std_bfs = [r['bf'] for r in grade_rows]
    std_pbcs = [r['pbc'] for r in grade_rows if not np.isnan(r['pbc'])]

    sh_nrps = [r['nrp'] for r in shielded_grade_rows]
    sh_bfs = [r['bf'] for r in shielded_grade_rows]
    sh_pbcs = [r['pbc'] for r in shielded_grade_rows if not np.isnan(r['pbc'])]

    print(f'{"Metric":>25s}  {"Standard":>12s}  {"Shielded":>12s}  {"Delta":>10s}')
    print('-' * 65)
    for name, sv, shv in [
        ('NRP (median)', np.median(std_nrps), np.median(sh_nrps)),
        ('BF  (median)', np.median(std_bfs), np.median(sh_bfs)),
        ('PBC (median)', np.median(std_pbcs), np.median(sh_pbcs)),
        ('NRP (mean)', np.mean(std_nrps), np.mean(sh_nrps)),
        ('BF  (mean)', np.mean(std_bfs), np.mean(sh_bfs)),
        ('PBC (mean)', np.mean(std_pbcs), np.mean(sh_pbcs)),
    ]:
        delta = shv - sv
        better = '  <<' if (('NRP' in name or 'BF' in name) and abs(delta) < abs(sv)*0.01) else \
                 ' (+)' if (('PBC' in name and delta > 0) or
                           (('NRP' in name or 'BF' in name) and delta < 0)) else ' (-)'
        print(f'{name:>25s}  {sv:12.3f}  {shv:12.3f}  {delta:+10.3f}{better}')

    # Distribution of chosen suppress percentiles
    from collections import Counter
    sp_counts = Counter(r['suppress_percentile'] for r in shielded_grade_rows)
    print(f'\nSuppress percentile distribution:')
    for sp, c in sorted(sp_counts.items()):
        bar = '#' * (c * 40 // len(shielded_grade_rows))
        print(f'  {sp:5.0f}%: {c:4d} bins ({100*c/len(shielded_grade_rows):5.1f}%) {bar}')

    # Spatial comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sh_nrp_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    sh_bf_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    sh_pbc_map = np.full((n_bin_rows, n_bin_cols), np.nan)
    sp_map = np.full((n_bin_rows, n_bin_cols), np.nan)

    for r in shielded_grade_rows:
        br, bc = r['bin']
        sh_nrp_map[br, bc] = r['nrp']
        sh_bf_map[br, bc] = r['bf']
        sh_pbc_map[br, bc] = r['pbc']
        sp_map[br, bc] = r['suppress_percentile']

    # Row 0: standard grades (reuse from section 11)
    im = axes[0, 0].imshow(nrp_map, cmap='RdBu_r', interpolation='nearest', aspect='auto')
    axes[0, 0].set_title('Standard NRP', fontsize=9)
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    im = axes[0, 1].imshow(bf_map, cmap='viridis_r', interpolation='nearest', aspect='auto')
    axes[0, 1].set_title('Standard BF', fontsize=9)
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    im = axes[0, 2].imshow(pbc_map, cmap='plasma', interpolation='nearest', aspect='auto')
    axes[0, 2].set_title('Standard PBC', fontsize=9)
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)

    # Row 1: shielded grades
    vmin_nrp = min(np.nanmin(nrp_map), np.nanmin(sh_nrp_map))
    vmax_nrp = max(np.nanmax(nrp_map), np.nanmax(sh_nrp_map))
    im = axes[1, 0].imshow(sh_nrp_map, cmap='RdBu_r', interpolation='nearest',
                            aspect='auto', vmin=vmin_nrp, vmax=vmax_nrp)
    axes[1, 0].set_title('Shielded NRP', fontsize=9)
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)
    # Match standard NRP colorbar range
    axes[0, 0].images[0].set_clim(vmin_nrp, vmax_nrp)

    vmin_bf = min(np.nanmin(bf_map), np.nanmin(sh_bf_map))
    vmax_bf = max(np.nanmax(bf_map), np.nanmax(sh_bf_map))
    im = axes[1, 1].imshow(sh_bf_map, cmap='viridis_r', interpolation='nearest',
                            aspect='auto', vmin=vmin_bf, vmax=vmax_bf)
    axes[1, 1].set_title('Shielded BF', fontsize=9)
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    axes[0, 1].images[0].set_clim(vmin_bf, vmax_bf)

    vmin_pbc = min(np.nanmin(pbc_map), np.nanmin(sh_pbc_map))
    vmax_pbc = max(np.nanmax(pbc_map), np.nanmax(sh_pbc_map))
    im = axes[1, 2].imshow(sh_pbc_map, cmap='plasma', interpolation='nearest',
                            aspect='auto', vmin=vmin_pbc, vmax=vmax_pbc)
    axes[1, 2].set_title('Shielded PBC', fontsize=9)
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
    axes[0, 2].images[0].set_clim(vmin_pbc, vmax_pbc)

    for ax in axes.flat:
        ax.set_xlabel('bin col')
        ax.set_ylabel('bin row')

    plt.suptitle('Standard (top) vs Shielded (bottom) Noise Model Grades', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Suppress percentile spatial map
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sp_map, cmap='coolwarm', interpolation='nearest', aspect='auto')
    ax.set_title('Optimal Suppress Percentile Per Bin', fontsize=10)
    ax.set_xlabel('bin col')
    ax.set_ylabel('bin row')
    plt.colorbar(im, ax=ax, shrink=0.8, label='suppress percentile')
    plt.tight_layout()
    plt.show()
