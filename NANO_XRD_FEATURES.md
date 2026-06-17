# Nano-XRD Data Analysis -- Script & Notebook Feature Reference

> **Project:** 2026-1 Luo -- ISN Beamline, Argonne National Laboratory
> **Beamline:** ISN 26-ID-C (also VelociProbe legacy scripts)
> **Sample system:** Perovskite / halide (PbI2, CdTe, ITO substrates)

---

## Table of Contents

1. [Overview](#overview)
2. [Python Scripts](#python-scripts)
   - [xrd_proc.py](#xrd_procpy)
   - [data_preprocess_isn_26c1.py](#data_preprocess_isn_26c1py)
   - [data_preprocess_isn_26c1_updated.py](#data_preprocess_isn_26c1_updatedpy)
   - [data_preprocess_MP_isn_26c1.py](#data_preprocess_mp_isn_26c1py)
   - [append_detectors_links.py](#append_detectors_linkspy)
   - [process_dp_velo_21c1.py](#process_dp_velo_21c1py)
   - [ptychi_recon.py / ptychi_recon_jd.py](#ptychi_reconpy--ptychi_recon_jdpy)
3. [Jupyter Notebooks](#jupyter-notebooks)
   - [data_view.ipynb](#data_viewipynb)
   - [data_view_2.ipynb](#data_view_2ipynb)
   - [data_view_gyl.ipynb](#data_view_gylipynb)
   - [data_view_gyl_2.ipynb](#data_view_gyl_2ipynb)
   - [data_proc_per_pixel.ipynb](#data_proc_per_pixelipynb)
   - [cdte_plotting.ipynb](#cdte_plottingipynb)
4. [Hotspot Detection Configs](#hotspot-detection-configs)
   - [baseline.py](#baselinepy)
   - [annotate.py](#annotatepy)
   - [reflections.py](#reflectionspy)
5. [CVEvolve Agent](#cvevolve-agent)
6. [Data Formats & Geometry](#data-formats--geometry)

---

## Overview

This project contains tools for processing nano-XRD data from the ISN 26-ID-C beamline at the Advanced Photon Source. The workflow spans:

1. **Raw data preprocessing** -- cropping, padding, and stacking CCD detector frames from per-position HDF5 files into analysis-ready datasets
2. **Visualization & exploration** -- interactive notebooks for viewing diffraction patterns, fluorescence maps, and detector images
3. **Hotspot / Bragg peak detection** -- identifying bright spots in summed CCD images using 2-theta masks and connected-component analysis
4. **Per-pixel spatial mapping** -- extracting ROI intensities at each scan position to create spatial maps of specific reflections
5. **Ptychographic reconstruction** -- phase retrieval from coherent diffraction patterns using ptychi/pear
6. **Automated algorithm optimization** -- CVEvolve agent for iteratively improving Bragg peak detection algorithms

---

## Python Scripts

### xrd_proc.py

**Location:** `python_scripts/xrd_proc.py`
**Purpose:** Core XRD processing library -- hotspot detection, ROI management, and visualization utilities.

| Feature | Description |
|---------|-------------|
| **Hotspot detection** | `detect_hotspot_rois()` -- identifies Bragg peak ROIs using 2-theta arc masking + percentile thresholding + connected-component labeling (scipy.ndimage) |
| **2-theta arc masking** | Creates binary masks for each reflection's Debye-Scherrer arc within `line_tol` of the target 2-theta angle |
| **Connected-component filtering** | Labels isolated bright regions, filters by `min_pixels`, and expands by `pad` pixels |
| **Edge/region exclusion** | `ignore_y_range`, `ignore_edge_rows`, `ignore_edge_cols` to mask beamstop shadows or detector edges |
| **ROI deduplication** | Removes fully-contained duplicate ROI bounding boxes |
| **ROI-to-object conversion** | `build_rois_from_detected()` converts bounding boxes to `mictools.roi_utils.Roi` objects |
| **Per-scan ROI mapping** | `get_detected_rois_on_scan()` uses `mictools.process_data.mesh_detector_data` to extract spatially-resolved ROI intensities |
| **Overlay visualization** | `plot_v_mask_overlay()` renders color-coded ROI class maps with legends keyed to reflection labels |
| **HDF5 I/O** | `save_roi_outputs_h5()` / `load_roi_outputs_h5()` -- persistent storage of ROI detection results, masks, and spatial coordinates |
| **Mesh plotting** | `plot_meshed_data()` -- 2D pcolormesh visualization with colorbar and equal-aspect ratio |

**Key parameters for `detect_hotspot_rois()`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `line_tol` | (required) | 2-theta tolerance for arc mask (degrees) |
| `hotspot_percentile` | 99.0 | Intensity threshold within the arc (higher = stricter) |
| `min_pixels` | 4 | Minimum connected-component size to keep |
| `pad` | 6 | Pixels to expand each ROI bounding box |
| `ignore_y_range` | None | (y0, y1) row range to exclude |
| `ignore_edge_rows` | 0 | Rows to exclude from top/bottom edges |
| `ignore_edge_cols` | 0 | Columns to exclude from left/right edges |

---

### data_preprocess_isn_26c1.py

**Location:** `python_scripts/data_preprocess_isn_26c1.py`
**Purpose:** Hardcoded batch preprocessing for ISN 26c1 diffraction data with multiprocessing.

| Feature | Description |
|---------|-------------|
| **Multiprocessing file loading** | Uses `multiprocessing.Pool.starmap()` to read HDF5 files in parallel |
| **Detector cropping** | Crops raw frames to 256x256 around center (517, 801) |
| **Bad pixel cleaning** | Clips negative values to 0, clips values > 1e7 to 0 |
| **Position loading** | Reads CSV position files (X/Y in micrometers), converts to meters |
| **Geometry calculation** | Computes real-space pixel size from energy (15 keV), detector distance (6.16 m), pixel size (75 um) |
| **Output format** | `data_roi{ROI}_dp.hdf5` (diffraction stack, gzip) + `data_roi{ROI}_para.hdf5` (geometry metadata) |

**Fixed geometry:** Energy = 15 keV, det-sample distance = 6.16 m, pixel size = 75 um, wavelength = 0.0826 nm

---

### data_preprocess_isn_26c1_updated.py

**Location:** `python_scripts/data_preprocess_isn_26c1_updated.py`
**Purpose:** CLI-driven variant with improved error handling and flexible crop parameters.

| Feature | Description |
|---------|-------------|
| **CLI interface** | `argparse` with `scan`, `--dataset`, `--det-npixel`, `--crop-center`, `--crop-size` |
| **Auto dataset discovery** | Falls back to searching for any 2D dataset if specified path not found in HDF5 |
| **2D/3D frame handling** | Automatically detects single-frame vs multi-frame HDF5 datasets |
| **Graceful position handling** | Zero-pads positions if fewer than patterns; continues if position file missing |
| **Configurable crop** | Center and size can be overridden via CLI (defaults: cx=396, cy=816) |

---

### data_preprocess_MP_isn_26c1.py

**Location:** `python_scripts/data_preprocess_MP_isn_26c1.py`
**Purpose:** Hardcoded multiprocessing preprocessing targeting the Luktuke user directory.

| Feature | Description |
|---------|-------------|
| **Same geometry** | 15 keV, 6.16 m, 75 um pixel, 256x256 output |
| **Hardcoded scans** | `scans=[302]` |
| **Parallel loading** | `multiprocessing.Pool` for file reading |
| **Different user directory** | Points to `2026-1-Luktuke` instead of `2026-1-Luo` |

---

### append_detectors_links.py

**Location:** `python_scripts/append_detectors_links.py`
**Purpose:** Creates external HDF5 links from aggregated scan files to raw detector data.

| Feature | Description |
|---------|-------------|
| **External HDF5 linking** | Creates `/detectors` group with `h5py.ExternalLink` references to raw detector files |
| **Auto-discovery** | Finds detector subdirectories under `Raw/Scan_XXXX/` |
| **Shape reporting** | `print_detector_link_sizes()` reports `/entry/data/data` shape for each linked file |
| **Filter/overwrite control** | `--detector` flag to target specific detectors; `--no-overwrite` to preserve existing links |

---

### process_dp_velo_21c1.py

**Location:** `python_scripts/process_dp_velo_21c1.py`
**Purpose:** Legacy preprocessing for VelociProbe 2021-1 beamline fly-scan data.

| Feature | Description |
|---------|-------------|
| **Different geometry** | Energy = 10 keV, det distance = 2.335 m, 64x64 output |
| **Fly-scan format** | Reads `fly{scanNo:03d}_data_{j:06d}.h5` pattern |
| **Position parsing** | Space-delimited CSV with column extraction |
| **Same cleaning pipeline** | Crop, clip negatives, clip >1e7, stack, save HDF5 |

---

### ptychi_recon.py / ptychi_recon_jd.py

**Location:** `python_scripts/ptychi_recon.py`, `python_scripts/ptychi_recon_jd.py`
**Purpose:** Ptychographic reconstruction drivers using the `ptychi.pear` engine.

| Feature | Description |
|---------|-------------|
| **Phase retrieval** | 2000-iteration ptychographic reconstruction from preprocessed diffraction patterns |
| **Multi-probe modes** | 5 probe modes for partial coherence modeling |
| **Position correction** | Fourier-gradient position refinement starting at iteration 500 |
| **Intensity correction** | Compensates for intensity variations across the dataset |
| **Batched processing** | 20 batches with uniform selection scheme |
| **Probe initialization** | Loads initial probe from a previous reconstruction |
| **ISN vs 2xfm** | `ptychi_recon.py` targets ISN instrument; `ptychi_recon_jd.py` targets 2xfm with momentum acceleration |

---

## Jupyter Notebooks

### data_view.ipynb

**Location:** `python_scripts/data_view.ipynb`
**Purpose:** Primary scan visualization and XRD calibration.

| Feature | Description |
|---------|-------------|
| **pyFAI calibration** | Loads `.poni` calibration file for 2-theta array computation |
| **2-theta mapping** | `ai.twoThetaArray()` or `ai.center_array('2th_rad')` for per-pixel 2-theta |
| **Multi-scan analysis** | `graph_run` and `analyze_run` on multiple scans |
| **KB optics analysis** | Examines sample z-position scanning (scans 54-65) |
| **mictools integration** | Uses `mictools.config` and `peak_modelling` for beamline-specific setup |

---

### data_view_2.ipynb

**Location:** `python_scripts/data_view_2.ipynb`
**Purpose:** Extended visualization with debugging and ROI testing.

| Feature | Description |
|---------|-------------|
| **ROI detection testing** | Tests `detect_hotspot_rois()` across multiple samples |
| **mesh_detector_data** | Uses `mictools.process_data.mesh_detector_data` for per-pixel ROI extraction |
| **XRD calibration loading** | External calibration file loading for 2-theta computation |
| **Debug helpers** | Utility functions for inspecting intermediate results |

---

### data_view_gyl.ipynb

**Location:** `python_scripts/data_view_gyl.ipynb`
**Purpose:** Fluorescence element mapping and CCD integration for GYL samples.

| Feature | Description |
|---------|-------------|
| **XRF ROI definitions** | Manual ROIs for Br, Cl, Pb, I fluorescence lines |
| **CCD sum integration** | `sum_detector_image()` for full-detector integrated intensity |
| **Fly-scan plotting** | `plot_flyscan()` for position-resolved fluorescence maps |
| **Multi-element overlay** | ROI overlay visualization with fixed color assignment |
| **Sample targeting** | 5%DI_Yes_GB perovskite samples |

---

### data_view_gyl_2.ipynb

**Location:** `python_scripts/data_view_gyl_2.ipynb`
**Purpose:** Advanced hotspot detection pipeline with CdTe ASU detector data.

| Feature | Description |
|---------|-------------|
| **Full hotspot pipeline** | detect -> extract ROIs -> spatial mapping -> overlay visualization |
| **Reflection labels** | PbI2, (001), (011), (111), (002), ITO, (012), (112) |
| **HDF5 ROI persistence** | Saves detected ROIs to HDF5 for downstream per-pixel analysis |
| **Multi-scan processing** | Tests scans 156-179 and 203 |
| **CdTe detector** | Processes data from CdTe ASU area detectors |

---

### data_proc_per_pixel.ipynb

**Location:** `python_scripts/data_proc_per_pixel.ipynb`
**Purpose:** Per-pixel detector data analysis with spatial mapping.

| Feature | Description |
|---------|-------------|
| **Full detector stacking** | `stack_detector_image()` loads all frames for a scan |
| **Per-position channels** | Processes individual detector channels at each spatial position |
| **Spatial mapping** | Maps detector signal to sample coordinates |

---

### cdte_plotting.ipynb

**Location:** `python_scripts/cdte_plotting.ipynb`
**Purpose:** Multi-channel XRF elemental mapping and registration overlays.

| Feature | Description |
|---------|-------------|
| **XRF channel extraction** | Loads NNLS-fitted counts/sec from `/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec` |
| **Element maps** | Se, As, Zn, Sn_L, Cd_L fluorescence lines |
| **XBIC normalization** | tetramm3 (ch=2) beam-induced current, I0 from tetramm1 (ch=1) |
| **Grid interpolation** | `scipy.griddata` for scattered-to-regular-grid conversion |
| **Registration overlay** | `plot_registration_overlay()` with dual-channel RGBA false-color comparison |
| **Percentile scaling** | Configurable percentile-based intensity normalization |

---

## Hotspot Detection Configs

**Location:** `hotspot_detection_split_configs/`

### baseline.py

Self-contained hotspot detection algorithm (standalone version of `xrd_proc.detect_hotspot_rois()`). Used as the baseline for CVEvolve optimization.

- CLI: `--image`, `--two-theta`, `--line-tol`
- Imports `reflections.py` for target angles
- Connected-component labeling with percentile thresholding
- Output: bounding boxes per reflection

### annotate.py

Interactive GUI annotation tool for manual Bragg peak labeling.

- `PlaneAnnotator` class with matplotlib click events
- Left-click to add, right-click to delete nearest point
- Iterates through all lattice planes from `reflections.py`
- Saves `annotations.json` with per-plane `[x, y]` pixel coordinate lists
- CLI: `--image scan_203_sum.tiff --tth tth.tiff --output annotations.json`

### reflections.py

Calibration constants for the perovskite/halide crystal system:

```python
degs = [6.81319, 7.51422, 10.61748, 13.00831, 15.01266,
        16.07224, 16.79944, 18.42549, 21.29655, 22.59817, 26.16205]
deg_labels = ['PbI2', '(001)', '(011)', '(111)', '(002)',
              'ITO', '(012)', '(112)']
```

Note: `degs` has 11 entries but `deg_labels` has only 8 -- the last 3 angles are unlabeled high-angle reflections.

---

## CVEvolve Agent

**Location:** `CVEvolve/`
**Purpose:** Automated algorithm optimization framework for Bragg peak detection.

CVEvolve is an agentic research framework that iteratively generates, evaluates, tunes, and evolves candidate detection algorithms. It uses an LLM agent (via LangGraph) with workspace tools to:

1. Read XRD images (TIFF), 2-theta maps, and reflection lists
2. Generate Python detection algorithms
3. Evaluate candidates against annotated ground truth (F1 score, 40-pixel tolerance)
4. Tune top performers and crossover/evolve parent algorithms
5. Track all experiments in SQLite + optional MLflow

**Key configuration (`config.yaml`):**

| Setting | Value |
|---------|-------|
| Model | Claude Opus via Argo API |
| Metric | F1 score (maximize) |
| Max rounds | 20 |
| Warmup rounds | 3 |
| Workers | 3 generate, 3 tune |
| Data | `test_data/` with `integrated_intensity_design.tiff` |
| Validation | Separate `validation_data/` with `integrated_intensity_val.tiff` |

**Expected output from candidate algorithms:** CSV with columns `reflection, x, y` (one row per detected Bragg peak).

---

## Data Formats & Geometry

### Raw Data Structure

```
Scan_XXXX/
  ME7/         -- Medipix detector frames
  XRD/         -- XRD area detector frames
  TETRAMM1/    -- Ion chamber / beam monitor
  SOCKETSERVER/-- Position files
```

Each subdirectory contains numbered HDF5 files: `scan_XXXX_NNNNN.h5`

### Preprocessed Data

```
results/scanNNN/
  data_roi{ROI}_dp.hdf5    -- Diffraction pattern stack (N, H, W) float32
  data_roi{ROI}_para.hdf5  -- Geometry: lambda, dx, ppX, ppY, N_files
```

### Hotspot Detection Data

```
hotspot_detection_split_configs/
  config.yaml               -- CVEvolve session configuration
  prompt.md                 -- Task prompt for the agent
  test_data/
    integrated_intensity_design.tiff  -- Summed XRD image
    tth.tiff                          -- Per-pixel 2-theta map
    annotations.json                  -- Ground truth Bragg peak locations
    reflections.py                    -- Target 2-theta angles and labels
    baseline.py                       -- Baseline detection algorithm
  validation_data/
    integrated_intensity_val.tiff     -- Validation XRD image
    tth.tiff                          -- Validation 2-theta map
    annotations.json                  -- Validation ground truth
    reflections.py                    -- Same reflection definitions
```

### ISN 26-ID-C Beamline Geometry

| Parameter | Value |
|-----------|-------|
| Beam energy | 15 keV |
| Wavelength | 0.0826 nm |
| Detector-sample distance | 6.16 m |
| Detector pixel size | 75 um |
| Detector crop size | 256 x 256 |
| Detector center (x, y) | ~(396-517, 801-816) depending on script |
| Real-space pixel size | lambda * dist / (pixel_size * N_pixels) |
