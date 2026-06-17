# Project Overview & Organization

## 🎯 Main Project Goal

**Bragg Peak Detection in X-ray Diffraction (XRD) Images**

Develop and optimize an algorithm to automatically detect bright spots (Bragg peaks) in XRD images of crystalline materials. The algorithm must:
- Locate peaks along specific 2-theta (lattice plane) arcs
- Distinguish real Bragg peaks from background noise
- Outperform the baseline algorithm
- Achieve high F1 score (detection tolerance: 40 pixels from ground truth)

---

## 📁 Repository Structure

### 1. **`hotspot_detection_split_configs/` — MAIN WORKING DIRECTORY**

This is where the Bragg peak detection task lives.

#### `prompt.md`
- **Purpose**: Complete task specification for the problem
- **Contains**: 
  - Problem statement (what are Bragg peaks, what to detect)
  - Test data description
  - Evaluation criteria (F1 score, 40-pixel tolerance)
  - Input/output specification (CSV format: `reflection, x, y`)
  - Key note: Images have bad pixels—use percentile range 10–99 when visualizing

#### `config.yaml`
- **Purpose**: Configuration for CVEvolve (agentic research framework)
- **Note**: Points to `/data/programs/CVEvolve/...` paths (likely from original environment)
- **Action needed**: Update paths to local paths if running locally

#### `test_data/` — Test/Training Dataset
| File | Purpose |
|------|---------|
| `integrated_intensity_design.tiff` | **Input XRD image** with Bragg peaks to detect |
| `tth.tiff` | **2-theta map**—pixel-to-angle mapping for identifying peak locations |
| `reflections.py` | **Lattice plane list**—2-theta values and names of planes to search |
| `annotations.json` | **Ground truth labels**—pixel coordinates of real Bragg peaks |
| `baseline.py` | **Baseline algorithm**—reference implementation to beat |
| `annotate.py` | Utility for manual annotation (if needed) |

#### `validation_data/` — Validation Dataset
- `integrated_intensity_val.tiff` — Validation XRD image
- `tth.tiff` — 2-theta map for validation
- `annotations.json` — Ground truth for validation
- `reflections.py` — Reflection list for validation
- **Purpose**: Test your algorithm on unseen data

---

### 2. **`python_scripts/` — Utility Scripts (Secondary)**

These scripts handle data preprocessing and reconstruction from synchrotron experiments. **These are not directly related to Bragg peak detection** but provide context for the experimental data pipeline.

| Script | Purpose |
|--------|---------|
| `data_preprocess_isn_26c1.py` | Preprocess raw scan data (ptychography) |
| `data_preprocess_isn_26c1_updated.py` | Updated version of preprocessing |
| `ptychi_recon.py` | Ptychography reconstruction (convert raw scans to diffraction patterns) |
| `ptychi_recon_jd.py` | Alternative reconstruction script |
| `append_detectors_links.py` | Link detector data in HDF5 files |
| `process_dp_velo_21c1.py` | Diffraction pattern velocity processing |

**Status**: These appear to be imported from a different project. Can likely be ignored for Bragg peak detection work.

---

### 3. **`results/` — Generated Output Data**

HDF5 and TIFF outputs from running reconstruction scripts.

| Subfolder | Contents |
|-----------|----------|
| `scan002/`, `scan004/`, `scan227/` | Diffraction pattern reconstructions (HDF5) and visualization TIFFs |
| `analysis/` | Analysis results |

**Status**: Reference/background data; not directly needed for Bragg peak detection.

---

### 4. **`Scan_0179/`, `Scan_0180/` — Raw Experimental Data**

Raw synchrotron facility data organized by detector:
- `ME7/` — Main detector (many .h5 files per scan)
- `XRD/`, `SOCKETSERVER/`, `TETRAMM1/` — Other detector/metadata files

**Status**: Raw experimental archive; likely not needed for this task unless you need to regenerate preprocessing.

---

### 5. **`CVEvolve/` — Agentic Research Framework (Background)**

A framework for running AI agents to research and optimize computer vision algorithms.

**Purpose**: The `config.yaml` in `hotspot_detection_split_configs/` is set up to use CVEvolve to:
1. Have an AI agent read the task
2. Generate algorithm candidates
3. Evaluate and tune candidates
4. Evolve to better solutions

**Status**: Available but not required for manual algorithm development. Use if you want agentic optimization.

---

## 🚀 How to Get Started

### Quick Start: Understand the Task

1. **Read** `hotspot_detection_split_configs/prompt.md` carefully
2. **Inspect test data**:
   ```python
   import tifffile
   import numpy as np
   
   # Load images
   xrd = tifffile.imread('hotspot_detection_split_configs/test_data/integrated_intensity_design.tiff')
   tth = tifffile.imread('hotspot_detection_split_configs/test_data/tth.tiff')
   
   # Load ground truth
   import json
   with open('hotspot_detection_split_configs/test_data/annotations.json') as f:
       labels = json.load(f)
   
   # Check reflections to search
   import importlib.util
   spec = importlib.util.spec_from_file_location("reflections", 
       'hotspot_detection_split_configs/test_data/reflections.py')
   reflections = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(reflections)
   print(reflections.reflections)
   ```

3. **Understand the baseline**:
   ```python
   from hotspot_detection_split_configs.test_data.baseline import detect_hotspot_rois
   
   # Read baseline implementation in baseline.py
   ```

4. **Develop your algorithm** in a new Python script following the input/output spec in `prompt.md`

---

## 📊 Evaluation Checklist

Your algorithm should:
- ✅ Accept: `--image`, `--two-theta`, `--reflections`, `--output` (+ optional flags)
- ✅ Output: CSV with columns `reflection, x, y`
- ✅ Calculate F1 score against ground truth if `--labels` provided
- ✅ Match ground truth within 40 pixels (use `scipy.spatial.distance` for efficiency)
- ✅ Handle bad pixels in images (set to 0 or mask them)
- ✅ Work on both test and validation datasets

---

## 🔧 Next Steps

### Phase 1: Analysis (1–2 hours)
- [ ] Inspect XRD images (visualize with percentile range 10–99)
- [ ] Understand 2-theta map structure
- [ ] Plot ground truth peaks over image
- [ ] Analyze baseline algorithm performance

### Phase 2: Development (variable)
- [ ] Implement initial algorithm (edge detection, peak finding, arc-based filtering)
- [ ] Test on small subset
- [ ] Refine based on false positives/negatives
- [ ] Optimize for F1 score

### Phase 3: Evaluation
- [ ] Run on full test set
- [ ] Run on validation set
- [ ] Compare to baseline
- [ ] Document results

### Optional: Use CVEvolve for automated optimization
- [ ] Set up environment with `uv`
- [ ] Run `cvevolve run --config config.yaml --prompt prompt.md`
- [ ] Let AI agent generate and tune candidates

---

## 📝 Key File Locations Quick Reference

| What | Where |
|------|-------|
| Task specification | `hotspot_detection_split_configs/prompt.md` |
| Config (CVEvolve) | `hotspot_detection_split_configs/config.yaml` |
| Test XRD image | `hotspot_detection_split_configs/test_data/integrated_intensity_design.tiff` |
| Test 2-theta map | `hotspot_detection_split_configs/test_data/tth.tiff` |
| Test ground truth | `hotspot_detection_split_configs/test_data/annotations.json` |
| Test reflections | `hotspot_detection_split_configs/test_data/reflections.py` |
| Baseline algorithm | `hotspot_detection_split_configs/test_data/baseline.py` |
| Validation data | `hotspot_detection_split_configs/validation_data/` |

---

## 💡 Recommended Algorithm Approach

Based on the task and baseline, consider:

1. **Preprocessing**:
   - Mask or handle bad pixels (negative values)
   - Normalize or apply percentile clipping (10–99)
   - Optional: Gaussian filter for smoothing

2. **Peak Detection**:
   - For each reflection (2-theta arc):
     - Extract pixels near that 2-theta value (tolerance window)
     - Apply local maxima detection (e.g., `scipy.ndimage.maximum_filter`)
     - Filter by intensity threshold (e.g., high percentile)

3. **Post-processing**:
   - Remove duplicates (nearby detections)
   - Filter by size/shape criteria
   - Keep only strong peaks

4. **Evaluation**:
   - Match detections to ground truth (within 40 pixels)
   - Calculate precision, recall, F1 score
   - Adjust thresholds based on scores

---

## 📚 Dependencies to Install

```bash
pip install numpy scipy pillow tifffile imageio matplotlib scikit-image opencv-python
```

For optional CVEvolve-based optimization:
```bash
cd CVEvolve
pip install -e .
# or use uv
uv sync
```

---

## 🎓 Summary

- **Main task**: Detect Bragg peaks in XRD images
- **Key files**: Organized in `hotspot_detection_split_configs/`
- **Workflow**: Read task → Analyze data → Implement algorithm → Evaluate → Iterate
- **Optional boost**: Use CVEvolve for agentic optimization
- **Other folders**: Supporting but not directly related to this task

Good luck! 🚀
