# Automated Bragg Peak Detection in Spatially-Resolved Nano-XRD Data via Agentic Algorithm Evolution

> **Project:** 2026-1 Luo | **Beamline:** ISN 26-ID-C, Advanced Photon Source, Argonne National Laboratory
> **Sample system:** Perovskite / halide thin-film devices (PbI2, CdTe, ITO substrates)

---

## 1. Introduction

This document describes the end-to-end methodology for detecting and mapping Bragg peaks in nano-X-ray diffraction (nano-XRD) data collected at the ISN 26-ID-C beamline. The workflow proceeds through six major phases:

1. **Data reduction** -- summing raw detector frames and constructing the 2-theta map
2. **Spatial binning** -- grouping frames into 5x5 and 3x3 superpixels on the sample grid
3. **Ground-truth labeling** -- manually annotating Bragg peaks at the 5x5 resolution
4. **Algorithm evolution** -- using CVEvolve to discover high-performance detection algorithms
5. **Cross-resolution transfer** -- applying the evolved algorithm to higher-resolution 3x3 bins and verifying hotspots via spatial feature analysis
6. **Device visualization** -- mapping confirmed features onto the physical sample grid with size and shape characterization

Each section details the mathematical formulation, algorithm design, and key parameters.

---

## 2. Experimental Setup and Raw Data

### 2.1 Beamline geometry

| Parameter | Value |
|-----------|-------|
| X-ray energy *E* | 15 keV |
| Wavelength | 0.0826 nm |
| Detector--sample distance *L* | 6.16 m |
| Detector pixel size *p* | 75 um |
| Detector dimensions | 1062 rows x 1028 columns |
| Crop region (ptychography) | 256 x 256 pixels centered at (517, 801) |

The wavelength is computed from the photon energy:

$$\lambda = \frac{hc}{E} = \frac{1.23984 \times 10^{-9}\ \text{m}}{E\ [\text{keV}]}$$

The real-space pixel size for ptychographic reconstruction:

$$\Delta x = \frac{\lambda \cdot L}{p \cdot N_{\text{pixels}}} = \frac{(8.265 \times 10^{-11})(6.16)}{(75 \times 10^{-6})(256)} \approx 2.65 \times 10^{-8}\ \text{m}$$

### 2.2 Scan configuration

Scan 203 rasters the focused X-ray beam across the sample in a serpentine pattern. Each probe position produces a 2D diffraction pattern on the area detector.

- **Total frames:** ~25,003 diffraction patterns
- **Storage:** 150 HDF5 files x 167 frames each (last file: 120 frames)
- **Dataset path:** `/entry/data/data` per file
- **Position data:** CSV file with (X, Y) stage coordinates in micrometers

### 2.3 Data cleaning

Raw detector frames undergo pixel-level corrections before any analysis:

$$I_{\text{clean}}[y, x] = \begin{cases} 0 & \text{if } I_{\text{raw}}[y, x] < 0 \\ 0 & \text{if } I_{\text{raw}}[y, x] > 10^9 \\ I_{\text{raw}}[y, x] & \text{otherwise} \end{cases}$$

Negative pixels arise from detector readout artifacts; extreme positive values indicate hot pixels or cosmic ray strikes.

---

## 3. Whole-Frame Summation and the 2-Theta Map

### 3.1 Integrated intensity image

The first step in identifying Bragg peaks across the full device is to sum all ~25,003 diffraction frames into a single integrated intensity image:

$$I_{\text{sum}}[y, x] = \sum_{k=1}^{N} I_k[y, x]$$

where *N* = 25,003 and each $I_k$ is a cleaned 1062 x 1028 detector frame. This produces an image with extremely high signal-to-noise ratio, making even faint Bragg peaks visible.

### 3.2 2-Theta map construction

The 2-theta map provides the scattering angle at every detector pixel. It is computed from the detector geometry using pyFAI calibration with a CeO2 standard:

```python
ai = pyFAI.load('CeO2_1_poni.poni')
tth_radians = ai.twoThetaArray()
tth_map = np.degrees(tth_radians)  # shape: (1062, 1028)
```

For each pixel at position $(y, x)$ relative to the direct beam center $(y_0, x_0)$, the 2-theta angle is:

$$2\theta[y, x] = \arctan\!\left(\frac{p \cdot \sqrt{(y - y_0)^2 + (x - x_0)^2}}{L}\right)$$

where *p* is the physical pixel size and *L* is the detector--sample distance. Pixels at the same 2-theta value form concentric arcs (Debye-Scherrer rings) centered on the direct beam position.

The resulting `tth.tiff` file (1062 x 1028, float32) maps each pixel to its 2-theta value in degrees. This map is constant across all spatial bins since the detector geometry does not change.

### 3.3 Bragg reflections

The perovskite sample produces diffraction from multiple lattice planes. Each plane diffracts at a characteristic 2-theta angle determined by Bragg's law:

$$n\lambda = 2d_{hkl} \sin\theta$$

where $d_{hkl}$ is the interplanar spacing for the $(hkl)$ reflection. The target reflections and their 2-theta positions:

| Reflection | 2-theta (deg) |
|------------|---------------|
| PbI2 | 6.813 |
| (001) | 7.514 |
| (011) | 10.617 |
| (111) | 13.008 |
| (002) | 15.013 |
| ITO | 16.072 |
| (012) | 16.799 |
| (112) | 18.425 |
| (022) | 21.297 |
| (003) | 22.598 |
| (122) | 26.162 |

On the detector, each reflection occupies a narrow band of pixels along its 2-theta arc. Bragg peaks appear as localized bright spots along these arcs where individual crystal grains satisfy the diffraction condition.

---

## 4. Spatial Binning

### 4.1 Scan grid reconstruction

The raw serpentine scan must be mapped to a regular 2D grid. The stage X-position for each frame is smoothed to identify direction-reversal points:

$$\bar{x}[i] = \frac{1}{K} \sum_{j=i-K/2}^{i+K/2} x[j], \quad K = 20$$

Local extrema (turning points) of $\bar{x}$ are found via `scipy.signal.argrelextrema` with `order=50`, identifying where the scan reverses direction. Frames between consecutive turning points form one row of the raster grid. For odd-numbered rows, the column index is reversed to account for the serpentine pattern:

$$\text{col}[s:e] = \begin{cases} [0, 1, 2, \ldots] & \text{even rows} \\ [\ldots, 2, 1, 0] & \text{odd rows} \end{cases}$$

Result: a 2D grid of approximately 53 rows x 74 columns = ~3,922 positions.

### 4.2 Bin construction

Spatial binning groups adjacent grid positions into superpixels to improve SNR at the cost of spatial resolution. For bin size $B$:

$$I_{\text{bin}}[r_b, c_b] = \sum_{dr=0}^{B-1} \sum_{dc=0}^{B-1} \sum_{k \in \text{frames}(r_b B + dr,\, c_b B + dc)} I_k$$

The number of bin-rows and bin-columns:

$$N_{r}^{\text{bin}} = \left\lceil \frac{N_r}{B} \right\rceil, \quad N_{c}^{\text{bin}} = \left\lceil \frac{N_c}{B} \right\rceil$$

| Bin size | Grid | Total bins | Frames/bin (approx.) |
|----------|------|-----------|---------------------|
| 5 x 5 | 32 x 45 | 1,188 | ~25 |
| 4 x 4 | 39 x 56 | 1,859 | ~16 |
| 3 x 3 | 52 x 74 | 3,253 | ~9 |

All binned images are pre-computed and stored as HDF5 files (`xrd_{B}x{B}_bins.h5`), keyed by `"{row}_{col}"`. Each dataset is a float32 array of shape (1062, 1028).

---

## 5. Ground-Truth Labeling

### 5.1 Labeling at 5x5 resolution

Ground-truth annotations were created using a custom interactive PyQt5 labeling tool (`labeling_tool.py`). At the 5x5 bin level, the higher SNR (~25 frames per bin) makes Bragg peaks clearly distinguishable.

For each bin image, the labeler applies radial background subtraction and displays the cleaned image with 2-theta arc overlays. The user clicks on visible Bragg peaks and assigns them to the appropriate reflection label. Annotations are stored as:

```json
{
  "3_7": {
    "(001)": [[450, 500], [460, 510]],
    "(011)": [[380, 420]],
    "__reviewed__": true
  }
}
```

### 5.2 Label transfer to 3x3 and 4x4 bins

Rather than re-labeling from scratch at finer resolutions, annotations are transferred from 5x5 bins to the target resolution using spatial overlap and signal verification (`generate_validation_labels.py`).

**Spatial overlap mapping.** For each annotated 5x5 bin at position $(r_5, c_5)$, the corresponding scan-grid rows span $[5r_5,\, 5r_5 + 4]$ and columns span $[5c_5,\, 5c_5 + 4]$. The overlapping target bins at bin size $B$ are:

$$r_{\text{start}} = \left\lfloor \frac{5 r_5}{B} \right\rfloor, \quad r_{\text{end}} = \left\lfloor \frac{5 r_5 + 4}{B} \right\rfloor$$

For each candidate target bin, the frame overlap is computed:

$$\text{overlap}(b_5, b_t) = |\text{frames}(b_5) \cap \text{frames}(b_t)|$$

The target bin with the largest overlap is selected.

**Peak verification.** Each transferred peak at pixel position $(p_x, p_y)$ is verified in the target bin image using two independent criteria (logical OR):

**Method 1 -- Local SNR via annular background:**

$$\text{peak\_val} = \max\left(I_{\text{cleaned}}[p_y - r : p_y + r,\; p_x - r : p_x + r]\right), \quad r = 5$$

The background is estimated from an annulus around the peak:

$$\text{annulus} = \{(y, x) : r_{\text{inner}} \leq \sqrt{(y-p_y)^2 + (x-p_x)^2} \leq r_{\text{outer}}\}$$

with $r_{\text{inner}} = 10$, $r_{\text{outer}} = 20$ pixels. The noise level is estimated using the Median Absolute Deviation (MAD):

$$\hat{\sigma} = 1.4826 \cdot \text{median}\left(|b_i - \text{median}(b_i)|\right)$$

where $b_i$ are the annulus pixel values. The factor 1.4826 makes MAD a consistent estimator for the standard deviation of a Gaussian distribution:

$$1.4826 = \frac{1}{\Phi^{-1}(3/4)}$$

The local SNR is then:

$$\text{SNR}_{\text{local}} = \frac{\text{peak\_val} - \text{median}(b_i)}{\hat{\sigma}}$$

The peak is confirmed if $\text{SNR}_{\text{local}} \geq 3.0$.

**Method 2 -- 99th percentile exceedance:**

$$\text{peak\_val} \geq P_{99}(I_{\text{cleaned}})$$

where $P_{99}$ is the 99th percentile of all finite pixel values in the background-subtracted image.

A peak is confirmed if either method passes. The background subtraction used here is a fast radial median subtraction (see Section 6.2).

---

## 6. Algorithm Evolution with CVEvolve

### 6.1 CVEvolve framework

CVEvolve is an agentic AI framework that evolves computer vision algorithms through iterative generation, tuning, and crossover. Each "round" spawns AI agents (powered by Claude Opus 4.6 via the Argo API) that:

1. **Generate** novel detection algorithms from the problem specification
2. **Tune** high-performing algorithms by optimizing hyperparameters
3. **Evolve** algorithms by crossing over successful lineages

The fitness metric is the F1 score computed with a 40-pixel matching tolerance:

$$\text{Precision} = \frac{|\text{TP}|}{|\text{TP}| + |\text{FP}|}, \quad \text{Recall} = \frac{|\text{TP}|}{|\text{TP}| + |\text{FN}|}$$

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

where a detection is a true positive (TP) if it falls within 40 pixels of any ground-truth point (via bipartite matching).

### 6.2 Baseline algorithm

The baseline applies global percentile thresholding after parametric background subtraction:

1. Compute the radial profile: for each 2-theta bin, take the median intensity across all pixels in that bin
2. Fit a Gaussian model to the radial profile:

$$f(2\theta) = A \exp\!\left(-\frac{(2\theta - \mu)^2}{2\sigma^2}\right) + C$$

Parameters: amplitude $A$, center $\mu$, width $\sigma$, offset $C$. Fitted via `scipy.optimize.curve_fit` with `maxfev=10000`.

3. Map the fitted profile back to 2D and subtract:

$$I_{\text{cleaned}}[y, x] = I_{\text{raw}}[y, x] - f\!\left(2\theta[y, x]\right)$$

4. Apply global percentile threshold:

$$\text{hotspot}[y, x] = \begin{cases} 1 & \text{if } I_{\text{cleaned}}[y, x] \geq P_{97}(I_{\text{cleaned}}) \\ 0 & \text{otherwise} \end{cases}$$

5. Connected-component labeling via `scipy.ndimage.label()`
6. Filter by component size: keep only components with $\geq 3$ pixels

**Baseline F1 score: 0.022** -- the global threshold is too crude to distinguish Bragg peaks from residual noise.

### 6.3 Alternative background models

Four radial background models are available in the noise reduction library:

**Gaussian:**

$$f(2\theta) = A \exp\!\left(-\frac{(2\theta - \mu)^2}{2\sigma^2}\right) + C$$

**Split Gaussian** (asymmetric):

$$f(2\theta) = \begin{cases} A \exp\!\left(-\frac{(2\theta - \mu)^2}{2\sigma_L^2}\right) + C & \text{if } 2\theta \leq \mu \\ A \exp\!\left(-\frac{(2\theta - \mu)^2}{2\sigma_R^2}\right) + C & \text{if } 2\theta > \mu \end{cases}$$

**Skewed Gaussian** (via `scipy.stats.skewnorm`):

$$f(2\theta) = A \cdot \frac{\text{skewnorm.pdf}\!\left(\frac{2\theta - \mu}{\sigma},\, \alpha\right)}{\text{skewnorm.pdf}\!\left(\bar{z},\, \alpha\right)} + C$$

where $\alpha$ is the skewness parameter, bounded in $[-20, 20]$.

**Fourier low-pass:**

$$\hat{f}[\omega] = \begin{cases} \text{FFT}[\text{profile}][\omega] & \text{if } \omega \leq \omega_{\text{cutoff}} \\ 0 & \text{otherwise} \end{cases}$$

$$f_{\text{smooth}} = \text{IFFT}(\hat{f})$$

Cutoff tested over $\omega_{\text{cutoff}} \in \{0.02, 0.03, 0.05, 0.07, 0.10\}$; best selected by minimum RMSE on valid bins (bins with $> 50$ pixels).

### 6.4 Winning algorithm: tophat_band_adaptive_snr (F1 = 0.790)

The top-performing algorithm discovered by CVEvolve in Round 1 uses a five-stage pipeline:

**Stage 1 -- Radial median background subtraction.**

Non-parametric: for each 2-theta bin $i$, compute:

$$\tilde{I}_i = \text{median}\!\left(\{I[y,x] : \text{bin}[y,x] = i\}\right)$$

Smooth the profile with a uniform filter of width $\min(15, N_{\text{bins}})$:

$$\bar{I}_i = \frac{1}{W}\sum_{j=i-W/2}^{i+W/2} \tilde{I}_j$$

Map back to 2D and subtract:

$$I_{\text{cleaned}}[y, x] = I_{\text{raw}}[y, x] - \bar{I}_{\text{bin}(y,x)}$$

**Stage 2 -- White top-hat morphological transform.**

The white top-hat isolates compact bright features smaller than a structuring element of size $s = 15$ pixels:

$$\text{opening}(I) = \delta_s(\varepsilon_s(I))$$

$$\text{tophat}(I) = I - \text{opening}(I) = I - \delta_s(\varepsilon_s(I))$$

where $\varepsilon_s$ is erosion (minimum filter) and $\delta_s$ is dilation (maximum filter), both with a square footprint of side $s$. The opening removes all bright features smaller than $s$; subtracting it from the original isolates exactly those features. Implemented efficiently as:

```python
eroded = minimum_filter(cleaned, size=15)
opened = maximum_filter(eroded, size=15)
tophat = np.clip(cleaned - opened, 0, None)
```

**Stage 3 -- 2-theta band masking.**

For each reflection at 2-theta value $d_j$, construct a binary band mask:

$$M_j[y, x] = \begin{cases} 1 & \text{if } |2\theta[y, x] - d_j| \leq \Delta_{2\theta} \\ 0 & \text{otherwise} \end{cases}$$

with tolerance $\Delta_{2\theta} = 0.4\degree$. This restricts peak detection to physically meaningful regions -- peaks can only appear along their expected 2-theta arcs.

**Stage 4 -- Per-band adaptive SNR thresholding.**

Within each band, the noise level is estimated using the MAD of all tophat values:

$$\mu_j = \text{median}\!\left(\text{tophat}[M_j]\right)$$

$$\text{MAD}_j = \text{median}\!\left(|\text{tophat}[M_j] - \mu_j|\right)$$

$$\hat{\sigma}_j = 1.4826 \cdot \text{MAD}_j$$

A pixel is a peak candidate if:

$$\text{tophat}[y, x] > \mu_j + \tau \cdot \hat{\sigma}_j$$

where $\tau = 4.0$ is the SNR threshold. Edge pixels (within 3 pixels of any image boundary) are excluded.

**Stage 5 -- Connected-component analysis and filtering.**

Peak candidates are clustered via connected-component labeling. Each component is filtered by:

- **Size:** $3 \leq N_{\text{pixels}} \leq 150$
- **Compactness:** rejects elongated Debye-Scherrer arc segments

$$\text{aspect} = \frac{\min(h, w)}{\max(h, w)}, \quad \text{fill} = \frac{N_{\text{pixels}}}{h \cdot w}$$

$$\text{compactness} = \text{aspect} \times \text{fill} \geq 0.12$$

The peak centroid is computed as the intensity-weighted center of mass:

$$c_x = \frac{\sum_{i} x_i \cdot w_i}{\sum_i w_i}, \quad c_y = \frac{\sum_{i} y_i \cdot w_i}{\sum_i w_i}$$

where $w_i = \max(I_{\text{cleaned}}[y_i, x_i],\, 0)$.

**Stage 6 -- Cross-band deduplication via greedy NMS.**

All peaks across all bands are sorted by descending SNR. A peak is kept only if no previously-kept peak is within 15 pixels:

$$\text{keep}(p_k) = \begin{cases} \text{True} & \text{if } \forall\, p_j \in \text{kept}: \|p_k - p_j\| > 15 \\ \text{False} & \text{otherwise} \end{cases}$$

Output is capped at 25 detections per bin.

### 6.5 Algorithm evolution summary

| Round | Type | Best F1 | Algorithm |
|-------|------|---------|-----------|
| 0 | Baseline | 0.022 | Gaussian bg + global percentile |
| 1 | Generate | **0.790** | tophat_band_adaptive_snr |
| 1 | Generate | 0.778 | tophat_dual_snr_detector |
| 2 | Generate | 0.768 | log_tophat_hybrid_detector |
| 4 | Tune | 0.771 | tuned_multiscale_tophat_tight_bands |
| 7 | Tune | 0.769 | tuned_multiscale_tophat_dual_snr |

Key innovations discovered:

- **Morphological top-hat** replaced parametric background fitting, providing model-free feature extraction
- **Per-band adaptive thresholding** (using MAD) replaced global percentile, accounting for varying noise levels across 2-theta arcs
- **Compactness filtering** automatically discriminated compact Bragg peaks from extended Debye-Scherrer ring segments
- **Multi-scale top-hat** variants (sizes 7, 11, 15) captured peaks at varying spatial scales

---

## 7. Hotspot Detection in 3x3 Binned Data

### 7.1 Applying the evolved algorithm at higher resolution

The top algorithm from the 5x5 CVEvolve run (`tophat_band_adaptive_snr`) is applied to all 3,253 bins in the 3x3 dataset. The same detection pipeline described in Section 6.4 is used with identical parameters.

For each bin, the algorithm outputs a list of detected peaks with:
- Pixel position $(x, y)$ on the detector
- Reflection label (assigned by nearest 2-theta match)
- SNR value
- Cleaned-image intensity at the peak location, measured as the maximum value in a 7x7 window:

$$I_{\text{peak}}^{(b)} = \max\left(I_{\text{cleaned}}^{(b)}[p_y - 3 : p_y + 4,\; p_x - 3 : p_x + 4]\right)$$

where superscript $(b)$ denotes the bin.

### 7.2 Spatial linking via Union-Find

Peaks detected in adjacent bins at approximately the same detector position likely correspond to the same physical Bragg reflection from the same crystal grain. These are linked using a Union-Find (disjoint set) algorithm.

**Node definition.** Each detection is a node $(b_k, i, r, c, x, y)$ where $b_k$ is the bin key, $i$ is the peak index within that bin, $(r, c)$ is the bin's row/column on the spatial grid, and $(x, y)$ is the detector pixel position.

**Linking criterion.** Two nodes in adjacent bins (8-connected neighborhood: $|\Delta r| \leq 1, |\Delta c| \leq 1$) are linked if their detector positions are within 5 pixels:

$$\sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} \leq 5$$

This produces connected components (features), each representing a single Bragg reflection visible across multiple spatial bins.

### 7.3 Gaussian profile filtering

Features are classified as real or noise based on their spatial intensity profile. A genuine Bragg peak should show highest intensity at its center bin and decreasing intensity with distance -- a Gaussian-like spatial profile arising from the X-ray beam's intensity distribution.

**Coefficient of variation test.** If all bins in the feature have nearly identical intensity, the detection is likely an artifact:

$$\text{CV} = \frac{\sigma(I)}{\bar{I}} < 0.05 \implies \text{reject (flat profile)}$$

**Monotonicity test.** For each pair of bins $(i, j)$ in the feature, check whether the closer bin to the center has higher intensity:

$$d_i = \sqrt{(r_i - r^*)^2 + (c_i - c^*)^2}$$

where $(r^*, c^*)$ is the center bin (bin with maximum intensity). The pair is "monotonic" if:

$$(d_i < d_j \text{ and } I_i > I_j) \quad \text{or} \quad (d_i > d_j \text{ and } I_i < I_j)$$

The monotonic fraction must exceed 40% for the feature to be kept:

$$f_{\text{mono}} = \frac{N_{\text{monotonic pairs}}}{N_{\text{total pairs}}} \geq 0.4$$

Single-bin detections are automatically rejected as isolated noise.

### 7.4 Beam center estimation and azimuthal angle

The beam center $(y_0, x_0)$ is estimated by fitting the 2-theta map to a radial distance model:

$$\min_{y_0, x_0} \sum_{y, x} \left(2\theta[y, x] - k \sqrt{(y - y_0)^2 + (x - x_0)^2}\right)^2$$

where the proportionality constant $k = \frac{\sum 2\theta \cdot d}{\sum d^2}$ is solved analytically. Optimization uses Nelder-Mead simplex on a downsampled grid (step = 10 pixels).

For each feature, the azimuthal angle (chi) relative to the beam center is:

$$\chi = \arctan\!\left(\frac{y_{\text{det}} - y_0}{x_{\text{det}} - x_0}\right) \cdot \frac{180}{\pi}$$

This characterizes the crystallographic orientation of each grain.

### 7.5 Feature output

Kept features are saved as a structured JSON catalog (`feature_catalog_3x3.json`):

```json
{
  "feature_id": 1,
  "reflection": "(001)",
  "detector_x": 500,
  "detector_y": 450,
  "peak_intensity": 1250.5,
  "mean_snr": 8.3,
  "n_bins": 7,
  "spatial_extent": ["3_4", "3_5", "4_4", "4_5", "4_6", "5_4", "5_5"],
  "center_bin": "4_4",
  "center_row": 4,
  "center_col": 4,
  "intensity_profile": {
    "3_4": {"intensity": 450.2, "integrated": 1200.5},
    "4_4": {"intensity": 1250.5, "integrated": 3200.0}
  },
  "chi_deg": -45.3,
  "reason": "Gaussian-like: 85% monotonic, 7 bins"
}
```

---

## 8. Feature Verification on the Device Grid

### 8.1 Feature size determination

Each confirmed feature spans a set of spatial bins on the device grid. The **feature size** is characterized by:

- **Number of bins** ($N_{\text{bins}}$): the count of distinct spatial bins in which the peak was detected
- **Spatial extent**: the bounding box of the feature on the bin grid, given by the min/max row and column indices

$$\text{extent} = (r_{\min}, r_{\max}, c_{\min}, c_{\max})$$

- **Equivalent diameter**: approximating the feature as circular:

$$d_{\text{eq}} = 2\sqrt{\frac{N_{\text{bins}}}{\pi}} \cdot \Delta_{\text{bin}}$$

where $\Delta_{\text{bin}}$ is the physical size of one bin (3 scan positions, each separated by the scan step size).

### 8.2 Feature center

The center of each feature on the device grid is the bin with maximum cleaned intensity:

$$(r^*, c^*) = \arg\max_{(r, c) \in \text{feature}} I_{\text{peak}}^{(r, c)}$$

This intensity-weighted definition is more robust than a geometric centroid because it places the center at the physical location of strongest diffraction, which corresponds to the crystal grain center.

### 8.3 Boundary expansion for visualization

To provide visual context around each feature, a boundary expansion algorithm creates a soft halo:

For each data-containing cell in the feature grid, its empty 8-neighbors are assigned an interpolated boundary value:

$$I_{\text{boundary}}[r', c'] = 0.3 \cdot \bar{I}_{\text{neighbors}}$$

where $\bar{I}_{\text{neighbors}}$ is the mean intensity of all data-containing neighbors of the empty cell. This creates a smooth transition from the feature to the background without altering the actual data values.

---

## 9. Device Visualization

### 9.1 Spatial heatmap

The feature viewer (`feature_viewer.py`) renders a three-panel interactive display:

**Left panel -- Device heatmap:** A 2D intensity map on the spatial bin grid, showing the feature's profile across adjacent bins. The intensity value at each bin can be displayed as either:
- **Integrated intensity**: area under the peak in the background-subtracted image (default)
- **Peak pixel intensity**: maximum pixel value at the peak location

The heatmap uses a padded bounding box ($\pm 3$ bins around the feature extent) with NaN masking for empty bins.

**Center panel -- Detector image:** The full 1062 x 1028 detector image for the selected bin, with the current feature circled and other features in that bin annotated. Optional 2-theta arc overlays show the expected positions of all reflections.

**Right panel -- Controls:** Navigation (category, reflection, feature index), visualization settings (colormap, contrast, log scale), and noise reduction parameters.

### 9.2 3D isometric profile

An isometric bar chart visualizes the intensity profile in 3D, with bin position on the x-y plane and intensity on the z-axis. This provides an intuitive view of the Gaussian spatial envelope expected for a genuine Bragg peak.

Clicking on a bin in the heatmap:
1. Highlights that bin in the 3D view (full opacity, others dimmed to 25%)
2. Loads the corresponding detector image
3. Rotates the 3D view to face the selected bin from the nearest corner:

$$\text{azimuth} = \arg\min_{a \in \{-135, -45, 45, 135\}} \left|\left(\theta_{\text{sel}} - a + 180\right) \bmod 360 - 180\right|$$

where $\theta_{\text{sel}} = \arctan\!\left(\frac{r_{\text{sel}} - \bar{r}}{c_{\text{sel}} - \bar{c}}\right) \cdot \frac{180}{\pi}$.

### 9.3 Surface interpolation mode

An optional continuous surface mode uses distance-weighted interpolation for empty bins:

$$I_{\text{surface}}[r, c] = I_{\text{nearest}}[r, c] \cdot \exp(-0.8 \cdot d_{\text{nearest}})$$

where $d_{\text{nearest}}$ is the Euclidean distance to the nearest data-containing bin, computed via `scipy.ndimage.distance_transform_edt`. At data locations, the original values are preserved.

### 9.4 Device location mini-map

A small overview panel shows the full device grid (52 x 74 bins for 3x3) with the current feature's center marked as a red dot, providing spatial context for where on the sample each feature resides.

### 9.5 Feature differentiation

Features are differentiated on the device visualization by:

1. **Reflection type**: each reflection (e.g., (001), (011), PbI2) can be selected independently, showing only features from that lattice plane
2. **Category**: kept vs. filtered features are separated, with counts per reflection
3. **Spatial signature**: the number of bins, spatial extent, and Gaussian profile shape distinguish large, well-defined features from small or noisy ones
4. **Intensity scale**: a shared colormap (default: inferno) maps intensity to color, with adjustable contrast percentiles ($v_{\min}$, $v_{\max}$) and optional log scaling via $\log(1 + I)$
5. **Interactive annotation**: hovering over the detector image reveals nearby features with their ID, reflection label, SNR, and bin count

---

## 10. Device Map: Full-Device Crystallographic Visualization

The device map (`device_map.py`) provides a whole-device overview by projecting all confirmed features onto the 52 x 74 spatial bin grid, with four switchable physical metrics. The visualization consists of a dual-panel layout: a pre-rendered 3D bar plot (rotatable across 8 cardinal viewing angles) and a live 2D heatmap that supports metric switching. Each metric reveals a different aspect of the crystalline microstructure.

### 10.1 Per-bin crystallographic measurements

During the spatial feature analysis (Section 7), each peak detection records not only its intensity but also its precise position on the detector. From this position, two crystallographic quantities are extracted per bin:

**Measured 2-theta.** The 2-theta value at the exact peak pixel, read directly from the pre-computed 2-theta map:

$$2\theta_{\text{meas}}^{(b)} = \text{tth\_map}\!\left[y_{\text{peak}}^{(b)},\; x_{\text{peak}}^{(b)}\right]$$

where superscript $(b)$ denotes the spatial bin and $(x_{\text{peak}}, y_{\text{peak}})$ is the intensity-weighted centroid of the detected peak. This value is stored to 5 decimal places (e.g., 18.29974 degrees) in the feature catalog under each bin's profile entry.

**Azimuthal angle (chi).** The angle of the peak's position relative to the beam center, measured in the detector plane:

$$\chi^{(b)} = \arctan\!\left(\frac{y_{\text{peak}}^{(b)} - y_0}{x_{\text{peak}}^{(b)} - x_0}\right) \cdot \frac{180}{\pi}$$

where $(y_0, x_0)$ is the estimated beam center (see Section 7.4). Chi ranges from $-180\degree$ to $+180\degree$ and encodes the orientation of each crystal grain on the Debye-Scherrer ring.

### 10.2 Metric 1 -- Integrated intensity

**Physical meaning.** The integrated intensity of a Bragg peak is proportional to the volume of crystalline material satisfying the diffraction condition at that spatial position, weighted by the structure factor of the reflection and the illumination intensity. For a given reflection $(hkl)$:

$$I_{\text{int}} \propto V_{\text{grain}} \cdot |F_{hkl}|^2 \cdot L_p \cdot A \cdot I_0$$

where $V_{\text{grain}}$ is the diffracting grain volume, $|F_{hkl}|^2$ is the structure factor, $L_p$ is the Lorentz-polarization factor, $A$ is the absorption correction, and $I_0$ is the incident beam intensity. Since $|F_{hkl}|^2$, $L_p$, and $I_0$ are constant across the device for a given reflection, spatial variations in integrated intensity directly map spatial variations in crystalline grain volume.

**Computation.** For each bin, the integrated intensity is summed over the connected component of the detected peak in the background-subtracted image:

$$I_{\text{int}}^{(b)} = \sum_{(y,x) \in \text{component}} I_{\text{cleaned}}^{(b)}[y, x]$$

**Visualization.** Each reflection is rendered in its own colormap (Blues for (001), Reds for (011)) with alpha-blended compositing. Non-zero bins are colored proportionally to their integrated intensity:

$$\alpha[r, c] = \text{clip}\!\left(\frac{I_{\text{int}}[r,c]}{\max(I_{\text{int}})} \times 0.9 + 0.1,\; 0,\; 1\right)$$

When multiple reflections overlap at the same spatial bin, the higher-intensity reflection takes visual precedence via max-alpha compositing.

### 10.3 Metric 2 -- Strain ($\Delta 2\theta$)

**Physical background.** In crystallography, lattice strain manifests as a shift in the measured diffraction angle relative to the unstrained reference value. This relationship is governed by the differential form of Bragg's law. Starting from:

$$\lambda = 2 d_{hkl} \sin\theta$$

and differentiating with respect to the interplanar spacing $d_{hkl}$ while holding $\lambda$ constant:

$$0 = 2 \sin\theta \cdot \Delta d + 2 d \cos\theta \cdot \Delta\theta$$

Rearranging gives the **strain-angle relationship**:

$$\varepsilon_{hkl} = \frac{\Delta d_{hkl}}{d_{hkl}} = -\cot\theta \cdot \Delta\theta = -\frac{\cos\theta}{\sin\theta} \cdot \Delta\theta$$

where $\varepsilon_{hkl}$ is the lattice strain along the $(hkl)$ direction. For small angular shifts, this is often expressed in terms of the full $2\theta$ shift:

$$\varepsilon_{hkl} \approx -\frac{\Delta(2\theta)}{2 \tan\theta}$$

A positive $\Delta(2\theta)$ (peak shifted to higher angle) corresponds to compressive strain ($\varepsilon < 0$, smaller $d$-spacing), while a negative $\Delta(2\theta)$ indicates tensile strain ($\varepsilon > 0$, larger $d$-spacing). This is a direct consequence of Bragg's law: compressing the lattice ($d$ decreases) requires a larger scattering angle to maintain constructive interference.

**Computation in this pipeline.** For each feature, the reference 2-theta value $2\theta_{\text{ref}}$ is the ideal position for the assigned reflection (e.g., 7.51422 degrees for (001)), taken from the known crystal structure. The strain proxy is the raw angular deviation:

$$\Delta(2\theta)^{(b)} = 2\theta_{\text{meas}}^{(b)} - 2\theta_{\text{ref}}$$

where $2\theta_{\text{meas}}^{(b)}$ is the 2-theta value at the peak pixel in bin $b$, and $2\theta_{\text{ref}}$ is stored in the feature catalog as `ref_tth`.

For example, a feature assigned to the (112) reflection with $2\theta_{\text{ref}} = 18.42549\degree$ showing a measured value of $2\theta_{\text{meas}} = 18.29974\degree$ yields:

$$\Delta(2\theta) = 18.29974 - 18.42549 = -0.12575\degree$$

This negative shift indicates tensile strain (lattice expanded relative to the reference). The corresponding strain magnitude:

$$\varepsilon_{112} \approx -\frac{-0.12575 \times \pi/180}{2 \tan(9.213\degree)} = +6.76 \times 10^{-3}$$

**Why strain varies across the device.** In perovskite thin-film photovoltaic devices, lattice strain arises from multiple sources:

- **Thermal mismatch**: differential thermal expansion between the perovskite film and the substrate (ITO/glass) during crystallization from solution introduces biaxial strain at the interface
- **Compositional gradients**: spatial variations in halide (I/Br) ratio cause local changes in lattice parameter; Vegard's law predicts $d(x) = x \cdot d_A + (1-x) \cdot d_B$
- **Grain boundary effects**: stress concentrations at grain boundaries and between grains with different orientations
- **Film thickness variations**: regions of different thickness experience different substrate constraint

Mapping $\Delta(2\theta)$ across the device reveals these strain distributions with spatial resolution set by the bin size (~3 scan steps, typically 100-300 nm per step).

**Visualization.** Strain is rendered using a diverging colormap (RdBu_r) centered at zero:

$$\text{norm}(v) = \frac{v - (-v_{\max})}{2 v_{\max}}, \quad v_{\max} = \max\!\left(|\Delta(2\theta)|\right)$$

Blue regions indicate tensile strain (negative $\Delta 2\theta$, expanded lattice), and red regions indicate compressive strain (positive $\Delta 2\theta$, compressed lattice). This symmetric normalization ensures that zero strain appears as white, making it immediately visually apparent which device regions are under tension versus compression.

### 10.4 Metric 3 -- Chi (azimuthal angle)

**Physical background.** In a 2D X-ray diffraction geometry, each Bragg peak from a crystalline grain appears as a spot along the Debye-Scherrer ring at the 2-theta angle corresponding to its lattice plane. The **azimuthal position** of the peak along that ring is determined by the orientation of the diffracting lattice plane relative to the incident beam and detector.

In the general diffraction geometry, the scattering vector $\mathbf{Q}$ for a Bragg peak at detector position $(x, y)$ can be decomposed into radial ($2\theta$) and azimuthal ($\chi$) components:

$$\mathbf{Q} = \frac{4\pi}{\lambda} \sin\theta \cdot \hat{n}(\chi)$$

where $\hat{n}(\chi)$ is a unit vector in the plane perpendicular to the incident beam, parameterized by the azimuthal angle $\chi$. In the small-angle, flat-detector approximation used here:

$$\chi = \arctan\!\left(\frac{y - y_0}{x - x_0}\right)$$

where $(x_0, y_0)$ is the direct beam position on the detector.

**Crystallographic interpretation.** For a given $(hkl)$ reflection, the chi angle encodes the in-plane orientation of the crystal grain's lattice. Two grains with the same $d_{hkl}$ spacing (same $2\theta$) but different in-plane rotations will produce peaks at different chi values along the same Debye-Scherrer ring. The relationship between chi and the grain's crystallographic orientation involves the rotation matrix $\mathbf{R}(\phi_1, \Phi, \phi_2)$ parameterized by Euler angles:

$$\chi_{hkl} = \arctan\!\left(\frac{(\mathbf{R} \cdot \mathbf{G}_{hkl})_y}{(\mathbf{R} \cdot \mathbf{G}_{hkl})_x}\right)$$

where $\mathbf{G}_{hkl}$ is the reciprocal lattice vector for the $(hkl)$ plane. In a polycrystalline film with random in-plane orientation, chi values would be uniformly distributed around the ring. Clustering of chi values at specific angles indicates preferred orientation (texture) in the film.

**Computation.** Chi is computed per-bin during the spatial feature analysis (Section 7) and stored in each bin's intensity profile entry. The feature-level chi (`chi_deg`) is computed from the mean detector position of all bins in the feature:

$$\chi_{\text{feat}} = \arctan\!\left(\frac{\bar{y}_{\text{det}} - y_0}{\bar{x}_{\text{det}} - x_0}\right) \cdot \frac{180}{\pi}$$

**Visualization.** Chi is rendered using the `twilight` cyclic colormap, which wraps smoothly from $-180\degree$ to $+180\degree$, appropriate for an angular quantity:

$$\text{norm}(\chi) = \frac{\chi - \chi_{\min}}{\chi_{\max} - \chi_{\min}}$$

Each feature is labeled with its chi value in the annotation overlay (e.g., "$\chi = -45\degree$"), enabling direct visual correlation between spatial position on the device and crystal grain orientation. Features sharing similar chi values across different spatial positions likely belong to the same extended grain or a family of grains with correlated orientations.

### 10.5 Metric 4 -- Chi deviation ($\Delta\chi$)

**Physical background.** While chi encodes absolute grain orientation, the **chi deviation** measures how much each spatial bin's azimuthal angle deviates from the feature's mean orientation:

$$\Delta\chi^{(b)} = \chi^{(b)} - \bar{\chi}_{\text{feat}}$$

where $\chi^{(b)}$ is the per-bin chi value and $\bar{\chi}_{\text{feat}}$ is the feature-level mean chi.

**Interpretation.** A non-zero chi deviation within a single feature indicates that the crystal grain's orientation varies spatially -- the grain is not a perfect single crystal but exhibits **mosaic spread** or **grain bending**. In perovskite thin films:

- **Mosaic spread**: polycrystalline grains are composed of slightly misoriented sub-blocks (mosaic blocks), each producing a Bragg peak at a slightly different chi. The mosaic spread $\eta$ is related to the FWHM of the chi distribution
- **Grain curvature**: continuous bending of a crystal grain (e.g., due to substrate curvature or thermal gradients during growth) produces a systematic spatial gradient in chi, appearing as a $\Delta\chi$ that varies monotonically across the feature
- **Twinning**: certain crystal structures (including tetragonal perovskites) can form twin domains related by specific rotation angles, which would appear as discrete chi jumps within a single spatial feature

The magnitude of $\Delta\chi$ is directly related to the crystallographic tilt angle between different parts of the grain. For small deviations:

$$\Delta\chi \approx \Delta\phi \cdot \sin\Phi$$

where $\Delta\phi$ is the change in the first Euler angle (in-plane rotation) and $\Phi$ is the polar tilt of the lattice plane normal relative to the sample normal.

**Visualization.** Chi deviation uses the same diverging colormap (RdBu_r) as strain, centered at zero:

$$\text{norm}(\Delta\chi) = \frac{\Delta\chi + |\Delta\chi|_{\max}}{2 |\Delta\chi|_{\max}}$$

White indicates bins aligned with the feature mean; blue and red indicate clockwise and counterclockwise deviations respectively. A uniform white feature indicates a well-oriented single grain, while color variation reveals internal misorientation.

### 10.6 3D bar chart rendering

The device map includes pre-rendered 3D bar chart views for the intensity metric. Each confirmed feature produces a bar at its spatial bin position on the 52 x 74 grid, with bar height proportional to integrated intensity.

Bars are colored by their reflection-specific colormap with intensity-proportional normalization:

$$\text{color}[r, c] = \text{cmap}\!\left(\frac{I_{\text{int}}[r,c]}{I_{\text{max}}^{\text{ref}}}\right)$$

Eight viewing angles are pre-rendered at 150 DPI to avoid real-time 3D rendering overhead:

| View | Elevation | Azimuth |
|------|-----------|---------|
| N | 30 deg | 0 deg |
| NE | 30 deg | -45 deg |
| E | 30 deg | -90 deg |
| SE | 30 deg | -135 deg |
| S | 30 deg | -180 deg |
| SW | 30 deg | -225 deg |
| W | 30 deg | -270 deg |
| NW | 30 deg | -315 deg |

The user rotates between views using arrow buttons or keyboard keys, cycling through the pre-rendered images.

### 10.7 Annotation overlay

Each feature is annotated on the 2D heatmap with:
- A colored marker at its center bin (blue for (001), red for (011))
- A text label showing feature ID, reflection name, and chi angle: `#5 (001) chi=-45 deg`
- An anti-collision algorithm prevents overlapping labels: features within 3 rows and 8 columns of an already-labeled feature are shown as small dots without text

---

## 11. Mathematical Reference

### 11.1 Core formulas

| Quantity | Formula |
|----------|---------|
| Wavelength | $\lambda = 1.23984 \times 10^{-9} / E$ (m) |
| Real-space pixel | $\Delta x = \lambda L / (p N)$ |
| 2-theta | $2\theta = \arctan(p \cdot r / L)$ |
| Bragg's law | $n\lambda = 2 d_{hkl} \sin\theta$ |
| Strain-angle relation | $\varepsilon_{hkl} = -\Delta(2\theta) / (2\tan\theta)$ |
| Strain (raw) | $\Delta(2\theta) = 2\theta_{\text{meas}} - 2\theta_{\text{ref}}$ |
| Radial profile | $\tilde{I}_i = \text{median}(\{I[y,x] : \text{bin}(y,x) = i\})$ |
| MAD noise estimate | $\hat{\sigma} = 1.4826 \cdot \text{median}(\|x_i - \text{median}(x_i)\|)$ |
| White top-hat | $\text{tophat}(I) = I - \delta_s(\varepsilon_s(I))$ |
| SNR | $\text{SNR} = (I_{\text{peak}} - \mu) / \hat{\sigma}$ |
| Compactness | $C = \frac{\min(h,w)}{\max(h,w)} \cdot \frac{N_{\text{px}}}{hw}$ |
| Weighted centroid | $c_x = \sum x_i w_i / \sum w_i$ |
| F1 score | $F_1 = 2 \cdot \text{Prec} \cdot \text{Rec} / (\text{Prec} + \text{Rec})$ |
| Azimuthal angle | $\chi = \arctan\!\left(\frac{y - y_0}{x - x_0}\right) \cdot 180/\pi$ |
| Chi deviation | $\Delta\chi^{(b)} = \chi^{(b)} - \bar{\chi}_{\text{feat}}$ |

### 11.2 Key parameters

| Parameter | Value | Used in |
|-----------|-------|---------|
| Top-hat size | 15 px | Morphological filtering |
| 2-theta tolerance | 0.4 deg | Band masking |
| SNR threshold | 4.0 | Per-band detection |
| Min component size | 3 px | Component filtering |
| Max component size | 150 px | Component filtering |
| Min compactness | 0.12 | Arc rejection |
| NMS distance | 15 px | Cross-band dedup |
| Max detections/bin | 25 | Output cap |
| Spatial link tolerance | 5 px | Union-Find linking |
| Monotonicity threshold | 40% | Gaussian profile filter |
| CV threshold | 0.05 | Flat profile rejection |
| Local SNR threshold | 3.0 | Label transfer verification |
| Annulus radii | 10--20 px | Label transfer verification |
| Matching tolerance | 40 px | F1 evaluation |

---

## 12. Complete Pipeline Summary

```
Raw HDF5 scan files (150 files x 167 frames = 25,003 patterns)
    |
    v
[1] Clean bad pixels (clamp negatives, remove hot pixels)
    |
    v
[2] Sum all frames --> Integrated intensity image (1062 x 1028)
    |
    v
[3] pyFAI calibration --> 2-theta map (tth.tiff)
    |
    v
[4] Reconstruct serpentine scan grid from stage positions
    |
    v
[5] Spatial binning: group into 5x5 and 3x3 superpixels
    |
    |-- 5x5 bins (1,188 bins, ~25 frames/bin)
    |       |
    |       v
    |   [6] Manual labeling of Bragg peaks (interactive tool)
    |       |
    |       v
    |   [7] CVEvolve: evolve detection algorithms (11 rounds, 33 candidates)
    |       |           Baseline F1 = 0.022 --> Winner F1 = 0.790
    |       |           Key innovation: top-hat + band-adaptive MAD SNR
    |       |
    |       v
    |   [8] Transfer labels to 3x3 bins via spatial overlap + SNR verification
    |
    |-- 3x3 bins (3,253 bins, ~9 frames/bin)
            |
            v
        [9] Apply winning algorithm to all 3,253 bins
            |
            v
       [10] Link peaks across adjacent bins (Union-Find, 5px tolerance)
            |
            v
       [11] Gaussian profile filtering (CV > 0.05, monotonicity > 40%)
            |       |
            |       +--> Filtered peaks (single-bin, flat, non-Gaussian)
            v
       [12] Feature catalog: kept features with spatial extent, intensity
            profile, SNR, chi angle, reflection assignment
            |
            v
       [13] Interactive visualization: device heatmap, detector image,
            3D intensity profile, feature differentiation by reflection
            and spatial signature
            |
            v
       [14] Device map: full-device crystallographic visualization
            with four switchable metrics:
              - Integrated intensity (grain volume)
              - Strain (Delta-2theta from Bragg's law)
              - Chi (azimuthal grain orientation)
              - Chi deviation (mosaic spread / grain bending)
```

---

## 13. Software and Dependencies

| Component | Tool / Library |
|-----------|---------------|
| Raw data I/O | h5py, hdf5plugin (LZ4) |
| 2-theta calibration | pyFAI |
| Image processing | scipy.ndimage (label, minimum/maximum_filter) |
| Curve fitting | scipy.optimize.curve_fit |
| Scan grid detection | scipy.signal.argrelextrema |
| Visualization | matplotlib, tifffile |
| Interactive labeling | PyQt5, matplotlib backends |
| Algorithm evolution | CVEvolve (Claude Opus 4.6 via Argo API) |
| Array computation | NumPy |
