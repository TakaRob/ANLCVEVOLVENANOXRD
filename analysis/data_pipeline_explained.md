# Nano-XRD Data Pipeline: Integrated CCD vs Per-Pixel Analysis

## 1. Raw Data Structure (Scan 203)

The beam rasters across the sample in a fly-scan. At each position, the XRD area
detector captures a full 2D diffraction pattern (1062 x 1028 pixels). These are
saved into chunked HDF5 files:

```
Scan_0203/XRD/
  scan_0203_00001.h5   →  entry/data/data: shape (167, 1062, 1028)
  scan_0203_00002.h5   →  shape (167, 1062, 1028)
  ...
  scan_0203_00150.h5   →  shape (167, 1062, 1028)
  scan_0203_00151.h5   →  shape (120, 1062, 1028)   ← partial final chunk
```

- **150 files**, each containing **167 frames** (last file has 120)
- **Total: ~25,003 diffraction patterns**, one per scan position
- Each frame is the full detector image: 1062 rows x 1028 columns, int32

The sample positions are recorded separately in `SOCKETSERVER/` files (195 files,
~835 positions each, 24 columns of encoder values including X/Y stage coordinates).

## 2. What Was Done: The Integrated CCD Image

The file `scan_203_sum.tiff` (also called `integrated_intensity_design.tiff` in the
CVEvolve test data) was created by **summing all ~25,003 frames pixel-by-pixel**:

```
integrated_ccd[y, x] = sum over all positions of frame_i[y, x]
```

This produces a single 1062 x 1028 image. Equivalently, in the notebook (cell c1):

```python
xrd_sum = np.sum(dp_stack, axis=0)   # dp_stack shape: (N_positions, 1062, 1028)
                                      # xrd_sum shape:  (1062, 1028)
```

**What this gives you:** A high-signal-to-noise composite diffraction pattern that
shows all the Bragg peaks present anywhere in the scanned area. Peaks that are bright
at even a few positions accumulate enough counts to stand out.

**What this loses:** All spatial information. You cannot tell *where* on the sample
a particular Bragg peak came from. A peak could be from one strong grain or many
weak ones spread across the scan area.

## 3. How Peaks Were Found (CVEvolve Hotspot Detection)

The hotspot detection algorithm works on the **integrated CCD image only**:

1. Load the integrated image (1062 x 1028) and the 2-theta map (`tth.tiff`, same shape).
   The 2-theta map gives the scattering angle at every detector pixel, computed from
   the detector geometry and calibration (via pyFAI).

2. For each reflection (e.g., (001) at 2-theta = 7.514 deg):
   - Create a mask of all detector pixels where `|tth - 7.514| < line_tol` (default 0.2 deg).
     This selects an arc on the detector.
   - Within that arc, find pixels above the `hotspot_percentile` (default 99th percentile).
   - Connected-component labeling groups adjacent bright pixels into spots.
   - Each spot becomes a bounding box (ROI).

3. Output: a dict of `{reflection_label: [(y0, y1, x0, x1), ...]}` bounding boxes.

**Ground truth labeling** (for CVEvolve evaluation): You used `annotate.py` to
manually click on spot centers in the integrated image for each reflection. These
are stored as `[x, y]` pixel coordinates in `annotations.json`. The F1 metric uses
a 40-pixel matching tolerance.

**Key point:** Both the detection algorithm and the ground truth labels operate
entirely on the single integrated CCD image. There is no spatial (sample position)
information involved.

## 4. What Per-Pixel Analysis Means

Instead of one summed image, you work with each of the ~25,003 individual frames.
Each frame maps to a known (X, Y) position on the sample.

The question becomes: **at each sample position, which Bragg peaks are present
and how strong are they?**

There are two levels of per-pixel analysis:

### Level 1: ROI Intensity Mapping (what the notebook does now)

1. Find Bragg peak ROIs on the integrated CCD (the existing hotspot detection).
2. For each ROI bounding box (e.g., the (001)-1 spot at pixels y:380-460, x:900-980):
   - Go through every individual frame in `dp_stack`.
   - Sum the pixel intensities within that bounding box.
   - Record that scalar intensity for the corresponding sample position.
3. Plot the intensity as a spatial map (X, Y on sample vs. ROI intensity).

This tells you *where on the sample* each grain/peak is located, but relies on
the integrated image to define the ROI locations.

### Level 2: Per-Frame Peak Detection

Run the actual peak detection algorithm on each individual frame independently.
This is much harder because:
- A single frame has ~25,000x less signal than the integrated image.
- Many peaks are invisible in any single frame.
- Background noise dominates.

This is the harder but more scientifically complete approach — it can detect peaks
that move position on the detector between frames (due to strain, for example).

## 5. What You Need to Do for Per-Pixel CVEvolve

### The labeling problem changes

For the **integrated CCD CVEvolve**, you labeled one image:
- 1 image to annotate
- ~30 total spots across all reflections
- Labeling took one sitting with `annotate.py`

For **per-pixel CVEvolve**, you have ~25,003 images to consider. You cannot label
all of them. Options:

#### Option A: Use integrated-CCD ROIs as fixed windows (no new labeling)

Keep the existing approach: detect peaks on the integrated image, define ROI boxes,
then just sum intensities per frame. CVEvolve optimizes the integrated-image detector
as before. The per-pixel mapping is a downstream step, not something CVEvolve
optimizes.

**Pros:** No new labeling. The existing CVEvolve setup works as-is.
**Cons:** Cannot find peaks that are only visible in individual frames or that shift
position.

#### Option B: Label a small representative subset of frames

Select ~20-50 frames that span the range of conditions (some with strong peaks,
some with weak, some with none). Run `annotate.py` on each. Use those as
training/validation data for CVEvolve to optimize a per-frame detector.

**What to label:** For each selected frame, mark the (x, y) pixel location of each
Bragg peak, keyed by reflection label — same format as now. The difference is the
input image is a single frame (low signal) instead of the integrated sum.

**Selecting frames:** Pick frames where you already know peaks exist (from the
Level 1 ROI intensity maps — positions with high ROI intensity should have visible
peaks). Also include frames from "quiet" areas as negative examples.

**Pros:** CVEvolve directly optimizes per-frame detection.
**Cons:** Labeling effort scales with the number of frames. Individual frames are
noisy, making manual annotation harder and less reliable.

#### Option C: Bootstrap labels from the integrated detection

1. Run peak detection on the integrated CCD (existing approach).
2. For each detected ROI, check which individual frames contribute significant
   intensity to that ROI region.
3. Use those as "pseudo-labels" — frames where the ROI is bright get a positive
   label at that location; frames where it's dim get no label.
4. Feed to CVEvolve as training data.

**Pros:** No manual labeling of individual frames. Scales to the full dataset.
**Cons:** Pseudo-labels inherit errors from the integrated detector. Peaks that
move between frames get mislabeled.

### Recommended path

Start with **Option A** — it requires no new labeling and answers the immediate
question of spatial distribution. The existing CVEvolve hotspot detection finds
peaks on the integrated image, and you project those ROIs onto individual frames
for spatial mapping.

If you need CVEvolve to optimize per-frame detection directly (Option B), the
labeling workflow would be:

1. Run the notebook through Section 5 to generate ROI intensity spatial maps.
2. Identify ~10 "interesting" positions (high intensity) and ~10 "background"
   positions from those maps.
3. Extract those individual frames as TIFFs.
4. Label each frame with `annotate.py` (or a modified version).
5. Set up a new CVEvolve session with per-frame test data instead of the
   integrated image.

## 6. Summary Table

| Aspect | Integrated CCD | Per-Pixel (Level 1) | Per-Pixel (Level 2) |
|--------|---------------|--------------------|--------------------|
| Input | Sum of all frames | Sum image + individual frames | Individual frames |
| Detection on | 1 image | 1 image (integrated) | Each frame separately |
| Output | Peak locations on detector | Spatial map of peak intensities | Peak presence at each position |
| Labeling | ~30 spots on 1 image | Same as integrated | ~30 spots on each of ~20-50 frames |
| Signal quality | Excellent (25k frame sum) | N/A (uses integrated peaks) | Very low (single frame) |
| CVEvolve target | Integrated detector algorithm | Same as integrated | Per-frame detector algorithm |
