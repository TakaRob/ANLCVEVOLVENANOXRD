Xrd-app gui is a single gui with tabs
using PyQt5
Creating Project Directories


Raw Data (Formats?) 
We use beamline 19, where we calculate the coordinates.csv from interferometer data.




Binning
- What type, what were the pros and cons? show perfect row image
Peak finding (Labeling, CVEvolve, Algorithm Testing?)
 - figure out if it will work on diff scans
 - Runs a detector on each bin.
Shape Finding (what is this, methods, tolerance adjustments)
Rocking analysis for 3D



Claude.md and it's respective .mds will be instructions for agents using the CLI tools, (agent.md? we can have multiple) you have general access to tools linking, etc etc. whatever.
use this format in responese, you have 15 features ...
view using the gui. 


Single GUI (PyQt5, tabbed)
  Setup │ Programs │ View/Label │ Shape/Verify │ Device │ Orientation



  File Structure

  ```
Directory
| Project Folder
    metadata
    binned
    etc
    etc


└── xrd_app/
    ├── pyproject.toml          # entry point: xrd-app = xrd_app.cli:main
    ├── cli.py  config.py  app.py
    ├── tabs/                   # setup, programs, view_label, shape_verify,



    Data Loading

    - Creating a project.

    - Load single hdf5, will try to autodetect 

    - Loading from PONI, creating your own theta thing.
        can we make that easier



    Gui as Standalone
    - prompt to select file, based on stuff

    Guide for each gui


    Flesh out Save Algorithm better, what does it actually do?(1) sets the chosen
sensitivity, (2) calls the chosen noise-reduction module, (3) delegates to the
base algorithm — conforming to the **detector contract** (`precompute_tth`,
`detect_in_band(...)`) that `core/processing.py` already imports. Template in
`core/save_algorithm.py`.


what was 
xrd-app grid --deskew-method faithful (square-pixel re-grid, true-Y columns) and --deskew-method perrow_offset (legacy "triangle", in xrd_app/core/deskew_legacy.py).
xrd-app shapes --algorithm gaussian_deskew — the linker-stage column-shift fix (xrd_app/ShapeAlgorithms/gaussian_deskew.py). Note: a linker fix cannot move a shape's center_bin, so it cannot actually de-skew positions; kept only for the record.
The experimental deskew data catalogs (*_perrowOffset*, *_preHybrid*, *_faithful*, *_deskew* peaks/shapes in Labels/) were left in place.

 Rocking-Series Study — `5%_DI_Yes_GB` (Scans 203–214)

> nano-XRD orientation/strain study at ISN 26-ID-C (APS). Same sample, same spot,
> swept through sample θ. Goal: detect Bragg features per scan, **predict** which
> shapes should recur across the θ sweep, **track** them, and **verify** the
> prediction — ending in a *combined* device view that fuses all orientations into
> one spatial map. GUI work is deferred; this plan builds the engine + data layers.

---

## 1. What we're studying

- **Sample:** `5%_DI_Yes_GB`, single spot, `samy = −1.61`.
- **Scans:** 203–214 (logbook id 10270). A θ rocking series — the sample is
  rotated through θ while the nano-beam rasters an (x,y) map at each θ.
- **θ table (the orientation variable):**

  | Scan | θ (°) | Scan | θ (°) |
  |------|-------|------|-------|
  | 203 | 20.5 | 209 | 4.5 |
  | 204 | 20.0 | 210 | 4.0 |
  | 205 | 19.5 | 211 | 3.5 |
  | 206 | 6.0 ⚠*incomplete* | 212 | 3.0 |
  | 207 | 5.5  | 213 | 0.0 |
  | 208 | 5.0  | 214 | 20.5 *(repeat of 203 — reproducibility check)* |

  Note the sampling is **clustered**, not uniform: a dense cluster near θ≈3–6° and
  a few points at 19.5–20.5° and 0°. Rocking-curve fits are only meaningful where
  θ is densely sampled (≈3–6°). 203/214 are a duplicate orientation — use them to
  estimate detection repeatability (an empirical noise floor for the comparison).

- **Geometry:** 15 keV, 75 µm pixel, area detector
  `(1062×1028)`. Reflections are fixed 2θ bands (PbI₂, (001), (011), (111), (002),
  ITO, (012), (112), + 3 unlabeled) — see `core/reflections.py`.

- **Physics of the sweep:** a crystalline grain diffracts a given reflection only
  when its orientation satisfies the Bragg condition for the incident beam. As θ
  steps, different grains "light up." For a fixed (x,y) grain, diffracted
  intensity vs θ traces a **rocking curve** peaked at θ_Bragg, whose FWHM is the
  mosaic / rocking spread. χ (azimuth on the detector) should vary *smoothly* with
  θ for a single grain. These are the signals the prediction + tracking exploit.

---

## 2. Scientific goals & hypotheses

- **H1 (recurrence):** a real grain reappears in adjacent-θ scans (within its
  rocking width), at the same de-skewed (X,Y), same reflection band, with χ moving
  smoothly. Isolated single-θ detections are likely noise.
- **H2 (predictability):** from the reflection set + geometry + the first strong
  detections, we can *forecast* the per-θ set of expected shapes (reflection,
  position, χ, approximate intensity) — and the actuals should match it.
- **H3 (rocking curve):** tracked features' intensity(θ) fits a peaked curve;
  θ_Bragg and FWHM are physically sane and reproducible (203 vs 214).
- **Deliverable hypothesis:** fusing all θ into one **combined device view**
  yields a spatial orientation/strain map richer than any single scan.

---