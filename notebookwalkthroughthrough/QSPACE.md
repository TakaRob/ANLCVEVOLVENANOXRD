# Q-space mapping — `xrd-app qspace`

The GUI for this is hidden in XRD-APP GUI
Convert detector pixels (+ the sample θ of a rocking scan) into **3D reciprocal-
space vectors** `Q = (qx, qy, qz)`. This is the third geometry layer of the app:
the pipeline otherwise works in 2θ-radial + χ-azimuth space
(`xrd_app/core/geometry.py`, `xrd_app/core/rocking.py`);
`xrd_app/core/qspace.py` adds the full 3D scattering vector so a θ
series can be assembled into a reciprocal-space map (RSM) and **lattice tilt**
separated from **microstrain**.

> Built on [`xrayutilities`](https://xrayutilities.sourceforge.io/) for the
> default 3D geometry path and [`pyFAI`](https://pyfai.readthedocs.io/) for
> calibrated `.poni` geometry. Both paths are cross-checked against independent
> lab-frame calculations in the q-space tests.

---

## Why: what q-space gives you

For each detector pixel the scattering vector splits into two physically distinct
directions:

| Direction | Physical meaning | How to read it |
|-----------|------------------|----------------|
| **Radial** — `\|Q\| = 4π·sin(θ)/λ` | lattice spacing `d = 2π/\|Q\|` | strain requires comparison with a justified unstrained/reference value |
| **Transverse** — direction of `Q` at fixed `\|Q\|` | grain orientation / tilt | spread or drift of the `Q` direction across sample theta |

`qmagnitude_map(tth, λ)` obtains radial `\|Q\|` directly from the supplied 2θ
map. The CLI output instead stores `q_mag = sqrt(qx²+qy²+qz²)` from the selected
3D geometry path, so the default flat-fit result can inherit fit residual from
unmodeled detector tilt. A calibrated `--poni` includes the supplied tilt model.

---

## Install

```bash
pip install -e '.[qspace]'          # xrayutilities — the default path
pip install -e '.[qspace,poni]'     # + pyFAI — enables --poni (tilt-accurate)
```

## Quick start

```bash
# One scan, geometry recovered from its 2θ map (flat-detector fit):
xrd-app qspace --root <project> --scans Scan_0203 --bin-size 3

# The whole rocking series (θ pulled from the built-in 203–214 table):
xrd-app qspace --root <project> \
  --scans "Scan_0203,Scan_0204,Scan_0205,Scan_0207,Scan_0208,Scan_0209,\
Scan_0210,Scan_0211,Scan_0212,Scan_0213,Scan_0214"

# Tilt-accurate directions from a pyFAI calibration:
xrd-app qspace --root <project> --scans Scan_0203 --poni Metadata/calib.poni
```

Useful options: `--energy` (eV, default 15000), `--pixel-size` (m, default 75e-6),
`--theta` (deg, override the table), `--tth-path`, `--out-dir` (default
`Study/qspace/`), `--no-intensity` (skip the summed-image layer if you only want
geometry — but `xrd-app rsm` needs it).

---

## Fusing the θ series into a 3D RSM — `xrd-app rsm`

Once each scan has a q-map, fuse them into one **binned 3D reciprocal-space
volume** `I(qx,qy,qz)` — the reciprocal-space analogue of `combined-device`:

```bash
xrd-app rsm --root <project> --bins 128            # all *_qmap.npz in Study/qspace/
xrd-app rsm --root <project> --scans "Scan_0203,Scan_0207" --out Study/rsm.npz
```

Each scan's summed detector intensity (stored in its `.npz` by `qspace`) is
median-subtracted, thresholded (`--min-intensity`), and histogrammed by its
`(qx,qy,qz)` into a shared grid, accumulated across θ. Options: `--bins` (voxels
per axis, default 128 → a 128³ volume), `--min-intensity`,
`--subtract-median/--no-subtract-median`, `--in-dir`, `--out`.

**Output** `Study/rsm.npz`: `volume` plus per-voxel `counts` of retained detector
pixels after finite-value filtering, subtraction, and thresholding. It is not a
threshold-independent geometric coverage map. The file also contains the
`qx/qy/qz` `_edges` and `_centers`, and three max-intensity
2D projections (`proj_qx_qy`, `proj_qx_qz`, `proj_qy_qz`) for quick viewing
without 3D tooling. A `.summary.json` reports grid shape, q-ranges, total/peak
intensity and fill fraction.

```python
import numpy as np, matplotlib.pyplot as plt
d = np.load("Study/rsm.npz")
plt.imshow(d["proj_qx_qy"].T, origin="lower",
           extent=[d["qx_edges"][0], d["qx_edges"][-1],
                   d["qy_edges"][0], d["qy_edges"][-1]])
plt.xlabel("qx (1/Å)"); plt.ylabel("qy (1/Å)")
```

> This RSM integrates over the whole illuminated (x,y) map at each θ (a
> sample-integrated RSM). For a **per-grain** reciprocal-space view, use the
> feature q-coordinates in `<scan>_features_q.csv` (each detected grain is one
> point in Q) rather than the fused volume.

---

## Geometry: flat fit vs. `.poni`

**Default (no `--poni`).** The beam center and sample-detector distance are
recovered by a least-squares flat-detector fit of the 2θ map
(`xrd_app.core.qspace.recover_geometry`). The printed `fit-RMS` measures mismatch
between that model and the supplied map. The resulting vectors and stored
`q_mag` are exact for the fitted flat geometry, not necessarily for every
original 2θ-map pixel.

**Calibrated (`--poni`).** A pyFAI `.poni` carries distance, beam-center
parameters, and rotations (`rot1/rot2/rot3`), allowing detector tilt to be
modeled. The resulting directions are as accurate as the supplied calibration.
The frame conversion is cross-checked in `xrd_app/tests/test_qspace.py`.

### Getting a real `.poni`
Calibrate on a standard (LaB₆ / CeO₂) with pyFAI:

```bash
pyFAI-calib2      # GUI: pick calibrant + energy, refine rings → save .poni
# or headless:
pyFAI-calib -e 15000 -c LaB6 -D Detector calibration_image.edf
```

Then point `--poni` at the result. An **example template** ships at
`xrd_app/assets/example.poni` — it matches this detector's size/energy but has
**zero tilt** (recovered from `tth.tiff`, not a calibration), so it reproduces the
default flat path. Use it to see the file format; replace it with a real
calibration for tilt accuracy.

> The detector distance for this setup is **~0.33 m** (recovered from the 2θ map:
> 2θ spanning 4.8→19.8° across ~77 mm of detector at 75 µm pixels *requires*
> ~0.33 m). An earlier "6.16 m" note in the docs was an error and has been removed.

---

## Outputs (per scan, in `--out-dir`)

| File | Contents |
|------|----------|
| `<scan>_qmap.npz` | Detector-shaped `qx, qy, qz, q_mag`, scan/geometry metadata, and optional `intensity` when requested and available |
| `<scan>_qmap.summary.json` | Energy, wavelength, theta, geometry, q range, and feature count |
| `<scan>_features_q.csv` | Written when the selected default shape/combined catalog has features with detector coordinates; tags each at rounded/clipped `(detector_y, detector_x)` |

Load a q-map:

```python
import numpy as np
d = np.load("Study/qspace/Scan_0203_qmap.npz")
qx, qy, qz, qmag = d["qx"], d["qy"], d["qz"], d["q_mag"]
print("distance", float(d["distance_m"]), "θ", float(d["theta_deg"]))
```

### Physics check (do this)
A feature labeled `(002)` (ref 2θ = 15.01°) should land near
`\|Q\| ≈ 1.96 Å⁻¹` (`d ≈ 3.16 Å`). Compare the stored geometry-derived `q_mag`
with the direct radial value `4π·sin(tth_pixel/2)/λ`. On the default path, a
systematic discrepancy can reveal flat-fit residual or detector tilt; a large
feature-specific discrepancy suggests incorrect pixel or catalog assignment.

---

## Programmatic use

```python
from xrd_app.core import qspace as qs
import tifffile

tth = tifffile.imread("tth.tiff").astype("float64")
lam = qs.wavelength_angstrom(15000.0)                 # Å

# exact |Q| (radial / strain axis) — no geometry needed
qmag = qs.qmagnitude_map(tth, lam)

# flat-fit geometry + 3D vectors at θ (xrayutilities)
geom = qs.recover_geometry(tth)                       # beam center, distance, RMS
qx, qy, qz = qs.q_vectors(tth, geom, energy_ev=15000.0, theta_deg=20.5)

# tilt-accurate 3D vectors from a .poni (pyFAI)
qx, qy, qz = qs.q_vectors_from_poni("calib.poni", energy_ev=15000.0, theta_deg=20.5)
```

---

## Notes & next steps
- **Theta source.** `--theta` overrides; otherwise theta comes from
  `xrd_app.core.tracking.THETA_BY_SCAN`. For an unknown scan the CLI warns and
  uses `0.0` degrees, so pass `--theta` explicitly to avoid a zero-angle map.
- **One frame.** The flat (xrayutilities) and `.poni` (pyFAI) paths return `Q` in
  the same lab frame (beam +x, vertical z; sample rocks about z), so their outputs
  are directly comparable.
- **Background.** The `rsm` intensity is the summed detector image over all (x,y)
  bins at each θ, so a broad radial background rides under the Bragg spots;
  `--subtract-median` (default) removes the flat part, `--min-intensity` trims the
  rest. For clean per-grain work prefer the feature q-points.
- **GUI.** The Reciprocal Space tab (`xrd-app gui`, or standalone
  `python -m xrd_app.tabs.rsm`) supports 2D max-intensity projections and a 3D
  volume with the feature cloud overlaid. Install the `gl` extra for the
  PyOpenGL-backed 3D view. Build data with `qspace` then `rsm`, or orchestrate the
  complete registered study with `xrd-app run-study --with-rsm`.
