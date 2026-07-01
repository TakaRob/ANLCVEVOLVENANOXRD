# Q-space mapping — `xrd-app qspace`

Convert detector pixels (+ the sample θ of a rocking scan) into **3D reciprocal-
space vectors** `Q = (qx, qy, qz)`. This is the third geometry layer of the app:
the pipeline otherwise works in 2θ-radial + χ-azimuth space (`core/geometry.py`,
`core/rocking.py`); `core/qspace.py` adds the full 3D scattering vector so a θ
series can be assembled into a reciprocal-space map (RSM) and **lattice tilt**
separated from **microstrain**.

> Built on [`xrayutilities`](https://xrayutilities.sourceforge.io/) (the geometry
> engine, validated to machine precision against an independent lab-frame
> calculation) and, for the tilt-accurate path, [`pyFAI`](https://pyfai.readthedocs.io/).

---

## Why: what q-space gives you

For each detector pixel the scattering vector splits into two physically distinct
directions:

| Direction | Physical meaning | How to read it |
|-----------|------------------|----------------|
| **Radial** — `\|Q\| = 4π·sin(θ)/λ` | **microstrain** (lattice spacing `d = 2π/\|Q\|`) | shift in `\|Q\|` vs a reference reflection = Δd/d |
| **Transverse** — direction of `Q` at fixed `\|Q\|` | **tilt / mosaicity** (grain orientation) | spread/drift of the `Q` direction across θ |

`\|Q\|` is **exact from the 2θ map alone** — no geometry fit needed. Splitting
`\|Q\|` into `(qx,qy,qz)` needs the beam center + sample-detector distance (and,
for full accuracy, the detector tilt); see *Geometry* below.

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

**Output** `Study/rsm.npz`: `volume` + per-voxel `counts` (coverage) on a
`(nx,ny,nz)` grid, the `qx/qy/qz` `_edges` and `_centers`, and three max-intensity
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
recovered by a least-squares **flat-detector** fit of the 2θ map
(`core/qspace.recover_geometry`). The per-scan `fit-RMS` printed to the console is
the residual — for this detector it is ~50 mdeg, i.e. **unmodeled detector
tilt**. `\|Q\|` is still exact; only the *direction* of `Q` carries this
~50 mdeg (~0.01 Å⁻¹) error.

**Tilt-accurate (`--poni`).** A pyFAI `.poni` carries the full geometry including
tilt (`rot1/rot2/rot3`), so per-pixel directions are exact. `pyFAI` computes the
lab-frame pixel positions; the app assembles `Q` and applies the θ rotation in the
same frame as the default path (cross-checked in `tests/test_qspace.py`).

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
| `<scan>_qmap.npz` | `qx, qy, qz, q_mag` (detector-shaped, 1/Å) + geometry/meta (`beam_row`, `beam_col`, `distance_m`, `pixel_m`, `rms_deg`, `geometry_source`, `theta_deg`, `energy_ev`, `wavelength_A`) |
| `<scan>_qmap.summary.json` | energy, λ, θ, geometry, `\|Q\|` range, #features annotated |
| `<scan>_features_q.csv` | every detected feature (from `Labels/<scan>`) tagged with `qx, qy, qz, q_mag` at its **detector** pixel (`detector_y`, `detector_x`) — *not* its spatial scan-grid bin |

Load a q-map:

```python
import numpy as np
d = np.load("Study/qspace/Scan_0203_qmap.npz")
qx, qy, qz, qmag = d["qx"], d["qy"], d["qz"], d["q_mag"]
print("distance", float(d["distance_m"]), "θ", float(d["theta_deg"]))
```

### Physics check (do this)
A feature labeled `(002)` (ref 2θ = 15.01°) must land near `\|Q\| ≈ 1.96 Å⁻¹`
(`d ≈ 3.16 Å`). If a feature's `q_mag` disagrees with `4π·sin(θ_com/2)/λ` for its
own `tth_com`, the geometry or the pixel mapping is wrong — not a real signal.
(Per `CLAUDE.md`: validate against physics, not just code.)

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
- **θ source.** `--theta` overrides; otherwise θ comes from
  `core/tracking.THETA_BY_SCAN` (the 203–214 table). Scans not in the table need
  `--theta`.
- **One frame.** The flat (xrayutilities) and `.poni` (pyFAI) paths return `Q` in
  the same lab frame (beam +x, vertical z; sample rocks about z), so their outputs
  are directly comparable.
- **Background.** The `rsm` intensity is the summed detector image over all (x,y)
  bins at each θ, so a broad radial background rides under the Bragg spots;
  `--subtract-median` (default) removes the flat part, `--min-intensity` trims the
  rest. For clean per-grain work prefer the feature q-points.
- **Proof of concept / validation:** `xrd_app/notebooks/qspace_poc.py`.
- **GUI.** The **Reciprocal Space** tab (`xrd-app gui`, or standalone
  `python -m xrd_app.tabs.rsm`) renders the `rsm.npz` projections with the
  per-grain feature cloud overlaid — pick the plane (qx–qy / qx–qz / qy–qz),
  toggle the heatmap/log/cloud, and color the cloud by reflection, θ, or
  intensity. It's render-only over `core.rsm.load_rsm` / `load_feature_cloud`;
  build the data with `xrd-app qspace` then `xrd-app rsm` first.
