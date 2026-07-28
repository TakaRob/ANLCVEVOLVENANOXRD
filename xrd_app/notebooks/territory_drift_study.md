# Detector-Drift Territorial Segmentation Study

## Question

Do fixed detector coordinates and square spatial search limits truncate real-space
feature footprints when a Bragg peak drifts smoothly on the detector?

## Initial findings

- The interactive expansion worker in `xrd_app/gui/viewer.py` used the detector
  coordinate of the clicked seed for every candidate cell.
- Candidates had to remain within 5 detector pixels of that fixed coordinate.
- Expansion was additionally clipped to Chebyshev distance 10 from the seed,
  imposing a maximum 21 x 21 square footprint.
- The batch territorial linker is different: it links pairwise across physical
  neighbors and therefore permits cumulative smooth detector drift.
- The Gaussian profile filter separately assumes intensity tends to decrease
  radially from the brightest real-space cell.

## Pilot data

Project: `/home/takaji/rocking_203_214`

Scan: `Scan_0203`

The saved territorial mapping has `target_size=1`, 25,170 one-frame cells, true
stage coordinates, saved per-cell peaks, and saved shape catalogs. The pilot can
therefore run without reading raw detector HDF5 files.

Initial candidate shapes from the existing territorial catalog:

| Feature | Reflection | Cells | Detector range dx,dy (px) | Linear drift R2 |
|---:|:---:|---:|:---:|---:|
| 1289 | (012) | 4523 | 20, 9 | 0.97 |
| 278 | (002) | 3771 | 20, 16 | 0.90 |
| 1209 | (001) | 3723 | 15, 16 | 0.96 |
| 1129 | (002) | 2592 | 17, 31 | 0.85 |
| 1250 | (012) | 2353 | 16, 6 | 0.94 |
| 1297 | (002) | 1159 | 17, 7 | 0.93 |

These total excursions exceed the old fixed 5-pixel seed tolerance.

## Hypotheses

1. Fixed-seed matching truncates smoothly drifting features.
2. The radius-10 Chebyshev cap creates square boundaries independently of peak
   matching.
3. Pairwise frontier matching restores the footprint while preserving a local
   5-pixel continuity constraint.
4. Unconstrained transitive matching may merge nearby same-reflection peaks; this
   must be measured before treating it as production behavior.
5. Radial Gaussian validation may reject valid anisotropic intensity profiles
   even after linking is corrected.

## Methods

- **Old**: candidate detector coordinate must be within 5 pixels of the original
  seed; regular-grid growth is limited to a radius-10 square.
- **New**: candidate detector coordinate must be within 5 pixels of the accepted
  peak in the adjoining frontier cell; there is no fixed square radius.
- Run both methods over saved regular-grid peaks and saved target-size-1
  territorial peaks.
- Display a 2 x 2 comparison: regular old/new and true-coordinate territory
  old/new.

## Implementation

- Added `core.territory.grow_peak_feature`, with controlled `seed` and `frontier`
  modes for testing the two hypotheses on identical detections.
- Changed the interactive expansion worker to compare a candidate with the
  accepted peak in the adjoining frontier cell.
- Removed its hard radius-10 square limit.
- Cached detected peaks so a cell rejected from one frontier can still be tested
  from another adjoining accepted cell.
- Kept the same reflection requirement and 5-pixel local detector tolerance.
- Added synthetic tests for smooth drift, fixed-seed truncation, reflection
  changes, and detector jumps.
- Added `compare_territory_drift.py`, which writes a 2 x 2 old/new figure and a
  JSON metrics file. Gray points are the existing batch footprint; colored
  points are those recovered by the tested growth method. Old and new panels
  use identical axes.

## Scan 203 results

All values compare growth from the catalog feature's brightest cell against the
existing coordinate-linked territorial footprint.

| Feature | Reflection | Reference frames | Territory old recall | Territory new recall | Territory new precision |
|---:|:---:|---:|---:|---:|---:|
| 1289 | (012) | 4523 | 4.71% | 99.89% | 100% |
| 1209 | (001) | 3723 | 6.88% | 100% | 100% |
| 1129 | (002) | 2592 | 8.91% | 100% | 100% |
| 1297 | (002) | 1159 | 18.46% | 100% | 100% |
| 82 | (012) | 1415 | 17.88% | 100% | 100% |

Independent validation on Scan 207 feature 731, reflection (001): the old
territorial method recovered 301 of 4,053 frames (7.43%); the new method
recovered all 4,053 with 100% precision. Its detector peak spans 7 x 15 pixels.
The regular method improved from 11.0% to 86.0% recall, again retaining some
mapping-related fragmentation.

Tolerance sweep for feature 1297:

| Local tolerance | Recall | Precision | Extra frames |
|---:|---:|---:|---:|
| 3 px | 99.40% | 100% | 0 |
| 5 px | 100% | 100% | 0 |
| 7 px | 100% | 99.91% | 1 |

Regular-grid behavior is less consistent. Feature 1209 improves from 10.0% to
99.8% recall, while features 1289, 1129, and 1297 remain severely fragmented.
The regular mapping has 19,916 bins for 25,170 frames, so nominal 1x1 cells can
contain multiple frames and do not preserve the target-size-1 physical
adjacency. Detector-drift following fixes one failure mode but cannot repair
that separate coordinate/bin-collision problem.

The saved peak labels already enforce the expected reflection during growth.
No cross-reflection links occurred. The 3/5/7-pixel sweep shows low merge risk at
5 pixels for this pilot; 7 pixels admitted one extra same-reflection frame.

## Progress

- [x] Located the fixed-seed and square-cap behavior.
- [x] Identified a complete local 1x1 pilot dataset.
- [x] Added reusable old and drift-following segmentation methods.
- [x] Updated interactive expansion to frontier-relative matching.
- [x] Added focused unit tests.
- [x] Created the old/new regular/territory comparison script.
- [x] Ran five Scan 203 features and a 3/5/7-pixel tolerance sweep.
- [x] Confirmed the result on independent Scan 207 feature 731.
- [x] Recorded overlap, detector excursion, and merge-risk measurements.
- [x] Confirmed that the batch territorial linker already uses pairwise links, so
  no batch-linker change is proposed from this result.

## Conclusions

1. The fixed detector center is the dominant cause of territorial truncation in
   the interactive contour path. Large tested features recover only 4.7% to
   18.5% of their established footprints with the old method.
2. Local frontier-relative matching recovers 99.9% to 100% of the same
   footprints while retaining a strict local detector continuity test.
3. The radius-10 limit directly imposes an artificial square search boundary and
   is unnecessary once growth is constrained by physical adjacency and detector
   continuity.
4. Five detector pixels is supported by this pilot: it recovers the complete
   tested footprint without extra frames. Seven pixels begins to show a small
   same-reflection merge risk.
5. Regular-grid fragmentation is not solely a detector-drift problem. Some
   features improve strongly, but others remain disconnected because the regular
   mapping compresses 25,170 frames into 19,916 cells and changes adjacency.
6. This study does not yet alter the radial Gaussian profile acceptance rule.
   Its possible bias against anisotropic intensity distributions remains a
   separate hypothesis for a subsequent controlled study.

## Intensity-edge follow-up

The first study measured whether detector drift fragmented feature membership. It
did not determine whether every linked low-intensity point belonged inside a
circular real-space intensity footprint. Feature 1209 was therefore analyzed
with background-subtracted 10%, 25%, and 50% intensity edges and circular versus
rotated-elliptical Gaussian models.

Centers in regular `(column, row)` coordinates:

| Center definition | Coordinate |
|:---|:---|
| Brightest detected cell | (131.0, 44.0) |
| Background-subtracted center of mass | (119.84, 40.95) |
| Circular Gaussian center | (131.67, 39.68) |
| Elliptical Gaussian center | (129.49, 40.32) |

The centers differ because the linked footprint contains a broad low-intensity
skirt to the left of a bright vertical ridge. Replacing the brightest point with
a center of mass therefore shifts too far into that skirt. The fitted elliptical
center is the most useful single center for the high-intensity core.

Measured smoothed edges in regular grid cells:

| Background-subtracted edge | Width | Height | Principal aspect |
|---:|---:|---:|---:|
| 10% | 36.9 | 49.9 | 2.21 |
| 25% | 17.3 | 44.9 | 5.71 |
| 50% | 5.0 | 39.1 | 10.80 |

The territorial representation gives the same physical ridge with axes rotated
by the mapping: its 50% edge is 47.8 by 5.8 normalized coordinate steps, with
principal aspect 11.86.

Model comparison:

| Coordinates | Circular Gaussian R2 | Elliptical Gaussian R2 | Fitted aspect |
|:---|---:|---:|---:|
| Regular | 0.421 | 0.638 | 6.94 |
| Territory | 0.477 | 0.708 | 7.30 |

Detector position is strongly and directionally coupled to real-space position:
a planar drift model explains 95.5% of detector coordinate variation in the
regular representation and 95.7% in the territorial representation. This
accounts for the detector movement seen at the feature's top and bottom, but it
does not make the real-space intensity circular after correction.

### Catalog-wide morphology check

Feature 1209 is not representative of every feature. The same 50%-core and
linear detector-drift measurements were applied to all 62 Scan 203 shapes with
at least 30 detected cells, with separate summaries for 35 shapes with at least
100 cells and the 18 largest shapes with at least 500 cells.

| Population | Shapes | Median 50%-core aspect | Near-circular (aspect <=1.5) | Elongated (aspect >3) | Very elongated (aspect >5) | Drift R2 >0.8 |
|:---|---:|---:|---:|---:|---:|---:|
| >=30 cells | 62 | 2.18 | 13% | 38% | 18% | 29% |
| >=100 cells | 35 | 2.94 | 14% | 49% | 23% | 51% |
| >=500 cells | 18 | 2.79 | 22% | 44% | 17% | 78% |

Thus the large features are not uniformly square or uniformly ridge-shaped.
Examples among the largest include nearly circular cores (feature 1250, aspect
1.0; feature 1287, aspect 1.2; feature 1129, aspect 1.4) as well as strongly
elongated cores (feature 1209, aspect 11.6; feature 82, aspect 7.8).

There is nevertheless a common directional acquisition signature. Among the 18
largest shapes, the detector drift is strongly systematic for 78%. Their core
axes have moderate alignment in territorial coordinates, and 50% have their
intensity center displaced by more than 10% of the footprint span toward the
same row direction; none have an equally large displacement in the opposite row
direction. The corresponding direction appears rotated in the regular grid
because regular rows/columns are not the true stage axes.

This same-side bias should not be interpreted as natural grain morphology yet.
Its consistency across reflections and features makes scan trajectory, rocking
angle, illumination/footprint geometry, or another acquisition-coordinate
coupling more plausible. A natural morphology claim requires checking whether
the bias rotates or reverses in another scan orientation or rocking frame.

### Edge conclusion

- The outer linked footprint is not an intensity edge. It includes valid,
  connected low-intensity detections.
- Recentring alone does not reveal a hidden circular spot. The measured 25% and
  50% cores are strongly elongated.
- A circular cutoff would discard measured high-intensity signal along the
  ridge. A rotated elliptical model is materially better, although its residuals
  show that even one ellipse is only an approximation.
- For display or quantitative area, report explicit background-subtracted
  intensity contours. The 25% edge is a reasonable working feature boundary;
  50% describes the bright core and 10% describes the diffuse skirt.
- Keep detector-drift continuity for linking, then define area from intensity.
  Do not use detector-distance failure as the real-space boundary.

The follow-up script is `analyze_intensity_edges.py`. Solid cyan/green/white
lines are smoothed measured 10/25/50% edges; dashed lines are the corresponding
elliptical-model contours. It writes all centers, edge dimensions, model fits,
and detector gradients to JSON.

## Reproduce

From the repository root:

```bash
python3 xrd_app/notebooks/compare_territory_drift.py --feature-id 1297
```

Primary drift example output:

- `xrd_app/notebooks/territory_drift_Scan_0203_1297.png`
- `xrd_app/notebooks/territory_drift_Scan_0203_1297.json`

Intensity-edge study:

```bash
python3 xrd_app/notebooks/analyze_intensity_edges.py --feature-id 1209
```

Outputs:

- `xrd_app/notebooks/intensity_edges_Scan_0203_1209.png`
- `xrd_app/notebooks/intensity_edges_Scan_0203_1209.json`
