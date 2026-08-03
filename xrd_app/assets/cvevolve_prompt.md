Find a robust and accurate algorithm that detects Bragg peaks independently in each binned X-ray diffraction detector image.

## 1. Problem statement

An X-ray diffraction experiment scans a sample with a focused X-ray beam across a 2D grid. At each grid position, a detector records a diffraction pattern. Crystalline material produces localized Bragg peaks at positions determined by lattice spacing (2-theta) and crystal orientation.

The selected development set may use any supported bin size. Treat each bin image as an independent peak-finding problem. Do not assume a particular bin size, detector-image count, scan-grid shape, or signal level.

The algorithm should detect peak positions within the known 2-theta reflection bands. Favor recall without producing an unbounded number of detections: missing a real peak loses it entirely, while excessive detections still reduce precision and the F2 score.

## 2. Useful peak-detection strategies

Useful approaches to evaluate include:

1. **Radial background subtraction**: Estimate intensity as a function of 2-theta and subtract the smooth radial background.
2. **White top-hat filtering**: Subtract a morphological opening to isolate compact bright features from broad intensity variations.
3. **2-theta band restriction**: Search only near the known reflection angles to reject off-band artifacts.
4. **Per-band adaptive thresholds**: Use robust local statistics such as the median and MAD so thresholds adapt to each reflection band.
5. **Connected-component filtering**: Reject components that are implausibly small, large, elongated, or diffuse, then calculate intensity-weighted centroids.
6. **Duplicate suppression**: Keep the strongest candidate when detections from overlapping bands or scales are too close together.
7. **Multi-scale detection**: Test multiple feature sizes when Bragg spots vary substantially in width.
8. **Azimuthal discrimination**: Within a reflection band, localized peaks in an intensity-versus-chi profile can distinguish Bragg spots from broad Debye-Scherrer ring segments.

Tune all thresholds and size parameters from the supplied development data rather than assuming values from another bin size.

## 3. Data

Inspect the files supplied in the CVEvolve `data_dir`. The development set includes the peak annotations and metadata needed by the evaluation harness. Common files include:

- **`bin_annotations.json`**: Ground-truth peaks grouped by bin and reflection.
- **`empty_bins.json`**: Reviewed bins containing no peaks.
- **`grid_mapping.json`**: Grid and bin metadata when available.
- **`tth.tiff`**: Per-pixel 2-theta map.
- **`reflections.py`**: Valid reflection names and expected 2-theta values.
- **`baseline.py`**: Baseline per-bin peak detector, when supplied. Use it as a starting reference and try to outperform it.
- **`evaluate.py`**: Development evaluation harness, when supplied.

Detector dimensions, image paths, annotation counts, and bin size are properties of the supplied session. Discover them from the files and command-line interface rather than hard-coding them.

Annotation coordinates are detector pixel positions. Confirm whether each supplied file represents points as `[row, col]` or `(x, y)` before converting between formats.

## 4. Evaluation

For each evaluation bin, detect peaks in that bin's detector image and report the detected reflection and detector position.

The primary metric is the **mean per-bin F2 score** (beta=2), which weights recall more strongly than precision. The evaluation harness defines the matching radius and exact handling of empty bins. Use its reported primary score when comparing candidates.

Ground truth may omit weak but real peaks, so do not sacrifice recall merely to maximize precision. However, an excessive number of proposals still lowers F2, and detections in reviewed empty bins are false positives.

Use the supplied `evaluate.py` harness and inspect `--help` before running it. A seeded subset may be used for quick development if supported, but confirm promising candidates on the complete development set with full-resolution settings before reporting a score.

## 5. Algorithm contract

Follow the callable or command-line contract implemented by the supplied `baseline.py` and `evaluate.py`. Preserve required argument names and output formats so the evaluator can run the candidate unchanged.

A typical peak output contains:

- `reflection`: reflection label
- `x`: detector column coordinate
- `y`: detector row coordinate

Do not depend on labels at inference time. Labels are available only to the evaluator for scoring.

## 6. Guidance

- Treat each bin image as an independent peak-finding input.
- Restrict candidates to physically valid 2-theta bands.
- Remove radial background before applying local thresholds.
- Use robust statistics because intensity and noise vary between reflections and scans.
- Tune SNR, component-size, and filter-scale parameters for the selected bin size.
- Keep component analysis vectorized; avoid rescanning the entire image once per component.
- Cache geometry-derived arrays such as band masks when evaluating many bins.
- Validate that reported peaks lie in the expected reflection bands, not merely that the numerical score improved.
