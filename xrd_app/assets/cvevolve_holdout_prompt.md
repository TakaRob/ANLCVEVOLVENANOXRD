Evaluate the submitted per-bin Bragg peak detector against the supplied holdout set.

## Holdout data

Inspect the holdout directory rather than assuming a particular bin size, detector shape, scan-grid shape, or number of examples. It commonly contains:

- **`bin_annotations.json`**: Ground-truth peaks grouped by bin and reflection. Bin entries map reflection names to detector pixel coordinates.
- **`empty_bins.json`**: Reviewed bins containing no peaks.
- **`grid_mapping.h5`**: Grid and bin metadata when available.
- **`tth.tiff`**: Per-pixel 2-theta map.
- **`reflections.json`**: Valid reflection names and expected 2-theta values, when supplied.
- **`labels/`**: Per-bin label files when required by the evaluator.
- **`evaluate.py`**: Holdout evaluation harness when supplied.

Use the supplied files and the evaluator's command-line help to discover paths and formats. Confirm whether annotation points are represented as `[row, col]` or `(x, y)` before comparing them with detector coordinates.

## Evaluation procedure

1. Evaluate every annotated bin and every reviewed empty bin in the holdout set.
2. Run the submitted algorithm on each bin's detector image independently.
3. Compare the reported reflection and detector position with the ground truth using the matching rule implemented by the supplied evaluator.
4. Enforce one-to-one matching: each ground-truth point and each detection can participate in at most one match.
5. Use the evaluator's handling of empty bins. A reviewed empty bin should produce no detections.
6. Report the aggregate primary metric and supporting precision, recall, and F1 values provided by the evaluator.

The primary metric is **mean per-bin F2** (beta=2), which emphasizes recall while still penalizing excessive proposals. Do not replace it with F1 or a global point-weighted score.

## Running the evaluation

Prefer the supplied holdout `evaluate.py` harness. Inspect its interface first:

```bash
python <holdout_dir>/evaluate.py --help
```

Then run the candidate over the complete holdout set using the evaluator's full-resolution defaults. Do not use development subsets, reduced resolution, or other approximate speed settings for the final holdout score.

If no evaluation harness is supplied, invoke the candidate according to its documented contract, aggregate one-to-one matches per bin, and compute:

```text
F2 = 5 * precision * recall / (4 * precision + recall)
```

Handle zero-denominator and empty-bin cases explicitly and consistently, then average the per-bin scores equally.

## Important notes

- This is final validation data. Do not tune algorithms or hyperparameters against it.
- Submit the final numeric result with CVEvolve's holdout metric tool. Use a metric name containing `f2` when reporting the primary mean F2 score so xrd-app records it as `holdout_f2` during automatic registration.
- CVEvolve writes the selected module to `reports/best_candidate.py`; xrd-app validates and registers that report output automatically after a successful GUI run. Do not create `top_algorithms.json` or copy candidates into xrd-app manually.
- This task evaluates each bin's peak detections independently.
- Do not hard-code a bin size, detector dimensions, matching radius, or data path.
- Ensure each reported peak lies in the expected 2-theta band for its reflection.
- Include all holdout bins in the final result and print the primary F2 score clearly.
