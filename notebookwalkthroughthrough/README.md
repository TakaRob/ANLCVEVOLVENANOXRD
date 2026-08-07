# xrd-app notebooks

These tracked percent-format Python notebooks are a CLI-first introduction to
xrd-app. Open them in order in VS Code or another editor that supports `# %%`
cells. They also compile and run as ordinary Python scripts.

1. `00_project_and_results.py`: inspect project status, algorithms, lineage, and studies.
2. `01_single_scan_pipeline.py`: build normalized bins, peaks, and shapes.
3. `02_territory_and_roi_maps.py`: true-coordinate territories and detector ROI maps.
4. `03_cross_scan_study.py`: tracks, rocking curves, and study summaries.
5. `04_qspace_and_rsm.py`: reciprocal-space maps and physics checks.
6. `05_xrf_and_linked_xrd.py`: canonical XRF selections and linked XRD.

Set `XRD_PROJECT`, `XRD_SCAN`, and other documented environment variables, or
edit each notebook's configuration cell. Metadata-only commands run by default.
Any command that reads detector frames or builds products requires explicitly
setting `RUN_HEAVY = True` after reviewing the printed command.

Persistent work is performed through `python -m xrd_app.cli` or
`python -m xrd_app.xrf_cli`, using the current notebook kernel's interpreter.
This is equivalent to the installed `xrd-app` and `xrf-app` entry points and
avoids stale shell-entry-point issues in editable environments.
