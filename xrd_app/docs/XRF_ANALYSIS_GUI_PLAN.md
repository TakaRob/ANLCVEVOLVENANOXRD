# XRF Analysis GUI and Project Integration Plan

## Architecture Decision

Keep XRF analysis isolated from the main xrd-app GUI.

There are two applications with one explicit file contract:

1. **XRF Analysis** owns raw ME7 discovery, calibration, emission ROIs, spectra,
   material thresholds, spatial previews, and creation of a finalized XRF
   selection artifact.
2. **xrd-app** only loads and validates that artifact, lets the user choose a
   material selection, and compiles its keep mask into ordinary XRD grid/bin/
   peak/shape variants.

The XRF application must not become an alternative XRD analysis pipeline. It may
use XRD file/frame metadata to validate registration and preview an individual
linked frame, but all XRD binning, peak detection, shape linking, and scientific
XRD GUIs remain in xrd-app.

The main xrd-app must not expose XRF calibration, ROI editing, raw ME7 processing,
or threshold controls. This keeps XRF setup independent and makes the handoff a
simple loading operation.

## Objective

Turn the XRF prefilter notebooks into a standalone XRF project/application that:

1. registers raw ME7 and matching position/XRD identity metadata;
2. calibrates the XRF energy axis and defines named emission ROIs;
3. displays spectra, count distributions, and spatial maps;
4. sets per-material count thresholds;
5. exports one reproducible, finalized XRF selection HDF5; and
6. allows xrd-app to load that HDF5 as an optional frame-selection source.

The exported artifact feeds the existing xrd-app grid, bin, peaks, shapes, and
catalog-driven GUI workflow described in `XRF_CUT_INTEGRATION_PLAN.md`.

## Existing Pieces to Reuse

The repository already contains most of the analysis engine:

- `core/xrf.py`: ME7 registration, spectra, calibration, ROI integration,
  element maps, point spectrum store, and serialization.
- `xrd-app xrf`: CLI command for element maps and optional per-frame spectra.
- `tabs/xrf_calibration_popup.py`: emission-line and calibration editor.
- `notebooks/xrf_prefilter_xrd.py`: threshold selection, spatial comparison,
  keep masks, and XRF-to-XRD links.
- `notebooks/processed_data_visualizations.py`: XRF distributions, spatial maps,
  XRF/XRD correlations, and detector COM diagnostics.
- `core/linked_xrd.py` and `linked-xrd-track`: exploration of retained raw XRD
  links and detector-peak motion.

The implementation should move reusable notebook logic into `core/xrf.py` or a
small `core/xrf_selection.py`, expose it through CLI commands, and keep the new
Qt tab as a client of those commands and saved products.

## Nested XRF Add-on Project Model

Every XRF project is a normal xrd-app project with an `XRF/` add-on directory.
There is one shared project root and scan registry, while XRF-specific setup and
analysis state remain isolated under `XRF/`.

Creating a project from `xrf-app` first creates the standard xrd-app project tree
and `config.yaml`, then creates the add-on. Opening an existing xrd-app project in
`xrf-app gui` prompts to create the missing add-on; approval creates only `XRF/`
and never replaces existing XRD configuration or data.

Suggested combined project tree:

```text
MyProject/
  config.yaml                         # standard xrd-app project
  Raw/
  Binned/
  Metadata/
  Labels/
  Figures/
  XRF/
    xrf_config.yaml
    Raw/
      Scan_0024/ME7/                  # links or copies of raw XRF inputs
    Metadata/
      Scan_0024/
        positions.h5
        xrd_registration.h5           # source file/local-frame identity only
        xrf_elements.json
    Cache/
      Scan_0024_spectrum.npz
      Scan_0024_points.npz
      Scan_0024_roi_intensities.npz
    Processed/
      Scan_0024_xrf_selection.h5      # finalized handoff to xrd-app
    Figures/
```

Suggested `XRF/xrf_config.yaml`:

```yaml
name: My XRF Project
scans: {}
data_sources:
  me7_root: null
  position_root: null
  xrd_identity_root: null
calibration:
  default: null
outputs:
  processed_dir: Processed
  cache_dir: Cache
```

The XRF project may discover matching XRD file names and frame counts to establish
identity, but it does not copy, bin, or analyze detector pixels.

The standalone XRF application has its own command surface while sharing the
parent project and Python package. Initialization semantics are:

```text
xrf-app init --root NEW_PROJECT      # create xrd-app project + XRF/ add-on
xrf-app init --root EXISTING_PROJECT # create only the missing XRF/ add-on
xrf-app scan-detect
xrf-app calibrate
xrf-app process
xrf-app threshold
xrf-app export
xrf-app gui
```

This separation prevents the main xrd-app Setup tab from becoming a mixed XRF/XRD
instrument configuration screen.

## Standalone XRF Setup GUI

The XRF application owns all source setup:

- create/open an XRF project;
- select one ME7 scan folder or a parent scan set;
- select position data;
- locate matching XRD file/frame identity metadata;
- choose link-in-place versus copy for ME7 data;
- configure channels and deadtime correction; and
- configure energy calibration and emission lines.

Its Setup page should show per-scan registration status:

```text
Scan 24
ME7 files: 141
XRD identity files: 141
Mapped XRF/XRD frames: 109109
Positions: 109109
Status: valid
```

Registration errors block export but do not block XRF-only spectrum inspection.

## Minimal xrd-app Loading UI

The main xrd-app receives only finalized processed XRF selections.

Add one small group to xrd-app Setup -> Load Data:

```text
Processed XRF selection
[ ] Use processed XRF selection
Path: /path/to/file-or-directory       [Select file...] [Select directory...]
Mode: Link in place | Copy metadata into project
Status: Scan 24; Br, Pb; registration valid
```

There is no raw ME7 picker, calibration button, spectrum editor, ROI editor, or
threshold editor in xrd-app.

The checkbox remains explicit:

- unchecked: xrd-app ignores the configured path and behaves exactly as today;
- checked: xrd-app validates the artifact against each active XRD scan and exposes
  its materials as data-selection variants;
- changing the path does not activate it automatically; and
- a registration mismatch disables XRF variants without affecting unfiltered XRD.

Because the add-on is nested, xrd-app can default the processed source to
`XRF/Processed/`. The explicit enable checkbox still controls whether selections
are used. An optional external source may be recorded when loading selections
from another project's add-on:

```yaml
processed_xrf:
  enabled: true
  source: XRF/Processed
  mode: project_addon
```

No XRF scientific configuration is duplicated in the parent `config.yaml`.
Material names, thresholds, ROI definitions, calibration, and source hashes are
read from finalized selection artifacts under the add-on.

### Processed file/directory discovery

The xrd-app picker accepts:

- one canonical `xrd-app-xrf-selection` HDF5;
- a directory of canonical per-scan selection files; and
- the current Scan 24 notebook bundle only through an explicit legacy import
  command that first creates canonical files.

Add a read-only xrd-app command:

```bash
xrd-app processed-xrf-check --source PATH [--scan 24]
```

It reports scans, materials, frame counts, source signatures, and registration
errors. Directory discovery is filename-pattern based followed by content
validation; it never chooses an arbitrary newest HDF5.

Default to linking in place. Copy mode copies only finalized selection metadata
into `Metadata/<scan>/`, never raw ME7 or XRD detector data.

## Canonical XRF Selection Product

Use one canonical per-scan HDF5 product:

```text
Metadata/Scan_0024/Scan_0024_xrf_selection.h5
```

Schema:

```text
attrs/
  format = "xrd-app-xrf-selection"
  format_version = 1
  scan
  n_total_frames
  source_kind
  source_signature
  xrd_registration_signature
  channels
  deadtime_correction
  energy_calibration_json

spectrum/
  energy_kev                float64 [4096]
  summed_counts             float64 [4096]

frames/
  global_frame_index        int64 [N]
  source_file_index         int32 [N]
  source_frame_index        int64 [N]
  x                         float64 [N]
  y                         float64 [N]

materials/<safe_name>/
  attrs:
    display_name
    minimum_counts
    roi_energy_kev          float64 [2]
    roi_pixel_range         int32 [2]
  intensity                 float64 [N]
  keep                      bool [N]
```

Optional registration data can include XRD file/local-frame identity, but the
stable join is the validated global frame index plus acquisition signature.

This single product replaces the notebook's separate ROI config, intensity
cache, mask NPZ, cut summary, and linker as the project-facing artifact. Those
files remain importable for existing Scan 24 work.

## Command Boundaries

### Standalone `xrf-app`

The XRF application owns these commands:

#### `xrf-app source-check`

Read-only discovery and validation of raw ME7, position, and XRD identity inputs.

#### `xrf-app import-legacy`

Convert the current notebook output bundle into an XRF project and canonical
selection product. This is migration tooling and does not become the normal
xrd-app loading path.

#### `xrf-app process`

Build the calibrated grand spectrum, point store, and per-material intensities.
It reuses `core/xrf.py` but writes into the XRF project.

#### `xrf-app threshold`

Update material keep masks atomically from cached per-frame intensities. This is
cheap and never reopens raw ME7 or XRD pixels.

#### `xrf-app export`

Validate registration and write the finalized canonical selection HDF5 under the
XRF project's `Processed/` directory. Export is the only supported handoff to
xrd-app.

### Main `xrd-app`

The main application owns only processed-selection commands:

#### `xrd-app processed-xrf-check`

Validate one finalized selection file or directory against the active xrd-app
project's scan registry/grid mappings.

#### `xrd-app processed-xrf-link`

Record or copy the finalized selection into the xrd-app project. It never opens
raw ME7 and never edits XRF thresholds.

#### `xrd-app xrf-cut-grid`

Compile a finalized material keep mask into an ordinary grid variant:

```bash
xrd-app xrf-cut-grid \
  --root XRD_PROJECT \
  --scan 24 \
  --material Br \
  --bin-size 3 \
  --variant xrf_br
```

#### `xrd-app xrf-cut-build`

Convenience chain:

```text
xrf-cut-grid -> bin -> optional peaks -> optional shapes
```

It invokes the same existing xrd-app core/CLI stages as the individual commands.

## Standalone XRF Analysis GUI

Create a separate application entry point:

```bash
xrf-app gui --root /path/to/XRF_PROJECT
```

It may reuse xrd-app's tab discovery/window infrastructure internally, but it has
its own built-in tabs and project configuration. Do not add an XRF Analysis tab
to the main xrd-app `_BUILTIN_TABS`.

### Layout

Use four coordinated areas rather than reproducing notebook cells vertically.

#### 1. Source and status strip

Show:

- active scan;
- source mode and path;
- raw/processed status;
- channels and deadtime correction;
- frame registration count;
- canonical product path; and
- `Configure source...` link back to Setup.

If XRF is disabled, show a focused empty state with an `Enable XRF...` button.

#### 2. Spectrum and ROI editor

Main spectrum plot:

- logarithmic counts versus calibrated energy;
- named ROI spans;
- detected emission peaks;
- optional nearby library labels;
- zoom to one or multiple selected materials.

Controls:

- material list with checkboxes;
- add/remove material;
- low/high energy or pixel bounds;
- enabled channels;
- deadtime correction;
- detect/refine peaks; and
- save calibration/ROIs.

Reuse `xrf_calibration_popup.py` logic. Either embed its editor or extract a
reusable QWidget from it; do not maintain two independent calibration models.

#### 3. Threshold and distribution panel

For each selected material:

- integer-aware count histogram;
- minimum-count spin box/line edit;
- retained/cut counts and percentages;
- percentile markers; and
- Apply/Revert controls.

Support `Focus: all`, one material, or multiple materials, matching the notebook
workflow.

Threshold edits should update a preview in memory. `Save thresholds` invokes
`xrf-threshold` and atomically updates the canonical selection product.

#### 4. Spatial map and export

Map modes:

- XRF intensity;
- retained mask;
- rejected mask;
- overlap count across selected materials; and
- optional processed XRD ROI correlation when available.

Use measured X/Y coordinates, equal aspect, robust color percentiles, and explicit
white/no-data positions as in the notebooks.

Export controls:

```text
Scans: active scan | selected scans | all ready scans
Materials: Br, Pb
Output: XRF_PROJECT/Processed
[Validate registration] [Export finalized selection]
```

Show the exact output file and a summary of materials, thresholds, retained
counts, and acquisition signature. The XRF GUI stops at export; it does not offer
XRD binning, peaks, or shapes.

## XRF GUI State and Project Switching

Persist only presentation state in the XRF project's GUI state:

- selected XRF material(s);
- spectrum zoom;
- active map mode; and
- active scan.

Persist scientific state in the canonical selection HDF5, XRF project caches,
and `xrf_elements.json`, never only in GUI state.

When the top-level scan selector changes:

- resolve the scan-specific XRF source/product;
- cancel outstanding workers;
- clear stale plots immediately;
- show missing-source status rather than retaining the prior scan; and
- preserve material names where they exist in the new scan.

When switching projects, reload `xrf.enabled` and source mode before constructing
the tab.

## End-to-End Workflows

### XRF project workflow

1. Launch `xrf-app gui` and create/open an XRF project.
2. Register raw ME7 file/directory, positions, and matching XRD identity metadata.
3. Compute or load the grand spectrum and per-frame point store.
4. Calibrate energy and define named material ROIs.
5. Integrate per-frame material intensities.
6. Set and save thresholds.
7. Validate XRF/XRD registration.
8. Export finalized selection HDF5 files to `Processed/`.

Heavy raw reads are explicit and cached. Once the point store exists, spectrum,
ROI, and threshold iteration is local and fast.

### xrd-app loading workflow

1. Open the normal XRD project in xrd-app.
2. Check `Use processed XRF selection`.
3. Select the XRF project's `Processed/` directory or one finalized file.
4. xrd-app validates scan/global-frame/source identity.
5. Choose an exported material such as Br or Pb.
6. Create a standard `xrf_<material>` grid variant.
7. Run the unchanged bin, peaks, and shapes pipeline.
8. Open tagged catalogs in the existing XRD GUIs.

xrd-app treats the processed selection as read-only. Threshold changes require
returning to the XRF application, exporting a new artifact, and rebuilding any
stale XRD variants.

## Core Refactoring from Notebooks

Move these operations into tested core functions:

- calibrated pixel/energy conversion, including quadratic calibration support;
- exact grand spectrum caching;
- per-frame ROI integration;
- XRF/XRD registration validation;
- threshold application and cut summaries;
- canonical selection serialization;
- legacy linker/mask/config import;
- material overlap calculation; and
- filtered grid mapping generation.

Retain notebook plotting as exploratory examples, but have notebooks call core or
CLI functions. The Qt tab should never import notebook modules.

Unify the current linear calibration in `core/xrf.py` with the quadratic Scan 24
calibration by defining a calibration model in config, for example:

```json
{"kind": "polynomial", "coefficients_kev": [5.263744e-7, 8.41967e-3, 1.136032]}
```

Continue supporting the existing linear `ev_per_bin`/`offset_ev` representation
by normalizing it to the same model internally.

## Validation and Safety

Before permitting XRD handoff, require:

- selection scan matches the active scan;
- selection `n_total_frames` equals the source grid `frame_map` length;
- every global index maps to the same source file/local frame;
- position arrays align with global indices;
- no material name collision after safe filename normalization; and
- thresholds/ROIs/calibration have a deterministic configuration hash.

Use atomic writes for canonical selection updates. Never partially overwrite a
valid selection if threshold application fails.

The GUI must distinguish:

- missing XRF source;
- unprocessed raw source;
- processed product available;
- processed masks only;
- registration mismatch; and
- stale XRD cut variants after threshold changes.

## Testing

### Core tests

- raw ME7 registration with uneven per-file frame counts;
- linear and quadratic energy calibration round trips;
- ROI integration against known synthetic spectra;
- threshold updates without raw rereads;
- multi-material overlap;
- canonical HDF5 round trip;
- current notebook linker import;
- masks-only import behavior;
- source/grid mismatch rejection; and
- filtered grid membership correctness.

### CLI tests

- every new `xrf-app` and xrd-app command `--help`;
- XRF raw source and project discovery;
- finalized processed file/directory discovery in xrd-app;
- link versus metadata-copy loading;
- XRF threshold and export transactions;
- `xrf-cut-grid` output naming and provenance; and
- failure messages for disabled/missing/mismatched data.

### XRF GUI tests

- XRF project creation/opening;
- ME7/position/XRD-identity setup and validation;
- spectrum, ROI, and threshold ready/error states;
- multi-material focus and threshold previews;
- scan/project switching clears stale data; and
- export invokes the expected `xrf-app` command and writes a finalized artifact.

### xrd-app GUI tests

- processed source remains inactive until explicitly checked;
- file and directory pickers validate before saving config;
- only finalized selections are accepted;
- materials and retained counts are displayed read-only;
- registration mismatch leaves unfiltered XRD usable; and
- variant/build buttons construct the expected xrd-app commands.

### Scan 24 acceptance checks

- imported Br/Pb intensities and masks match notebook results exactly;
- material retained counts match the cut summary;
- spectrum and spatial maps match notebook figures;
- selected XRD file/frame links resolve to the original detector frames;
- Br/Pb overlap is preserved; and
- a generated `xrf_br` grid/bin/catalog opens in existing xrd-app views.

## Delivery Phases

### Phase 1: XRF project and canonical export

- Implement the standalone XRF project config/tree.
- Implement the finalized selection HDF5 schema.
- Add `xrf-app init`, source discovery, and legacy notebook import.
- Import current Scan 24 notebook outputs and export canonical Br/Pb selections.

Exit criterion: an XRF project can reproducibly create the only file contract
that xrd-app needs.

### Phase 2: Standalone read-only XRF GUI

- XRF project setup/status.
- Spectrum with ROI overlays.
- Material distributions and saved thresholds.
- XRF intensity/keep spatial maps.
- Export status.

Exit criterion: `xrf-app gui` reproduces the important notebook visualizations
without changing the main xrd-app GUI.

### Phase 3: Standalone editable XRF analysis

- Extract/reuse the calibration editor.
- Add material ROI editing and peak refinement.
- Add threshold preview/save.
- Build/cache raw ME7 point products through `xrf-app` CLI commands.
- Export finalized selections.

Exit criterion: raw ME7 can be processed, thresholded, and exported entirely in
the isolated XRF application.

### Phase 4: Minimal xrd-app loading integration

- Add the explicit processed-selection checkbox and file/directory picker.
- Add `processed-xrf-check`, `processed-xrf-link`, and `xrf-cut-grid`.
- Add selection/provenance hashes.
- Generate ordinary tagged bins/peaks/shapes.
- Route View/Label consistently through selected variants.

Exit criterion: XRF-selected XRD appears in existing xrd-app GUIs without raw ME7
controls, XRF editing, or a new XRD data backend.

### Phase 5: Performance and polish

- Measure `/mnt/z` behavior.
- Add optional local sparse union archive only if needed.
- Keep XRF/XRD correlation and linked peak-tracking analysis in the appropriate
  application or as read-only result products.
- Add batch multi-scan XRF export and xrd-app selection summaries.

## Recommended First Implementation Slice

1. Define and implement the finalized selection HDF5 contract.
2. Create an XRF project from the current Scan 24 notebook outputs.
3. Build a small standalone `xrf-app gui` that reads the project and shows Br/Pb
   spectra, distributions, masks, and spatial maps.
4. Export Scan 24 Br/Pb finalized selections.
5. Add only the processed-selection checkbox and picker to xrd-app Setup.
6. Implement `processed-xrf-check` and `xrf-cut-grid` for Br 3x3.
7. Run the unchanged XRD bin/peaks/shapes pipeline and open its tagged catalog.

This slice proves the isolated XRF project/application and the narrow processed
file handoff before adding raw-ME7 editing or performance caches.
