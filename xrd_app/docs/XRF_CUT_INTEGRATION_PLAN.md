# XRF-Cut XRD Integration Plan

## Goal

Make XRF-selected XRD data behave like ordinary xrd-app data without creating a
second analysis pipeline or teaching every GUI a new linker-table format.

The XRF crop must preserve:

- the original XRD detector values for retained frames;
- the original global acquisition indices;
- the measured X/Y coordinates and spatial lattice;
- material overlap, where one frame may pass more than one XRF selection; and
- reproducible threshold and source provenance.

After selection, the existing `bin -> peaks -> shapes` pipeline and existing
catalog-driven GUIs should operate normally.

## Decision

Use an **XRF-filtered grid-mapping variant** as the authoritative XRD cut.

For each material, copy the ordinary grid mapping and filter only its bin
membership lists:

```text
source mapping bins[bin_key] = [all global frame indices in the spatial bin]
XRF keep set             = {global indices passing the material threshold}
cut mapping bins[bin_key] = source mapping bins[bin_key] intersect keep set
```

Keep these source-mapping fields unchanged:

- `xrd_files`
- `frame_map`
- `n_total_frames`
- `n_rows`, `n_cols`, `n_bin_rows`, and `n_bin_cols`
- coordinate source and position provenance
- territory geometry, when filtering a territory mapping

Only contributing frame membership changes. Empty bins may be omitted while the
full lattice dimensions remain unchanged, so rejected sample positions appear as
no data rather than detector images containing artificial zeros.

Example products for Br at 3x3:

```text
Metadata/Scan_0024/grid_mapping_3x3_xrf_br.h5
Binned/Scan_0024/xrd_3x3_bins_xrf_br.h5
Labels/Scan_0024/<detector>_peaks_3x3_xrf_br.h5
Labels/Scan_0024/<shape>_shapes_3x3_xrf_br.h5
```

This matches the existing xrd-app variant mechanism. `DataManager`, binning,
peak detection, shape finding, lineage, and catalog source resolution already
understand tagged mappings and products.

## Why Not Make Rejected Frames Zero?

A full-index, zero-filled copy can preserve array dimensions, but zero is a
valid detector value. Existing xrd-app readers cannot distinguish:

- a real acquired frame whose pixels happen to be zero; and
- a frame replaced with zeros because XRF rejected it.

If an unfiltered mapping includes those placeholders:

- sum binning happens to produce the same sum, but contributor counts are wrong;
- mean-per-frame binning is biased downward;
- `n_frames` metadata is wrong;
- HD maps show zero intensity instead of no data; and
- the GUI presents rejected positions as measured black frames.

A sparse, zero-fill HDF5 cache is safe only when paired with the filtered grid
mapping, because the mapping prevents rejected chunks from being read or counted.
It is therefore a performance cache, not the definition of the crop.

## Storage Model

### Authoritative inputs

1. Original raw XRD files or the complete lossless unbinned archive.
2. The XRF selection artifact containing material, global frame index, source
   file/local frame index, X/Y, integrated XRF intensity, and threshold metadata.
3. The unfiltered xrd-app grid mapping built from true measured X/Y positions.

The original XRD data remain immutable. Changing an XRF threshold only rebuilds
small selection/mapping metadata and its downstream binned products.

### Derived products

1. Material-specific filtered grid mapping.
2. Standard material-specific binned HDF5.
3. Standard peak and shape catalogs carrying the material variant in lineage.

These products are disposable and reproducible from the authoritative inputs.

### Optional local union cache

If repeated selected reads over `/mnt/z` are too slow, add one local cache for
the union of all material selections, not one detector copy per material.

The cache should:

- preserve the complete global-index axis;
- use the original detector dtype;
- use one-frame HDF5 chunks;
- physically write only frames selected by at least one material;
- leave other chunks unallocated with fill value zero;
- store original source file/local frame metadata; and
- always be read through a filtered material mapping.

Material overlap then costs no additional detector storage. This cache is a
later optimization; the first implementation should read selected frames from
the complete archive or raw files and materialize normal binned products.

Do not use HDF5 virtual datasets as the primary solution. They preserve `/mnt/z`
latency, introduce fragile source-path dependencies, and do not remove the need
for explicit selection membership.

## Proposed CLI Workflow

### 1. Create or update the XRF selection

The existing prefilter writes the linker and masks. Move this reusable operation
behind an eventual CLI command, while retaining the notebook as a visualization
and threshold-selection client:

```bash
xrd-app xrf-cut \
  --root /path/to/project \
  --scan 24 \
  --xrf-roi Br=11.761:12.061 \
  --minimum-counts Br=40 \
  --output Metadata/Scan_0024/xrf_cut.h5
```

The immediate integration can consume the current
`Scan_0024_xrf_xrd_links.h5` without first implementing `xrf-cut`.

### 2. Compile the selection into a standard grid variant

Add a command such as:

```bash
xrd-app xrf-cut-grid \
  --root /path/to/project \
  --scan 24 \
  --material Br \
  --selection /path/to/Scan_0024_xrf_xrd_links.h5 \
  --source-grid Metadata/Scan_0024/grid_mapping_3x3.h5 \
  --bin-size 3 \
  --variant xrf_br
```

The command should call a small pure core function:

```python
filter_grid_mapping(source_mapping, selected_global_indices, provenance)
```

It must not read detector pixels.

### 3. Run the unchanged pipeline

```bash
xrd-app bin \
  --root /path/to/project \
  --scan 24 \
  --bin-size 3 \
  --variant xrf_br

xrd-app peaks \
  --root /path/to/project \
  --scan 24 \
  --bin-size 3 \
  --variant xrf_br

xrd-app shapes \
  --root /path/to/project \
  --scan 24 \
  --bin-size 3 \
  --variant xrf_br
```

A convenience command can later chain these stages:

```bash
xrd-app xrf-cut-build --scan 24 --material Br --bin-size 3 --variant xrf_br
```

The individual commands remain the engine and are required before adding a GUI
button.

## Selection Schema

The current linker is sufficient for a first adapter, but the stable project
artifact should use full-index selection data rather than only retained rows.

Suggested `Metadata/<scan>/xrf_cut.h5` schema:

```text
attrs/
  format = "xrd-app-xrf-cut"
  format_version = 1
  scan
  n_total_frames
  source_dataset
  channels
  deadtime_correction
  energy_calibration_json
  source_signature

frames/
  global_frame_index       int64 [N]       # normally 0..N-1
  source_file_index        int32 [N]
  source_frame_index       int64 [N]
  x                        float64 [N]
  y                        float64 [N]

materials/<safe_name>/
  keep                     bool [N]
  xrf_intensity            float64 [N]
  minimum_counts           float64 attr
  roi_energy_kev           float64 [2] attr
  roi_pixel_range          int32 [2] attr
```

A frame may have `keep=True` for multiple materials. Selections are not assumed
to be mutually exclusive.

The adapter for the current link table should reconstruct each material keep set
from `links/Material` plus `links/Global Frame Index` and validate file/frame
identity against the source grid's `frame_map`.

## Filtered Grid Provenance

Add these fields to the copied grid mapping's `metadata_json`:

```text
variant = "xrf_br"
frame_selection_kind = "xrf_threshold"
frame_selection_source = absolute or project-relative artifact path
frame_selection_material = "Br"
frame_selection_hash = deterministic hash of the Boolean keep mask
frame_selection_config_hash = hash of ROI/calibration/threshold settings
source_grid = path or identity of the unfiltered mapping
source_grid_hash = deterministic mapping hash
n_frames_acquired
n_frames_selected
n_frames_rejected
n_bins_selected
```

`n_total_frames` continues to mean the number of acquired frames represented by
`frame_map`. `n_frames_selected` records contributors after the cut. Do not
renumber selected frames.

Copying the mapping rather than rebuilding coordinates is important: every
material variant must share the same spatial lattice, allowing direct comparison
of Br, Pb, and unfiltered results.

## Core Changes

### Phase 1: Selection-to-grid adapter

Add a focused core module, for example `core/frame_selection.py`, with:

- `load_xrf_selection(path, material)`
- `validate_selection_against_grid(selection, grid)`
- `filter_grid_mapping(grid, selected_indices, provenance)`
- deterministic selection and source-grid hashing

Validation must reject:

- scan mismatch;
- selected global indices outside `frame_map`;
- source file/local frame mismatch;
- duplicate conflicting records;
- detector source signature mismatch; and
- a selection generated from a different acquisition ordering.

Add `xrf-cut-grid` to `cli.py` as a thin wrapper.

### Phase 2: Product identity and stale-data protection

A changed threshold must not silently reuse old binned data.

Add the selection hash and source-grid hash to binned HDF5 root attributes in
`build_bins()`. Extend binned-file validation to compare these values with the
selected grid mapping.

Propagate the variant and selection identity through peak and shape lineage.
The existing variant lineage remains the human-readable identity; hashes prevent
stale artifacts from masquerading as current results.

### Phase 3: Consistent image-source use

Most GUIs already use `open_bin_source()` or catalog source resolution. Migrate
View/Label's remaining direct binned/raw read paths to `BinImageSource` so every
view honors the exact selected mapping and source pair.

This is a cleanup of an existing bypass, not a new data backend. The public GUI
contract remains:

```python
source.keys()
source.image(bin_key)
source.region(bin_key, y0, y1, x0, x1)
```

No GUI should read the XRF link table directly.

Also add grid integrity checks at `open_bin_source()`:

- `n_total_frames == len(frame_map)`;
- every bin member is a valid global index;
- archive source identity matches the mapping when an archive is used; and
- selection provenance matches a prebuilt binned product.

## GUI Integration

### Programs tab

Add an "XRF-Cut XRD" section after the CLI stages are stable:

1. Selection artifact picker.
2. Material selector populated from the selection file.
3. Source grid/bin-size selector.
4. Read-only summary: acquired, selected, rejected, occupied bins, overlap.
5. "Create cut grid" button invoking `xrf-cut-grid`.
6. "Create cut bins" button invoking ordinary `bin --variant ...`.
7. Optional "Run peaks and shapes" button invoking the ordinary commands.

The GUI must show the exact command and output paths, following the existing
CLI-is-the-engine architecture.

### View/Label and ROI > Shape

Add a data-variant selector only where no feature catalog has yet selected the
variant:

```text
Data: unfiltered | XRF Br | XRF Pb
```

Resolve it to the tagged grid and binned HDF5 through `DataManager`. Missing
sample positions should display as no-data cells, not synthetic black frames.
The detector image itself remains unchanged for retained positions.

ROI > Shape currently opens the default source directly. Pass the chosen variant
and exact grid path to `open_bin_source()`.

### Shape/Verify, Device View, Orientation, and Territory Map

These views should derive the variant from the selected feature catalog. Existing
catalog source resolution already maps a tagged catalog to its tagged grid and
binned HDF5. Add a visible badge such as:

```text
Data selection: XRF Br, minimum 40 counts, retained 27.1%
```

The badge reads lineage/provenance; it does not alter pixel access.

Territory support should filter an existing territory mapping's membership while
preserving its polygons and neighbor graph. Do not independently regrow territory
geometry for each material, because that would make material maps spatially
incomparable.

### XRF overlays

Keep XRF intensity overlays independent from the XRD frame-selection variant.
A Br-selected XRD catalog may still display Br, Pb, or other XRF overlays. The
selection determines which XRD frames contributed; it does not redefine the XRF
visualization product.

## Behavior of Empty and Partial Bins

- A bin with no retained frames is absent/no-data.
- A bin with some retained frames sums only those frames.
- `n_frames` means retained contributing frames.
- Mean-per-frame normalization divides by retained contributing frames.
- A retained detector frame containing real zero counts remains valid.
- Rejected positions are never inferred from detector pixel values.

Optionally add `BinImageSource.bin_info(key)` later to report acquired versus
selected contributors, but this is not required for the first integration.

## Testing Strategy

### Unit tests

1. Filter a synthetic mapping and verify global indices are preserved.
2. Verify selected membership is the exact intersection for every bin.
3. Preserve grid dimensions, `frame_map`, coordinate provenance, and territory
   metadata.
4. Reject out-of-range and mismatched file/local indices.
5. Allow one global frame in multiple material variants.
6. Verify an empty selected bin is omitted and a partial bin remains.
7. Verify sum and mean binning use only selected contributors.
8. Verify a real all-zero retained frame is counted.
9. Verify selection/config hashes change when thresholds change.
10. Reject stale binned products whose selection hash differs.

### Backend parity tests

For the same filtered mapping, compare bin images from:

- loose raw HDF5;
- complete unbinned archive; and
- prebuilt tagged binned HDF5.

All three must have identical keys and numerically equivalent images.

### GUI tests

1. View/Label and ROI > Shape open the selected variant.
2. Empty selected cells show no data rather than a black frame.
3. Catalog switching changes grid/HDF5 sources together.
4. Shape/Verify displays the material/threshold badge.
5. Device maps retain the same lattice dimensions across unfiltered, Br, and Pb.

### Physics checks

For Scan 24:

1. Confirm selected counts match the XRF cut summary exactly.
2. Randomly sample retained links and verify the tagged source reads the exact raw
   XRD frame identified by file and local index.
3. Confirm rejected global indices never contribute to tagged bins.
4. Compare a tagged bin against a direct raw-frame sum of its selected links.
5. Verify Br/Pb material overlap is preserved rather than forced exclusive.
6. Confirm detected XRD peaks remain in expected 2-theta bands.

## Delivery Phases

### Phase A: CLI proof of integration

- Implement selection validation and filtered mapping creation.
- Add `xrf-cut-grid`.
- Build one Scan 24 Br 3x3 variant.
- Run standard `bin`, `peaks`, and `shapes`.
- Verify direct selected-frame sums and physics.

Exit criterion: the standard CLI pipeline completes without reading a new pixel
format or modifying existing detector algorithms.

### Phase B: Source consistency and lineage

- Add selection hashes to mapping, bins, peaks, and shapes lineage.
- Add stale-product validation.
- Route View/Label through `open_bin_source()`.
- Fix exact variant/grid propagation in GUI source opening.

Exit criterion: every GUI and CLI consumer resolves the same tagged mapping and
pixel source.

### Phase C: GUI controls

- Add Programs controls to create and process XRF-cut variants.
- Add data-variant selectors to pre-catalog image views.
- Add selection badges to catalog-driven views.

Exit criterion: a user can create, process, and inspect Br/Pb selections without
manually entering paths.

### Phase D: Performance optimization

Measure first. If selected raw reads from `/mnt/z` remain a bottleneck:

- add a full-global-index sparse union archive on local WSL storage;
- teach archive resolution to select that cache for XRF variants; and
- verify parity against the complete archive/raw sources.

Do not add this cache before Phase A establishes correct membership and lineage.

## Recommended First Slice

Implement only this vertical slice first:

1. Generate an ordinary unfiltered 3x3 grid for Scan 24.
2. Compile the current Br linker rows into `grid_mapping_3x3_xrf_br.h5`.
3. Build `xrd_3x3_bins_xrf_br.h5` from raw data or the complete archive.
4. Run the existing peak and shape commands with `--variant xrf_br`.
5. Open the resulting catalog in Shape/Verify and Device View.
6. Compare several bins to direct sums from the original linker.

This proves the integration with minimal code. Pb, additional bin sizes,
territories, GUI controls, and optional caching then reuse the same mechanism.
