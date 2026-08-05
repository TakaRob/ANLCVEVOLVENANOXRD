# XRD Binning and Storage Strategies

The two storage approaches in `xrd-app` preserve different levels of
information. The "lossless 1x1" approach is more accurately described as a
**lossless, grid-neutral, frame-level archive**. It is not a precomputed 1x1
grid file.

| | Lossless unbinned archive | Single pre-summed NxN product |
|---|---|---|
| Output | `Binned/<scan>/xrd_unbinned_archive.h5` | `Binned/<scan>/xrd_NxN_bins.h5` |
| Stored data | Every detector frame separately, in acquisition order | One summed detector image per spatial NxN bin |
| Grid dependency | None when archived; mappings are applied later | Permanently tied to the selected grid mapping |
| Detector dtype | Original source dtype, usually `uint16` | Converted to `float32`; values clipped to `[0, 1e9]` |
| Spatial resolution | Full per-frame resolution remains available | Reduced by approximately N in each spatial dimension |
| Can make another bin size later? | Yes, without the original raw files | No |
| Can change deskew/grid/territory assignment? | Yes | No; rebuild from raw data or an archive |
| Storage | Comparable to a compressed copy of all raw detector frames | Up to roughly `1/N^2` as many full detector images; actual ratio depends on compression |
| Read speed | Frames must be summed on demand unless an NxN cache is built | Fast; detector images are already summed |

## Lossless, Grid-Neutral Archive

Build the archive while the raw data is available:

```bash
xrd-app archive-unbinned \
  --root "/path/to/project" \
  --scan 203
```

This writes:

```text
Binned/Scan_0203/xrd_unbinned_archive.h5
```

The archive stores every frame independently in acquisition order, using the
original detector dtype. It does not assign frames to a spatial grid.

Create whichever grid mappings are needed later:

```bash
xrd-app grid \
  --root "/path/to/project" \
  --scan 203 \
  --bin-size 1

xrd-app grid \
  --root "/path/to/project" \
  --scan 203 \
  --bin-size 3
```

These write small mapping files such as:

```text
Metadata/Scan_0203/grid_mapping_1x1.h5
Metadata/Scan_0203/grid_mapping_3x3.h5
```

The mapping says which acquisition-order frames belong to each spatial bin.
When a bin is requested, the app can gather and sum those archived frames on
demand. If two frames occupy one true-position 1x1 cell, they remain separate in
the archive and are summed only when that cell is read.

A fast NxN cache can subsequently be materialized from the archive, even after
the loose raw files have been removed:

```bash
xrd-app bin \
  --root "/path/to/project" \
  --scan 203 \
  --bin-size 3
```

This creates:

```text
Binned/Scan_0203/xrd_3x3_bins.h5
```

### Advantages

- Retains original frame-level information.
- Supports later 1x1, 3x3, 5x5, territory, or revised deskew mappings.
- Allows recovery if the original grid assignment is found to be wrong.
- Supports true per-frame and coordinate-linking analysis.
- Preserves the original integer detector values.

### Costs

- Does not substantially reduce the fundamental amount of detector data.
- On-demand NxN reads require summing multiple archived frames.
- The conventional `xrd-app peaks` command expects a materialized
  `xrd_NxN_bins.h5`, so build the desired bin size before running it.

## Single Pre-Summed NxN Product

If storage is the priority and only one finalized bin size and grid are needed,
generate the grid and directly sum the raw frames:

```bash
xrd-app grid \
  --root "/path/to/project" \
  --scan 203 \
  --bin-size 3

xrd-app bin \
  --root "/path/to/project" \
  --scan 203 \
  --bin-size 3
```

If an unbinned archive does not exist, `bin` reads the loose raw HDF5 files
directly. Retain these products:

```text
Metadata/Scan_0203/grid_mapping_3x3.h5
Binned/Scan_0203/xrd_3x3_bins.h5
```

The app can then browse and analyze the 3x3 data after the raw files are removed:

```bash
xrd-app peaks --root "/path/to/project" --scan 203 --bin-size 3
xrd-app shapes --root "/path/to/project" --scan 203 --bin-size 3
```

Each output image is the sum of every frame assigned to that spatial bin. The
sum retains most total counts, aside from float32 rounding and clipping, but it
cannot retain where those counts originated within the NxN area.

Once both the raw files and lossless archive are gone, the pre-summed product
cannot reconstruct:

- Individual 1x1 frames.
- A different NxN bin size.
- A corrected grid or deskew assignment.
- Territory bins.
- Frame-level intensity variation.

## Important `make-bins` Behavior

The convenience command:

```bash
xrd-app make-bins \
  --root "/path/to/project" \
  --scan 203 \
  --bin-size 3
```

runs all three operations:

```text
archive-unbinned -> grid -> bin
```

It therefore creates both the large lossless archive and the requested 3x3
cache. This is the safest and most reusable workflow, but it is not the
minimum-storage workflow.

To retain only one space-saving NxN product, use the separate `grid` and `bin`
commands and do not run `make-bins` or `archive-unbinned`.

## Practical Recommendation

- For scientifically important or actively analyzed scans, keep
  `xrd_unbinned_archive.h5` and only the NxN caches currently needed.
- For archival scans where storage dominates and the grid is finalized, keep
  only `grid_mapping_NxN[_variant].h5` and `xrd_NxN_bins.h5`.
- Do not refer to the archive as a 1x1 binned file. A true
  `xrd_1x1_bins.h5` is already grid-assigned and may contain collision sums.
  When both exist, the app preferentially uses the lossless archive for ordinary
  1x1 access.
