# Preprocessing Process Guide

This guide is a simple map of what the preprocessing scripts do. The raw files in the scan folders are not the final dataset. The pipeline reads them, transforms them, and writes a smaller processed file that the next step uses.

## Big Picture

Raw scan files are taken from a scan directory, processed one by one, and saved into one result directory.

```text
Raw scan folder
    -> read all matching .h5 files
    -> inspect the first file to learn the data shape
    -> crop each diffraction pattern
    -> pad or resize to the target size
    -> clean bad pixels
    -> stack everything into one array
    -> save a processed .hdf5 file
    -> save a matching metadata .hdf5 file
```

## Example Input Folders

The workspace contains examples from different tests, so the exact names are not always consistent. The idea is the same in every case: a scan folder contains many numbered raw files.

```text
Scan_0179/ME7/
    scan_0179_00001.h5
    scan_0179_00002.h5
    scan_0179_00003.h5
    ...

Scan_0180/ME7/
    scan_0180_00001.h5
    scan_0180_00002.h5
    ...
```

That means the pipeline is not looking for one special file. It is looking for the whole numbered sequence in the scan folder.

## What The Code Does

### 1. Find the raw files

The script searches for files like `scan_0179_*.h5` in the scan directory.

Example:

```text
/net/micdata/data1/isn/2026-1/2026-1-Luo/Raw/Scan_0179/PTYCHO/
    scan_0179_00001.h5
    scan_0179_00002.h5
    scan_0179_00003.h5
```

### 2. Read the first file

The first file is opened only to learn:

- whether the data is 2D or 3D
- how many frames are inside each file
- the frame height and width

This is why the code reads `files[0]` first.

### 3. Read every file in the scan

After the shape is known, the script loops over all matching `.h5` files and loads the diffraction patterns from each one.

### 4. Crop the useful region

Only the detector region around the pattern is kept.

Example crop settings in the script:

```text
center = (396, 816)
size = 512 x 512
```

### 5. Make every frame the same size

Each cropped frame is placed into a fixed output size such as `256 x 256`.

### 6. Clean bad pixels

Negative values and very large values are set to zero.

### 7. Save one processed data file

The cleaned frames are stacked and written to one output file.

Example output:

```text
/net/micdata/data1/isn/2026-1/2026-1-Luo/results/scan0179/data_roi1_Ndp256_dp.hdf5
```

### 8. Save one metadata file

The script also writes scan metadata such as:

- wavelength
- pixel size in real space
- scan positions from the CSV file

Example output:

```text
/net/micdata/data1/isn/2026-1/2026-1-Luo/results/scan0179/data_roi1_Ndp256_para.hdf5
```

## Simple File Flow

For one scan, the flow looks like this:

```text
Raw files
    /Raw/Scan_0179/PTYCHO/scan_0179_00001.h5
    /Raw/Scan_0179/PTYCHO/scan_0179_00002.h5
    /Raw/Scan_0179/PTYCHO/scan_0179_00003.h5

Processing
    read all raw files
    crop each frame
    pad/resize to 256 x 256
    clean bad pixels
    stack all frames

Processed files
    /results/scan0179/data_roi1_Ndp256_dp.hdf5
    /results/scan0179/data_roi1_Ndp256_para.hdf5
```

## One-Sentence Summary

Many raw files live in the scan folder, but the pipeline turns them into one processed diffraction stack plus one metadata file.

## Important Note

The files in this workspace came from different tests and do not all match perfectly. This guide is intentionally written as a general process guide, so the directory names and filenames are examples rather than exact one-to-one references for every test folder.

