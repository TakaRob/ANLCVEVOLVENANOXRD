from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import h5py


SCAN_FILE_RE = re.compile(r"Scan_(\d{4})\.h5$")


def append_detectors_group(
    scan_h5_path: str | os.PathLike[str],
    raw_scan_dir: str | os.PathLike[str] | None = None,
    *,
    overwrite: bool = True,
    detector_names: Iterable[str] | None = None,
) -> int:
    """Append or refresh /detectors external links in a Scan_xxxx.h5 file.

    Example:
        Scan_0314.h5/detectors/ME7/scan_0314_00001.h5
        -> ./Raw/Scan_0314/ME7/scan_0314_00001.h5

    Returns the number of external links written.
    """

    scan_h5_path = Path(scan_h5_path).expanduser().resolve()
    match = SCAN_FILE_RE.search(scan_h5_path.name)
    if match is None:
        raise ValueError(
            f"Expected a scan file named like 'Scan_0314.h5', got '{scan_h5_path.name}'."
        )

    scan_id = match.group(1)
    if raw_scan_dir is None:
        raw_scan_dir = scan_h5_path.parent / "Raw" / f"Scan_{scan_id}"
    raw_scan_dir = Path(raw_scan_dir).expanduser().resolve()

    if not scan_h5_path.exists():
        raise FileNotFoundError(f"Scan file not found: {scan_h5_path}")
    if not raw_scan_dir.is_dir():
        raise FileNotFoundError(f"Raw scan directory not found: {raw_scan_dir}")

    selected_detectors = set(detector_names) if detector_names is not None else None
    link_count = 0

    with h5py.File(scan_h5_path, "r+") as scan_h5:
        if "detectors" in scan_h5 and not isinstance(scan_h5["detectors"], h5py.Group):
            raise TypeError(f"'detectors' already exists in {scan_h5_path} but is not a group.")

        detectors_group = scan_h5.require_group("detectors")

        for detector_dir in sorted(raw_scan_dir.iterdir()):
            if not detector_dir.is_dir():
                continue
            if selected_detectors is not None and detector_dir.name not in selected_detectors:
                continue

            detector_group = detectors_group.require_group(detector_dir.name)

            for detector_file in sorted(detector_dir.glob("*.h5")):
                link_name = detector_file.name
                relative_target = os.path.relpath(detector_file, start=scan_h5_path.parent)

                if link_name in detector_group:
                    if not overwrite:
                        continue
                    del detector_group[link_name]

                detector_group[link_name] = h5py.ExternalLink(relative_target, "/")
                link_count += 1

    return link_count


def print_detector_link_sizes(
    scan_h5_path: str | os.PathLike[str],
    *,
    detector_names: Iterable[str] | None = None,
) -> None:
    """Print the /entry/data/data dataset shape for each external link under /detectors."""

    scan_h5_path = Path(scan_h5_path).expanduser().resolve()
    selected_detectors = set(detector_names) if detector_names is not None else None

    with h5py.File(scan_h5_path, "r") as scan_h5:
        if "detectors" not in scan_h5:
            raise KeyError(f"No 'detectors' group found in {scan_h5_path}")

        detectors_group = scan_h5["detectors"]
        if not isinstance(detectors_group, h5py.Group):
            raise TypeError(f"'detectors' exists in {scan_h5_path} but is not a group.")

        for detector_name in sorted(detectors_group.keys()):
            if selected_detectors is not None and detector_name not in selected_detectors:
                continue

            detector_group = detectors_group[detector_name]
            print(f"[{detector_name}]")

            for link_name in sorted(detector_group.keys()):
                link = detector_group.get(link_name, getlink=True)
                if not isinstance(link, h5py.ExternalLink):
                    print(f"  {link_name}: not an external link")
                    continue

                target_path = (scan_h5_path.parent / link.filename).resolve()
                if not target_path.exists():
                    print(f"  {link_name}: missing -> {link.filename}")
                    continue

                with h5py.File(target_path, "r") as target_h5:
                    if "/entry/data/data" not in target_h5:
                        print(f"  {link_name}: missing dataset /entry/data/data -> {link.filename}")
                        continue

                    data_shape = target_h5["/entry/data/data"].shape
                    print(f"  {link_name}: shape={data_shape} -> {link.filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append /detectors external links to a Scan_xxxx.h5 file."
    )
    parser.add_argument("scan_h5_path", help="Path to the Scan_xxxx.h5 file to update.")
    parser.add_argument(
        "--raw-scan-dir",
        help="Optional Raw/Scan_xxxx directory. Defaults to ./Raw/Scan_xxxx next to the scan file.",
    )
    parser.add_argument(
        "--detector",
        action="append",
        dest="detectors",
        help="Detector name to include. Repeat to restrict to multiple detectors.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing detector links instead of replacing same-named entries.",
    )
    parser.add_argument(
        "--print-sizes",
        action="store_true",
        help="Print the /entry/data/data dataset shape for each external link under /detectors.",
    )
    args = parser.parse_args()

    if args.print_sizes:
        print_detector_link_sizes(
            args.scan_h5_path,
            detector_names=args.detectors,
        )
    else:
        link_count = append_detectors_group(
            args.scan_h5_path,
            raw_scan_dir=args.raw_scan_dir,
            overwrite=not args.no_overwrite,
            detector_names=args.detectors,
        )
        print(f"Wrote {link_count} detector external links into {args.scan_h5_path}.")


if __name__ == "__main__":
    main()
