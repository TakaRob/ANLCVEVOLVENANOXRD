#!/usr/bin/env python3
"""Convert pre-HDF5 grid mappings (grid_mapping_*.json) to the HDF5 form the
current pipeline requires (grid_mapping_*.h5).

Older project trees carry grid mappings as JSON. After the HDF5 migration the
CLI resolves and loads only ``grid_mapping_*.h5`` (config.grid_mapping /
io.load_grid_mapping), so ``bin --normalize-frames`` hard-fails on a JSON-only
tree with:

    Error: grid mapping (run 'xrd-app grid' first) not found: .../grid_mapping_3x3.h5

The JSON already holds the full frame->bin assignment (bins/frame_map/xrd_files/
territories) plus provenance (positions_real, coordinate_source, ...), so this is
a pure format conversion — no raw frames needed, and the exact original binning
is preserved.

Usage:
    python3 migrate_grids_to_h5.py <project_root> [--force]

Only converts a JSON whose sibling .h5 is missing (or all, with --force). Each
written .h5 is round-tripped through io.load_grid_mapping to verify it reads back.
"""
import argparse
import json
import sys
from pathlib import Path

from xrd_app.core import io


def convert(json_path: Path, force: bool) -> str:
    h5_path = json_path.with_suffix(".h5")
    if h5_path.exists() and not force:
        return "skip (h5 exists)"
    try:
        mapping = json.loads(json_path.read_text())
    except (OSError, ValueError) as exc:
        return f"FAIL read json: {exc}"
    try:
        io.save_grid_mapping(h5_path, mapping)
        reloaded = io.load_grid_mapping(h5_path)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return f"FAIL convert: {exc}"

    src_frames = sum(len(v) for v in mapping.get("bins", {}).values())
    dst_frames = sum(len(v) for v in reloaded.get("bins", {}).values())
    if (len(reloaded.get("bins", {})) != len(mapping.get("bins", {}))
            or src_frames != dst_frames
            or bool(reloaded.get("positions_real")) != bool(mapping.get("positions_real"))):
        return "FAIL verify: reloaded content does not match json"
    return f"OK ({len(mapping.get('bins', {}))} bins, {src_frames} frames)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Project root (contains Metadata/Scan_*/)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing .h5 grid mappings")
    args = parser.parse_args()

    meta = Path(args.root) / "Metadata"
    if not meta.is_dir():
        print(f"No Metadata dir under {args.root}", file=sys.stderr)
        return 2

    jsons = sorted(meta.glob("Scan_*/grid_mapping_*.json"))
    if not jsons:
        print(f"No grid_mapping_*.json under {meta}")
        return 0

    print(f"Found {len(jsons)} JSON grid mapping(s) under {meta}\n")
    failures = 0
    for json_path in jsons:
        status = convert(json_path, args.force)
        rel = json_path.relative_to(meta)
        print(f"  {rel} -> {json_path.with_suffix('.h5').name}: {status}")
        if status.startswith("FAIL"):
            failures += 1

    print(f"\nDone. {len(jsons) - failures}/{len(jsons)} converted, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
