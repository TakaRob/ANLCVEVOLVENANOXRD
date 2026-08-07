"""Standalone command-line interface for XRF project analysis and export."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import click
import h5py
import numpy as np

from .config import DataManager
from .core import xrf as xrf_core
from .core import xrf_selection
from .xrf_project import XRFProject, scan_name


@click.group(context_settings={"show_default": True})
def main():
    """XRF Analysis: prepare finalized material selections for xrd-app."""


@main.command()
@click.option("--name", required=True, help="Name of the xrd-app project")
@click.option("--scan-number", type=click.IntRange(min=0), default=None,
              help="Initial scan number for a newly created xrd-app project")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="xrd-app project root")
def init(name, scan_number, root):
    """Create an xrd-app project and its XRF add-on.

    If ROOT is already an xrd-app project, only the missing ``XRF/`` add-on is
    created. Existing xrd-app configuration and data are never replaced.
    """
    project = XRFProject.load(root)
    try:
        if project.xrd_exists():
            project.create_addon(name)
            created = "XRF add-on"
        else:
            project = XRFProject(root).create(name, scan_number=scan_number)
            created = "xrd-app project and XRF add-on"
    except (FileExistsError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Created {created} at {project.root}")
    click.echo(f"  xrd-app config: {project.xrd_config_path}")
    click.echo(f"  XRF config: {project.config_path}")


@main.command(name="load-data")
@click.option("--source", required=True, type=click.Path(exists=True, path_type=Path),
              help="Raw ME7 scan folder, scans directory, or canonical selection file/directory")
@click.option("--scan", default=None, help="Scan number/name (auto-detected when possible)")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="xrd-app project root")
def load_data(source, scan, root):
    """Register raw ME7 data or import processed XRF selection data."""
    project = XRFProject.load(root)
    if not project.exists():
        raise click.ClickException(f"No XRF add-on found at {project.config_path}")
    source = Path(source).resolve()

    if source.is_file():
        try:
            selection = xrf_selection.load(source)
        except (KeyError, OSError, ValueError) as exc:
            raise click.ClickException(
                f"Unsupported processed file {source}; choose a canonical XRF selection "
                f"or a directory containing canonical selections ({exc})"
            ) from exc
        scan_value = scan or selection["attrs"].get("scan")
        if scan_value is None:
            raise click.ClickException("Selection has no scan identity; pass --scan")
        name = scan_name(scan_value)
        destination = project.selection_path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copy2(source, destination)
        loaded = xrf_selection.load(destination)
        info = xrf_selection.summary(loaded)
        project.register_selection(name, destination, info["materials"], info["selection_hash"])
        click.echo(f"Loaded canonical selection {name} -> {destination}")
        return

    canonical = sorted(source.glob("Scan_*_xrf_selection.h5"))
    if canonical:
        for path in canonical:
            selection = xrf_selection.load(path)
            scan_value = selection["attrs"].get("scan")
            if scan is not None and scan_name(scan_value) != scan_name(scan):
                continue
            name = scan_name(scan_value)
            destination = project.selection_path(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path != destination:
                shutil.copy2(path, destination)
            info = xrf_selection.summary(selection)
            project.register_selection(
                name, destination, info["materials"], info["selection_hash"], save=False
            )
            click.echo(f"Loaded canonical selection {name} -> {destination}")
        project.save()
        return

    scans = XRFProject.discover_scan_folders(source)
    if scan is not None:
        scans = [item for item in scans if item["name"] == scan_name(scan)]
    if not scans:
        raise click.ClickException(
            f"No Scan_*/ME7 data or canonical Scan_*_xrf_selection.h5 files in {source}"
        )
    for item in scans:
        project.register_raw_me7(item["name"], item["me7_dir"], save=False)
        click.echo(
            f"Registered {item['name']}: {item['n_files']} ME7 files, "
            f"{item['n_points']:,} points -> {item['me7_dir']}"
        )
    project.save()


@main.command(name="process-raw")
@click.option("--scan", required=True, help="Registered scan number/name")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="xrd-app project root")
def process_raw(scan, root):
    """Build an editable canonical selection from registered raw ME7 data."""
    from .core import io

    project = XRFProject.load(root)
    if not project.exists():
        raise click.ClickException(f"No XRF add-on found at {project.config_path}")
    name = scan_name(scan)
    record = (project.data.get("scans") or {}).get(name, {})
    me7_dir = record.get("me7_dir")
    if not me7_dir:
        raise click.ClickException(f"No raw ME7 directory registered for {name}")
    me7_files = xrf_core.me7_files(me7_dir)
    if not me7_files:
        raise click.ClickException(f"No scan_*.h5 ME7 files in {me7_dir}")
    config_path = project.metadata_scan_dir(name) / "xrf_elements.json"
    config = xrf_core.read_config(config_path)
    project.metadata_scan_dir(name).mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config["channels"] = list(range(6))
        config["deadtime_correction"] = False
        xrf_core.write_config(config, config_path)
    click.echo(f"Summing complete ME7 spectrum for {name}...")
    spectrum = xrf_core.grand_sum_spectrum(
        me7_dir, config["channels"], config["deadtime_correction"],
        progress=lambda done, total: click.echo(f"PROGRESS {done}/{total} files"),
    )
    calibration = project.data.get("calibration") or config["calibration"]
    energy = xrf_selection.pixel_to_kev(np.arange(xrf_core.N_BINS), calibration)
    sample_theta_deg = None
    y_position_offset = None
    definitions = {}
    for element in config["elements"]:
        center_kev = float(element.get("observed_ev", element["line_ev"])) / 1000.0
        window = element.get("window_ev")
        if window is not None and len(window) == 2:
            energy_range = [center_kev + float(window[0]) / 1000,
                            center_kev + float(window[1]) / 1000]
        else:
            half_width = float(element.get("half_width_ev", 150.0)) / 1000.0
            energy_range = [center_kev - half_width, center_kev + half_width]
        pixels = xrf_selection.kev_to_pixel(energy_range, calibration)
        lo = max(0, int(np.floor(pixels[0])))
        hi = min(xrf_core.N_BINS, int(np.ceil(pixels[1])))
        definitions[element["name"].split("_", 1)[0]] = {
            "display_name": element["name"].split("_", 1)[0],
            "pixel_range": [lo, hi],
            "energy_range_kev": [
                float(xrf_selection.pixel_to_kev(lo, calibration)),
                float(xrf_selection.pixel_to_kev(hi, calibration)),
            ],
            "minimum_counts": None,
        }
    source_files = [str(path) for path in me7_files]
    base = {
        "attrs": {
            "scan": int(name.split("_")[1]),
            "n_total_frames": 0,
            "source_kind": "raw_me7_spectrum",
            "linked_dataset": False,
            "channels": config["channels"],
            "deadtime_correction": config["deadtime_correction"],
            "energy_calibration": calibration,
            "sample_theta_deg": sample_theta_deg,
            "y_position_offset": y_position_offset,
        },
        "source_files": source_files,
        "frames": {
            "global_frame_index": np.asarray([], dtype=np.int64),
            "source_file_index": np.asarray([], dtype=np.int32),
            "source_frame_index": np.asarray([], dtype=np.int64),
            "x": np.asarray([], dtype=float),
            "y": np.asarray([], dtype=float),
        },
        "materials": {
            material: {
                "intensity": np.asarray([], dtype=float),
                "keep": np.asarray([], dtype=bool),
                "attrs": attrs,
            }
            for material, attrs in definitions.items()
        },
        "spectrum": {"energy_kev": energy, "summed_counts": spectrum},
    }
    selection = xrf_selection.validate(base)
    destination = project.selection_path(name)
    xrf_selection.save(destination, selection)
    info = xrf_selection.summary(selection)
    project.register_selection(name, destination, info["materials"], info["selection_hash"])
    click.echo(f"Saved complete ME7 spectrum for {name} -> {destination}")


@main.command(name="link-dataset")
@click.option("--scan", required=True, help="Processed scan number/name")
@click.option("--definitions", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Material range definitions JSON")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="xrd-app project root")
def link_dataset(scan, definitions, root):
    """Register ME7/XRD frames and build per-point material intensities."""
    from .core import io

    project = XRFProject.load(root)
    project.restore_position_offset()
    name = scan_name(scan)
    record = (project.data.get("scans") or {}).get(name, {})
    me7_dir = record.get("me7_dir")
    selection_path = project.selection_path(name)
    if not me7_dir or not selection_path.exists():
        raise click.ClickException("Compute the complete ME7 spectrum first")
    selection = xrf_selection.load(selection_path)
    xrd_files = io.scan_h5_files(Path(me7_dir).parent / "XRD", int(name.split("_")[1]))
    registration = xrf_core.register_raw_frames(me7_dir, xrd_files)
    frame_map = np.asarray(registration["frame_map"], dtype=np.int64)
    global_indices = np.asarray(registration["global_frame_indices"], dtype=np.int64)
    selection["source_files"] = registration["xrd_files"]
    selection["frames"] = {
        "global_frame_index": global_indices,
        "source_file_index": frame_map[:, 0],
        "source_frame_index": frame_map[:, 1],
        "x": np.full(global_indices.size, np.nan),
        "y": np.full(global_indices.size, np.nan),
    }
    selection["attrs"]["n_total_frames"] = registration["n_total_xrd_frames"]
    selection["attrs"]["linked_dataset"] = True
    dm = DataManager(project.root, scan=name)
    raw_root = Path(me7_dir).parent.parent.parent
    positions = raw_root / "processed" / "SOCKETSERVER" / f"{name}_position.h5"
    if positions.exists():
        all_x, all_y = io.load_positions_xy(positions, registration["n_total_xrd_frames"])
        master = raw_root / f"{name}.h5"
        offset_path = project.position_offset_path()
        if master.exists() and offset_path.exists():
            with h5py.File(master, "r") as handle:
                theta = np.asarray(handle[
                    "entry/instrument/bluesky/streams/baseline/sample_theta/value"
                ][:], dtype=float)
            all_y += xrf_core.position_offset_at_theta(offset_path, float(np.nanmean(theta)))
        selection["frames"]["x"] = all_x[global_indices]
        selection["frames"]["y"] = all_y[global_indices]
    with definitions.open() as stream:
        material_definitions = json.load(stream)
    selection["materials"] = {
        material: {
            "intensity": np.full(global_indices.size, np.nan),
            "keep": np.ones(global_indices.size, dtype=bool),
            "attrs": attrs,
        }
        for material, attrs in material_definitions.items()
    }
    selection = xrf_selection.integrate_material_rois(
        selection, me7_dir, material_definitions,
        progress=lambda done, total: click.echo(f"PROGRESS {done}/{total} frames"),
    )
    selection["attrs"]["linked_dataset"] = True
    xrf_selection.save(selection_path, selection)
    info = xrf_selection.summary(selection)
    project.register_selection(name, selection_path, info["materials"], info["selection_hash"])
    click.echo(
        f"Built XRF/XRD frame registration for {name}: {global_indices.size:,} matched frames; "
        f"unmatched ME7/XRD {registration['unmatched_me7']}/"
        f"{registration['unmatched_xrd']}"
    )


@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="Optional xrd-app project root (otherwise choose in Setup)")
def gui(root):
    """Launch XRF setup and analysis; no project path is required."""
    from .app import display_preflight_error, schedule_x11_failure_notice
    display_error = display_preflight_error()
    if display_error:
        raise click.ClickException(display_error)
    schedule_x11_failure_notice()
    from .xrf_gui import launch

    raise SystemExit(launch(root))


@main.command()
@click.option("--scan", default=None, help="Optional scan number/name")
@click.option("--json-output", is_flag=True, help="Print machine-readable JSON")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="XRF project root")
def status(scan, json_output, root):
    """Show XRF project scans and finalized selection status."""
    project = XRFProject.load(root)
    if not project.exists():
        raise click.ClickException(f"No XRF add-on found at {project.config_path}")
    scans = project.data.get("scans") or {}
    if scan is not None:
        name = scan_name(scan)
        scans = {name: scans.get(name, {})}
    report = {"project": project.data.get("name"), "root": str(project.root), "scans": {}}
    for name, record in scans.items():
        path = project.selection_path(name)
        entry = {"selection": str(path), "exists": path.exists()}
        if path.exists():
            try:
                entry.update(xrf_selection.summary(xrf_selection.load(path)))
                entry["valid"] = True
            except (KeyError, OSError, ValueError) as exc:
                entry["valid"] = False
                entry["error"] = str(exc)
        report["scans"][name] = entry
    if json_output:
        click.echo(json.dumps(report, indent=2))
        return
    click.echo(f"XRF project: {report['project']} ({report['root']})")
    if not report["scans"]:
        click.echo("  No scans imported.")
    for name, entry in report["scans"].items():
        state = "valid" if entry.get("valid") else "missing/invalid"
        click.echo(f"  {name}: {state} -> {entry['selection']}")
        for material, values in entry.get("materials", {}).items():
            click.echo(
                f"    {material}: {values['retained_frames']:,} retained "
                f"({values['retained_percent']:.2f}%)"
            )


if __name__ == "__main__":
    main()
