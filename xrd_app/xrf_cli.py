"""Standalone command-line interface for XRF project analysis and export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
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
                f"or the directory containing a legacy notebook bundle ({exc})"
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
@click.option("--grid-mapping", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Parent xrd-app 1x1 grid mapping")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="xrd-app project root")
def process_raw(scan, grid_mapping, root):
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
    dm = DataManager(project.root, scan=name)
    grid_path = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=1, scan=name)
    if not grid_path.exists():
        raise click.ClickException(
            f"No 1x1 grid mapping for {name}: {grid_path}. Build it in xrd-app first."
        )
    grid = io.load_grid_mapping(grid_path)
    frame_map = np.asarray(grid.get("frame_map", []), dtype=np.int64)
    if frame_map.size == 0:
        raise click.ClickException("Grid mapping has no frame registration")
    config_path = project.metadata_scan_dir(name) / "xrf_elements.json"
    config = xrf_core.read_config(config_path)
    project.metadata_scan_dir(name).mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        xrf_core.write_config(config, config_path)
    click.echo(f"Building per-frame XRF spectrum store for {name}...")
    store = xrf_core.build_point_store(
        me7_dir, grid, config["channels"], config["deadtime_correction"], log=click.echo
    )
    calibration = project.data.get("calibration") or config["calibration"]
    energy = xrf_selection.pixel_to_kev(np.arange(xrf_core.N_BINS), calibration)
    x = np.full(len(frame_map), np.nan, dtype=float)
    y = np.full(len(frame_map), np.nan, dtype=float)
    positions = dm.position_csv(scan=name)
    if positions.exists():
        x, y = io.load_positions_xy(positions, len(frame_map))
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
    source_files = [str(path) for path in grid.get("xrd_files", [])]
    base = {
        "attrs": {
            "scan": int(name.split("_")[1]),
            "n_total_frames": len(frame_map),
            "source_kind": "raw_me7",
            "channels": config["channels"],
            "deadtime_correction": config["deadtime_correction"],
            "energy_calibration": calibration,
        },
        "source_files": source_files,
        "frames": {
            "global_frame_index": np.arange(len(frame_map)),
            "source_file_index": frame_map[:, 0],
            "source_frame_index": frame_map[:, 1],
            "x": x,
            "y": y,
        },
        "materials": {
            material: {
                "intensity": np.zeros(len(frame_map), dtype=float),
                "keep": np.ones(len(frame_map), dtype=bool),
                "attrs": attrs,
            }
            for material, attrs in definitions.items()
        },
        "spectrum": {"energy_kev": energy, "summed_counts": store.sum(axis=0)},
    }
    selection = xrf_selection.validate(base)
    for material, attrs in definitions.items():
        lo, hi = attrs["pixel_range"]
        values = store[:, lo:hi].sum(axis=1, dtype=np.float64)
        selection["materials"][material]["intensity"] = values
        selection["materials"][material]["keep"] = np.ones(values.size, dtype=bool)
    selection = xrf_selection.validate(selection)
    destination = project.selection_path(name)
    xrf_selection.save(destination, selection)
    info = xrf_selection.summary(selection)
    project.register_selection(name, destination, info["materials"], info["selection_hash"])
    point_path = project.cache_scan_dir(name) / f"{name}_xrf_points.npz"
    cache_calibration = config["calibration"] if "ev_per_bin" not in calibration else calibration
    xrf_core.save_point_store(
        point_path, store, config["channels"], config["deadtime_correction"],
        cache_calibration,
    )
    click.echo(f"Processed {name} -> {destination}")


@main.command(name="import-legacy")
@click.option("--linker", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Legacy Scan_NNNN_xrf_xrd_links.h5")
@click.option("--masks", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Legacy threshold mask NPZ (auto-resolved by default)")
@click.option("--roi-config", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Legacy xrf_rois.json")
@click.option("--spectrum", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Legacy ME7 spectrum NPZ")
@click.option("--registration", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Legacy XRF/XRD registration NPZ")
@click.option("--intensities", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Legacy full ROI intensity NPZ")
@click.option("--scan", required=True, help="Scan number/name")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Canonical output (default: Processed/<scan>_xrf_selection.h5)")
@click.option("--root", type=click.Path(path_type=Path), default=Path("."),
              help="XRF project root")
def import_legacy(linker, masks, roi_config, spectrum, registration, intensities,
                  scan, output, root):
    """Import the current prefilter notebook outputs into an XRF project."""
    project = XRFProject.load(root)
    if not project.exists():
        raise click.ClickException(
            f"No XRF add-on found at {project.config_path}; run xrf-app init"
        )
    name = scan_name(scan)
    out = output or project.selection_path(name)
    try:
        selection = xrf_selection.import_legacy_linker(
            linker, mask_path=masks, roi_config_path=roi_config,
            spectrum_path=spectrum, registration_path=registration,
            intensity_path=intensities, scan=int(name.split("_")[1]),
        )
        xrf_selection.save(out, selection)
        loaded = xrf_selection.load(out)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    info = xrf_selection.summary(loaded)
    project.register_selection(name, out, info["materials"], info["selection_hash"])
    click.echo(f"Imported {name} -> {out}")
    for material, values in info["materials"].items():
        click.echo(
            f"  {material}: {values['retained_frames']:,} retained "
            f"({values['retained_percent']:.2f}%)"
        )
    unresolved = loaded["attrs"].get("unresolved_frame_identities", 0)
    if unresolved:
        click.echo(
            f"  Warning: {unresolved:,} frames rejected by every material have no "
            "source/local identity in the retained-only legacy linker."
        )


@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="Optional xrd-app project root (otherwise choose in Setup)")
def gui(root):
    """Launch XRF setup and analysis; no project path is required."""
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
        selection_record = record.get("selection") or {}
        path = Path(selection_record.get("path", project.selection_path(name)))
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
