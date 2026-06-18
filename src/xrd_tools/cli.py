"""xrd-tools command-line interface."""

import os
import shutil
import click
from pathlib import Path

from .config import ProjectConfig, DataManager, default_config


@click.group()
def main():
    """XRD Tools CLI for data processing and CVEvolve integration."""
    pass


# ─────────────────────────────────────────────────────────────────────
# init
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--project-name', prompt='Project Name', help='Name of the project')
@click.option('--scan-number', type=int, default=None, help='Scan number (e.g. 203 for Scan_0203)')
@click.option('--root', default='.', help='Root directory for the project')
def init(project_name, scan_number, root):
    """Initialize a new XRD experiment project structure."""
    cfg = ProjectConfig(root, data=default_config(project_name, root, scan_number))
    cfg.create_tree()
    cfg.save()

    click.echo(f"Project '{project_name}' initialized at {cfg.root}")
    if scan_number is not None:
        click.echo(f"  Scan: Scan_{scan_number:04d}")
    click.echo(f"  Config: {cfg.config_path}")
    click.echo("  Next: 'xrd-tools link' to point at your raw data, tth map, and grid mapping.")


# ─────────────────────────────────────────────────────────────────────
# link
# ─────────────────────────────────────────────────────────────────────
# Maps a --link option to (config data_sources key, destination sub-dir under data/).
_LINK_TARGETS = {
    'raw': ('raw_scan_dir', 'raw_scans'),
    'positions': ('position_csv', 'holdout'),
    'tth': ('tth_map', 'holdout'),
    'grid': ('grid_mapping', 'holdout'),
    'reflections': ('reflections', 'holdout'),
    'detector': ('detector_script', None),  # links into hutch/
}

# Roots for multi-scan setups: recorded as absolute paths, not copied/linked.
_LINK_ROOTS = {
    'raw_root': 'raw_root',
    'position_root': 'position_root',
}


@main.command()
@click.option('--raw', help='Path to external raw scan directory (or HDF5 scan)')
@click.option('--positions', help='Path to the scan position CSV')
@click.option('--raw-root', help='Parent dir containing many Scan_NNNN/ dirs (multi-scan)')
@click.option('--position-root', help='Dir containing scan_NNNN_position.csv files (multi-scan)')
@click.option('--tth', help='Path to external 2-theta TIFF map')
@click.option('--grid', help='Path to external grid mapping JSON')
@click.option('--reflections', help='Path to external reflections.py')
@click.option('--detector', help='Path to a detector / evolved-algorithm .py script')
@click.option('--copy', is_flag=True, help='Copy files instead of symlinking')
@click.option('--root', default='.', help='Project root directory')
def link(raw, positions, raw_root, position_root, tth, grid, reflections, detector, copy, root):
    """Link external data files into the project and record them in config.yaml."""
    cfg = ProjectConfig.load(root)
    if not cfg.exists():
        click.echo("Error: no config.yaml found. Run 'xrd-tools init' first.")
        raise SystemExit(1)
    cfg.data.setdefault('data_sources', {})

    data_dir = cfg.root / cfg.get('paths', 'data_dir', default='data')
    hutch_dir = cfg.root / cfg.get('paths', 'hutch_dir', default='hutch')

    # Roots are just recorded as absolute paths (no copy/symlink).
    for opt, key in (('raw_root', raw_root), ('position_root', position_root)):
        if key:
            p = Path(key).resolve()
            if not p.exists():
                click.echo(f"Warning: {p} does not exist — recording anyway.")
            cfg.data['data_sources'][_LINK_ROOTS[opt]] = str(p)
            click.echo(f"  {_LINK_ROOTS[opt]}: {p}")

    provided = {'raw': raw, 'positions': positions, 'tth': tth, 'grid': grid,
                'reflections': reflections, 'detector': detector}
    if not any(provided.values()) and not (raw_root or position_root):
        click.echo("Nothing to link. Provide at least one of "
                   "--raw/--positions/--raw-root/--position-root/--tth/--grid/"
                   "--reflections/--detector.")
        return

    for opt, source_path in provided.items():
        if not source_path:
            continue
        key, sub_dir = _LINK_TARGETS[opt]
        source = Path(source_path).resolve()
        if not source.exists():
            click.echo(f"Warning: {source} does not exist — skipping.")
            continue

        dest_dir = hutch_dir if sub_dir is None else (data_dir / sub_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name

        stored = _place(source, dest, copy)
        cfg.data['data_sources'][key] = str(stored)
        click.echo(f"  {key}: {stored}")

    cfg.save()
    click.echo(f"Configuration updated: {cfg.config_path}")


def _place(source: Path, dest: Path, copy: bool) -> Path:
    """Copy or symlink ``source`` to ``dest``; return the path to store in config.

    Falls back to recording the original absolute path if linking/copying fails.
    """
    try:
        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if copy:
            if source.is_dir():
                shutil.copytree(source, dest)
            else:
                shutil.copy2(source, dest)
        else:
            os.symlink(source, dest)
        return dest.resolve()
    except Exception as e:
        click.echo(f"  ({'copy' if copy else 'link'} failed: {e}; storing original path)")
        return source


# ─────────────────────────────────────────────────────────────────────
# status — show resolved paths
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--root', default='.', help='Project root directory')
@click.option('--bin-size', type=int, default=3, help='Bin size to resolve bins/grid for')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
def status(root, bin_size, scan):
    """Show the project configuration and resolved data paths."""
    cfg = ProjectConfig.load(root)
    if not cfg.exists():
        click.echo("No config.yaml found. Run 'xrd-tools init' first.")
        raise SystemExit(1)
    dm = DataManager(root, cfg, scan=scan)

    click.echo(f"Project: {cfg.get('name')}")
    click.echo(f"Root:    {cfg.root}")
    click.echo(f"Scan:    {dm.scan_name}")
    click.echo("\nResolved paths (✓ exists / ✗ missing):")
    entries = [
        ("raw_scan_dir", dm.raw_scan_dir()),
        ("xrd_frames_dir", dm.xrd_frames_dir()),
        ("position_csv", dm.position_csv()),
        ("tth_map", dm.tth_map()),
        ("grid_mapping", dm.grid_mapping(bin_size=bin_size)),
        ("reflections", dm.reflections()),
        ("detector_script", dm.detector_script()),
        (f"bins_{bin_size}x{bin_size}", dm.bins_h5(bin_size)),
        ("results_dir", dm.results_dir()),
    ]
    for name, path in entries:
        mark = "✓" if path and Path(path).exists() else "✗"
        click.echo(f"  [{mark}] {name:16s} {path}")


# ─────────────────────────────────────────────────────────────────────
# grid — generate the spatial grid mapping from raw frames
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--shape', default=None,
              help='Synthesize a grid with no positions: ROWSxCOLS or just COLS')
@click.option('--xrd-dir', help='Directory of raw per-frame H5 files (defaults to resolved)')
@click.option('--positions', help='Scan position CSV (defaults to resolved)')
@click.option('--output', help='Output grid_mapping JSON (defaults to per-scan holdout dir)')
@click.option('--root', default='.', help='Project root directory')
def grid(bin_size, scan, shape, xrd_dir, positions, output, root):
    """Generate grid_mapping.json assigning raw frames to a spatial bin grid.

    Uses the position CSV if available; otherwise pass --shape to synthesize a
    regular serpentine grid (uniform-raster fallback).
    """
    from .core import io
    dm = DataManager(root, scan=scan)
    scan_no = dm.scan_number() or 203
    xdir = Path(xrd_dir) if xrd_dir else dm.xrd_frames_dir()
    pos = Path(positions) if positions else dm.position_csv()
    out = Path(output) if output else dm.grid_mapping(bin_size=bin_size)
    out.parent.mkdir(parents=True, exist_ok=True)

    _require(xdir, "raw frames directory")

    n_cols = _parse_shape_cols(shape)
    if not Path(pos).exists() and n_cols is None:
        click.echo(f"Error: no position CSV at {pos} and no --shape given.")
        click.echo("  Provide --positions, link --position-root, or pass --shape ROWSxCOLS.")
        raise SystemExit(1)

    io.generate_grid_mapping(xdir, pos if Path(pos).exists() else None, bin_size,
                             scan_number=scan_no, output=out, n_cols=n_cols, log=click.echo)
    click.echo(f"Wrote grid_mapping -> {out}")


# ─────────────────────────────────────────────────────────────────────
# bin — pre-build the binned HDF5 from raw frames + grid mapping
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--grid-mapping', help='Grid mapping JSON (defaults to resolved)')
@click.option('--output', help='Output binned HDF5 path (defaults to per-scan data/bins/)')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--root', default='.', help='Project root directory')
def bin(bin_size, scan, grid_mapping, output, compression, root):
    """Pre-build the binned HDF5 (xrd_NxN_bins.h5) used by 'process'."""
    from .core import io
    dm = DataManager(root, scan=scan)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size)
    out = Path(output) if output else dm.bins_h5(bin_size)
    _require(gm, "grid mapping (run 'xrd-tools grid' first)")

    io.build_bins(gm, out, bin_size=bin_size, compression=compression, log=click.echo)
    click.echo(f"Wrote bins -> {out}")


# ─────────────────────────────────────────────────────────────────────
# process — run the spatial feature analysis pipeline
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--link-tolerance', type=int, default=5, help='Cross-bin link tolerance (px)')
@click.option('--h5-path', help='Binned HDF5 file (defaults to resolved bins)')
@click.option('--tth-path', help='2-theta TIFF map (defaults to resolved tth)')
@click.option('--detector-script', help='Peak detector script (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.py (defaults to resolved)')
@click.option('--grid-mapping', help='Grid mapping JSON (defaults to resolved)')
@click.option('--output-dir', help='Results directory (defaults to results/<scan>/)')
@click.option('--root', default='.', help='Project root directory')
def process(bin_size, scan, snr, link_tolerance, h5_path, tth_path, detector_script,
            reflections_path, grid_mapping, output_dir, root):
    """Run spatial feature analysis: detect -> link -> filter -> catalog."""
    from .core import processing
    dm = DataManager(root, scan=scan)

    h5 = dm.bins_h5(bin_size, h5_path)
    tth = dm.tth_map(tth_path)
    det = dm.detector_script(detector_script)
    refl = dm.reflections(reflections_path)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size)
    out = Path(output_dir) if output_dir else dm.results_dir()

    for label, path in [("bins", h5), ("tth", tth), ("detector", det),
                        ("reflections", refl), ("grid_mapping", gm)]:
        _require(path, label)

    click.echo(f"[process] bins:     {h5}")
    click.echo(f"[process] detector: {det}")
    click.echo(f"[process] output:   {out}\n")

    summary = processing.run_analysis(
        bins_h5=h5, tth_path=tth, detector_path=det, reflections_path=refl,
        grid_mapping=gm, output_dir=out, bin_size=bin_size,
        snr_threshold=snr, link_tolerance=link_tolerance, log=click.echo)

    click.echo(f"\nDone: {summary['n_kept']} kept, {summary['n_filtered']} filtered")
    click.echo(f"  Catalog: {summary['feature_catalog']}")


# ─────────────────────────────────────────────────────────────────────
# GUIs — one command each
# ─────────────────────────────────────────────────────────────────────
def _resolve_gui_scan(root, scan):
    """Resolve which scan a GUI should open.

    Returns the scan to use (or None to accept the GUI's own default). When no
    ``--scan`` is given and the project has no configured scan, auto-pick the
    only processed scan, or exit with the list when several exist.
    """
    if scan is not None:
        return scan
    dm = DataManager(root)
    if dm.config.get("scan", "name"):   # project has a default scan configured
        return None
    scans = dm.discover_scans()
    if len(scans) == 1:
        click.echo(f"[gui] no --scan given; using the only scan found: {scans[0]}")
        return scans[0]
    if len(scans) > 1:
        click.echo("Multiple scans in this project — pick one with --scan:")
        for s in scans:
            click.echo(f"  {s}  (--scan {dm.scan_number_of(s)})")
        raise SystemExit(2)
    return None   # nothing discovered; let the GUI surface the missing data


def _launch_gui(tool, root, scan=None, bin_size=3):
    from .gui import launch
    return launch(tool, root, scan=_resolve_gui_scan(root, scan), bin_size=bin_size)


# tool name -> (gui module, accepts --bin-size)
_ALL_GUIS = [
    ('view', 'xrd_tools.gui.viewer', True),
    ('label', 'xrd_tools.gui.labeling', False),
    ('device-map', 'xrd_tools.gui.device_map', True),
    ('orientation', 'xrd_tools.gui.orientation', True),
]


@main.command()
@click.option('--root', default='.', help='Project root directory')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--bin-size', type=int, default=3, help='Bin size to view')
@click.option('--only', default=None,
              help='Comma-separated subset, e.g. "view,device-map" (default: all four)')
def gui(root, scan, bin_size, only):
    """Launch all four GUIs at once, each in its own window."""
    import subprocess
    import sys

    root = str(Path(root).resolve())
    scan = _resolve_gui_scan(root, scan)
    wanted = {t.strip() for t in only.split(',')} if only else None

    procs = []
    for name, module, takes_bin in _ALL_GUIS:
        if wanted and name not in wanted:
            continue
        cmd = [sys.executable, '-m', module, '--project-root', root]
        if scan is not None:
            cmd += ['--scan', str(scan)]
        if takes_bin:
            cmd += ['--bin-size', str(bin_size)]
        click.echo(f"[gui] launching {name}: {' '.join(cmd)}")
        procs.append((name, subprocess.Popen(cmd)))

    if not procs:
        click.echo("No GUIs selected. Check --only.")
        raise SystemExit(1)

    click.echo(f"\nLaunched {len(procs)} GUI(s). Close the windows (or Ctrl-C here) to exit.")
    try:
        for _, p in procs:
            p.wait()
    except KeyboardInterrupt:
        for _, p in procs:
            p.terminate()


@main.command()
@click.option('--root', default='.', help='Project root directory')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--bin-size', type=int, default=3, help='Bin size to view')
def label(root, scan, bin_size):
    """Launch the interactive labeling GUI."""
    _launch_gui('label', root, scan, bin_size)


@main.command()
@click.option('--root', default='.', help='Project root directory')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--bin-size', type=int, default=3, help='Bin size to view')
def view(root, scan, bin_size):
    """Launch the feature viewer GUI."""
    _launch_gui('view', root, scan, bin_size)


@main.command(name='device-map')
@click.option('--root', default='.', help='Project root directory')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--bin-size', type=int, default=3, help='Bin size to view')
def device_map(root, scan, bin_size):
    """Launch the device/detector map GUI."""
    _launch_gui('device-map', root, scan, bin_size)


@main.command()
@click.option('--root', default='.', help='Project root directory')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--bin-size', type=int, default=3, help='Bin size to view')
def orientation(root, scan, bin_size):
    """Launch the orientation map GUI."""
    _launch_gui('orientation', root, scan, bin_size)


# ─────────────────────────────────────────────────────────────────────
# run-cvevolve — wrapper around the CVEvolve algorithm search
# ─────────────────────────────────────────────────────────────────────
@main.command(name='run-cvevolve')
@click.option('--config', 'config_path', required=True, help='CVEvolve config.yaml')
@click.option('--prompt', 'prompt_path', help='CVEvolve task prompt .md (optional)')
@click.option('--engine', type=click.Choice(['local', 'podman', 'docker']),
              default='podman', help='Where to run CVEvolve (default: podman container)')
@click.option('--cvevolve-dir', help='Path to the CVEvolve checkout (for venv or image build)')
@click.option('--image', default='cvevolve', help='Container image tag')
@click.option('--build', is_flag=True, help='Build the image from --cvevolve-dir before running')
@click.option('--mount', 'mounts', multiple=True,
              help='Host dir to mount at the same path inside the container (repeatable). '
                   'Defaults to the project root so config.yaml absolute paths resolve.')
@click.option('--env', 'envs', multiple=True, default=('ARGO_API_KEY',),
              help='Environment variable name to pass through (repeatable).')
@click.option('--root', default='.', help='Project root directory')
def run_cvevolve(config_path, prompt_path, engine, cvevolve_dir, image, build,
                 mounts, envs, root):
    """Run CVEvolve with the given config (wraps `cvevolve run`).

    Defaults to running inside a Podman container — the recommended isolation
    boundary for CVEvolve, since the agent executes LLM-generated code. The
    container mounts your host paths at the SAME absolute path so the absolute
    paths inside config.yaml resolve unchanged.
    """
    import subprocess
    import sys
    config_path = Path(config_path).resolve()
    _require(config_path, "CVEvolve config")
    if prompt_path:
        prompt_path = Path(prompt_path).resolve()
        _require(prompt_path, "CVEvolve prompt")

    inner = ["cvevolve", "run", "--config", str(config_path)]
    if prompt_path:
        inner += ["--prompt", str(prompt_path)]

    # ----- local: run directly in a Python environment -----
    if engine == 'local':
        if cvevolve_dir:
            py = Path(cvevolve_dir) / ".venv" / "bin" / "python"
            exe = str(py) if py.exists() else sys.executable
        else:
            exe = sys.executable
        cmd = [exe, "-m", *inner]
        click.echo(f"[run-cvevolve:local] {' '.join(cmd)}")
        raise SystemExit(subprocess.call(cmd))

    # ----- podman / docker: run inside a container -----
    if shutil.which(engine) is None:
        click.echo(f"Error: '{engine}' not found on PATH.")
        raise SystemExit(1)

    if build:
        if not cvevolve_dir or not Path(cvevolve_dir).exists():
            click.echo("Error: --build requires --cvevolve-dir pointing at the CVEvolve checkout.")
            raise SystemExit(1)
        build_cmd = [engine, "build", "-t", image, str(Path(cvevolve_dir).resolve())]
        click.echo(f"[run-cvevolve:{engine}] {' '.join(build_cmd)}")
        rc = subprocess.call(build_cmd)
        if rc != 0:
            raise SystemExit(rc)

    # Mount host paths at identical paths so absolute config paths resolve.
    mount_dirs = [Path(m).resolve() for m in mounts] or [Path(root).resolve()]
    run_cmd = [engine, "run", "--rm", "-it"]
    for name in envs:
        run_cmd += ["-e", name]            # pass through from the host environment
    for d in mount_dirs:
        run_cmd += ["-v", f"{d}:{d}"]
    run_cmd += ["-w", str(config_path.parent), image, *inner]

    click.echo(f"[run-cvevolve:{engine}] {' '.join(run_cmd)}")
    raise SystemExit(subprocess.call(run_cmd))


# ─────────────────────────────────────────────────────────────────────
# batch — run grid -> bin -> process over many scans
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scan numbers/names, e.g. "203,204,205"')
@click.option('--all', 'all_scans', is_flag=True, help='Discover all Scan_NNNN dirs under raw-root')
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--shape', default=None, help='Synthesize grids with no positions: ROWSxCOLS or COLS')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--skip-existing', is_flag=True, help='Skip a scan whose feature catalog already exists')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def batch(ctx, scans, all_scans, bin_size, snr, shape, compression, skip_existing, root):
    """Run grid -> bin -> process for many scans, each in its own per-scan dirs."""
    scan_list = _resolve_scan_list(scans, all_scans, root)
    if not scan_list:
        click.echo("No scans to process. Use --scans \"203,204\" or --all with a linked --raw-root.")
        raise SystemExit(1)

    click.echo(f"Batch over {len(scan_list)} scan(s): {', '.join(scan_list)}\n")
    failures = []
    for name in scan_list:
        click.echo(f"{'='*60}\n  {name}\n{'='*60}")
        dm = DataManager(root, scan=name)
        if skip_existing and (dm.results_dir() /
                              f"feature_catalog_{bin_size}x{bin_size}.json").exists():
            click.echo("  catalog exists — skipping (--skip-existing)\n")
            continue
        try:
            ctx.invoke(grid, bin_size=bin_size, scan=name, shape=shape, root=root)
            ctx.invoke(bin, bin_size=bin_size, scan=name, compression=compression, root=root)
            ctx.invoke(process, bin_size=bin_size, scan=name, snr=snr, root=root)
        except SystemExit as e:
            if e.code:
                click.echo(f"  ✗ {name} failed (exit {e.code})\n")
                failures.append(name)
                continue
        click.echo(f"  ✓ {name} done\n")

    done = len(scan_list) - len(failures)
    click.echo(f"Batch complete: {done}/{len(scan_list)} succeeded"
               + (f", failed: {', '.join(failures)}" if failures else ""))
    if failures:
        raise SystemExit(1)


@main.command()
@click.option('--scans', help='Comma-separated scans to include (default: all in results/)')
@click.option('--bin-size', type=int, default=None, help='Only this bin size (default: all)')
@click.option('--output', help='Output directory (default: results/summary/)')
@click.option('--format', 'fmt', type=click.Choice(['both', 'csv', 'db']), default='both',
              help='What to write (default: both CSV and SQLite)')
@click.option('--root', default='.', help='Project root directory')
def aggregate(scans, bin_size, output, fmt, root):
    """Combine all scans' feature catalogs into one comparable CSV + SQLite DB.

    Produces a `features` table (intensity / prevalence / shape per feature) and
    a long `device_map` table (per scan, reflection, spatial bin).
    """
    from .core import aggregate as agg
    dm = DataManager(root)
    results_dir = dm._abs(dm.config.get('paths', 'results_dir', default='results'))
    out_dir = Path(output) if output else (results_dir / 'summary')
    scan_list = [DataManager.scan_name_of(s.strip()) for s in scans.split(',')] if scans else None

    features, device_map = agg.aggregate(results_dir, scan_list, bin_size, log=click.echo)
    if not features:
        click.echo(f"No feature catalogs found under {results_dir}. Run 'process' first.")
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ('csv', 'both'):
        f_csv = agg.write_csv(features, agg.FEATURE_COLUMNS, out_dir / 'features.csv')
        d_csv = agg.write_csv(device_map, agg.DEVICEMAP_COLUMNS, out_dir / 'device_map.csv')
        click.echo(f"  CSV: {f_csv}")
        click.echo(f"  CSV: {d_csv}  ({len(device_map)} rows)")
    if fmt in ('db', 'both'):
        db = agg.write_sqlite(out_dir / 'analysis.db', features, device_map)
        click.echo(f"  DB:  {db}  (tables: features, device_map)")
    click.echo(f"\nDone: {len(features)} features across "
               f"{len(set(r['scan'] for r in features))} scan(s).")


def _resolve_scan_list(scans, all_scans, root):
    """Build the list of scan names from --scans or by discovering --raw-root."""
    if scans:
        return [DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
    if all_scans:
        dm = DataManager(root)
        raw_root = dm.config.get('data_sources', 'raw_root')
        base = Path(raw_root) if raw_root else (dm.data_dir / 'raw_scans')
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir()
                      if p.is_dir() and p.name.lower().startswith('scan_'))
    return []


def _parse_shape_cols(shape):
    """Parse --shape 'ROWSxCOLS' or 'COLS' into the column count (or None)."""
    if not shape:
        return None
    s = str(shape).lower().replace('×', 'x')
    cols = s.split('x')[-1]
    return int(cols)


# ─────────────────────────────────────────────────────────────────────
def _require(path, label):
    """Abort with a clear message if a required input path is missing."""
    if not path or not Path(path).exists():
        click.echo(f"Error: {label} not found: {path}")
        click.echo("  Check 'xrd-tools status' and 'xrd-tools link'.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
