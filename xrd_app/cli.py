"""xrd-app command-line interface.

The CLI is the engine: every "big button" in the GUI shells out to one of these
commands, and everything is usable headless. Commands are added phase by phase
(see ``PackageDraft2/IMPLEMENTATION.md``).
"""

import os
import shutil
from pathlib import Path

import click

from . import __version__
from .config import ProjectConfig, DataManager, default_config


@click.group()
@click.version_option(__version__, prog_name="xrd-app")
def main():
    """XRD App — peak/shape finding, labeling, and CVEvolve over one workflow."""
    pass


# ─────────────────────────────────────────────────────────────────────
# init
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--name', 'project_name', prompt='Project Name', help='Name of the project')
@click.option('--scan-number', type=int, default=None, help='Scan number (e.g. 203 → Scan_0203)')
@click.option('--root', default='.', help='Root directory for the project')
def init(project_name, scan_number, root):
    """Initialize a new XRD project (creates the standard directory tree)."""
    cfg = ProjectConfig(root, data=default_config(project_name, root, scan_number))
    cfg.create_tree()
    cfg.save()

    click.echo(f"Project '{project_name}' initialized at {cfg.root}")
    if scan_number is not None:
        click.echo(f"  Scan: Scan_{scan_number:04d}")
    click.echo(f"  Config: {cfg.config_path}")
    click.echo("  Next: 'xrd-app scan-detect --scans-dir <dir>' to register your scans,")
    click.echo("        then 'xrd-app link --tth <tiff> --reflections <json>'.")


# ─────────────────────────────────────────────────────────────────────
# link — record external calibration / reflections / detector / roots
# ─────────────────────────────────────────────────────────────────────
# Maps a --link option to (config data_sources key, destination sub-dir).
# Sub-dir None means "record the absolute path only" (no copy/symlink).
_LINK_TARGETS = {
    'tth': ('tth_map', 'Metadata'),
    'reflections': ('reflections', 'Metadata'),
    'detector': ('detector_script', None),
}
_LINK_ROOTS = {
    'raw_root': 'raw_root',
    'position_root': 'position_root',
}


@main.command()
@click.option('--tth', help='Path to a 2θ-per-pixel TIFF map')
@click.option('--reflections', help='Path to reflections.json or reflections.py')
@click.option('--detector', help='Path to a detector / evolved-algorithm .py script')
@click.option('--raw-root', help='Parent dir containing many Scan_NNNN/ dirs (multi-scan)')
@click.option('--position-root', help='Dir containing scan_NNNN_position.csv files (multi-scan)')
@click.option('--poni', help='Path to a pyFAI .poni (recorded; conversion deferred)')
@click.option('--copy', is_flag=True, help='Copy files instead of symlinking')
@click.option('--root', default='.', help='Project root directory')
def link(tth, reflections, detector, raw_root, position_root, poni, copy, root):
    """Link external calibration/reflections/detector files into the project.

    Scan discovery lives in 'xrd-app scan-detect'; this command records the
    shared inputs (tth, reflections, detector, multi-scan roots).
    """
    cfg = ProjectConfig.load(root)
    if not cfg.exists():
        click.echo("Error: no config.yaml found. Run 'xrd-app init' first.")
        raise SystemExit(1)
    cfg.data.setdefault('data_sources', {})

    metadata_dir = cfg.root / cfg.get('paths', 'metadata_dir', default='Metadata')

    for opt, key in (('raw_root', raw_root), ('position_root', position_root)):
        if key:
            p = Path(key).resolve()
            if not p.exists():
                click.echo(f"Warning: {p} does not exist — recording anyway.")
            cfg.data['data_sources'][_LINK_ROOTS[opt]] = str(p)
            click.echo(f"  {_LINK_ROOTS[opt]}: {p}")

    if poni:
        p = Path(poni).resolve()
        cfg.data['data_sources'].setdefault('poni', None)
        cfg.data['data_sources']['poni'] = str(p)
        click.echo(f"  poni: {p}  (note: .poni→tth conversion is not yet implemented)")

    provided = {'tth': tth, 'reflections': reflections, 'detector': detector}
    if not any(provided.values()) and not (raw_root or position_root or poni):
        click.echo("Nothing to link. Provide --tth/--reflections/--detector/"
                   "--raw-root/--position-root/--poni.")
        return

    for opt, source_path in provided.items():
        if not source_path:
            continue
        config_key, sub_dir = _LINK_TARGETS[opt]
        source = Path(source_path).resolve()
        if not source.exists():
            click.echo(f"Warning: {source} does not exist — skipping.")
            continue
        if sub_dir is None:
            cfg.data['data_sources'][config_key] = str(source)
            click.echo(f"  {config_key}: {source}")
            continue
        dest_dir = cfg.root / sub_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        stored = _place(source, dest_dir / source.name, copy)
        cfg.data['data_sources'][config_key] = str(stored)
        click.echo(f"  {config_key}: {stored}")

    cfg.save()
    click.echo(f"Configuration updated: {cfg.config_path}")


def _place(source: Path, dest: Path, copy: bool) -> Path:
    """Copy or symlink ``source`` to ``dest``; return the path to store in config."""
    try:
        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if copy:
            shutil.copytree(source, dest) if source.is_dir() else shutil.copy2(source, dest)
        else:
            os.symlink(source, dest)
        return dest.resolve()
    except Exception as e:
        click.echo(f"  ({'copy' if copy else 'link'} failed: {e}; storing original path)")
        return source


# ─────────────────────────────────────────────────────────────────────
# detectors — list the bundled / saved algorithm library
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=None, help='Filter to one bin size')
@click.option('--root', default='.', help='Project root directory')
def detectors(bin_size, root):
    """List the peak-detector algorithms and their holdout scores."""
    dm = DataManager(root)
    entries = dm.list_detectors(bin_size)
    if not entries:
        click.echo("No detectors found.")
        return
    click.echo(f"Detectors ({dm.detectors_dir()}):\n")
    click.echo(f"  {'bin':>4}  {'f1':>7}  {'f2':>7}  {'src':>8}  name")
    click.echo(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*30}")
    for d in sorted(entries, key=lambda d: (d['bin_size'], -(d.get('holdout_f1') or -1))):
        f1 = f"{d['holdout_f1']:.4f}" if d.get('holdout_f1') is not None else "—"
        f2 = f"{d['holdout_f2']:.4f}" if d.get('holdout_f2') is not None else "—"
        click.echo(f"  {d['bin_size']:>4}  {f1:>7}  {f2:>7}  "
                   f"{str(d.get('source') or '—'):>8}  {d['name']}")


# ─────────────────────────────────────────────────────────────────────
# save-algorithm — freeze a tuned detector variant into the library
# ─────────────────────────────────────────────────────────────────────
@main.command(name='save-algorithm')
@click.option('--base', required=True, help='Base detector name (see `xrd-app detectors`)')
@click.option('--sensitivity', type=float, required=True, help='Baked-in SNR threshold')
@click.option('--bin-size', type=int, required=True, help='Bin size this variant targets')
@click.option('--noise-reduction', default=None, help='Optional noise-reduction label')
@click.option('--name', default=None, help='Output name (default: <base>__sens<NN>__nr-<...>)')
@click.option('--kind', type=click.Choice(['peak', 'shape', 'combined']), default='peak')
@click.option('--root', default='.', help='Project root directory')
def save_algorithm_cmd(base, sensitivity, bin_size, noise_reduction, name, kind, root):
    """Generate a runnable detector that bakes in a sensitivity + noise reduction."""
    from .core import save_algorithm
    out = save_algorithm.save_algorithm(
        base, sensitivity=sensitivity, bin_size=bin_size,
        noise_reduction=noise_reduction, name=name, kind=kind, source="manual")
    click.echo(f"Saved algorithm -> {out}")
    click.echo(f"Run it with: xrd-app peaks --bin-size {bin_size} --algorithm {out.stem}")


# ─────────────────────────────────────────────────────────────────────
# convert-poni — pyFAI .poni → tth.tiff
# ─────────────────────────────────────────────────────────────────────
@main.command(name='convert-poni')
@click.option('--poni', required=True, help='Path to a pyFAI .poni calibration file')
@click.option('--shape', default=None, help='ROWSxCOLS (default: config detector.shape, else from .poni)')
@click.option('--output', default=None, help='Output tth.tiff (default: Metadata/tth.tiff)')
@click.option('--scan', default=None, help='Scan number/name (for a per-scan tth)')
@click.option('--root', default='.', help='Project root directory')
def convert_poni(poni, shape, output, scan, root):
    """Convert a pyFAI .poni calibration into a 2θ-per-pixel tth.tiff."""
    from .core import geometry
    cfg = ProjectConfig.load(root)
    dm = DataManager(root, cfg, scan=scan)
    _require(poni, "poni file")

    sh = None
    if shape:
        s = str(shape).lower().replace('×', 'x')
        rows, cols = s.split('x')
        sh = (int(rows), int(cols))
    elif cfg.get('detector', 'shape'):
        sh = tuple(cfg.get('detector', 'shape'))

    out = Path(output) if output else (dm.metadata_dir / "tth.tiff")
    try:
        geometry.convert_poni_file(poni, out, sh)
    except ImportError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    cfg.data.setdefault('data_sources', {})['tth_map'] = str(out.resolve())
    cfg.data.setdefault('data_sources', {})['poni'] = str(Path(poni).resolve())
    cfg.save()
    click.echo(f"Wrote tth map -> {out}  (shape={sh or 'from .poni'})")


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
        click.echo("No config.yaml found. Run 'xrd-app init' first.")
        raise SystemExit(1)
    dm = DataManager(root, cfg, scan=scan)

    click.echo(f"Project: {cfg.get('name')}")
    click.echo(f"Root:    {cfg.root}")
    click.echo(f"Scan:    {dm.scan_name}")
    shape = cfg.get('detector', 'shape')
    click.echo(f"Frame:   {tuple(shape) if shape else '— (run scan-detect)'}")
    scans = dm.discover_scans()
    click.echo(f"Scans:   {len(scans)} registered" + (f" ({', '.join(scans)})" if scans else ""))

    click.echo("\nResolved paths (✓ exists / ✗ missing):")
    entries = [
        ("scans.json", dm.scans_registry_path()),
        ("raw_scan_dir", dm.raw_scan_dir()),
        ("tth_map", dm.tth_map()),
        ("reflections", dm.reflections()),
        ("grid_mapping", dm.grid_mapping(bin_size=bin_size)),
        ("detector_script", dm.detector_script(bin_size=bin_size)),
        (f"binned_{bin_size}x{bin_size}", dm.binned_h5(bin_size)),
        ("labels_dir", dm.labels_dir()),
    ]
    for label, path in entries:
        mark = "✓" if path and Path(path).exists() else "✗"
        click.echo(f"  [{mark}] {label:16s} {path}")

    from .core import io
    warning = io.slow_mount_warning(dm.binned_dir_root)
    if warning:
        click.echo(f"\n⚠ WSL: {warning}")


# ─────────────────────────────────────────────────────────────────────
# build-holdout — make a CVEvolve dev/holdout split from a labeled source
# ─────────────────────────────────────────────────────────────────────
@main.command(name='build-holdout')
@click.option('--source', type=click.Choice(['verified', 'peaks', 'shapes']), required=True,
              help='Bins from verified labels, or an algorithm peak/shape set')
@click.option('--algorithm', default=None, help='Algorithm name (for peaks/shapes source)')
@click.option('--bin-size', type=int, default=3)
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--holdout-pct', type=float, default=20.0, help='% of labeled bins → holdout')
@click.option('--seed', type=int, default=42, help='Seed for the reproducible split')
@click.option('--dest', default=None, help='CVEvolve dir (default: project CVEvolve/)')
@click.option('--root', default='.', help='Project root directory')
def build_holdout(source, algorithm, bin_size, scan, holdout_pct, seed, dest, root):
    """Build a seeded dev (test_data/) + holdout (holdout_data/) split.

    The source determines the algorithm kind CVEvolve evolves: a peak set → peak
    algorithm, a shape set → shape/combined.
    """
    from .core import holdout as H
    dm = DataManager(root, scan=scan)
    suffix = f"{bin_size}x{bin_size}"

    if source == 'verified':
        src_path = dm.labels_dir(scan) / f"bin_annotations_{suffix}.json"
        _require(src_path, "verified labels (label bins in View/Label first)")
        ann, empty = H.bins_from_verified(src_path)
    elif source == 'peaks':
        if not algorithm:
            click.echo("Error: --algorithm <name> required for --source peaks.")
            raise SystemExit(1)
        src_path = dm.peaks_json(algorithm, bin_size, scan)
        _require(src_path, "peaks JSON")
        ann, empty = H.bins_from_peaks(str(src_path))
    else:  # shapes
        if not algorithm:
            click.echo("Error: --algorithm <name> required for --source shapes.")
            raise SystemExit(1)
        src_path = dm.shapes_json(algorithm, bin_size, scan)
        _require(src_path, "shapes JSON")
        ann, empty = H.bins_from_shapes(str(src_path))

    dest_dir = Path(dest) if dest else dm.cvevolve_dir
    grid = dm.grid_mapping(bin_size=bin_size, scan=scan)
    counts = H.build_split(
        ann, empty, holdout_pct=holdout_pct, seed=seed,
        dest_dev=dest_dir / "test_data", dest_holdout=dest_dir / "holdout_data",
        grid_mapping=grid if Path(grid).exists() else None)

    click.echo(f"[holdout] source={source} kind={'peak' if source!='shapes' else 'shape'}")
    click.echo(f"[holdout] ({counts['holdout_bins']}/{counts['total_bins']}) bins → holdout, "
               f"{counts['dev_bins']} → dev  ({counts['total_points']} points, seed={seed})")
    click.echo(f"[holdout] wrote {dest_dir}/test_data and {dest_dir}/holdout_data")


# ─────────────────────────────────────────────────────────────────────
# run-cvevolve — wrapper around the CVEvolve algorithm search
# ─────────────────────────────────────────────────────────────────────
@main.command(name='run-cvevolve')
@click.option('--config', 'config_path', required=True, help='CVEvolve config.yaml')
@click.option('--prompt', 'prompt_path', default=None, help='CVEvolve task prompt .md (optional)')
@click.option('--engine', type=click.Choice(['local', 'podman', 'docker']), default='podman')
@click.option('--cvevolve-dir', default=None, help='Path to the CVEvolve checkout')
@click.option('--image', default='cvevolve', help='Container image tag')
@click.option('--build', is_flag=True, help='Build the image from --cvevolve-dir first')
@click.option('--mount', 'mounts', multiple=True, help='Host dir to mount at the same path (repeatable)')
@click.option('--env', 'envs', multiple=True, default=('ARGO_API_KEY',), help='Env var to pass through')
@click.option('--root', default='.', help='Project root directory')
def run_cvevolve(config_path, prompt_path, engine, cvevolve_dir, image, build, mounts, envs, root):
    """Run CVEvolve with the given config (Podman by default — LLM-generated code)."""
    import subprocess
    import sys
    config_path = Path(config_path).resolve()
    _require(config_path, "CVEvolve config")
    inner = ["cvevolve", "run", "--config", str(config_path)]
    if prompt_path:
        prompt_path = Path(prompt_path).resolve()
        _require(prompt_path, "CVEvolve prompt")
        inner += ["--prompt", str(prompt_path)]

    if engine == 'local':
        exe = sys.executable
        if cvevolve_dir:
            py = Path(cvevolve_dir) / ".venv" / "bin" / "python"
            exe = str(py) if py.exists() else sys.executable
        cmd = [exe, "-m", *inner]
        click.echo(f"[run-cvevolve:local] {' '.join(cmd)}")
        raise SystemExit(subprocess.call(cmd))

    if shutil.which(engine) is None:
        click.echo(f"Error: '{engine}' not found on PATH.")
        raise SystemExit(1)
    if build:
        if not cvevolve_dir or not Path(cvevolve_dir).exists():
            click.echo("Error: --build requires --cvevolve-dir.")
            raise SystemExit(1)
        rc = subprocess.call([engine, "build", "-t", image, str(Path(cvevolve_dir).resolve())])
        if rc != 0:
            raise SystemExit(rc)

    mount_dirs = [Path(m).resolve() for m in mounts] or [Path(root).resolve()]
    run_cmd = [engine, "run", "--rm", "-it"]
    for name in envs:
        run_cmd += ["-e", name]
    for d in mount_dirs:
        run_cmd += ["-v", f"{d}:{d}"]
    run_cmd += ["-w", str(config_path.parent), image, *inner]
    click.echo(f"[run-cvevolve:{engine}] {' '.join(run_cmd)}")
    raise SystemExit(subprocess.call(run_cmd))


# ─────────────────────────────────────────────────────────────────────
# gui — launch the single-window app
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--root', default='.', help='Project root directory')
@click.option('--scan', default=None, help='Initial scan (defaults to config/last-used)')
@click.option('--bin-size', type=int, default=3, help='Initial bin size')
def gui(root, scan, bin_size):
    """Launch the single-window GUI (Setup / Programs / viewers as tabs)."""
    from .app import launch_app
    raise SystemExit(launch_app(root, scan=scan, bin_size=bin_size))


# ─────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────
def _require(path, label):
    """Abort with a clear message if a required input path is missing."""
    if not path or not Path(path).exists():
        click.echo(f"Error: {label} not found: {path}")
        click.echo("  Check 'xrd-app status' and 'xrd-app link'.")
        raise SystemExit(1)


def _make_progress(noun):
    """Return a progress(i, n) callback that prints a throttled (i/n) count.

    Emits at most ~100 updates plus a final line. The ``PROGRESS i/n`` prefix is
    machine-parseable by the GUI; the ``(i/n) noun`` text is human-readable.
    """
    def progress(i, n):
        step = max(1, n // 100)
        if i == n or i % step == 0:
            click.echo(f"PROGRESS {i}/{n}  ({i}/{n}) {noun}")
    return progress


def _write_json(path, data):
    from .core import io
    return io.atomic_write_json(path, data)


# ─────────────────────────────────────────────────────────────────────
# scan-detect — discover + validate scans, write the registry
# ─────────────────────────────────────────────────────────────────────
@main.command(name='scan-detect')
@click.option('--scans-dir', help='Parent dir of Scan_*/ (or one scan dir)')
@click.option('--scan-file', help='A single .hdf5 file → its scan dir is registered')
@click.option('--deep', is_flag=True, help='Open every file (exact counts, catches corrupt files). Slow on WSL/OneDrive.')
@click.option('--root', default='.', help='Project root directory')
def scan_detect(scans_dir, scan_file, deep, root):
    """Discover scans from a file or directory, validate them, write Raw/scans.json.

    Fast by default: samples the first file per scan (frame count is then an
    estimate). Use --deep for exact counts and full corruption checks.
    """
    from .core import io
    if not scans_dir and not scan_file:
        click.echo("Provide --scans-dir <dir> or --scan-file <hdf5>.")
        raise SystemExit(1)

    cfg = ProjectConfig.load(root)
    if not cfg.exists():
        click.echo("Error: no config.yaml found. Run 'xrd-app init' first.")
        raise SystemExit(1)
    dm = DataManager(root, cfg)

    target = scan_file or scans_dir
    if not Path(target).exists():
        click.echo(f"Error: {target} does not exist.")
        raise SystemExit(1)

    found = io.discover_scans(target, deep=deep)
    if not found:
        click.echo(f"No scans found under {target}.")
        raise SystemExit(1)

    # Adopt the first valid scan's frame shape as the project detector shape.
    proj_shape = cfg.get('detector', 'shape')
    if not proj_shape:
        for s in found:
            if s.get('shape'):
                proj_shape = s['shape']
                break

    registry = dm.scans_registry()
    n_ok = 0
    click.echo(f"{len(found)} scan(s) detected under {target}:\n")
    for s in found:
        problems = io.validate_scan(s, expected_shape=proj_shape)
        mark = "✓" if not problems else "⚠"
        if not problems:
            n_ok += 1
        approx = "~" if s.get('frames_estimated') else ""
        click.echo(f"  [{mark}] {s['name']}  ({s['n_files']} files / "
                   f"{approx}{s['n_frames']} frames, shape={s['shape']})")
        for p in problems:
            click.echo(f"         - {p}")
        registry[s['name']] = {k: s[k] for k in
                               ('dir', 'frames_dir', 'n_files', 'n_frames', 'shape')}

    dm.write_scans_registry(registry)
    cfg.data.setdefault('detector', {})['shape'] = proj_shape
    cfg.data['scans'] = registry
    # If the project has no configured scan and exactly one was found, adopt it.
    if not cfg.get('scan', 'name') and len(found) == 1:
        name = found[0]['name']
        cfg.data['scan'] = {'number': DataManager.scan_number_of(name), 'name': name}
    cfg.save()

    click.echo(f"\n{n_ok}/{len(found)} OK. Frame shape: {proj_shape}.")
    click.echo(f"Registry: {dm.scans_registry_path()}")


# ─────────────────────────────────────────────────────────────────────
# grid — assign raw frames to a spatial bin grid
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--shape', default=None, help='Synthesize a grid with no positions: ROWSxCOLS or COLS')
@click.option('--xrd-dir', help='Directory of raw per-frame H5 files (defaults to resolved)')
@click.option('--positions', help='Scan position CSV (defaults to resolved)')
@click.option('--output', help='Output grid_mapping JSON (defaults to per-scan Metadata dir)')
@click.option('--root', default='.', help='Project root directory')
def grid(bin_size, scan, shape, xrd_dir, positions, output, root):
    """Generate grid_mapping.json assigning raw frames to a spatial bin grid."""
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
# bin — pre-build the binned HDF5
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--grid-mapping', help='Grid mapping JSON (defaults to resolved)')
@click.option('--output', help='Output binned HDF5 path (defaults to per-scan Binned/)')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--root', default='.', help='Project root directory')
def bin(bin_size, scan, grid_mapping, output, compression, root):
    """Pre-build the binned HDF5 (xrd_NxN_bins.h5) used by 'peaks'."""
    from .core import io
    dm = DataManager(root, scan=scan)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size)
    out = Path(output) if output else dm.binned_h5(bin_size)
    _require(gm, "grid mapping (run 'xrd-app grid' first)")
    out.parent.mkdir(parents=True, exist_ok=True)

    io.build_bins(gm, out, bin_size=bin_size, compression=compression, log=click.echo)
    click.echo(f"Wrote bins -> {out}")


# ─────────────────────────────────────────────────────────────────────
# peaks — Phase 1: per-bin detection
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', default=None, help='Detector path OR bundled name (see status)')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--name', 'out_name', default=None, help='Algorithm name for the output file (default: detector stem)')
@click.option('--h5-path', help='Binned HDF5 (defaults to resolved bins)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.py (defaults to resolved)')
@click.option('--root', default='.', help='Project root directory')
def peaks(bin_size, scan, algorithm, snr, out_name, h5_path, tth_path,
          reflections_path, root):
    """Phase 1: run a detector over every bin → per-bin peaks (Labels/<scan>/)."""
    from .core import processing
    dm = DataManager(root, scan=scan)
    h5 = dm.binned_h5(bin_size, h5_path)
    tth = dm.tth_map(tth_path)
    det = dm.detector_script(algorithm, bin_size=bin_size)
    refl = dm.reflections(reflections_path)
    for label, p in [("bins", h5), ("tth", tth), ("detector", det), ("reflections", refl)]:
        _require(p, label)

    algo = out_name or Path(det).stem
    click.echo(f"[peaks] detector: {det}\n[peaks] bins: {h5}\n")
    result = processing.run_peaks(
        bins_h5=h5, tth_path=tth, detector_path=det, reflections_path=refl,
        bin_size=bin_size, snr_threshold=snr,
        progress=_make_progress("peaks"), log=click.echo)
    result["scan"] = dm.scan_name
    result["algorithm"] = algo

    out = dm.peaks_json(algo, bin_size, scan)
    _write_json(out, result)
    click.echo(f"\nDone: {result['n_peaks']} peaks in "
               f"{result['n_bins_with_peaks']} bins -> {out}")


# ─────────────────────────────────────────────────────────────────────
# shapes — Phase 2: link + gaussian filter + characterize
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', default='gaussian', help='Shape algorithm name (output label)')
@click.option('--from-peaks', help='Path to a saved *_peaks.json (else --peak-algo)')
@click.option('--peak-algo', help='Name of a saved peak set in Labels/<scan>/')
@click.option('--link-tolerance', type=int, default=5, help='Cross-bin link tolerance (px)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.py (defaults to resolved)')
@click.option('--grid-mapping', help='Grid mapping JSON (defaults to resolved)')
@click.option('--root', default='.', help='Project root directory')
def shapes(bin_size, scan, algorithm, from_peaks, peak_algo, link_tolerance,
           tth_path, reflections_path, grid_mapping, root):
    """Phase 2: link peaks across bins → gaussian-like shapes (Labels/<scan>/)."""
    import json
    from .core import processing
    dm = DataManager(root, scan=scan)

    peaks_path = Path(from_peaks) if from_peaks else (
        dm.peaks_json(peak_algo, bin_size, scan) if peak_algo else None)
    if not peaks_path:
        click.echo("Error: provide --from-peaks <json> or --peak-algo <name>.")
        raise SystemExit(1)
    _require(peaks_path, "peaks JSON (run 'xrd-app peaks' first)")
    with open(peaks_path) as f:
        peaks_data = json.load(f)

    tth = dm.tth_map(tth_path)
    refl = dm.reflections(reflections_path)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size)
    for label, p in [("tth", tth), ("reflections", refl), ("grid_mapping", gm)]:
        _require(p, label)

    click.echo(f"[shapes] peaks: {peaks_path}\n")
    result = processing.run_shapes(
        peaks=peaks_data, tth_path=tth, grid_mapping=gm, reflections_path=refl,
        bin_size=bin_size, link_tolerance=link_tolerance,
        progress=_make_progress("shapes"), log=click.echo)
    result["scan"] = dm.scan_name
    result["shape_algo"] = algorithm
    result["peak_source"] = peaks_data.get("algorithm", str(peaks_path.name))

    out = dm.shapes_json(algorithm, bin_size, scan)
    _write_json(out, result)

    # Also emit legacy-format catalog + CSVs so the embedded GUIs (viewer,
    # device-map, orientation) read this scan's shapes unchanged.
    suffix = f"{bin_size}x{bin_size}"
    ldir = dm.labels_dir(scan)
    ldir.mkdir(parents=True, exist_ok=True)
    processing.write_feature_catalog(result["kept"], ldir / f"feature_catalog_{suffix}.json", click.echo)
    processing.write_peak_table(result["kept"], ldir / f"kept_peaks_{suffix}.csv", "kept peaks", click.echo)
    processing.write_peak_table(result["filtered"], ldir / f"filtered_peaks_{suffix}.csv", "filtered peaks", click.echo)

    click.echo(f"\nDone: {result['n_kept']} shapes kept, "
               f"{result['n_filtered']} filtered -> {out}")


# ─────────────────────────────────────────────────────────────────────
# batch — grid -> bin -> peaks -> shapes over many scans
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scan numbers/names, e.g. "203,204"')
@click.option('--all', 'all_scans', is_flag=True, help='All scans in the registry')
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--algorithm', default=None, help='Peak detector path OR bundled name')
@click.option('--shape-algo', default='gaussian', help='Shape algorithm name')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--shape', 'grid_shape', default=None, help='Synthesize grids: ROWSxCOLS or COLS')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--skip-existing', is_flag=True, help='Skip a scan whose shapes already exist')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def batch(ctx, scans, all_scans, bin_size, algorithm, shape_algo, snr, grid_shape,
          compression, skip_existing, root):
    """Run grid -> bin -> peaks -> shapes for many scans, each in its own dirs."""
    scan_list = _resolve_scan_list(scans, all_scans, root)
    if not scan_list:
        click.echo('No scans. Use --scans "203,204" or --all (after scan-detect).')
        raise SystemExit(1)

    click.echo(f"Batch over {len(scan_list)} scan(s): {', '.join(scan_list)}\n")
    failures = []
    for name in scan_list:
        click.echo(f"{'='*60}\n  {name}\n{'='*60}")
        dm = DataManager(root, scan=name)
        algo = algorithm or Path(dm.detector_script(algorithm, bin_size=bin_size)).stem
        if skip_existing and dm.shapes_json(shape_algo, bin_size, name).exists():
            click.echo("  shapes exist — skipping (--skip-existing)\n")
            continue
        try:
            ctx.invoke(grid, bin_size=bin_size, scan=name, shape=grid_shape, root=root)
            ctx.invoke(bin, bin_size=bin_size, scan=name, compression=compression, root=root)
            ctx.invoke(peaks, bin_size=bin_size, scan=name, algorithm=algorithm,
                       snr=snr, root=root)
            ctx.invoke(shapes, bin_size=bin_size, scan=name, algorithm=shape_algo,
                       peak_algo=algo, root=root)
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


def _resolve_scan_list(scans, all_scans, root):
    if scans:
        return [DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
    if all_scans:
        return DataManager(root).discover_scans()
    return []


def _parse_shape_cols(shape):
    """Parse --shape 'ROWSxCOLS' or 'COLS' into the column count (or None)."""
    if not shape:
        return None
    s = str(shape).lower().replace('×', 'x')
    return int(s.split('x')[-1])


if __name__ == "__main__":
    main()

