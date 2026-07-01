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

    # Seed an editable default reflection set so the project resolves reflections
    # from its own Metadata/ (not the hidden bundled fallback) out of the box.
    from .core import reflections as refl_io
    mdir = cfg.root / cfg.get('paths', 'metadata_dir', default='Metadata')
    refl_io.save(refl_io.default_reflections(),
                 mdir / "reflections.json", mdir / "reflections.py")

    click.echo(f"Project '{project_name}' initialized at {cfg.root}")
    click.echo(f"  Reflections: {mdir / 'reflections.json'} "
               "(default perovskite set — edit in Setup → Reflections)")
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
@click.option('--position-csv', help='A single scan position CSV → Metadata/<scan>/positions.csv')
@click.option('--poni', help='Path to a pyFAI .poni (recorded; conversion deferred)')
@click.option('--copy', is_flag=True, help='Copy files instead of symlinking')
@click.option('--scan', default=None, help='Scan number/name (for per-scan --position-csv)')
@click.option('--root', default='.', help='Project root directory')
def link(tth, reflections, detector, raw_root, position_root, position_csv,
         poni, copy, scan, root):
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

    if position_csv:
        src = Path(position_csv).resolve()
        if not src.exists():
            click.echo(f"Warning: {src} does not exist — skipping positions.")
        else:
            dm = DataManager(root, scan=scan)
            dest_dir = dm.metadata_scan_dir(scan) if scan else dm.metadata_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            stored = _place(src, dest_dir / "positions.csv", copy)
            click.echo(f"  positions: {stored}")

    provided = {'tth': tth, 'reflections': reflections, 'detector': detector}
    if not any(provided.values()) and not (raw_root or position_root or position_csv or poni):
        click.echo("Nothing to link. Provide --tth/--reflections/--detector/"
                   "--raw-root/--position-root/--position-csv/--poni.")
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
@click.option('--kind', type=click.Choice(['peak', 'shape', 'combined']), default='peak',
              help='Library to list (peak detectors, shape finders, or combined algos)')
@click.option('--root', default='.', help='Project root directory')
def detectors(bin_size, kind, root):
    """List the algorithm library and holdout scores (peak/shape/combined)."""
    dm = DataManager(root)
    if kind == 'combined':
        entries = dm.list_combined()
        lib_dir = dm.combined_dir()
    elif kind == 'shape':
        entries = dm.list_shapes()
        lib_dir = dm.shapes_dir()
    else:
        entries = dm.list_detectors(bin_size)
        lib_dir = dm.detectors_dir()
    if not entries:
        click.echo("No detectors found.")
        return
    click.echo(f"Detectors ({lib_dir}):\n")
    click.echo(f"  {'bin':>4}  {'f1':>7}  {'f2':>7}  {'src':>8}  name")
    click.echo(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*30}")
    for d in sorted(entries, key=lambda d: (d.get('bin_size') or '', -(d.get('holdout_f1') or -1))):
        f1 = f"{d['holdout_f1']:.4f}" if d.get('holdout_f1') is not None else "—"
        f2 = f"{d['holdout_f2']:.4f}" if d.get('holdout_f2') is not None else "—"
        bin_lbl = d.get('bin_size') or 'any'
        click.echo(f"  {bin_lbl:>4}  {f1:>7}  {f2:>7}  "
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
@click.option('--root', default=None,
              help='Project root (default: last-opened project, or pick one in Setup)')
@click.option('--scan', default=None, help='Initial scan (defaults to config/last-used)')
@click.option('--bin-size', type=int, default=3, help='Initial bin size')
@click.option('--fresh', is_flag=True,
              help='Ignore saved state (last project + last tab/scan); start at Setup.')
def gui(root, scan, bin_size, fresh):
    """Launch the single-window GUI (Setup / Programs / viewers as tabs).

    With no ``--root``, the app reopens the last-used project (remembered in
    ``~/.xrd-app/settings.json``); if there is none, the Setup tab prompts you to
    choose a workspace and create or open a project.

    ``--fresh`` starts a clean session: it does not reopen the last project and
    does not restore the last-used tab/scan/bin size. The workspace is still
    remembered so you can pick a project in Setup. Useful when the remembered
    project is broken or slow to load.
    """
    from .app import launch_app
    raise SystemExit(launch_app(root, scan=scan, bin_size=bin_size, fresh=fresh))


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
@click.option('--rawgrid', is_flag=True,
              help='Bypass (X,Y) de-skew: use the legacy serpentine X-only grid.')
@click.option('--deskew-method',
              type=click.Choice(['commanded', 'faithful', 'faithful_native', 'perrow_offset']),
              default='commanded',
              help='Column assignment for file-per-row scans: commanded (default, '
                   'align by rank), faithful (snap columns to true Y on a square-pixel '
                   'lattice — to-scale, de-skewed, best for viewing), faithful_native '
                   '(snap to true Y at native frame density — de-skewed with ~1 frame/'
                   'cell, best for detection/recall), or perrow_offset (DEPRECATED).')
@click.option('--variant', default=None,
              help='Tag appended to default output names (e.g. "faithful") so a '
                   'coordinate variant sits alongside the default instead of overwriting it.')
@click.option('--output', help='Output grid_mapping JSON (defaults to per-scan Metadata dir)')
@click.option('--root', default='.', help='Project root directory')
def grid(bin_size, scan, shape, xrd_dir, positions, rawgrid, deskew_method, variant, output, root):
    """Generate grid_mapping.json assigning raw frames to a spatial bin grid.

    Coordinate source (auto-selected, recorded in the output JSON):
    a real position CSV → de-skewed ``positions_xy`` (or ``--rawgrid`` for the
    legacy serpentine X-only grid); no CSV → reconstructed from the one-file-per-
    row layout (``file_per_row``); ``--shape`` → ``synthetic`` raster.

    When no position CSV exists, one is created automatically: first from the
    **real SOCKETSERVER interferometry** stream if present (see
    'xrd-app create-positions'), otherwise **recreated** from the file-per-row
    layout (so downstream never zero-pads positions). Pass ``--shape`` to
    synthesize a raster instead.
    """
    from .core import io
    dm = DataManager(root, scan=scan)
    scan_no = dm.scan_number() or 203
    xdir = Path(xrd_dir) if xrd_dir else dm.xrd_frames_dir()
    pos = Path(positions) if positions else dm.position_csv()
    out = Path(output) if output else dm.grid_mapping(bin_size=bin_size, variant=variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    _require(xdir, "raw frames directory")

    if not io.has_raw_frames(xdir, scan_no):
        click.echo(f"Error: no raw frame files (scan_{scan_no:04d}_*.h5) in {xdir}.")
        click.echo("  This scan looks incomplete (no XRD frames). Skip it or check the raw data.")
        raise SystemExit(1)

    n_cols = _parse_shape_cols(shape)
    pos_real = Path(pos).exists() and not io.is_recreated_csv(pos)

    # When we don't have a real position CSV (and weren't asked to synthesize a
    # raster shape), build one from the REAL stage positions in the SOCKETSERVER
    # interferometry stream. With no interferometry data we pass no CSV and let
    # generate_grid_mapping reconstruct the grid straight from the one-file-per-
    # row layout (exact for these scans) — no synthetic lattice is fabricated.
    if not pos_real and n_cols is None:
        from .core import positions as P
        sdir = dm.socketserver_dir(scan=scan)
        if P.has_socketserver(sdir, scan_no):
            dest = dm.metadata_scan_dir(scan) / "positions.csv"
            click.echo("No real position CSV — building one from SOCKETSERVER "
                       f"interferometry ({sdir}) ...")
            try:
                P.build_positions_csv(sdir, dest, scan_number=scan_no, log=click.echo)
                pos, pos_real = dest, True
            except (FileNotFoundError, ValueError) as e:
                click.echo(f"  SOCKETSERVER positions failed ({e}); "
                           "using the one-file-per-row layout for the grid.")
        else:
            click.echo(f"No SOCKETSERVER interferometry at {sdir} — using the "
                       "one-file-per-row layout for the grid.")

    io.generate_grid_mapping(xdir, pos if pos_real else None, bin_size,
                             scan_number=scan_no, output=out, n_cols=n_cols,
                             deskew=not rawgrid, deskew_method=deskew_method,
                             log=click.echo)
    click.echo(f"Wrote grid_mapping -> {out}")


# ─────────────────────────────────────────────────────────────────────
# territory-grid — skew-free reference binning by true (X, Y) territories
# ─────────────────────────────────────────────────────────────────────
@main.command(name='territory-grid')
@click.option('--target-size', type=int, default=9,
              help='Frames per territory before it stops growing (sweepable; '
                   'small ≈ 1×1 resolution, large = higher per-cell SNR).')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--xrd-dir', help='Directory of raw per-frame H5 files (defaults to resolved)')
@click.option('--positions', help='Real scan position CSV (defaults to resolved)')
@click.option('--variant', default='territory',
              help='Tag for the output names so the territorial mapping sits '
                   'alongside the grid ones (default "territory").')
@click.option('--output', help='Output grid_mapping JSON (defaults to per-scan Metadata dir)')
@click.option('--root', default='.', help='Project root directory')
def territory_grid(target_size, scan, xrd_dir, positions, variant, output, root):
    """Build a territorial (cell-model) grid mapping — the skew-free source of truth.

    Groups frames by **true (X, Y) stage positions** into irregular territories
    that grow until they hit ``--target-size`` frames, bypassing the serpentine
    reconstruction that skews the N×N grid. Requires a *real* position CSV
    (X_Position/Y_Position); it will not fall back to a recreated lattice.

    Then run the standard pipeline on the variant (bin_size is nominally 1×1)::

        xrd-app bin    --bin-size 1 --variant territory
        xrd-app peaks  --bin-size 1 --variant territory
        xrd-app shapes --bin-size 1 --variant territory --algorithm territory
    """
    from .core import io, territory
    dm = DataManager(root, scan=scan)
    scan_no = dm.scan_number() or 203
    xdir = Path(xrd_dir) if xrd_dir else dm.xrd_frames_dir()
    pos = Path(positions) if positions else dm.position_csv()
    out = Path(output) if output else dm.grid_mapping(bin_size=1, variant=variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    _require(xdir, "raw frames directory")

    if not io.has_raw_frames(xdir, scan_no):
        click.echo(f"Error: no raw frame files (scan_{scan_no:04d}_*.h5) in {xdir}.")
        raise SystemExit(1)

    try:
        territory.build_territory_mapping(
            xdir, pos, target_size=target_size, scan_number=scan_no,
            output=out, log=click.echo)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    click.echo(f"Wrote territorial grid_mapping -> {out}")


# ─────────────────────────────────────────────────────────────────────
# create-positions — build a REAL position CSV from SOCKETSERVER interferometry
# ─────────────────────────────────────────────────────────────────────
@main.command(name='create-positions')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--socket-dir', help='SOCKETSERVER interferometry dir (defaults to resolved)')
@click.option('--method', type=click.Choice(['averaging', 'basic']), default='averaging',
              help='averaging (default, self-contained) or basic (needs --theta)')
@click.option('--theta', type=float, default=None,
              help='Sample theta in degrees — only used by --method basic')
@click.option('--reduction', type=int, default=1,
              help='Use every Nth interferometer sample (speed; 1 = all)')
@click.option('--output', help='Output CSV (default: Metadata/<scan>/positions.csv)')
@click.option('--force', is_flag=True, help='Overwrite an existing CSV')
@click.option('--root', default='.', help='Project root directory')
def create_positions(scan, socket_dir, method, theta, reduction, output, force, root):
    """Build a REAL per-frame position CSV from the SOCKETSERVER interferometry stream.

    Reduces the interferometer encoder samples to one true (X, Y) stage position
    per trigger and writes ``Metadata/<scan>/positions.csv`` — the *real*
    measured positions. 'xrd-app grid' calls this automatically when no position
    CSV is found; run it directly to (re)generate one. When a scan has no
    SOCKETSERVER stream, the grid is reconstructed from the one-file-per-row
    layout instead (no positions file needed).
    """
    from .core import io, positions as P
    dm = DataManager(root, scan=scan)
    scan_no = dm.scan_number() or 203
    sdir = Path(socket_dir) if socket_dir else dm.socketserver_dir(scan=scan)
    out = Path(output) if output else (dm.metadata_scan_dir(scan) / "positions.csv")

    if out.exists() and not force:
        kind = "recreated" if io.is_recreated_csv(out) else "existing (real?)"
        click.echo(f"Refusing to overwrite {kind} CSV: {out}")
        click.echo("  Pass --force to overwrite, or --output to write elsewhere.")
        raise SystemExit(1)

    if not P.has_socketserver(sdir, scan_no):
        click.echo(f"Error: no SOCKETSERVER files (scan_{scan_no:04d}_*.h5) in {sdir}.")
        click.echo("  This scan has no interferometry stream — 'xrd-app grid' will "
                   "reconstruct the grid from the one-file-per-row layout instead.")
        raise SystemExit(1)

    try:
        info = P.build_positions_csv(
            sdir, out, scan_number=scan_no, method=method, theta_deg=theta,
            reduction=reduction, log=click.echo)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    click.echo(f"Wrote {info['n_positions']} real positions "
               f"(span {info['x_span_um']:.1f} x {info['y_span_um']:.1f} um) "
               f"-> {info['path']}")


# ─────────────────────────────────────────────────────────────────────
# bin — pre-build the binned HDF5
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--grid-mapping', help='Grid mapping JSON (defaults to resolved)')
@click.option('--variant', default=None,
              help='Coordinate variant tag (e.g. "faithful") — resolves the matching '
                   'tagged grid mapping and writes a tagged binned HDF5.')
@click.option('--output', help='Output binned HDF5 path (defaults to per-scan Binned/)')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--root', default='.', help='Project root directory')
def bin(bin_size, scan, grid_mapping, variant, output, compression, root):
    """Pre-build the binned HDF5 (xrd_NxN_bins.h5) used by 'peaks'."""
    from .core import io
    dm = DataManager(root, scan=scan)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size, variant=variant)
    out = Path(output) if output else dm.binned_h5(bin_size, variant=variant)
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
@click.option('--variant', default=None,
              help='Coordinate variant tag (e.g. "faithful") — reads the tagged bins '
                   'and writes a tagged peaks JSON.')
@click.option('--h5-path', help='Binned HDF5 (defaults to resolved bins)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.py (defaults to resolved)')
@click.option('--root', default='.', help='Project root directory')
def peaks(bin_size, scan, algorithm, snr, out_name, variant, h5_path, tth_path,
          reflections_path, root):
    """Phase 1: run a detector over every bin → per-bin peaks (Labels/<scan>/)."""
    from .core import processing
    dm = DataManager(root, scan=scan)
    h5 = dm.binned_h5(bin_size, h5_path, variant=variant)
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
    from .core import lineage
    result["lineage"] = lineage.peak_lineage(
        scan=dm.scan_name, bin_size=bin_size, algorithm=algo,
        detector_file=det, snr=snr)

    out = dm.peaks_json(algo, bin_size, scan, variant=variant)
    _write_json(out, result)
    from .core import catalogs
    catalogs.record_catalog(dm.labels_dir(scan), out.name, result["lineage"])
    click.echo(f"\nDone: {result['n_peaks']} peaks in "
               f"{result['n_bins_with_peaks']} bins -> {out}")


# ─────────────────────────────────────────────────────────────────────
# shapes — Phase 2: link + gaussian filter + characterize
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', default='gaussian', help='Shape algorithm path OR bundled name (see "detectors --kind shape")')
@click.option('--from-peaks', help='Path to a saved *_peaks.json (else --peak-algo)')
@click.option('--peak-algo', help='Name of a saved peak set in Labels/<scan>/')
@click.option('--link-tolerance', type=int, default=5, help='Cross-bin link tolerance (px)')
@click.option('--variant', default=None,
              help='Coordinate variant tag (e.g. "faithful") — resolves the tagged '
                   'peaks/grid and writes a tagged shapes JSON + CSVs.')
@click.option('--coordinate/--grid-link', 'coordinate', default=None,
              help='Linking mode. Gridless coordinate linking (across true (X,Y) '
                   'neighbors) is the DEFAULT at 1×1 — the skew-free path, no grid '
                   'to skew; binned sizes (≥2×2) default to grid linking, where '
                   'backlash is already averaged out. --grid-link / --coordinate '
                   'force the choice. Coordinate mode reuses the standard peaks '
                   '(only linking changes) and writes a "_coord" shapes file.')
@click.option('--positions', help='Position CSV for coordinate linking (defaults to resolved)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.py (defaults to resolved)')
@click.option('--grid-mapping', help='Grid mapping JSON (defaults to resolved)')
@click.option('--root', default='.', help='Project root directory')
def shapes(bin_size, scan, algorithm, from_peaks, peak_algo, link_tolerance, variant,
           coordinate, positions, tth_path, reflections_path, grid_mapping, root):
    """Phase 2: link peaks → shapes (Labels/<scan>/).

    Links peaks into shapes. At 1×1 the default is gridless **coordinate**
    linking (across true (X,Y) physical neighbors) — the skew-free path, since
    the serpentine/backlash skew is a grid artefact and there is no grid here.
    Binned sizes default to grid linking. Coordinate mode reuses the standard
    peaks and only changes the linking stage.
    """
    import json
    from .core import processing
    dm = DataManager(root, scan=scan)

    # Gridless coordinate linking is the skew-free default at 1×1; binned sizes
    # keep grid linking. An explicit --coordinate/--grid-link overrides.
    if coordinate is None:
        coordinate = (bin_size == 1)

    # Coordinate linking needs positions; degrade to grid linking if absent so
    # position-less projects still run (with a clear note).
    pos = None
    if coordinate:
        pos = Path(positions) if positions else dm.position_csv(scan=scan)
        if not Path(pos).exists():
            click.echo(f"Note: no position CSV ({pos}) — falling back to grid linking.")
            coordinate = False
    if coordinate and algorithm in (None, 'gaussian'):
        # Coordinate linking needs a neighbor-graph-capable linker. 'gaussian'
        # (grid-only) maps to 'territory' = same gaussian verification, coordinate
        # linking. An explicit coordinate-capable algo (e.g. voigt) is kept.
        algorithm = 'territory'

    peaks_path = Path(from_peaks) if from_peaks else (
        dm.peaks_json(peak_algo, bin_size, scan, variant=variant) if peak_algo else None)
    if not peaks_path:
        click.echo("Error: provide --from-peaks <json> or --peak-algo <name>.")
        raise SystemExit(1)
    _require(peaks_path, "peaks JSON (run 'xrd-app peaks' first)")
    with open(peaks_path) as f:
        peaks_data = json.load(f)

    tth = dm.tth_map(tth_path)
    refl = dm.reflections(reflections_path)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size, variant=variant)
    shape = dm.shape_script(algorithm)
    for label, p in [("tth", tth), ("reflections", refl), ("grid_mapping", gm),
                     ("shape algorithm", shape)]:
        _require(p, label)
    algo = Path(shape).stem

    # Gridless coordinate linking: augment the grid mapping with true-(X,Y)
    # neighbors and route to a "_coord" output so it never clobbers grid shapes.
    out_variant = variant
    grid_for_run = gm
    if coordinate:
        from .core import io as core_io, territory
        gm_dict = core_io.load_grid_mapping(gm)
        n_total = gm_dict.get("n_total_frames") or len(gm_dict.get("frame_map", []))
        fx, fy = core_io.load_positions_xy(pos, n_total)
        if not (fx == fx).any():
            click.echo("Error: positions have no usable X — cannot link by coordinate.")
            raise SystemExit(1)
        territory.add_coordinate_neighbors(gm_dict, fx, fy, log=click.echo)
        grid_for_run = gm_dict
        out_variant = f"{variant}_coord" if variant else "coord"

    click.echo(f"[shapes] algorithm: {shape}\n[shapes] peaks: {peaks_path}\n")
    result = processing.run_shapes(
        peaks=peaks_data, tth_path=tth, grid_mapping=grid_for_run, reflections_path=refl,
        bin_size=bin_size, link_tolerance=link_tolerance, shape_path=shape,
        progress=_make_progress("shapes"), log=click.echo)
    result["scan"] = dm.scan_name
    result["shape_algo"] = algo
    result["peak_source"] = peaks_data.get("algorithm", str(peaks_path.name))
    from .core import lineage
    result["lineage"] = lineage.shape_lineage(
        scan=dm.scan_name, bin_size=bin_size, shape_algorithm=algo,
        link_tolerance=link_tolerance,
        peak_source=lineage.from_peaks_data(peaks_data, fallback_file=peaks_path.name),
        peak_source_file=peaks_path.name)

    out = dm.shapes_json(algo, bin_size, scan, variant=out_variant)
    _write_json(out, result)

    # Emit the kept/filtered CSVs alongside the shapes file. The shapes JSON is
    # the catalog the GUIs (viewer, device-map, orientation) read directly via
    # core.catalogs — no separate feature_catalog copy is written anymore.
    suffix = f"{bin_size}x{bin_size}" + (f"_{out_variant}" if out_variant else "")
    ldir = dm.labels_dir(scan)
    ldir.mkdir(parents=True, exist_ok=True)
    processing.write_peak_table(result["kept"], ldir / f"kept_peaks_{suffix}.csv", "kept peaks", click.echo)
    processing.write_peak_table(result["filtered"], ldir / f"filtered_peaks_{suffix}.csv", "filtered peaks", click.echo)

    # The shapes file carries its own in-file lineage block.
    from .core import catalogs
    catalogs.record_catalog(ldir, out.name, result["lineage"])

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
@click.option('--rawgrid', is_flag=True,
              help='Bypass (X,Y) de-skew: use the legacy serpentine X-only grid.')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--skip-existing', is_flag=True, help='Skip a scan whose shapes already exist')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def batch(ctx, scans, all_scans, bin_size, algorithm, shape_algo, snr, grid_shape,
          rawgrid, compression, skip_existing, root):
    """Run grid -> bin -> peaks -> shapes for many scans, each in its own dirs."""
    scan_list = _resolve_scan_list(scans, all_scans, root)
    if not scan_list:
        click.echo('No scans. Use --scans "203,204" or --all (after scan-detect).')
        raise SystemExit(1)

    from .core import io
    click.echo(f"Batch over {len(scan_list)} scan(s): {', '.join(scan_list)}\n")
    failures, skipped = [], []
    for name in scan_list:
        click.echo(f"{'='*60}\n  {name}\n{'='*60}")
        dm = DataManager(root, scan=name)
        # Skip incomplete scans (no XRD/ frame files) rather than crashing — many
        # Scan_NNNN/ dirs on the beamline mount have no frames yet.
        if not io.has_raw_frames(dm.xrd_frames_dir(scan=name), dm.scan_number(name) or 0):
            click.echo("  no raw frames (incomplete scan) — skipping\n")
            skipped.append(name)
            continue
        algo = algorithm or Path(dm.detector_script(algorithm, bin_size=bin_size)).stem
        if skip_existing and dm.shapes_json(shape_algo, bin_size, name).exists():
            click.echo("  shapes exist — skipping (--skip-existing)\n")
            continue
        try:
            ctx.invoke(grid, bin_size=bin_size, scan=name, shape=grid_shape,
                       rawgrid=rawgrid, root=root)
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
        except Exception as e:  # one bad scan must not abort the whole batch
            click.echo(f"  ✗ {name} errored: {e}\n")
            failures.append(name)
            continue
        click.echo(f"  ✓ {name} done\n")

    done = len(scan_list) - len(failures) - len(skipped)
    click.echo(f"Batch complete: {done}/{len(scan_list)} succeeded"
               + (f", {len(skipped)} skipped (incomplete)" if skipped else "")
               + (f", failed: {', '.join(failures)}" if failures else ""))
    if failures:
        raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────
# run-pipeline — peaks -> shapes for ONE scan in a single process
# ─────────────────────────────────────────────────────────────────────
@main.command(name='run-pipeline')
@click.option('--bin-size', type=int, default=3, help='Bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', default=None, help='Peak detector path OR bundled name')
@click.option('--shape-algo', default='gaussian', help='Shape algorithm name (output label)')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def run_pipeline(ctx, bin_size, scan, algorithm, shape_algo, snr, root):
    """Run Peak Finding then Shape Finding for one scan, back to back."""
    dm = DataManager(root, scan=scan)
    # Same naming peaks uses for its output set, so shapes can pick it up.
    algo = algorithm or Path(dm.detector_script(algorithm, bin_size=bin_size)).stem
    ctx.invoke(peaks, bin_size=bin_size, scan=scan, algorithm=algorithm,
               snr=snr, root=root)
    ctx.invoke(shapes, bin_size=bin_size, scan=scan, algorithm=shape_algo,
               peak_algo=algo, root=root)
    click.echo("\nPipeline complete: peaks → shapes")


# ─────────────────────────────────────────────────────────────────────
# make-bins — grid mapping -> binned HDF5 for ONE scan
# ─────────────────────────────────────────────────────────────────────
@main.command(name='make-bins')
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--shape', 'grid_shape', default=None,
              help='Synthesize a grid (no positions): ROWSxCOLS or COLS')
@click.option('--rawgrid', is_flag=True,
              help='Bypass (X,Y) de-skew: use the legacy serpentine X-only grid.')
@click.option('--compression', type=click.Choice(['gzip', 'lz4', 'none']), default='gzip')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def make_bins(ctx, bin_size, scan, grid_shape, rawgrid, compression, root):
    """Build the binned HDF5 for one scan: grid mapping, then bins."""
    ctx.invoke(grid, bin_size=bin_size, scan=scan, shape=grid_shape,
               rawgrid=rawgrid, root=root)
    ctx.invoke(bin, bin_size=bin_size, scan=scan, compression=compression, root=root)
    dm = DataManager(root, scan=scan)
    click.echo(f"\nBins ready: {dm.binned_h5(bin_size)}")


# ─────────────────────────────────────────────────────────────────────
# run-combined — peak + shape in one per-frame pass (CombinedAlgorithms)
# ─────────────────────────────────────────────────────────────────────
@main.command(name='run-combined')
@click.option('--bin-size', type=int, default=1, help='Bin size (combined algos are 1x1)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', required=True,
              help='Combined algorithm name (see `xrd-app detectors --kind combined`)')
@click.option('--root', default='.', help='Project root directory')
def run_combined_cmd(bin_size, scan, algorithm, root):
    """Run a combined (peak+shape) per-frame algorithm over a scan.

    Combined detectors do detection + cross-bin linking + Voigt verification in
    one pass and emit final validated features. Output is feature-level (no per-bin
    intensities), so View/Label shows the features but Device/Orientation heatmaps
    are not populated.
    """
    from .core import processing
    dm = DataManager(root, scan=scan)
    det = dm.combined_script(algorithm)
    h5 = dm.binned_h5(bin_size)
    tth = dm.tth_map()
    refl = dm.reflections()
    gm = dm.grid_mapping(bin_size=bin_size)
    if not Path(h5).exists():
        click.echo(f"Error: no {bin_size}x{bin_size} bins at {h5}.")
        click.echo("  Build them first: Programs → Create bins at 1x1 "
                   "(or `xrd-app make-bins --bin-size 1`).")
        raise SystemExit(1)
    for label, p in [("combined detector", det), ("tth", tth),
                     ("reflections", refl), ("grid_mapping", gm)]:
        _require(p, label)

    # Output identity: a sub-foldered detector.py is named by its folder
    # (e.g. "1x1_global_perframe_uf_voigt"); a flat algorithm by its file stem.
    detp = Path(det)
    algo = detp.parent.name if detp.stem == "detector" else detp.stem
    click.echo(f"[combined] detector: {det}\n[combined] bins: {h5}\n")
    result = processing.run_combined(
        detector_path=det, tth_path=tth, reflections_path=refl,
        bins_h5=h5, grid_mapping=gm,
        progress=_make_progress("combined"), log=click.echo)
    result["scan"] = dm.scan_name
    from .core import lineage
    result["lineage"] = lineage.combined_lineage(
        scan=dm.scan_name, bin_size=bin_size, algorithm=algo, detector_file=det)

    suffix = f"{bin_size}x{bin_size}"
    ldir = dm.labels_dir(scan)
    ldir.mkdir(parents=True, exist_ok=True)
    _write_json(ldir / f"{algo}_combined_{suffix}.json", result)
    # The combined JSON is itself a feature source (load_features_any reads its
    # "features" list) — the GUIs read it directly; no feature_catalog copy.
    from .core import catalogs
    catalogs.record_catalog(ldir, f"{algo}_combined_{suffix}.json", result["lineage"])
    click.echo(f"\nDone: {result['n_features']} features in "
               f"{len(result['by_bin'])} bins.")


# ─────────────────────────────────────────────────────────────────────
# lineage — show the provenance of result JSONs (peaks/shapes/combined)
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.argument('target', required=False)
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--root', default='.', help='Project root directory')
def lineage(target, scan, root):
    """Show the lineage/provenance of result JSONs.

    With no TARGET, summarizes every peaks/shapes/combined JSON in
    Labels/<scan>/. TARGET may be a path or a file name within that folder.
    """
    import json
    from .core import lineage as L
    dm = DataManager(root, scan=scan)
    ldir = dm.labels_dir(scan)
    if target:
        p = Path(target)
        if not p.exists():
            p = ldir / target            # try as a name inside Labels/<scan>
        paths = [p]
    else:
        paths = (sorted(ldir.glob("*_peaks_*.json"))
                 + sorted(ldir.glob("*_shapes_*.json"))
                 + sorted(ldir.glob("*_combined_*.json")))
    if not paths:
        click.echo(f"No result JSONs found in {ldir}.")
        return
    for p in paths:
        if not p.exists():
            click.echo(f"\n{p}: not found")
            continue
        with open(p) as f:
            data = json.load(f)
        click.echo(f"\n{p.name}")
        lin = data.get("lineage")
        if isinstance(lin, dict):
            for line in L.format_lineage(lin):
                click.echo("  " + line)
        else:
            click.echo("  (no lineage block — legacy file)")
            for k in ("algorithm", "shape_algo", "peak_source", "bin_size", "scan"):
                if k in data:
                    click.echo(f"    {k}: {data[k]}")


# ─────────────────────────────────────────────────────────────────────
# aggregate — fuse per-scan shape catalogs into cross-scan tables
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scan numbers/names (default: all in Labels/)')
@click.option('--bin-size', type=int, default=None, help='Filter to one bin size (default: all)')
@click.option('--out', 'out_dir', default='Study', help='Output directory (default: Study/)')
@click.option('--root', default='.', help='Project root directory')
def aggregate(scans, bin_size, out_dir, root):
    """Aggregate per-scan shape catalogs → features.csv, device_map.csv, study.db.

    Walks Labels/<scan>/ across scans (and bin sizes) via the canonical
    shapes/combined catalog per (scan, bin) and emits two tidy tables plus a
    combined SQLite db — the cross-scan foundation for track/rocking/predict.
    """
    from .core import aggregate as agg
    dm = DataManager(root)
    results_dir = dm.labels_dir_root
    _require(results_dir, "Labels/ directory (run 'xrd-app peaks'/'shapes' first)")

    scan_list = ([DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
                 if scans else None)
    features, device_map = agg.aggregate(
        results_dir, scans=scan_list, bin_size=bin_size, log=click.echo)
    if not features:
        click.echo("No features found — run peaks/shapes first, or check --scans/--bin-size.")
        raise SystemExit(1)

    out = Path(out_dir)
    if not out.is_absolute():
        out = Path(root) / out
    fcsv = agg.write_csv(features, agg.FEATURE_COLUMNS, out / "features.csv")
    dcsv = agg.write_csv(device_map, agg.DEVICEMAP_COLUMNS, out / "device_map.csv")
    db = agg.write_sqlite(out / "study.db", features, device_map)
    click.echo(f"\nWrote {len(features)} features, {len(device_map)} device-map rows:")
    click.echo(f"  {fcsv}\n  {dcsv}\n  {db}")


# ─────────────────────────────────────────────────────────────────────
# track — link shapes across θ into grain tracks
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scan numbers/names (default: all in Labels/)')
@click.option('--bin-size', type=int, default=3, help='Bin size to track (default: 3)')
@click.option('--match-tol', type=float, default=2.0,
              help='Max spatial distance (bins) to call two shapes the same grain across θ')
@click.option('--min-theta', type=int, default=2,
              help='Distinct θ a track needs to be flagged "recurrent" (H1)')
@click.option('--out', 'out_path', default='Study/tracks.json', help='Output tracks JSON')
@click.option('--root', default='.', help='Project root directory')
def track(scans, bin_size, match_tol, min_theta, out_path, root):
    """Link shapes across the θ sweep into grain tracks (Study/tracks.json + .csv).

    Same reflection band + spatial proximity within --match-tol bins (the grid is
    identical across θ, so de-skewed bin coords compare directly). Emits a full
    JSON (per-track θ membership, χ(θ), intensity(θ)) and a one-row-per-track CSV.
    """
    from .core import aggregate as agg, tracking
    dm = DataManager(root)
    results_dir = dm.labels_dir_root
    _require(results_dir, "Labels/ directory (run 'xrd-app peaks'/'shapes' first)")

    scan_list = ([DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
                 if scans else None)
    features, _ = agg.aggregate(results_dir, scans=scan_list, bin_size=bin_size, log=click.echo)
    if not features:
        click.echo("No features to track — run peaks/shapes first.")
        raise SystemExit(1)

    tracks = tracking.build_tracks(
        features, match_tol=match_tol, min_theta=min_theta, log=click.echo)

    out = Path(out_path)
    if not out.is_absolute():
        out = Path(root) / out
    _write_json(out, {
        "bin_size": bin_size, "match_tol": match_tol, "min_theta": min_theta,
        "n_tracks": len(tracks), "tracks": tracks,
    })
    csv_path = out.with_suffix(".csv")
    from .core import aggregate as _agg
    _agg.write_csv(tracking.track_summary_rows(tracks), tracking.TRACK_COLUMNS, csv_path)
    n_rec = sum(1 for t in tracks if t["is_recurrent"])
    click.echo(f"\nWrote {len(tracks)} tracks ({n_rec} recurrent):\n  {out}\n  {csv_path}")


# ─────────────────────────────────────────────────────────────────────
# rocking — fit intensity(θ) per track → θ_Bragg, FWHM (mosaicity)
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--tracks', 'tracks_path', default='Study/tracks.json', help='tracks JSON from `xrd-app track`')
@click.option('--min-points', type=int, default=4, help='Distinct θ needed to attempt a Gaussian fit')
@click.option('--all-tracks', is_flag=True, help='Fit non-recurrent (single/sparse-θ) tracks too')
@click.option('--out', 'out_path', default='Study/rocking_curves.csv', help='Output rocking-curves CSV')
@click.option('--root', default='.', help='Project root directory')
def rocking(tracks_path, min_points, all_tracks, out_path, root):
    """Fit each track's rocking curve (intensity vs θ) → Study/rocking_curves.csv.

    Gaussian in θ: θ_Bragg (peak), FWHM (mosaicity), amplitude, R². Tracks too
    sparsely sampled in θ are emitted with moment descriptors and a 'too_sparse'
    status (the θ sampling is clustered — fits are only meaningful near θ≈3–6°).
    """
    import json
    from .core import rocking as rk, aggregate as agg
    tp = Path(tracks_path)
    if not tp.is_absolute():
        tp = Path(root) / tp
    _require(tp, "tracks JSON (run 'xrd-app track' first)")
    with open(tp) as f:
        tracks = json.load(f).get("tracks", [])

    rows = rk.fit_tracks(tracks, min_points=min_points,
                         only_recurrent=not all_tracks, log=click.echo)
    out = Path(out_path)
    if not out.is_absolute():
        out = Path(root) / out
    agg.write_csv(rows, rk.ROCKING_COLUMNS, out)
    click.echo(f"\nWrote {len(rows)} rocking curves -> {out}")


# ─────────────────────────────────────────────────────────────────────
# predict — forecast per-θ shapes, compare predicted vs observed
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--tracks', 'tracks_path', default='Study/tracks.json', help='tracks JSON from `xrd-app track`')
@click.option('--scans', help='Comma-separated scans (default: all in Labels/)')
@click.option('--bin-size', type=int, default=3, help='Bin size to aggregate features for')
@click.option('--match-tol', type=float, default=2.0, help='Match tolerance (bins) for the repeatability floor')
@click.option('--rocking', 'rocking_path', default='Study/rocking_curves.csv',
              help='Optional rocking_curves.csv to fold fit quality into the report')
@click.option('--repeat-pair', default='203,214', help='Same-orientation scan pair for the noise floor')
@click.option('--out', 'out_path', default='Study/prediction_report.md', help='Output report (.md; .json written alongside)')
@click.option('--root', default='.', help='Project root directory')
def predict(tracks_path, scans, bin_size, match_tol, rocking_path, repeat_pair, out_path, root):
    """Compare predicted (recurrent-track) shapes vs observed → prediction_report.{md,json}.

    Headline metrics: recall (do predicted shapes appear?), precision (are
    detections predicted vs noise?), the 203-vs-214 repeatability floor, χ(θ)
    smoothness, and rocking-fit quality.
    """
    import csv as _csv, json
    from .core import aggregate as agg, prediction as pred
    dm = DataManager(root)

    tp = Path(tracks_path)
    if not tp.is_absolute():
        tp = Path(root) / tp
    _require(tp, "tracks JSON (run 'xrd-app track' first)")
    with open(tp) as f:
        tracks = json.load(f).get("tracks", [])

    scan_list = ([DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
                 if scans else None)
    features, _ = agg.aggregate(dm.labels_dir_root, scans=scan_list, bin_size=bin_size, log=click.echo)

    rocking_rows = None
    rp = Path(rocking_path)
    if not rp.is_absolute():
        rp = Path(root) / rp
    if rp.exists():
        with open(rp) as f:
            rocking_rows = [
                {k: (float(v) if k not in ("reflection", "status") and v not in ("", None) else v)
                 for k, v in row.items()}
                for row in _csv.DictReader(f)]

    pair = tuple(DataManager.scan_name_of(s.strip()) for s in repeat_pair.split(','))
    report = pred.build_report(tracks, features, match_tol=match_tol,
                              repeat_pair=pair, rocking_rows=rocking_rows)

    out = Path(out_path)
    if not out.is_absolute():
        out = Path(root) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred.to_markdown(report))
    _write_json(out.with_suffix(".json"), report)
    click.echo(f"\n{report['verdict']}\n")
    click.echo(f"Wrote:\n  {out}\n  {out.with_suffix('.json')}")


# ─────────────────────────────────────────────────────────────────────
# combined-device — fuse per-θ device maps into one spatial canvas
# ─────────────────────────────────────────────────────────────────────
@main.command(name='combined-device')
@click.option('--device-map', 'device_map_path', default='Study/device_map.csv',
              help='device_map.csv from `xrd-app aggregate`')
@click.option('--tracks', 'tracks_path', default='Study/tracks.json',
              help='Optional tracks JSON for the centroid overlay')
@click.option('--intensity', 'intensity_key', type=click.Choice(['integrated', 'intensity']),
              default='integrated', help='Which column drives the max/argmax canvases')
@click.option('--out', 'out_path', default='Study/combined_device.npz', help='Output .npz')
@click.option('--root', default='.', help='Project root directory')
def combined_device(device_map_path, tracks_path, intensity_key, out_path, root):
    """Fuse all θ into one spatial device-view dataset (Study/combined_device.npz).

    Per (row,col) bin: max intensity over θ, the argmax-θ orientation map, the
    recurrence count, and per-reflection layers — plus track centroids. Pure data
    layer for a future Combined Device View tab (no GUI here).
    """
    import csv as _csv, json
    from .core import combined_device as cd
    dmp = Path(device_map_path)
    if not dmp.is_absolute():
        dmp = Path(root) / dmp
    _require(dmp, "device_map.csv (run 'xrd-app aggregate' first)")
    with open(dmp) as f:
        rows = list(_csv.DictReader(f))

    tracks = None
    tp = Path(tracks_path)
    if not tp.is_absolute():
        tp = Path(root) / tp
    if tp.exists():
        with open(tp) as f:
            tracks = json.load(f).get("tracks", [])

    combined = cd.build_combined(rows, intensity_key=intensity_key,
                                 tracks=tracks, log=click.echo)
    out = Path(out_path)
    if not out.is_absolute():
        out = Path(root) / out
    cd.save_npz(out, combined)
    _write_json(out.with_suffix(".summary.json"), cd.summary(combined))
    click.echo(f"\nWrote combined device view:\n  {out}\n  {out.with_suffix('.summary.json')}")


# ─────────────────────────────────────────────────────────────────────
# qspace — pixel → 3D reciprocal-space (q) mapping
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scan numbers/names (default: config scan)')
@click.option('--bin-size', type=int, default=3, help='Bin size to resolve features to annotate')
@click.option('--energy', type=float, default=None, help='Photon energy in eV (default: 15000)')
@click.option('--pixel-size', type=float, default=None, help='Detector pixel size in metres (default: 75e-6)')
@click.option('--theta', type=float, default=None,
              help='Sample θ (deg) override; default from the rocking θ-table per scan')
@click.option('--poni', 'poni_path', default=None,
              help='pyFAI .poni for tilt-accurate Q directions (else a flat fit of the 2θ map)')
@click.option('--intensity/--no-intensity', default=True,
              help='Store the summed detector image at that θ in the .npz (needed by `xrd-app rsm`)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--out-dir', default='Study/qspace', help='Output directory (default: Study/qspace/)')
@click.option('--root', default='.', help='Project root directory')
def qspace(scans, bin_size, energy, pixel_size, theta, poni_path, intensity, tth_path, out_dir, root):
    """Map detector pixels (+ sample θ) into 3D reciprocal space (q-space).

    For each scan: resolve the detector geometry, build the per-pixel
    Q=(qx,qy,qz) field at that scan's θ, and (if features exist) tag each
    detected feature with its (qx,qy,qz,|Q|). Writes one .npz q-map +
    .summary.json per scan, plus features_q.csv.

    Geometry: with --poni the directions are tilt-accurate (pyFAI); otherwise a
    flat-detector fit of the 2θ map is used (exact |Q|, ~tens-of-mdeg direction
    error from unmodeled tilt — see the fit-RMS printed per scan).

    Needs `pip install 'xrd-app[qspace]'` (xrayutilities); --poni also needs
    `xrd-app[poni]` (pyFAI).
    """
    import tifffile
    from .core import qspace as qs
    from .core import aggregate as agg
    from .core.tracking import theta_of

    energy_ev = energy if energy is not None else qs.DEFAULT_ENERGY_EV
    pixel_m = pixel_size if pixel_size is not None else qs.DEFAULT_PIXEL_M
    lam = qs.wavelength_angstrom(energy_ev)
    if poni_path:
        _require(poni_path, "poni file")

    scan_list = ([DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
                 if scans else None)
    if not scan_list:
        one = DataManager(root).scan_name
        if not one:
            click.echo("Error: no scan given. Use --scans or set a config scan.")
            raise SystemExit(1)
        scan_list = [one]

    out_base = Path(out_dir)
    if not out_base.is_absolute():
        out_base = Path(root) / out_base

    src = "poni (tilt-accurate)" if poni_path else "flat fit of 2θ map"
    click.echo(f"[qspace] E={energy_ev/1000:.1f} keV  λ={lam:.5f} Å  "
               f"pixel={pixel_m*1e6:.0f} µm  geometry: {src}")
    for scan in scan_list:
        dm = DataManager(root, scan=scan)
        tth = dm.tth_map(tth_path)
        tth_deg = tifffile.imread(str(tth)).astype('float64') if Path(tth).exists() else None

        th = theta if theta is not None else theta_of(scan)
        if th is None:
            click.echo(f"  {scan}: θ unknown (not in table; pass --theta) — using 0.0")
            th = 0.0

        if poni_path:
            geom = qs.geometry_from_poni(poni_path)
            shape = tth_deg.shape if tth_deg is not None else None
            qx, qy, qz = qs.q_vectors_from_poni(poni_path, energy_ev=energy_ev,
                                                theta_deg=th, shape=shape)
        else:
            if tth_deg is None:
                _require(tth, f"2θ map for {scan}")  # aborts with guidance
            geom = qs.recover_geometry(tth_deg, pixel_m=pixel_m)
            qx, qy, qz = qs.q_vectors(tth_deg, geom, energy_ev=energy_ev, theta_deg=th)
        qmag = (qx ** 2 + qy ** 2 + qz ** 2) ** 0.5

        # annotate detected features for this scan, if any exist
        features, _ = agg.aggregate(dm.labels_dir_root, scans=[scan],
                                    bin_size=bin_size, log=lambda *_: None)
        tagged = qs.annotate_features(features, qx, qy, qz)

        meta = {"scan": scan, "theta_deg": float(th),
                "energy_ev": float(energy_ev), "wavelength_A": float(lam)}
        has_intensity = False
        if intensity:
            from .core import io
            try:
                src = io.open_bin_source(dm, bin_size, scan=scan)
                try:
                    meta["intensity"] = src.sum_all().astype('float32')
                    has_intensity = True
                finally:
                    src.close()
            except Exception as e:
                click.echo(f"  {scan}: no intensity (skipping): {e}")

        npz = out_base / f"{scan}_qmap.npz"
        qs.save_qmap(npz, qx, qy, qz, geom, meta=meta)
        _write_json(npz.with_suffix(".summary.json"),
                    qs.summary(geom, th, energy_ev, lam, qmag, n_features=len(tagged)))
        if tagged:
            cols = list(tagged[0].keys())
            agg.write_csv(tagged, cols, out_base / f"{scan}_features_q.csv")

        geo_note = "poni" if geom.source == "poni" else f"fit-RMS={geom.rms_deg*1000:.0f} mdeg"
        click.echo(
            f"  {scan}: θ={th:>5.1f}°  D={geom.distance_m:.4f} m  {geo_note}  "
            f"|Q|={qmag.min():.3f}–{qmag.max():.3f} 1/Å  "
            f"features={len(tagged)}{'  +I' if has_intensity else ''} -> {npz.name}")

    click.echo(f"\nWrote q-maps to {out_base}")


# ─────────────────────────────────────────────────────────────────────
# rsm — fuse per-scan q-maps into one binned 3D reciprocal-space map
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scans (default: all *_qmap.npz in --in-dir)')
@click.option('--in-dir', default='Study/qspace', help='Dir of <scan>_qmap.npz (from `xrd-app qspace`)')
@click.option('--bins', 'nbins', type=int, default=128, help='Voxels per axis in the 3D grid')
@click.option('--min-intensity', type=float, default=0.0,
              help='Drop detector pixels at/below this (after median subtraction)')
@click.option('--subtract-median/--no-subtract-median', default=True,
              help='Baseline-subtract each scan by its median before binning')
@click.option('--out', 'out_path', default='Study/rsm.npz', help='Output .npz')
@click.option('--root', default='.', help='Project root directory')
def rsm(scans, in_dir, nbins, min_intensity, subtract_median, out_path, root):
    """Fuse per-scan q-maps into one binned 3D reciprocal-space map (RSM).

    Reads the ``<scan>_qmap.npz`` files from `xrd-app qspace` (which must carry an
    intensity layer — run qspace with --intensity), histograms each scan's summed
    detector intensity into a shared 3D (qx,qy,qz) grid, and accumulates across θ.
    Writes the volume + per-voxel counts + max-projections to one .npz.
    """
    import numpy as np
    from .core import rsm as R

    base = Path(in_dir)
    if not base.is_absolute():
        base = Path(root) / base
    if scans:
        names = [DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
        files = [base / f"{n}_qmap.npz" for n in names]
    else:
        files = sorted(base.glob("*_qmap.npz"))
    files = [f for f in files if Path(f).exists()]
    if not files:
        click.echo(f"Error: no *_qmap.npz in {base}. Run 'xrd-app qspace' first.")
        raise SystemExit(1)

    qmaps = [R.load_qmap(f) for f in files]
    have = [m for m in qmaps if m.intensity is not None]
    if not have:
        click.echo("Error: no q-map has an intensity layer. "
                   "Re-run 'xrd-app qspace' with --intensity.")
        raise SystemExit(1)
    if len(have) < len(qmaps):
        click.echo(f"Note: {len(qmaps) - len(have)} scan(s) lack intensity; skipped.")

    click.echo(f"[rsm] fusing {len(have)} scans into a {nbins}³ grid "
               f"(median-subtract={subtract_median}) …")
    edges = R.common_grid(have, nbins=nbins)
    volume, counts = R.accumulate(have, edges, min_intensity=min_intensity,
                                  subtract_median=subtract_median,
                                  progress=_make_progress("rsm"))
    out = Path(out_path)
    if not out.is_absolute():
        out = Path(root) / out
    scan_names = [m.scan for m in have]
    thetas = [m.theta_deg for m in have]
    meta = {"scans": np.array(scan_names),
            "thetas": np.array([np.nan if t is None else t for t in thetas], float)}
    R.save_npz(out, volume, counts, edges, meta=meta)
    s = R.summary(volume, counts, edges, scan_names, thetas)
    _write_json(out.with_suffix(".summary.json"), s)
    qr = s["q_ranges"]
    click.echo(f"\nRSM {tuple(volume.shape)}  nonzero={s['nonzero_voxels']} "
               f"({100*s['fill_fraction']:.1f}%)  ΣI={s['total_intensity']:.3g}  "
               f"peak={s['peak_voxel_intensity']:.3g}")
    click.echo(f"  qx {qr['qx'][0]:.2f}..{qr['qx'][1]:.2f}  "
               f"qy {qr['qy'][0]:.2f}..{qr['qy'][1]:.2f}  "
               f"qz {qr['qz'][0]:.2f}..{qr['qz'][1]:.2f} 1/Å")
    click.echo(f"Wrote:\n  {out}\n  {out.with_suffix('.summary.json')}")


def _resolve_scan_list(scans, all_scans, root):
    if scans:
        return [DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
    if all_scans:
        return DataManager(root).discover_scans(usable_only=True)
    return []


def _parse_shape_cols(shape):
    """Parse --shape 'ROWSxCOLS' or 'COLS' into the column count (or None)."""
    if not shape:
        return None
    s = str(shape).lower().replace('×', 'x')
    return int(s.split('x')[-1])


if __name__ == "__main__":
    main()

