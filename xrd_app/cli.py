"""xrd-app command-line interface.

The CLI is the engine: every "big button" in the GUI shells out to one of these
commands, and everything is usable headless. See ``README.md`` and
``xrd_app/PATHWAYS.md`` for the supported workflows.
"""

import os
import shutil
from pathlib import Path

import click

from . import __version__
from .config import ProjectConfig, DataManager, default_config, safe_component


@click.group(context_settings={"show_default": True})
@click.version_option(__version__, prog_name="xrd-app")
def main():
    """XRD App: reproducible nano-XRD analysis from setup through studies.

    Run ``xrd-app COMMAND --help`` for command-specific options. The usual
    single-scan workflow is ``init``, ``scan-detect``, ``link``, ``make-bins``,
    then ``run-pipeline``. See README.md for categorized workflows.
    """
    pass


# ─────────────────────────────────────────────────────────────────────
# init
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--name', 'project_name', required=True, help='Name of the project')
@click.option('--scan-number', type=click.IntRange(min=0), default=None,
              help='Scan number (e.g. 203 -> Scan_0203)')
@click.option('--root', type=click.Path(path_type=Path), default=Path('.'),
              help='Root directory for the project')
def init(project_name, scan_number, root):
    """Initialize a new XRD project and its standard directory tree.

    ROOT may be an existing empty directory, but this command never overwrites
    an existing config.yaml. This makes it safe to use in scripts.
    """
    root = root.resolve()
    config_path = root / "config.yaml"
    if config_path.exists():
        raise click.ClickException(
            f"Project already exists at {root}; refusing to overwrite {config_path}.")
    cfg = ProjectConfig(root, data=default_config(project_name, root, scan_number))
    cfg.create_tree()
    cfg.save()

    # Seed an editable default reflection set so the project resolves reflections
    # from its own Metadata/ (not the hidden bundled fallback) out of the box.
    from .core import reflections as refl_io
    mdir = DataManager(config=cfg).metadata_dir
    refl_io.save(refl_io.default_reflections(), mdir / "reflections.json")

    click.echo(f"Project '{project_name}' initialized at {cfg.root}")
    click.echo(f"  Reflections: {mdir / 'reflections.json'} "
               "(default perovskite set — edit in Setup → Reflections)")
    if scan_number is not None:
        click.echo(f"  Scan: Scan_{scan_number:04d}")
    click.echo(f"  Config: {cfg.config_path}")
    click.echo("  Next: 'xrd-app scan-detect --scans-dir <dir>' to register your scans,")
    click.echo("        then 'xrd-app link --tth <tiff> --reflections <json>'.")


@main.command(name='whole-frame-reflections')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--project', is_flag=True, default=False,
              help='Write the project-wide default (Metadata/) instead of per-scan.')
@click.option('--root', default='.', help='Project root directory')
def whole_frame_reflections(scan, project, root):
    """Write one unlimited-width "(no reflections)" reflection.

    Detectors recognize this reserved reflection as a detector-spanning band, so
    peaks/shape/territory search the whole frame. For datasets with no known Bragg
    reflections. Consider ROI → Shape for that workflow too.
    """
    from .core import reflections as refl_io
    dm = DataManager(root, scan=scan)
    refls = refl_io.whole_frame_reflections()
    mdir = dm.metadata_dir if project else dm.metadata_scan_dir(scan)
    out = refl_io.save(refls, mdir / "reflections.json")
    scope = "project default" if project else f"scan {dm.scan_name}"
    click.echo(f"[whole-frame-reflections] wrote one unlimited-width reflection ({scope}) → {out}")
    click.echo(f"  label: {refl_io.WHOLE_FRAME_LABEL} — detector will search the whole frame")


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
@click.option('--reflections', help='Path to reflections.json')
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
        raise click.ClickException("No config.yaml found. Run 'xrd-app init' first.")
    cfg.data.setdefault('data_sources', {})
    metadata_dir = DataManager(config=cfg).metadata_dir

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
            # Preserve the source kind: a Lozano position h5 must land as
            # positions.h5 (loaders dispatch on suffix); everything else → .csv.
            dest_name = ("positions.h5"
                         if src.suffix.lower() in (".h5", ".hdf5")
                         else "positions.csv")
            # Drop a stale sibling of the other kind so the resolver isn't torn
            # between positions.h5 and positions.csv.
            other = dest_dir / ("positions.csv" if dest_name == "positions.h5"
                                else "positions.h5")
            if other.exists() or other.is_symlink():
                other.unlink()
            stored = _place(src, dest_dir / dest_name, copy)
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
        dest_dir = metadata_dir if sub_dir == 'Metadata' else cfg.root / sub_dir
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


def _require_scan_no(dm) -> int:
    """Resolve the scan number to operate on, or hard-fail with a clear message.

    Replaces the old ``dm.scan_number() or 203`` silent fallback: in a
    multi-scan project ``scan.number`` is null, so an unresolved scan would
    quietly glob ``scan_0203_*.h5`` in the wrong directory and find nothing.
    Better to tell the user to pass ``--scan`` than to operate on the wrong one.
    """
    no = dm.scan_number()
    if no is None:
        raise click.ClickException(
            "Could not determine which scan to use. Pass --scan <number/name> "
            "(this project has no global scan.number set).")
    return no


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
    click.echo(f"  {'bin':>4}  {'f2':>7}  {'f1':>7}  {'src':>8}  name")
    click.echo(f"  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*30}")
    for d in sorted(entries, key=lambda d: (
            d.get('bin_size') or '',
            -(d.get('holdout_f2') if d.get('holdout_f2') is not None else -1),
            -(d.get('holdout_f1') if d.get('holdout_f1') is not None else -1))):
        f1 = f"{d['holdout_f1']:.4f}" if d.get('holdout_f1') is not None else "—"
        f2 = f"{d['holdout_f2']:.4f}" if d.get('holdout_f2') is not None else "—"
        bin_lbl = d.get('bin_size') or 'any'
        click.echo(f"  {bin_lbl:>4}  {f2:>7}  {f1:>7}  "
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
        noise_reduction=noise_reduction, name=name, kind=kind, source="manual",
        project_root=root)
    click.echo(f"Saved algorithm -> {out}")
    click.echo(f"Run it with: xrd-app peaks --bin-size {bin_size} --algorithm {out.stem}")


# ─────────────────────────────────────────────────────────────────────
# convert-poni — pyFAI .poni → tth.tiff
# ─────────────────────────────────────────────────────────────────────
@main.command(name='convert-poni')
@click.option('--poni', required=True, help='Path to a pyFAI .poni calibration file')
@click.option('--shape', default=None, help='ROWSxCOLS (default: config detector.shape, else from .poni)')
@click.option('--output', default=None,
              help='Output tth.tiff (default: Metadata[/<scan>]/tth.tiff)')
@click.option('--scan', default=None, help='Scan number/name for a per-scan tth map')
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

    out = Path(output) if output else (
        dm.metadata_scan_dir(scan) / "tth.tiff" if scan else dm.metadata_dir / "tth.tiff")
    try:
        geometry.convert_poni_file(poni, out, sh)
    except ImportError as e:
        raise click.ClickException(str(e))
    if not scan:
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
        ("unbinned_archive", dm.unbinned_archive_h5()),
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
        _require(src_path, "peaks catalog")
        ann, empty = H.bins_from_peaks(str(src_path))
    else:  # shapes
        if not algorithm:
            click.echo("Error: --algorithm <name> required for --source shapes.")
            raise SystemExit(1)
        src_path = dm.shapes_json(algorithm, bin_size, scan)
        _require(src_path, "shapes catalog")
        ann, empty = H.bins_from_shapes(str(src_path))

    dest_dir = Path(dest) if dest else dm.cvevolve_dir
    grid = dm.grid_mapping(bin_size=bin_size, scan=scan)
    try:
        counts = H.build_split(
            ann, empty, holdout_pct=holdout_pct, seed=seed,
            dest_dev=dest_dir / "test_data", dest_holdout=dest_dir / "holdout_data",
            grid_mapping=grid if Path(grid).exists() else None)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    click.echo(f"[holdout] source={source} kind={'peak' if source!='shapes' else 'shape'}")
    click.echo(f"[holdout] ({counts['holdout_bins']}/{counts['total_bins']}) bins → holdout, "
               f"{counts['dev_bins']} → dev  ({counts['total_points']} points, seed={seed})")
    click.echo(f"[holdout] wrote {dest_dir}/test_data and {dest_dir}/holdout_data")


# ─────────────────────────────────────────────────────────────────────
# cvevolve-init — scaffold a CVEvolve project from bundled defaults
# ─────────────────────────────────────────────────────────────────────
@main.command(name='cvevolve-init')
@click.option('--name', default=None, help='Session name (default: project CVEvolve dir name)')
@click.option('--force', is_flag=True, help='Overwrite existing config/prompt files')
@click.option('--root', default='.', help='Project root directory')
def cvevolve_init(name, force, root):
    """Create default config.yaml + prompt.md + holdout_test_prompt.md.

    Beamline users won't arrive with these files; this seeds the project's
    ``CVEvolve/`` dir with sane defaults (paths filled in) that they can then
    tweak — the metric description in config.yaml and the prompt .md are the two
    places to change "what the model is looking for".
    """
    from .core import cvevolve_setup as CV
    dm = DataManager(root)
    dest = dm.cvevolve_dir
    session_name = name or dest.name or "cvevolve"
    res = CV.scaffold_project(dest, session_name, force=force)
    for p in res["written"]:
        click.echo(f"[cvevolve-init] wrote {p}")
    for p in res["skipped"]:
        click.echo(f"[cvevolve-init] skipped (exists) {p}  — use --force to overwrite")
    click.echo(f"[cvevolve-init] project ready at {dest}  (name={session_name})")


# ─────────────────────────────────────────────────────────────────────
# register-cvevolve — import a completed winner into the project library
# ─────────────────────────────────────────────────────────────────────
@main.command(name='register-cvevolve')
@click.option('--config', 'config_path', required=True, help='CVEvolve config.yaml')
@click.option('--name', default=None, help='Registered algorithm name (default: candidate name)')
@click.option('--bin-size', type=int, default=None, help='Bin size this winner was trained on')
@click.option('--root', default='.', help='Project root directory')
def register_cvevolve(config_path, name, bin_size, root):
    """Validate and register a completed CVEvolve winner in Algorithms/."""
    from .core import cvevolve_results
    try:
        result = cvevolve_results.register_winner(
            config_path, root, name=name, bin_size=bin_size)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"[CVEvolve] registered winner -> {result['path']}")


# ─────────────────────────────────────────────────────────────────────
# run-cvevolve — wrapper around the CVEvolve algorithm search
# ─────────────────────────────────────────────────────────────────────
@main.command(name='run-cvevolve')
@click.option('--config', 'config_path', required=True, help='CVEvolve config.yaml')
@click.option('--prompt', 'prompt_path', default=None, help='CVEvolve task prompt .md (default: prompt.md beside config)')
@click.option('--holdout-test-prompt', 'holdout_prompt_path', default=None,
              help='Holdout prompt .md (default: holdout_test_prompt.md beside config)')
@click.option('--engine', type=click.Choice(['local', 'podman', 'docker']), default='podman')
@click.option('--cvevolve-dir', default=None, help='Path to the CVEvolve checkout')
@click.option('--image', default='cvevolve', help='Container image tag')
@click.option('--build', is_flag=True, help='Build the image from --cvevolve-dir first')
@click.option('--mount', 'mounts', multiple=True, help='Host dir to mount at the same path (repeatable)')
@click.option('--env', 'envs', multiple=True, default=('ARGO_API_KEY',), help='Env var to pass through')
@click.option('--hutch/--no-hutch', default=False,
              help='Enable Hutch (SQLite) tracking before launching; prints HUTCH_DB <path>')
@click.option('--root', default='.', help='Project root directory')
def run_cvevolve(config_path, prompt_path, holdout_prompt_path, engine, cvevolve_dir,
                 image, build, mounts, envs, hutch, root):
    """Run CVEvolve with the given config (Podman by default — LLM-generated code)."""
    import subprocess
    import sys
    config_path = Path(config_path).resolve()
    _require(config_path, "CVEvolve config")
    if hutch:
        from .core import cvevolve_setup as CV
        db_path = CV.set_hutch(config_path, True)
        # Emit a machine-readable marker so the GUI can point its live SQL view
        # at the same DB without re-parsing the config.
        click.echo(f"HUTCH_DB {db_path}")
    inner = ["cvevolve", "run", "--config", str(config_path)]
    if prompt_path is None:
        default_prompt = config_path.parent / "prompt.md"
        prompt_path = default_prompt if default_prompt.exists() else None
    if prompt_path:
        prompt_path = Path(prompt_path).resolve()
        _require(prompt_path, "CVEvolve prompt")
        inner += ["--prompt", str(prompt_path)]
    if holdout_prompt_path is None:
        default_holdout_prompt = config_path.parent / "holdout_test_prompt.md"
        holdout_prompt_path = default_holdout_prompt if default_holdout_prompt.exists() else None
    if holdout_prompt_path:
        holdout_prompt_path = Path(holdout_prompt_path).resolve()
        _require(holdout_prompt_path, "CVEvolve holdout test prompt")
        inner += ["--holdout-test-prompt", str(holdout_prompt_path)]

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

    from .core import cvevolve_setup as CV
    mount_dirs = [Path(m).resolve() for m in mounts] or [Path(root).resolve()]
    referenced_dirs = CV.config_mount_dirs(config_path)
    for path in (prompt_path, holdout_prompt_path):
        if path:
            referenced_dirs.append(path.parent)
    for directory in referenced_dirs:
        if not any(directory.is_relative_to(d) for d in mount_dirs):
            mount_dirs.append(directory)
    run_cmd = [engine, "run", "--rm", "-i"]
    for name in envs:
        run_cmd += ["-e", name]
    for d in dict.fromkeys(mount_dirs):
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
@click.option('--bin-size', type=int, default=None,
              help='Initial bin size (default: restore last-used, otherwise 3)')
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


@main.command(name='roifeature')
@click.option('--source', default=None,
              help='Existing project folder or raw Scan_NNNN folder (otherwise choose in GUI)')
@click.option('--bin-size', type=int, default=3, help='Spatial bin size for feature finding')
def roi_feature_gui(source, bin_size):
    """Open the focused ROI-feature GUI, optionally directly from a raw scan."""
    from .gui.roifeature import launch
    raise SystemExit(launch(source=source, bin_size=bin_size))


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
@click.option('--scans', help='Comma-separated scan numbers/names to keep from --scans-dir')
@click.option('--deep', is_flag=True, help='Open every file (exact counts, catches corrupt files). Slow on WSL/OneDrive.')
@click.option('--root', default='.', help='Project root directory')
def scan_detect(scans_dir, scan_file, scans, deep, root):
    """Discover scans from a file or directory, validate them, write Raw/scans.json.

    Fast by default: samples the first file per scan (frame count is then an
    estimate). Use --deep for exact counts and full corruption checks.
    """
    from .core import io
    if not scans_dir and not scan_file:
        raise click.UsageError("Provide --scans-dir <dir> or --scan-file <hdf5>.")
    if scans_dir and scan_file:
        raise click.UsageError("Use only one of --scans-dir or --scan-file.")

    cfg = ProjectConfig.load(root)
    if not cfg.exists():
        raise click.ClickException("No config.yaml found. Run 'xrd-app init' first.")
    dm = DataManager(root, cfg)

    target = scan_file or scans_dir
    if not Path(target).exists():
        raise click.BadParameter(f"path does not exist: {target}",
                                 param_hint='--scan-file/--scans-dir')

    found = io.discover_scans(target, deep=deep)
    if scans:
        requested = {DataManager.scan_name_of(s) or str(s).strip()
                     for s in scans.split(',') if str(s).strip()}
        found = [s for s in found if s.get('name') in requested]
    if not found:
        suffix = f" matching --scans {scans}" if scans else ""
        click.echo(f"No scans{suffix} found under {target}.")
        raise SystemExit(1)

    # Adopt the first valid scan's frame shape as the project detector shape.
    proj_shape = cfg.get('detector', 'shape')
    if not proj_shape:
        for s in found:
            if s.get('shape'):
                proj_shape = s['shape']
                break

    registry = dm.scans_registry()
    valid = []
    click.echo(f"{len(found)} scan(s) detected under {target}:\n")
    for s in found:
        problems = io.validate_scan(s, expected_shape=proj_shape)
        mark = "✓" if not problems else "⚠"
        if not problems:
            valid.append(s)
        approx = "~" if s.get('frames_estimated') else ""
        click.echo(f"  [{mark}] {s['name']}  ({s['n_files']} files / "
                   f"{approx}{s['n_frames']} frames, shape={s['shape']})")
        for p in problems:
            click.echo(f"         - {p}", err=True)
    for s in valid:
        registry[s['name']] = {k: s[k] for k in
                               ('dir', 'frames_dir', 'n_files', 'n_frames', 'shape')}

    dm.write_scans_registry(registry)
    cfg.data.setdefault('detector', {})['shape'] = proj_shape
    cfg.data['scans'] = registry
    # If the project has no configured scan and exactly one valid scan was found,
    # adopt it. Invalid scans are reported but never become processing inputs.
    if not cfg.get('scan', 'name') and len(valid) == 1:
        name = valid[0]['name']
        cfg.data['scan'] = {'number': DataManager.scan_number_of(name), 'name': name}
    cfg.save()

    click.echo(f"\n{len(valid)}/{len(found)} OK. Frame shape: {proj_shape}.")
    click.echo(f"Registry: {dm.scans_registry_path()}")
    if len(valid) != len(found):
        raise click.ClickException(
            f"Rejected {len(found) - len(valid)} invalid scan(s); valid scans were registered.")


# ─────────────────────────────────────────────────────────────────────
# grid — assign raw frames to a spatial bin grid
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--shape', default=None, help='Synthesize a grid with no positions: ROWSxCOLS or COLS')
@click.option('--xrd-dir', help='Directory of raw per-frame H5 files (defaults to resolved)')
@click.option('--positions', help='Scan position CSV (defaults to resolved)')
@click.option('--deskew-method',
              type=click.Choice(['auto', 'positions_xy', 'faithful', 'faithful_native',
                                 'commanded']),
              default='auto',
              help='How frames map to the lattice, from the real (X, Y) CSV. '
                   'auto (DEFAULT): positions_xy at 1x1 (both axes snapped to true '
                   '(X, Y) — skew-free, avoids the file-index-row bowtie/"X"), '
                   'faithful at >=2x2 (already clean there). Override with: '
                   'positions_xy (force the true-(X,Y) grid at any bin), faithful '
                   '(snap columns to true Y on a square-pixel, file-index-row '
                    'lattice), faithful_native (faithful at native frame density), or '
                    'commanded (align columns by rank). For zero-collision irregular '
                    "cells use 'xrd-app territory-grid'.")
@click.option('--variant', default=None,
              help='Tag appended to default output names (e.g. "faithful") so a '
                   'coordinate variant sits alongside the default instead of overwriting it.')
@click.option('--output', help='Output grid mapping HDF5 (defaults to per-scan Metadata dir)')
@click.option('--root', default='.', help='Project root directory')
def grid(bin_size, scan, shape, xrd_dir, positions, deskew_method, variant, output, root):
    """Generate an HDF5 grid mapping assigning raw frames to a spatial bin grid.

    The grid is built from the **real (X, Y) coordinate CSV** — required. When
    no real CSV exists one is created automatically from the **real SOCKETSERVER
    interferometry** stream (see 'xrd-app create-positions'). If neither a real
    CSV nor a SOCKETSERVER stream is available this command **hard-fails** rather
    than silently reconstructing the grid from the one-file-per-row layout (that
    silent fallback is what skewed rocking 203-214). The chosen ``deskew_method``
    (default ``auto``) and the positions provenance (``positions_csv`` /
    ``positions_real``) are recorded in the output mapping, and ``bin`` refuses any
    mapping whose ``positions_real`` is not true.

    The only opt-in bypass is ``--shape ROWSxCOLS``, which synthesizes a raster
    with no positions; its output is flagged ``positions_real=false`` and cannot
    be binned.
    """
    from .core import io
    dm = DataManager(root, scan=scan)
    scan_no = _require_scan_no(dm)
    xdir = Path(xrd_dir) if xrd_dir else dm.xrd_frames_dir()
    pos = Path(positions) if positions else dm.position_csv()
    out = Path(output) if output else dm.grid_mapping(bin_size=bin_size, variant=variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = dm.unbinned_archive_h5(scan=scan)
    if not archive.exists() and not io.has_raw_frames(xdir, scan_no):
        click.echo(f"Error: no unbinned archive ({archive}) or raw frame files "
                   f"(scan_{scan_no:04d}_*.h5) in {xdir}.")
        click.echo("  Build the archive while raw data is connected, or check the scan source.")
        raise SystemExit(1)

    n_cols = _parse_shape_cols(shape)
    pos_real = ((Path(pos).exists() and not io.is_recreated_csv(pos)) or
                (archive.exists() and io.archive_has_real_positions(archive)))

    # The real coordinate CSV is REQUIRED. When we don't have one (and weren't
    # asked to synthesize a raster shape), build it from the REAL stage positions
    # in the SOCKETSERVER interferometry stream. If there is no interferometry
    # stream — or the build fails — we HARD-FAIL instead of silently
    # reconstructing the grid from the one-file-per-row layout: that silent
    # fallback is exactly what skewed rocking 203-214.
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
                click.echo(f"Error: could not build real positions from the "
                           f"SOCKETSERVER stream ({e}).")
                raise SystemExit(1)
        else:
            click.echo("Error: no real coordinate CSV and no SOCKETSERVER "
                       f"interferometry at {sdir}.")
            click.echo("  The grid must be built from true (X, Y) positions. "
                       "Provide a real --positions CSV, run 'xrd-app "
                       "create-positions', or pass --shape ROWSxCOLS to "
                       "synthesize an (unbinnable) raster.")
            raise SystemExit(1)

    io.generate_grid_mapping(xdir, pos if pos_real else None, bin_size,
                             scan_number=scan_no, output=out, n_cols=n_cols,
                             deskew=True, deskew_method=deskew_method,
                             log=click.echo, archive=archive if archive.exists() else None)
    click.echo(f"Wrote grid_mapping -> {out}")


# ─────────────────────────────────────────────────────────────────────
# territory-grid — skew-free reference binning by true (X, Y) territories
# ─────────────────────────────────────────────────────────────────────
@main.command(name='territory-grid')
@click.option('--target-size', type=int, default=1,
              help='Frames per territory before it stops growing (sweepable; '
                   'small ≈ 1×1 resolution, large = higher per-cell SNR).')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--xrd-dir', help='Directory of raw per-frame H5 files (defaults to resolved)')
@click.option('--positions', help='Real scan position CSV (defaults to resolved)')
@click.option('--variant', default='territory',
              help='Tag for the output names so the territorial mapping sits '
                   'alongside the grid ones (default "territory").')
@click.option('--output', help='Output grid mapping HDF5 (defaults to per-scan Metadata dir)')
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
    scan_no = _require_scan_no(dm)
    xdir = Path(xrd_dir) if xrd_dir else dm.xrd_frames_dir()
    pos = Path(positions) if positions else dm.position_csv()
    out = Path(output) if output else dm.grid_mapping(bin_size=1, variant=variant)
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = dm.unbinned_archive_h5(scan=scan)
    if not archive.exists() and not io.has_raw_frames(xdir, scan_no):
        click.echo(f"Error: no unbinned archive ({archive}) or raw frame files "
                   f"(scan_{scan_no:04d}_*.h5) in {xdir}.")
        raise SystemExit(1)

    try:
        territory.build_territory_mapping(
            xdir, pos, target_size=target_size, scan_number=scan_no,
            output=out, log=click.echo,
            archive=archive if archive.exists() else None)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    click.echo(f"Wrote territorial grid_mapping -> {out}")


# ─────────────────────────────────────────────────────────────────────
# territory-build — the whole skew-free territorial reference in one command
# ─────────────────────────────────────────────────────────────────────
@main.command(name='territory-build')
@click.option('--target-size', type=int, default=1,
              help='Frames per territory before it stops growing (see territory-grid).')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', default=None, help='Peak detector path OR bundled name')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--compression', type=click.Choice(['zstd', 'gzip', 'lz4', 'none']), default='zstd')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def territory_build(ctx, target_size, scan, algorithm, snr, compression, root):
    """Build the whole skew-free territorial reference in one command.

    Chains territory-grid → bin → peaks → shapes (all ``--variant territory`` at
    1×1, with coordinate linking) so the Territory Map and the Device-View
    "Territorial reference available →" button can be produced without running
    the four steps by hand. This is the one-button version of the TERRITORY.md
    chain — the GUI's "Build territorial reference" button shells out to it.
    """
    dm = DataManager(root, scan=scan)
    # The peak set name shapes will pick up (same rule peaks/batch use).
    algo = algorithm or Path(dm.detector_script(algorithm, bin_size=1)).stem
    ctx.invoke(archive_unbinned, scan=scan, compression=compression, root=root)
    ctx.invoke(territory_grid, target_size=target_size, scan=scan, root=root)
    ctx.invoke(bin, bin_size=1, scan=scan, variant='territory',
               compression=compression, root=root)
    ctx.invoke(peaks, bin_size=1, scan=scan, algorithm=algorithm, snr=snr,
               variant='territory', root=root)
    ctx.invoke(shapes, bin_size=1, scan=scan, algorithm='territory',
               variant='territory', peak_algo=algo, root=root)
    click.echo("\nTerritorial reference complete: "
               "territory-grid → bin → peaks → shapes (--variant territory)")


# ─────────────────────────────────────────────────────────────────────
# create-positions — build a REAL position CSV from SOCKETSERVER interferometry
# ─────────────────────────────────────────────────────────────────────
@main.command(name='create-positions')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--socket-dir', help='SOCKETSERVER interferometry dir (defaults to resolved)')
@click.option('--reduction', type=int, default=1,
              help='Use every Nth interferometer sample (speed; 1 = all)')
@click.option('--from-h5', 'from_h5', default=None,
              help='Build from a Lozano position h5 (entry/data/Position) instead '
                   'of SOCKETSERVER. Pass a path, or omit for auto-discovery.')
@click.option('--output', help='Output CSV (default: Metadata/<scan>/positions.csv)')
@click.option('--force', is_flag=True, help='Overwrite an existing CSV')
@click.option('--root', default='.', help='Project root directory')
def create_positions(scan, socket_dir, reduction, from_h5, output, force, root):
    """Build a REAL per-frame position CSV from the beamline's stage positions.

    Two sources, tried in order (or forced): the **SOCKETSERVER interferometry
    stream** (reduced to one true (X, Y) per trigger), or a **Lozano position h5**
    (``Scan_NNNN.h5`` with an already-reduced ``entry/data/Position`` group). Both
    write ``Metadata/<scan>/positions.csv`` — the *real* measured positions.
    'xrd-app grid' calls this automatically when no position file is found; run it
    directly to (re)generate one. When neither source exists, the grid is
    reconstructed from the one-file-per-row layout instead (no file needed).
    """
    from .core import io, positions as P
    dm = DataManager(root, scan=scan)
    scan_no = _require_scan_no(dm)
    sdir = Path(socket_dir) if socket_dir else dm.socketserver_dir(scan=scan)
    out = Path(output) if output else (dm.metadata_scan_dir(scan) / "positions.csv")

    if out.exists() and not force:
        kind = "recreated" if io.is_recreated_csv(out) else "existing (real?)"
        click.echo(f"Refusing to overwrite {kind} CSV: {out}")
        click.echo("  Pass --force to overwrite, or --output to write elsewhere.")
        raise SystemExit(1)

    # ---- source selection --------------------------------------------------
    # Explicit --from-h5 (path or auto), else SOCKETSERVER if present, else a
    # Lozano position h5 discovered by scan number.
    lozano_h5 = None
    if from_h5:
        lozano_h5 = Path(from_h5)
        if not lozano_h5.is_file():
            click.echo(f"Error: --from-h5 file not found: {lozano_h5}")
            raise SystemExit(1)
    elif not P.has_socketserver(sdir, scan_no):
        raw = dm.raw_scan_dir(scan=scan)
        search_dirs = [raw, raw / "data", raw.parent, raw.parent / "data",
                       dm.xrd_frames_dir(scan=scan), dm.root, dm.root / "data"]
        lozano_h5 = P.find_position_h5(search_dirs, scan_no)
        if lozano_h5 is None:
            click.echo(f"Error: no SOCKETSERVER files (scan_{scan_no:04d}_*.h5) in "
                       f"{sdir}, and no Lozano position h5 (Scan_{scan_no:04d}.h5 "
                       f"with {io.H5_POSITION_GROUP}) found nearby.")
            click.echo("  Pass --from-h5 <path>, or let 'xrd-app grid' reconstruct "
                       "the grid from the one-file-per-row layout instead.")
            raise SystemExit(1)

    try:
        if lozano_h5 is not None:
            info = P.build_positions_csv_from_h5(lozano_h5, out, log=click.echo)
        else:
            info = P.build_positions_csv(
                sdir, out, scan_number=scan_no, reduction=reduction, log=click.echo)
    except (FileNotFoundError, ValueError, KeyError) as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    click.echo(f"Wrote {info['n_positions']} real positions "
               f"(span {info['x_span_um']:.1f} x {info['y_span_um']:.1f} um) "
               f"-> {info['path']}")


# ─────────────────────────────────────────────────────────────────────
# archive-unbinned — lossless acquisition-order detector archive
# ─────────────────────────────────────────────────────────────────────
@main.command(name='archive-unbinned')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--xrd-dir', help='Directory of raw per-frame H5 files (defaults to resolved)')
@click.option('--positions', help='Per-frame position CSV/H5 embedded as x/y metadata')
@click.option('--output', help='Output HDF5 (default: Binned/<scan>/xrd_unbinned_archive.h5)')
@click.option('--compression', type=click.Choice(['zstd', 'gzip', 'lz4', 'none']), default='zstd')
@click.option('--force', is_flag=True, help='Replace an existing archive')
@click.option('--root', default='.', help='Project root directory')
def archive_unbinned(scan, xrd_dir, positions, output, compression, force, root):
    """Archive every detector frame losslessly, once, independent of any grid."""
    from .core import io
    dm = DataManager(root, scan=scan)
    scan_no = _require_scan_no(dm)
    xdir = Path(xrd_dir) if xrd_dir else dm.xrd_frames_dir(scan=scan)
    pos = Path(positions) if positions else dm.position_csv(scan=scan)
    out = Path(output) if output else dm.unbinned_archive_h5(scan=scan)
    if out.exists() and not force:
        try:
            _, _, n_frames = io.archive_metadata(out)
            click.echo(f"Unbinned archive exists ({n_frames} frames), skipping: {out}")
            return
        except (OSError, KeyError, ValueError):
            raise click.ClickException(
                f"Existing archive is invalid: {out}. Re-run with --force to replace it.")
    io.build_unbinned_archive(
        xdir, out, scan_no, positions=pos if pos.exists() else None,
        compression=compression, log=click.echo)
    click.echo(f"Wrote unbinned archive -> {out}")


# ─────────────────────────────────────────────────────────────────────
# bin — pre-build the binned HDF5
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Spatial bin size (NxN)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--grid-mapping', help='Grid mapping HDF5 (defaults to resolved)')
@click.option('--variant', default=None,
              help='Coordinate variant tag (e.g. "faithful") — resolves the matching '
                   'tagged grid mapping and writes a tagged binned HDF5.')
@click.option('--output', help='Output binned HDF5 path (defaults to per-scan Binned/)')
@click.option('--compression', type=click.Choice(['zstd', 'gzip', 'lz4', 'none']), default='zstd')
@click.option('--normalize-frames/--sum-frames', default=False,
              help='Write the mean per contributing frame instead of the frame sum.')
@click.option('--root', default='.', help='Project root directory')
def bin(bin_size, scan, grid_mapping, variant, output, compression,
        normalize_frames, root):
    """Pre-build the binned HDF5 (xrd_NxN_bins.h5) used by 'peaks'."""
    from .core import io
    dm = DataManager(root, scan=scan)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size, variant=variant)
    out = Path(output) if output else dm.binned_h5(bin_size, variant=variant)
    _require(gm, "grid mapping (run 'xrd-app grid' first)")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Require the grid to have been built from a REAL (X, Y) coordinate CSV.
    # Layout-reconstructed or synthetic grids without position provenance are
    # rejected so we never bin on the skew that mis-binned rocking 203-214.
    gm_data = io.load_grid_mapping(gm)
    if not gm_data.get("positions_real", False):
        src = gm_data.get("coordinate_source", "unknown")
        click.echo(f"Error: {gm} was not built from a real coordinate CSV "
                   f"(coordinate_source={src}, positions_real="
                   f"{gm_data.get('positions_real', False)}).")
        click.echo("  Binning requires a grid built from true (X, Y) positions. "
                   "Rebuild it with 'xrd-app grid' (now requires a real "
                   "positions.csv / SOCKETSERVER stream), then re-run bin.")
        raise SystemExit(1)

    archive = dm.unbinned_archive_h5(scan=scan)
    io.build_bins(gm_data, out, bin_size=bin_size, compression=compression,
                  log=click.echo, archive=archive if archive.exists() else None,
                  normalize_frames=normalize_frames)
    click.echo(f"Wrote bins -> {out}")


# ─────────────────────────────────────────────────────────────────────
# peaks — Phase 1: per-bin detection
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--bin-size', type=int, default=3, help='Bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--algorithm', default=None, help='Detector path OR bundled name (see status)')
@click.option('--snr', type=float, default=4.0, help='SNR threshold for detection')
@click.option('--workers', type=click.IntRange(min=1), default=None,
              help='Detector worker processes (default: up to 4)')
@click.option('--name', 'out_name', default=None, help='Algorithm name for the output file (default: detector stem)')
@click.option('--variant', default=None,
              help='Coordinate variant tag (e.g. "faithful") — reads the tagged bins '
                   'and writes a tagged peaks HDF5 catalog.')
@click.option('--h5-path', help='Binned HDF5 (defaults to resolved bins)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.json (defaults to resolved)')
@click.option('--root', default='.', help='Project root directory')
def peaks(bin_size, scan, algorithm, snr, workers, out_name, variant, h5_path, tth_path,
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
        bin_size=bin_size, snr_threshold=snr, n_workers=workers,
        progress=_make_progress("peaks"), log=click.echo)
    result["scan"] = dm.scan_name
    result["algorithm"] = algo
    from .core import lineage
    result["lineage"] = lineage.peak_lineage(
        scan=dm.scan_name, bin_size=bin_size, algorithm=algo,
        detector_file=det, snr=snr, variant=variant)

    out = dm.peaks_json(algo, bin_size, scan, variant=variant)
    from .core import catalogs
    catalogs.save_result(out, result)
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
@click.option('--from-peaks', help='Path to a saved peaks HDF5 catalog (else --peak-algo)')
@click.option('--peak-algo', help='Name of a saved peak set in Labels/<scan>/')
@click.option('--link-tolerance', type=int, default=5, help='Cross-bin link tolerance (px)')
@click.option('--variant', default=None,
              help='Coordinate variant tag (e.g. "faithful") — resolves the tagged '
                   'peaks/grid and writes a tagged shapes HDF5 catalog + CSVs.')
@click.option('--coordinate/--grid-link', 'coordinate', default=None,
              help='Linking mode. Gridless coordinate linking (across true (X,Y) '
                   'neighbors) is the DEFAULT at 1×1 — the skew-free path, no grid '
                   'to skew; binned sizes (≥2×2) default to grid linking, where '
                   'backlash is already averaged out. --grid-link / --coordinate '
                   'force the choice. Coordinate mode reuses the standard peaks '
                   '(only linking changes) and writes a "_coord" shapes file.')
@click.option('--positions', help='Position CSV for coordinate linking (defaults to resolved)')
@click.option('--tth-path', help='2θ TIFF map (defaults to resolved)')
@click.option('--reflections', 'reflections_path', help='reflections.json (defaults to resolved)')
@click.option('--grid-mapping', help='Grid mapping HDF5 (defaults to resolved)')
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
    from .core import processing
    dm = DataManager(root, scan=scan)

    # Gridless coordinate linking is the skew-free default at 1x1; binned sizes
    # keep grid linking. An explicit --coordinate must never silently change the
    # requested scientific method.
    coordinate_explicit = coordinate is True
    if coordinate is None:
        coordinate = (bin_size == 1)

    pos = None
    if coordinate:
        pos = Path(positions) if positions else dm.position_csv(scan=scan)
        if not Path(pos).exists():
            if coordinate_explicit:
                raise click.ClickException(
                    f"Coordinate linking requires a position CSV; not found: {pos}")
            click.echo(f"Warning: no position CSV ({pos}); using grid linking.", err=True)
            coordinate = False
    if coordinate and algorithm in (None, 'gaussian'):
        # Coordinate linking needs a neighbor-graph-capable linker. 'gaussian'
        # (grid-only) maps to 'territory' = same gaussian verification, coordinate
        # linking. An explicit coordinate-capable algo (e.g. voigt) is kept.
        algorithm = 'territory'

    peaks_path = Path(from_peaks) if from_peaks else (
        dm.peaks_json(peak_algo, bin_size, scan, variant=variant) if peak_algo else None)
    if not peaks_path:
        raise click.UsageError("Provide --from-peaks <catalog> or --peak-algo <name>.")
    _require(peaks_path, "peaks catalog (run 'xrd-app peaks' first)")
    from .core import catalogs
    peaks_data = catalogs.load_result(peaks_path)
    try:
        catalogs.validate_result_identity(
            peaks_path, expected_scan=dm.scan_name,
            expected_bin_size=bin_size, expected_variant=variant)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    tth = dm.tth_map(tth_path)
    refl = dm.reflections(reflections_path)
    gm = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size, variant=variant)
    shape = dm.shape_script(algorithm)
    for label, p in [("tth", tth), ("reflections", refl), ("grid_mapping", gm),
                     ("shape algorithm", shape)]:
        _require(p, label)
    try:
        from .core import io as core_io
        core_io.validate_grid_mapping_bin_size(gm, bin_size)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
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
        peak_source=lineage.from_peaks_data(peaks_data),
        peak_source_file=peaks_path.name)

    out = dm.shapes_json(algo, bin_size, scan, variant=out_variant)
    catalogs.save_result(out, result)

    # Emit the kept/filtered CSVs alongside the shapes file. The shapes HDF5 file is
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


@main.command(name='roi')
@click.option('--root', default=None,
              help='Project root (default: last-opened project)')
@click.option('--scan', default=None, help='Initial scan (defaults to config/last-used)')
@click.option('--bin-size', type=int, default=3, help='Initial bin size')
def roi(root, scan, bin_size):
    """Open the ROI > Shape window on its own (Setup + ROI > Shape tabs).

    Shortcut for ``python -m xrd_app.tabs.roi_shape``: a focused two-tab window
    (Setup + ROI > Shape) with the same project/scan switching as the full app.
    """
    from .tabs._standalone import launch_tab
    raise SystemExit(launch_tab("roi_shape", project=root, scan=scan,
                                bin_size=bin_size))


@main.command(name='roi-cvevolve-init')
@click.option('--dest', default=None, help='Session directory (default: CVEvolve/roi_summed_detection)')
@click.option('--holdout-pct', type=float, default=20.0, help='Percentage of labeled scans held out')
@click.option('--seed', type=int, default=42, help='Seeded scan split')
@click.option('--root', default='.', help='Project root directory')
def roi_cvevolve_init(dest, holdout_pct, seed, root):
    """Create a summed-image ROI detector CVEvolve session from manual catalogs."""
    from .core import roi_cvevolve
    dm = DataManager(root)
    try:
        result = roi_cvevolve.create_session(
            dm, dest=dest, holdout_pct=holdout_pct, seed=seed)
    except ValueError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Created ROI CVEvolve session at {result['dest']} from "
               f"{result['examples']} labeled scans")
    click.echo(f"Dev: {result['splits']['test_data']}")
    click.echo(f"Holdout: {result['splits']['holdout_data']}")


# ─────────────────────────────────────────────────────────────────────
# roi-detect — propose ROIs on a fully summed detector image
# ─────────────────────────────────────────────────────────────────────
@main.command(name='roi-detect')
@click.option('--scan', default=None, help='Scan number/name')
@click.option('--algorithm', default=None, help='ROI detector script (default: baseline)')
@click.option('--sensitivity', type=float, default=None,
              help='Detection threshold override (default: algorithm setting)')
@click.option('--max-rois', type=int, default=None,
              help='Maximum proposals override (default: algorithm setting)')
@click.option('--output', required=True, help='Output candidate JSON')
@click.option('--root', default='.', help='Project root directory')
def roi_detect(scan, algorithm, sensitivity, max_rois, output, root):
    """Detect candidate feature ROIs on the scan's fully summed image."""
    from .core import reflection_sum, roi_detection

    dm = DataManager(root, scan=scan)
    sum_path = reflection_sum.sum_path(dm, scan)
    _require(sum_path, "reflection_sum.npz (compute the full scan sum first)")
    import numpy as np
    with np.load(sum_path) as saved:
        image = saved["image"].astype(np.float64)
    algorithm_path = Path(algorithm) if algorithm else roi_detection.default_algorithm()
    _require(algorithm_path, "ROI detector")
    overrides = {}
    if sensitivity is not None:
        overrides["sensitivity"] = sensitivity
    if max_rois is not None:
        overrides["max_rois"] = max_rois
    candidates = roi_detection.detect(image, algorithm_path, **overrides)
    result = {"kind": "roi_candidates", "scan": dm.scan_name,
              "algorithm": algorithm_path.stem, "n_candidates": len(candidates),
              "candidates": candidates}
    _write_json(output, result)
    click.echo(f"Detected {len(candidates)} candidate ROIs -> {output}")


# ─────────────────────────────────────────────────────────────────────
# roi-shapes — detector ROI intensity preview for the manual ROI catalog
# ─────────────────────────────────────────────────────────────────────
@main.command(name='roi-shapes')
@click.option('--bin-size', type=int, default=3, help='Spatial bin size to process')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--roi', 'rois', multiple=True, required=True,
              help='Detector rectangle X0,Y0,X1,Y1; repeat to batch multiple ROIs')
@click.option('--name', required=True, help='Manual ROI catalog name')
@click.option('--preview-output', default=None,
              help='Write one unsaved batch preview here instead of updating a catalog')
@click.option('--fast', is_flag=True, help='Approximate coarse-to-fine preview (never use for save)')
@click.option('--stride', type=click.IntRange(2, 10), default=3,
              help='Spatial stride for --fast preview')
@click.option('--normalize-frames/--sum-frames', default=False,
              help='Divide each spatial bin by its contributing frame count. Use '
                   'for intensity maps when true-position cells have unequal occupancy.')
@click.option('--sample-crop', 'sample_crops', multiple=True,
              help='Per-ROI sample rectangle X0,Y0,X1,Y1, or none; repeat in ROI order')
@click.option('--root', default='.', help='Project root directory')
def roi_shapes(bin_size, scan, rois, name, preview_output, fast, stride,
               normalize_frames, sample_crops, root):
    """Batch detector ROIs into spatial maps with one pass over scan bins."""
    from .core import io, processing, roi_map

    detector_rois = []
    for roi in rois:
        try:
            detector_roi = tuple(int(v.strip()) for v in roi.split(','))
            if len(detector_roi) != 4:
                raise ValueError
            detector_rois.append(detector_roi)
        except ValueError:
            raise click.BadParameter('expected X0,Y0,X1,Y1', param_hint='--roi')
    try:
        tag = safe_component(name, normalize=True, label="catalog name")
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint='--name')
    crops = []
    for value in sample_crops:
        if value.strip().lower() in ("none", "full"):
            crops.append(None)
            continue
        try:
            crop = tuple(int(v.strip()) for v in value.split(','))
            if len(crop) != 4:
                raise ValueError
            crops.append(crop)
        except ValueError:
            raise click.BadParameter(
                'expected X0,Y0,X1,Y1 or none', param_hint='--sample-crop')
    if crops and len(crops) != len(detector_rois):
        raise click.BadParameter(
            'repeat once per --roi, using none for the full scan', param_hint='--sample-crop')
    if not crops:
        crops = [None] * len(detector_rois)

    dm = DataManager(root, scan=scan)
    tth = dm.tth_map(scan=scan)
    gm_path = dm.grid_mapping(bin_size=bin_size, scan=scan)
    for label, path in (("tth", tth), ("grid mapping", gm_path)):
        _require(path, label)
    try:
        gm = io.validate_grid_mapping_bin_size(gm_path, bin_size)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        source = io.open_bin_source(dm, bin_size, scan, grid_mapping=gm_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Cannot open ROI bin source: {exc}") from exc
    try:
        sampled = roi_map.sample_rois(
            source, detector_rois, grid_mapping=gm, metric="integrated",
            normalize_frames=normalize_frames, sample_crops=crops, fast=fast, stride=stride,
            progress=_make_progress("ROI intensity maps"), log=click.echo)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        source.close()
    tth_map = io.load_tth_map(tth)
    beam_center = processing.estimate_beam_center(tth_map)
    features = [roi_map.to_shape_feature(
        result, "manual ROI", feature_id=index,
        tth_map=tth_map, beam_center=beam_center)
        for index, result in enumerate(sampled, 1)]
    for feature in features:
        feature["n_bin_rows"] = int(gm.get("n_bin_rows", 0))
        feature["n_bin_cols"] = int(gm.get("n_bin_cols", 0))
    preview_result = {
        "kind": "manual_roi_preview",
        "scan": dm.scan_name,
        "bin_size": bin_size,
        "n_bin_rows": int(gm.get("n_bin_rows", 0)),
        "n_bin_cols": int(gm.get("n_bin_cols", 0)),
        "approximate": bool(fast and any(r.get("approximate") for r in sampled)),
        "stride": stride if fast else 1,
        "normalize_frames": bool(normalize_frames),
        "sample_crops": [list(crop) if crop is not None else None for crop in crops],
        "intensity_definition": (
            "mean detector counts per contributing frame inside ROI per spatial bin"
            if normalize_frames else
            "total detector counts inside ROI per spatial bin"),
        "features": features,
    }
    output = Path(preview_output) if preview_output else dm.roi_map_json(tag, bin_size, scan)
    if preview_output:
        _write_json(output, preview_result)
    else:
        from .core import roi_catalog
        roi_catalog.save_previews(output, features, scan=dm.scan_name,
                                  bin_size=bin_size, name=tag)
    click.echo(f"\nDone: {len(features)} ROI feature(s) sampled in one pass -> {output}")


@main.command(name='roi-save')
@click.option('--roi', 'rois', multiple=True, required=True,
              help='ROI X0,Y0,X1,Y1; repeat for every ready feature')
@click.option('--name', required=True, help='Manual ROI catalog name')
@click.option('--bin-size', type=int, default=3, help='Spatial bin size')
@click.option('--scan', default=None, help='Scan number/name')
@click.option('--sample-crop', 'sample_crops', multiple=True,
              help='Per-ROI sample rectangle X0,Y0,X1,Y1, or none; repeat in ROI order')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def roi_save(ctx, rois, name, bin_size, scan, sample_crops, root):
    """Recompute all ready ROIs exactly in one batch, then save the catalog."""
    import tempfile
    try:
        tag = safe_component(name, normalize=True, label="catalog name")
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint='--name')
    preview = Path(tempfile.gettempdir()) / f"xrd_app_roi_exact_{os.getpid()}.json"
    try:
        ctx.invoke(roi_shapes, bin_size=bin_size, scan=scan, rois=rois, name=tag,
                   preview_output=str(preview), fast=False, stride=3,
                   sample_crops=sample_crops, normalize_frames=False, root=root)
        import json
        from .core import roi_catalog
        with open(preview) as handle:
            features = json.load(handle).get("features", [])
        dm = DataManager(root, scan=scan)
        output = dm.roi_map_json(tag, bin_size, scan)
        result = roi_catalog.save_previews(
            output, features, scan=dm.scan_name, bin_size=bin_size, name=tag)
        click.echo(f"Saved {len(features)} exact ROI features "
                   f"({result['n_features']} total) -> {output}")
    finally:
        try:
            preview.unlink()
        except OSError:
            pass


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
@click.option('--compression', type=click.Choice(['zstd', 'gzip', 'lz4', 'none']), default='zstd')
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

    from .core import io
    click.echo(f"Batch over {len(scan_list)} scan(s): {', '.join(scan_list)}\n")
    failures, skipped = [], []
    for name in scan_list:
        click.echo(f"{'='*60}\n  {name}\n{'='*60}")
        dm = DataManager(root, scan=name)
        # Skip incomplete scans (no XRD/ frame files) rather than crashing — many
        # Scan_NNNN/ dirs on the beamline mount have no frames yet.
        if (not dm.unbinned_archive_h5(scan=name).exists() and
                not io.has_raw_frames(dm.xrd_frames_dir(scan=name),
                                      dm.scan_number(name) or 0)):
            click.echo("  no unbinned archive or raw frames (incomplete scan) — skipping\n")
            skipped.append(name)
            continue
        algo = algorithm or Path(dm.detector_script(algorithm, bin_size=bin_size)).stem
        if skip_existing and dm.shapes_json(shape_algo, bin_size, name).exists():
            click.echo("  shapes exist — skipping (--skip-existing)\n")
            continue
        try:
            ctx.invoke(archive_unbinned, scan=name, compression=compression, root=root)
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
@click.option('--workers', type=click.IntRange(min=1), default=None,
              help='Peak detector worker processes (default: up to 4)')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def run_pipeline(ctx, bin_size, scan, algorithm, shape_algo, snr, workers, root):
    """Run Peak Finding then Shape Finding for one scan, back to back."""
    dm = DataManager(root, scan=scan)
    # Same naming peaks uses for its output set, so shapes can pick it up.
    algo = algorithm or Path(dm.detector_script(algorithm, bin_size=bin_size)).stem
    ctx.invoke(peaks, bin_size=bin_size, scan=scan, algorithm=algorithm,
               snr=snr, workers=workers, root=root)
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
@click.option('--compression', type=click.Choice(['zstd', 'gzip', 'lz4', 'none']), default='zstd')
@click.option('--normalize-frames/--sum-frames', default=False,
              help='Write the mean per contributing frame instead of the frame sum.')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def make_bins(ctx, bin_size, scan, grid_shape, compression, normalize_frames, root):
    """Archive raw frames once, then build the requested grid and bins."""
    ctx.invoke(archive_unbinned, scan=scan, compression=compression, root=root)
    ctx.invoke(grid, bin_size=bin_size, scan=scan, shape=grid_shape, root=root)
    ctx.invoke(bin, bin_size=bin_size, scan=scan, compression=compression,
               normalize_frames=normalize_frames, root=root)
    dm = DataManager(root, scan=scan)
    click.echo(f"\nBins ready: {dm.binned_h5(bin_size)}")

    # A completed bin build is exactly when the fast (h5) grand-sum becomes
    # available, so refresh the manual-reflections sum here (best-effort: a
    # sum failure must not fail the bin build the user actually asked for).
    from .core import reflection_sum
    try:
        res = reflection_sum.compute_and_save(dm, scan, overwrite=True,
                                              progress=_make_progress("summing bins"))
        click.echo(f"Reflection sum: {res['path']}  {res['shape']}")
    except Exception as e:
        click.echo(f"(reflection sum skipped: {e})")


# ─────────────────────────────────────────────────────────────────────
# reflection-sum — grand-sum a scan's bins (the "Compute histogram" artifact)
# ─────────────────────────────────────────────────────────────────────
@main.command(name='reflection-sum')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--all-scans', is_flag=True,
              help='Process every discovered scan that has built bins.')
@click.option('--include-raw', is_flag=True,
              help='With --all-scans, also sum scans that lack built bins '
                   '(raw frames — slow over a network share).')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Recompute even if reflection_sum.npz already exists.')
@click.option('--max-bins', type=int, default=0,
              help='Cap bins summed (0 = all); recorded in the file.')
@click.option('--root', default='.', help='Project root directory')
def reflection_sum_cmd(scan, all_scans, include_raw, max_bins, overwrite, root):
    """Sum all of a scan's bins into Metadata/<scan>/reflection_sum.npz.

    This is the artifact behind Setup → manual reflections → "Compute histogram":
    the 2θ histogram is re-derived from it instantly, so precomputing it here
    means the GUI opens with the histogram ready. Reads from a prebuilt NxN h5
    when present (fast), else sums raw frames (slower).
    """
    from .core import reflection_sum
    dm = DataManager(root, scan=scan)
    scans = dm.discover_scans(usable_only=True) if all_scans else [scan]
    if not scans:
        raise click.ClickException("No scans to process.")
    done = skipped = failed = 0
    for name in scans:
        # A "bin sum" needs bins: in batch mode, skip scans with no built h5
        # rather than silently summing raw frames over the network share.
        if all_scans and not include_raw:
            bdir = dm.binned_dir(name)
            if not (bdir.is_dir() and any(bdir.glob("xrd_*x*_bins.h5"))):
                skipped += 1
                continue
        try:
            res = reflection_sum.compute_and_save(
                dm, name, max_bins=max_bins, overwrite=overwrite,
                progress=_make_progress(f"summing {name or dm.scan_name}"))
        except Exception as e:
            failed += 1
            click.echo(f"  {name}: FAILED — {e}")
            continue
        if res["skipped"]:
            skipped += 1
            click.echo(f"  {res['scan']}: exists, skipped (use --overwrite)")
        else:
            done += 1
            src = "raw" if res["is_raw"] else f"{res['bin_size']}x{res['bin_size']}"
            click.echo(f"  {res['scan']}: {res['shape']} from {src} → {res['path']}")
    click.echo(f"\nreflection-sum: {done} computed, {skipped} skipped, {failed} failed.")
    if failed:
        raise click.ClickException(f"{failed} scan(s) failed during reflection summation.")


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
    if bin_size != 1:
        raise click.UsageError("Combined algorithms require --bin-size 1.")
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
    output = ldir / f"{algo}_combined_{suffix}.h5"
    from .core import catalogs
    catalogs.save_result(output, result)
    catalogs.record_catalog(ldir, output.name, result["lineage"])
    click.echo(f"\nDone: {result['n_features']} features in "
               f"{len(result['by_bin'])} bins.")


# ─────────────────────────────────────────────────────────────────────
# lineage — show provenance of peaks/shapes/combined catalogs
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.argument('target', required=False)
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--root', default='.', help='Project root directory')
def lineage(target, scan, root):
    """Show the lineage/provenance of result catalogs.

    With no TARGET, summarizes every peaks/shapes/combined HDF5 catalog in
    Labels/<scan>/. TARGET may be a path or a file name within that folder.
    """
    from .core import catalogs, lineage as L
    dm = DataManager(root, scan=scan)
    ldir = dm.labels_dir(scan)
    if target:
        p = Path(target)
        if not p.exists():
            p = ldir / target            # try as a name inside Labels/<scan>
        if not p.exists():
            raise click.ClickException(f"Result catalog not found: {p}")
        paths = [p]
    else:
        paths = (catalogs.list_catalogs(ldir, "peaks")
                 + catalogs.list_catalogs(ldir, "shapes")
                 + catalogs.list_catalogs(ldir, "combined"))
    if not paths:
        click.echo(f"No result catalogs found in {ldir}.")
        return
    for p in paths:
        if not p.exists():
            click.echo(f"\n{p}: not found")
            continue
        data = catalogs.load_result(p) or {}
        click.echo(f"\n{p.name}")
        lin = data.get("lineage")
        if isinstance(lin, dict):
            for line in L.format_lineage(lin):
                click.echo("  " + line)
        else:
            click.echo("  (no lineage block)")


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
# scan-table — one summary row per scan (cross-scan comparison)
# ─────────────────────────────────────────────────────────────────────
@main.command(name='scan-table')
@click.option('--bin-size', type=int, default=3, help='Bin size (default: 3)')
@click.option('--type', 'type_match', default=None,
              help='Catalog type to compare (substring of the lineage label, '
                   'e.g. "gaussian" or "territory"); default: first available')
@click.option('--refl', default=None,
              help='Comma-separated reflections to filter to (e.g. "(001),(111)")')
@click.option('--all-reflections', 'all_refl', is_flag=True,
              help='Break out every reflection: an "(all)" row per scan plus one '
                   'row per reflection (adds a Reflection column). Ignores --refl.')
@click.option('--bandwidth', type=float, default=5.0, help='χ KDE bandwidth (°)')
@click.option('--out', 'out_dir', default='Study', help='Output directory (Study/)')
@click.option('--root', default='.', help='Project root directory')
def scan_table(bin_size, type_match, refl, all_refl, bandwidth, out_dir, root):
    """One summary row per scan → prints a table + writes Study/scan_summary.csv.

    For a bin size and catalog TYPE (the lineage shared across scans — e.g.
    the gaussian shapes at 3×3, or a territorial mapping) reports per scan:
    feature count, footprint area (sum + union), coverage %, the preferred χ
    (dominant azimuthal cluster) ± range, and shape fill % (solidity). A
    territorial type reports areas in coordinate-CSV units.
    """
    from .core import scan_table as st, aggregate as agg
    dm = DataManager(root)
    types = st.catalog_types(dm, bin_size)
    if not types:
        click.echo(f"No catalogs at {bin_size}x{bin_size} — run peaks/shapes first.")
        raise SystemExit(1)
    if type_match:
        matches = [t for t in types if type_match.lower() in t["label"].lower()]
        if not matches:
            click.echo(f"No catalog type matches {type_match!r}. Available:")
            for t in types:
                click.echo(f"  {t['label']}  ({t['scans']} scan(s))")
            raise SystemExit(1)
        chosen = matches[0]
    else:
        chosen = types[0]
    click.echo(f"Catalog type: {chosen['label']}  ({chosen['scans']} scan(s))")

    refs = [r.strip() for r in refl.split(',') if r.strip()] if refl else None
    rows, meta = st.scan_table_rows(dm, bin_size, chosen["key"], refs=refs,
                                    bandwidth=bandwidth, breakdown=all_refl)
    if not rows:
        click.echo("No matching scans found for this type.")
        raise SystemExit(1)
    click.echo("")
    click.echo(st.format_table(rows, meta))

    out = Path(out_dir)
    if not out.is_absolute():
        out = Path(root) / out
    csv = agg.write_csv(rows, st.COLUMNS, out / "scan_summary.csv")
    click.echo(f"\nWrote {len(rows)} scan rows:\n  {csv}")


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
@click.option('--out', 'out_path', default='Study/tracks.h5', help='Output tracks HDF5')
@click.option('--root', default='.', help='Project root directory')
def track(scans, bin_size, match_tol, min_theta, out_path, root):
    """Link shapes across the θ sweep into grain tracks (Study/tracks.h5 + .csv).

    Same reflection band + spatial proximity within --match-tol bins (the grid is
    identical across θ, so de-skewed bin coords compare directly). Emits a full
    HDF5 (per-track theta membership, chi(theta), intensity(theta)) and a one-row-per-track CSV.
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
    from .core import result_store
    result_store.save(out, {
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
@click.option('--tracks', 'tracks_path', default='Study/tracks.h5', help='tracks HDF5 from `xrd-app track`')
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
    from .core import rocking as rk, aggregate as agg, result_store
    tp = Path(tracks_path)
    if not tp.is_absolute():
        tp = Path(root) / tp
    _require(tp, "tracks HDF5 (run 'xrd-app track' first)")
    tracks = (result_store.load(tp) or {}).get("tracks", [])

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
@click.option('--tracks', 'tracks_path', default='Study/tracks.h5', help='tracks HDF5 from `xrd-app track`')
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
    import csv as _csv
    from .core import aggregate as agg, prediction as pred, result_store
    dm = DataManager(root)

    tp = Path(tracks_path)
    if not tp.is_absolute():
        tp = Path(root) / tp
    _require(tp, "tracks HDF5 (run 'xrd-app track' first)")
    tracks = (result_store.load(tp) or {}).get("tracks", [])

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
@click.option('--tracks', 'tracks_path', default='Study/tracks.h5',
              help='Optional tracks HDF5 for the centroid overlay')
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
    import csv as _csv
    from .core import combined_device as cd, result_store
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
        tracks = (result_store.load(tp) or {}).get("tracks", [])

    combined = cd.build_combined(rows, intensity_key=intensity_key,
                                 tracks=tracks, log=click.echo)
    out = Path(out_path)
    if not out.is_absolute():
        out = Path(root) / out
    cd.save_npz(out, combined)
    _write_json(out.with_suffix(".summary.json"), cd.summary(combined))
    click.echo(f"\nWrote combined device view:\n  {out}\n  {out.with_suffix('.summary.json')}")


def _same_grid_lattice(child, source):
    """Whether two grid mappings share the acquisition and coordinate lattice."""
    keys = ("coordinate_source", "positions_real", "xrd_files", "frame_map")
    compared = False
    for key in keys:
        if key not in child or key not in source:
            continue
        compared = True
        if child[key] != source[key]:
            return False
    return compared


def _ensure_1x1_grid_mapping(dm, scan, source_bin_size, log=click.echo):
    """Build a lattice-compatible 1×1 mapping so HD real (x, y) can attach.

    The HD real-position layer needs a 1×1 grid mapping (cell → raw frame →
    stage (x, y)). It must share the source N×N catalog's lattice — same
    ``deskew_method`` — so its cells are a clean N× refinement (else
    :func:`hd_map.build_cell_xy`'s keys miss the sub-bin keys and no positions
    attach). We mirror the N×N grid's ``coordinate_source`` rather than the
    ``auto`` default (which would pick ``positions_xy`` at 1×1).

    Graceful: returns True if a usable mapping now exists, False (with a warning)
    when raw frames or real positions are unavailable — the HD build then
    proceeds intensity-only, exactly as before. Never raises.
    """
    from .core import io
    from .core import positions as P

    gm_path = dm.grid_mapping(bin_size=1, scan=scan)
    nxn_path = dm.grid_mapping(bin_size=source_bin_size, scan=scan)
    source_grid = None
    if nxn_path and Path(nxn_path).exists():
        source_grid = io.load_grid_mapping(nxn_path)
    if gm_path and Path(gm_path).exists():
        child_grid = io.load_grid_mapping(gm_path)
        if source_bin_size == 1 or (source_grid and _same_grid_lattice(child_grid, source_grid)):
            return True
        log(f"[hd-device-map] existing 1x1 grid {gm_path} does not match the "
            f"{source_bin_size}x{source_bin_size} source lattice; rebuilding it.")

    try:
        scan_no = _require_scan_no(dm)
        xdir = dm.xrd_frames_dir(scan=scan)
        archive = dm.unbinned_archive_h5(scan=scan)
        if not archive.exists() and not io.has_raw_frames(xdir, scan_no):
            log("[hd-device-map] no 1×1 grid mapping, unbinned archive, or raw "
                f"frames ({xdir}) — real-position scatter unavailable.")
            return False

        # Real positions are required for the (x, y) layer. Build them from the
        # SOCKETSERVER interferometry stream when there's no real CSV (same
        # source the 'grid' command uses); if neither exists, skip gracefully.
        pos = dm.position_csv(scan=scan)
        pos_real = ((Path(pos).exists() and not io.is_recreated_csv(pos)) or
                    (archive.exists() and io.archive_has_real_positions(archive)))

        if not pos_real:
            sdir = dm.socketserver_dir(scan=scan)
            if P.has_socketserver(sdir, scan_no):
                dest = dm.metadata_scan_dir(scan) / "positions.csv"
                log("[hd-device-map] building real positions from SOCKETSERVER "
                    f"interferometry ({sdir}) …")
                P.build_positions_csv(sdir, dest, scan_number=scan_no, log=log)
                pos, pos_real = dest, True
            else:
                log("[hd-device-map] no 1×1 grid mapping and no real positions "
                    "(CSV or SOCKETSERVER) — real-position scatter unavailable.")
                return False

        # Match the source N×N catalog's lattice so sub-bin keys align.
        coordinate_source = (source_grid or {}).get("coordinate_source")
        deskew = io.deskew_method_for_source(coordinate_source)

        gm_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"[hd-device-map] no 1×1 grid mapping — building one ({deskew}, to "
            f"match the {source_bin_size}×{source_bin_size} catalog) for real "
            "positions …")
        io.generate_grid_mapping(
            xdir, pos, 1, scan_number=scan_no, output=gm_path,
            deskew_method=deskew, log=log,
            archive=archive if archive.exists() else None)
        return True
    except Exception as e:  # never let grid-building abort the HD map
        log(f"[hd-device-map] could not auto-build 1×1 grid mapping ({e}); "
            "continuing without real positions.")
        return False


# ─────────────────────────────────────────────────────────────────────
# hd-device-map — 1×1 intensity sampled beneath a binned feature map
# ─────────────────────────────────────────────────────────────────────
@main.command(name='hd-device-map')
@click.option('--bin-size', type=int, default=3, help='Source feature-map bin size (e.g. 3)')
@click.option('--scan', default=None, help='Scan number/name (defaults to config scan)')
@click.option('--catalog', default=None,
              help='3×3 shapes/feature catalog (path or name in Labels/<scan>). '
                   'Default: newest shapes/combined for --bin-size.')
@click.option('--win', type=int, default=4,
              help='Detector-peak sampling half-window (px); widen for broad peaks')
@click.option('--max-cells', type=int, default=0,
              help='Cap 1×1 cells per feature (0 = unlimited)')
@click.option('--name', 'out_name', default=None,
              help='Algorithm name for the output file (default: source catalog algo)')
@click.option('--root', default='.', help='Project root directory')
def hd_device_map(bin_size, scan, catalog, win, max_cells, out_name, root):
    """Sample raw 1×1 intensity at each feature's detector peak (Labels/<scan>/).

    Reads a binned (N×N) feature catalog, and for every 1×1 pixel inside each
    feature's footprint reads that raw frame and records the intensity at the
    feature's detector peak — the high-def layer the HD Device View renders
    beneath the N×N outlines. Real stage (x, y) per pixel is attached when the
    scan has a real position CSV, for the real-position scatter mode.

    Heavy (reads thousands of raw frames); run once — the JSON is cached.
    """
    import json
    from .core import catalogs, hd_map, io

    dm = DataManager(root, scan=scan)

    # Source catalog: explicit path/name, else newest shapes/combined for the bin.
    if catalog:
        cat_path = Path(catalog)
        if not cat_path.exists():
            cat_path = dm.labels_dir(scan) / catalog
    else:
        cat_path = catalogs.default_feature_source(dm.labels_dir(scan), bin_size)
    _require(cat_path, f"{bin_size}×{bin_size} feature catalog "
                       "(run 'xrd-app shapes' first, or pass --catalog)")
    try:
        catalogs.validate_result_identity(
            cat_path, expected_scan=dm.scan_name, expected_bin_size=bin_size)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    features, _ = catalogs.load_features_any(cat_path)
    if not features:
        click.echo(f"Error: no features in {cat_path}")
        raise SystemExit(1)
    algo = out_name or (catalogs.parse_name(Path(cat_path).name) or {}).get("algo") or "hd"

    # 1×1 raw-frame source (h5 if built, else summed raw frames on demand).
    click.echo(f"[hd-device-map] catalog: {cat_path}\n[hd-device-map] "
               f"{len(features)} features, win={win}px")
    # Ensure the 1×1 grid mapping exists (built to match this catalog's lattice)
    # so the real (x, y) scatter layer can attach — auto-built once, then cached.
    _ensure_1x1_grid_mapping(dm, scan, bin_size)
    source = io.open_bin_source(dm, 1, scan)
    try:
        # 1×1 grid dims + real per-cell (x, y) from the 1×1 grid mapping.
        gm_path = dm.grid_mapping(bin_size=1, scan=scan)
        gm = {}
        if gm_path and Path(gm_path).exists():
            gm = io.load_grid_mapping(gm_path)
        # Real per-cell (x, y): try any resolvable position CSV. build_cell_xy
        # returns {} for X-only / missing CSVs (older grid mappings omit the
        # positions_real flag, so don't gate on it).
        # Prefer the grid mapping's stored CSV, but it may be an absolute path
        # from another machine (e.g. /net/micdata on the LAN host vs /mnt/z on a
        # laptop). Fall back to the project-relative CSV when it doesn't resolve
        # here, so real (x, y) still attaches regardless of where we run.
        pos_csv = gm.get("positions_csv")
        if not pos_csv or not Path(pos_csv).exists():
            pos_csv = dm.position_csv(scan=scan)
        cell_xy = hd_map.build_cell_xy(
            gm, pos_csv, archive=dm.unbinned_archive_h5(scan=scan)
        ) if gm.get("bins") else {}

        hd_features = hd_map.sample_hd_intensity(
            features, source, bin_size, win=win, cell_xy=cell_xy,
            max_cells_per_feature=(max_cells or None),
            progress=_make_progress("hd-map"), log=click.echo)
    finally:
        source.close()

    # 1×1 grid dims: prefer the 1×1 grid mapping, else derive from sampled cells.
    n_rows = gm.get("n_bin_rows") or gm.get("n_rows")
    n_cols = gm.get("n_bin_cols") or gm.get("n_cols")
    if not (n_rows and n_cols):
        mr = mc = -1
        for f in hd_features:
            for k in f["hd_profile"]:
                r, c = (int(p) for p in k.split("_"))
                mr, mc = max(mr, r), max(mc, c)
        n_rows, n_cols = mr + 1, mc + 1

    summary = hd_map.summarize(hd_features)
    result = {
        "kind": "hd_map",
        "scan": dm.scan_name,
        "bin_size": bin_size,
        "win": win,
        "source_catalog": Path(cat_path).name,
        "n_bin_rows_1x1": int(n_rows),
        "n_bin_cols_1x1": int(n_cols),
        "positions_real": bool(cell_xy),
        "features": hd_features,
    }
    out = dm.hd_map_json(algo, bin_size, scan)
    catalogs.save_result(out, result)
    click.echo(
        f"\nDone: {summary['n_features']} features, {summary['n_cells']} 1×1 cells "
        f"sampled ({summary['n_features_empty']} empty, "
        f"{summary['n_cells_with_position']} with (x,y)) -> {out}")


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


# ─────────────────────────────────────────────────────────────────────
# xrf — ME7 XSPRESS3 fluorescence → per-element spatial maps on the grid
# ─────────────────────────────────────────────────────────────────────
@main.command()
@click.option('--scans', help='Comma-separated scans (default: config scan)')
@click.option('--bin-size', type=int, default=1,
              help='Grid bin size to map XRF onto (default 1: finest, underlays any bin size)')
@click.option('--me7-dir', help='ME7 directory (defaults to <scan raw>/ME7)')
@click.option('--config', 'config_path', help='XRF elements/calibration JSON (defaults to Metadata)')
@click.option('--no-deadtime', is_flag=True, help='Disable XSPRESS3 deadtime correction')
@click.option('--refine-roi', is_flag=True,
              help='Data-driven ROI: center windows on observed peaks in the grand-sum spectrum')
@click.option('--save-points', is_flag=True,
              help='Also write <scan>_xrf_points.npz: compact per-frame spectra so the '
                   'GUI histogram needs no raw ME7 (~10x smaller than raw)')
@click.option('--grid-mapping', help='Grid mapping HDF5 (defaults to resolved)')
@click.option('--out-dir', default=None, help='Output dir (default: per-scan Metadata dir)')
@click.option('--root', default='.', help='Project root directory')
def xrf(scans, bin_size, me7_dir, config_path, no_deadtime, refine_roi, save_points, grid_mapping, out_dir, root):
    """ME7 (XSPRESS3) fluorescence → per-element spatial maps on the XRD grid.

    For each scan: read the ME7 MCA spectra, deadtime-correct + sum the enabled
    channels, integrate per-element energy ROIs (from the config JSON), and
    accumulate onto the same de-skewed bins as the XRD via the grid mapping.
    Writes ``<scan>_xrf.npz`` (+ .summary.json). Elements/calibration come from a
    small JSON (auto-created with perovskite defaults on first run).
    """
    import json
    from .core import xrf as xrf_core

    # resolve / seed the config JSON (project-level default)
    cfg_default = ProjectConfig.load(root)  # noqa: F841 (ensures project exists)
    scan_list = ([DataManager.scan_name_of(s.strip()) for s in scans.split(',') if s.strip()]
                 if scans else None)
    if not scan_list:
        one = DataManager(root).scan_name
        if not one:
            click.echo("Error: no scan given. Use --scans or set a config scan.")
            raise SystemExit(1)
        scan_list = [one]

    for scan in scan_list:
        dm = DataManager(root, scan=scan)
        # config: explicit → per-scan → project (seed project default if none)
        if config_path:
            cfg_file = Path(config_path)
        else:
            per_scan = dm.metadata_scan_dir(scan) / "xrf_elements.json"
            proj = dm.metadata_dir / "xrf_elements.json"
            cfg_file = per_scan if per_scan.exists() else proj
            if not cfg_file.exists():
                xrf_core.write_config(xrf_core.default_config(), proj)
                cfg_file = proj
                click.echo(f"[xrf] seeded default element config -> {proj}")
        cfg = xrf_core.read_config(cfg_file)
        if no_deadtime:
            cfg["deadtime_correction"] = False

        # ME7 dir: override → local copy → config raw sibling
        if me7_dir:
            me7 = Path(me7_dir)
        else:
            me7 = dm.me7_dir(scan=scan)
        _require(me7, f"ME7 directory for {scan}")

        gm_path = Path(grid_mapping) if grid_mapping else dm.grid_mapping(bin_size=bin_size, scan=scan)
        _require(gm_path, "grid mapping (run 'xrd-app grid' first)")
        from .core import io as core_io
        gm = core_io.load_grid_mapping(gm_path)

        click.echo(f"[xrf] {scan}: ME7={me7.name}  channels={cfg['channels']}  "
                   f"deadtime={cfg['deadtime_correction']}  "
                   f"cal={cfg['calibration']['ev_per_bin']} eV/bin")

        refine_diag = None
        if refine_roi:
            click.echo("[xrf] refining ROIs from the data-driven grand-sum spectrum…")
            cfg, refine_diag = xrf_core.refine_rois(me7, cfg, log=click.echo)
            click.echo(f"  found {refine_diag['n_peaks_found']} peaks; line → observed:")
            for d in refine_diag["elements"]:
                if d["matched"]:
                    click.echo(f"    {d['name']:6} {d['line_ev']:7.0f} → "
                               f"{d['observed_ev']:7.0f} eV  (Δ{d['shift_ev']:+.0f})")
                else:
                    click.echo(f"    {d['name']:6} {d['line_ev']:7.0f} eV → no peak matched "
                               f"(using theoretical)")
            if refine_diag["overlaps"]:
                pairs = ", ".join(f"{a}/{b}" for a, b in refine_diag["overlaps"])
                click.echo(f"  ⚠ ROI overlap(s): {pairs} — tighten window_ev to reduce crosstalk")

        result = xrf_core.element_maps(me7, gm, cfg, log=click.echo)

        if out_dir:
            out_base = Path(out_dir)
            if not out_base.is_absolute():
                out_base = Path(root) / out_base
            npz = out_base / f"{scan}_xrf.npz"
        else:
            npz = dm.xrf_product(scan=scan)
        xrf_core.save_npz(npz, result)
        smry = xrf_core.summary(result)
        if refine_diag is not None:
            smry["refinement"] = refine_diag
        _write_json(npz.with_suffix(".summary.json"), smry)

        nr, nc = result["shape"]
        tot = {n: f"{result['maps'][n].sum():.3g}" for n in result["elements"]}
        click.echo(f"  {scan}: grid {nr}x{nc}  dropped={result['dropped']}  "
                   f"totals={tot} -> {npz.name}")

        if save_points:
            click.echo("[xrf] building per-frame spectrum store (--save-points)…")
            store = xrf_core.build_point_store(
                me7, gm, cfg["channels"], cfg["deadtime_correction"], log=click.echo)
            if out_dir:
                pnpz = out_base / f"{scan}_xrf_points.npz"
            else:
                pnpz = dm.xrf_points_product(scan=scan)
            xrf_core.save_point_store(pnpz, store, cfg["channels"],
                                      cfg["deadtime_correction"], cfg["calibration"])
            mb = pnpz.stat().st_size / 1e6
            click.echo(f"  {scan}: {store.shape[0]} frames -> {pnpz.name} ({mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────
# studies — catalog / run whole rocking-study result sets
# ─────────────────────────────────────────────────────────────────────
@main.command(name='list-studies')
@click.option('--root', default='.', help='Project root directory')
def list_studies(root):
    """List the rocking-study result sets discovered under the project.

    A *study* is a directory carrying the cross-scan artifacts
    (rsm.npz / rocking_curves.csv / combined_device.npz / tracks.h5 /
    features.csv). Merges any names/notes from ``studies.json``. This is what the
    Reciprocal-Space and Rocking-Study tabs offer in their study selector.
    """
    from .core import studies
    found = studies.list_studies(root)
    if not found:
        click.echo("No studies found. Run 'xrd-app run-study' (or the "
                   "aggregate→track→rocking→predict→combined-device chain).")
        return
    click.echo(f"{len(found)} stud{'y' if len(found) == 1 else 'ies'} under {root}:\n")
    for e in found:
        click.echo(f"  {e['name']}  [{e['path']}]")
        desc = studies.describe(e)
        if desc:
            click.echo(f"      {desc}")
        if e.get("notes"):
            click.echo(f"      note: {e['notes']}")


@main.command(name='register-study')
@click.argument('path')
@click.option('--name', default=None, help='Human-friendly study name (default: dir name)')
@click.option('--notes', default=None, help='Free-text note stored in studies.json')
@click.option('--root', default='.', help='Project root directory')
def register_study_cmd(path, name, notes, root):
    """Record a study directory in ``studies.json`` (name + notes overlay).

    Discovery already lists any directory with artifacts; this just attaches a
    friendly name/notes so it reads nicely in the study selector.
    """
    from datetime import datetime
    from .core import studies
    p = Path(path)
    if not p.is_absolute():
        p = Path(root) / p
    if not studies.is_study_dir(p):
        click.echo(f"Warning: {p} has no study artifacts yet "
                   f"({', '.join(studies.PRIMARY_ARTIFACTS)}).")
    entry = studies.register_study(
        root, p, name=name, notes=notes,
        created=datetime.now().isoformat(timespec="seconds"))
    click.echo(f"Registered '{entry.get('name', p.name)}' -> {studies.registry_path(root)}")


@main.command(name='run-study')
@click.option('--scans', help='Comma-separated scans (default: all in Labels/)')
@click.option('--bin-size', type=int, default=3, help='Bin size to analyze (default: 3)')
@click.option('--out', 'out_dir', default='Study', help='Study output directory (default: Study/)')
@click.option('--name', 'study_name', default=None, help='Study name for studies.json (default: dir name)')
@click.option('--notes', default=None, help='Free-text note for studies.json')
@click.option('--match-tol', type=float, default=2.0, help='Track match tolerance (bins)')
@click.option('--repeat-pair', default='203,214', help='Same-orientation scan pair for the noise floor')
@click.option('--with-rsm', is_flag=True,
              help='Also build the 3D reciprocal-space map (runs qspace + rsm; needs xrd-app[qspace])')
@click.option('--rsm-bins', type=int, default=128, help='Voxels per axis for --with-rsm')
@click.option('--root', default='.', help='Project root directory')
@click.pass_context
def run_study(ctx, scans, bin_size, out_dir, study_name, notes, match_tol,
              repeat_pair, with_rsm, rsm_bins, root):
    """Run the whole rocking study in one command, then register it.

    Chains aggregate → track → rocking → predict → combined-device (and, with
    --with-rsm, qspace → rsm) into ``--out``, then writes a ``studies.json``
    entry so the analysis shows up, named, in the GUI study selectors. This is
    the one-button version of the per-θ-series pipeline shell script.
    """
    from datetime import datetime
    from .core import studies

    out = Path(out_dir)
    if not out.is_absolute():
        out = Path(root) / out
    out.mkdir(parents=True, exist_ok=True)
    tracks_json = f"{out_dir}/tracks.h5"
    fails = []

    def _step(label, cmd, **kw):
        click.echo(f"\n─── {label} ─────────────────────────────────────────")
        try:
            ctx.invoke(cmd, **kw)
        except SystemExit as e:
            if e.code not in (0, None):
                click.echo(f"[{label}] FAILED (exit {e.code})")
                fails.append(label)
        except Exception as e:  # keep the chain going; report at the end
            click.echo(f"[{label}] ERROR: {type(e).__name__}: {e}")
            fails.append(label)

    _step("aggregate", aggregate, scans=scans, bin_size=bin_size,
          out_dir=out_dir, root=root)
    _step("track", track, scans=scans, bin_size=bin_size, match_tol=match_tol,
          out_path=tracks_json, root=root)
    _step("rocking", rocking, tracks_path=tracks_json,
          out_path=f"{out_dir}/rocking_curves.csv", root=root)
    _step("predict", predict, tracks_path=tracks_json, scans=scans,
          bin_size=bin_size, match_tol=match_tol,
          rocking_path=f"{out_dir}/rocking_curves.csv", repeat_pair=repeat_pair,
          out_path=f"{out_dir}/prediction_report.md", root=root)
    _step("combined-device", combined_device,
          device_map_path=f"{out_dir}/device_map.csv", tracks_path=tracks_json,
          out_path=f"{out_dir}/combined_device.npz", root=root)

    if with_rsm:
        _step("qspace", qspace, scans=scans, bin_size=bin_size,
              out_dir=f"{out_dir}/qspace", root=root)
        _step("rsm", rsm, in_dir=f"{out_dir}/qspace", nbins=rsm_bins,
              out_path=f"{out_dir}/rsm.npz", root=root)

    # Register whatever got produced so the GUI can select it.
    entry = studies.register_study(
        root, out, name=study_name, notes=notes,
        created=datetime.now().isoformat(timespec="seconds"),
        extra={"bin_size": bin_size})
    rel = studies._rel(root, out)
    desc = next((studies.describe(e) for e in studies.list_studies(root)
                 if e["path"] == rel), "")
    click.echo(f"\nStudy '{entry.get('name', out.name)}' registered  {desc}")
    click.echo(f"  {out}")
    if fails:
        click.echo(f"  incomplete steps: {', '.join(fails)}")
        raise SystemExit(1)
    click.echo("  all steps OK.")


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

