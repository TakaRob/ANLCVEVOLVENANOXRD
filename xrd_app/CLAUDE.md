# CLAUDE.md — xrd-app

Notes for Claude when working inside `xrd_app/` — the single-GUI nano-XRD
Bragg-peak analysis app (ISN 26-ID-C / APS). General nano-XRD scientific-
computing conventions live in the repo-root `../CLAUDE.md`; **this file is the
app specifics**.

## The one architectural rule

**The CLI is the engine; the GUI is a face over it.** Every "big button" in the
GUI maps to a `xrd-app <command>`. Put real logic in `core/` (pure, importable,
testable) and `cli.py` (the command surface). Tabs/widgets should call into
those — never embed analysis logic in a `tabs/` or `gui/` module. If you add a
capability, add/extend the CLI command first, then wire the button to it.

Layers (keep the dependency direction one-way):
```
core/   pure analysis + IO + config logic. No PyQt. No click.   ← logic lives here
cli.py  click commands; thin wrappers over core/.               ← the engine
tabs/   QWidget builders, one per tab (make_tab + TAB_META).    ← GUI shell
gui/    reusable embeddable widgets (viewer, device_map, …).
app.py  single tabbed window; discovers tabs lazily.
```

## Run / check

```bash
pip install -e .                 # editable; rerun only if deps/entry points change
xrd-app --help                   # command surface
xrd-app gui                      # launch the single-window app
xrd-app gui --fresh              # ignore saved gui_state
python3 -c "import py_compile,glob; [py_compile.compile(f,doraise=True) for f in glob.glob('xrd_app/**/*.py',recursive=True)]"
```

- **Don't launch the GUI to "verify" headless work** — it needs a display and
  blocks. Prefer running the underlying `xrd-app` CLI command, or `py_compile`
  for a syntax check. Only open the GUI when the user is driving it.
- Pipeline order: `init → scan-detect → link → peaks` (Phase 1, per-bin) `→
  shapes` (Phase 2, link across bins). Project layout and command list are in
  `README.md`; design/phasing in `docs/PLAN.md` + `docs/IMPLEMENTATION.md`.

## Tab contract

A tab module in `xrd_app/tabs/` must expose:
- `TAB_META = {"title": ..., "order": N, ...}`
- `make_tab(project_root=".", scan=None, bin_size=3) -> QWidget`

`app.py::_discover_tabs` imports built-ins in `_BUILTIN_TABS` and any
`xrd_app.tabs` entry points; a tab that raises shows a placeholder instead of
crashing the window — so wrap data-dependent setup defensively. Bin sizes are
`_BIN_SIZES = [1, 3, 4, 5]`; the header scan/bin selectors drive every
scan-dependent tab.

## Conventions

- All paths resolve through `config.ProjectConfig` / `DataManager` — never
  hardcode project paths. The project tree is `Raw/ Binned/ Metadata/ Labels/
  Figures/ CVEvolve/`.
- `pyFAI` is an **optional** dep (`.[poni]`); the convert-poni path must degrade
  gracefully when it's missing. Core install stays light.
- Reflection resolution order: per-scan `Metadata/<scan>/` → project
  `Metadata/` → bundled default. Keep that fallback chain intact.
- Match surrounding style: module docstrings, the `─────` section banners in
  `cli.py`, lazy PyQt imports inside functions (so `core`/CLI import without Qt).

## Environment gotchas (WSL2 + OneDrive)

- Working dir has **spaces** (`OneDrive - Argonne National Laboratory/…`) —
  always quote paths in Bash.
- WSL2 on Windows; files sync via OneDrive. Filesystem is slow-ish and case
  -insensitive in places. Avoid churn-y mass file ops.
- Python via `.venv/` (or `python3`). PyQt5 GUI needs an X server / display.

## Domain quick-ref

Perovskite/halide thin films; reflections (PbI2, (001)/(011)/(111)/(002), ITO,
(012)/(112)) at 2-theta angles in `core/reflections.py`. Beamline: 15 keV,
75 µm pixel. A *peak* is per-bin; a *shape* is a peak
that holds up linked across neighboring bins (Union-Find), characterized by
`rocking_fwhm` / `strain_breadth` / `chi_deg`. See `TERMINOLOGY.md` for the full
glossary. CVEvolve work uses **mean F2** (recall-weighted) as the primary metric.
