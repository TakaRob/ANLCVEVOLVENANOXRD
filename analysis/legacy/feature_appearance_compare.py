# %% [markdown]
# # Compare reflection appearance across the six focus scans
#
# One interactive window that lets us compare **where linked shapes sit in
# orientation (χ)** across our three experimental conditions, on a common,
# θ-normalized χ axis and a device-area (not raw-pixel) y-axis.
#
# ## The six scans (two per condition — a θ≈20.5° and a θ≈6° member each)
#
# | independent variable | scan  | θ (deg) | samy (mm) |
# |----------------------|-------|---------|-----------|
# | `No_DI_Yes_GB`       | 0179  | 20.5    | −2.3135   |
# | `No_DI_Yes_GB`       | 0182  | 6.0     | −2.3136   |
# | `5%_DI_Yes_GB`       | 0203  | 20.5    | −1.61     |
# | `5%_DI_Yes_GB`       | 0207  | 5.5     | −1.61     |
# | `5%_DI_No_GB`        | 0215  | 20.5    | −1.61     |
# | `5%_DI_No_GB`        | 0218  | 6.0     | −1.61     |
#
# ## What is normalized, and why
#
# Absolute χ (azimuth around the beam) drifts with the sample rocking angle θ, so
# two scans of the *same* grain taken at different θ report different χ. To compare
# conditions we remove that drift. Two modes (`NORM_MODE` below):
#
# * `"simple"` **(default)** — `χ_norm = χ − θ`. The rigid-rotation, first-order
#   geometric correction: rotate the sample by θ, the whole pattern's azimuth shifts
#   by θ, so subtract it. Depends on nothing but θ.
# * `"tilt_rate"` — `χ_norm = χ − (dχ/dθ)·θ`, per-reflection rate from `DCHI_DTHETA`.
#   The defaults there are the median `chi_tilt_rate` from the 203–214 rocking fits
#   (≈0 for (001), ≈−1.3°/° for (111)) — but that series has heavy beam damage
#   (perovskite reflections collapse scan-to-scan), so those few-point fits are
#   unreliable; prefer `"simple"` unless you have trustworthy rates.
#
# Either way χ_norm is wrapped back into (−180, 180].
#
# ## The two connected views
#
# * **Histogram** — χ_norm on x, **fraction of device area** on y (max 1). Each
#   scan trace is `Σ shape n_bins per χ slice ÷ that scan's total grid bins`, so
#   scans of different grid size compare directly. One coloured trace per scan;
#   solid = (001), dashed = (111). Tick scans on/off; drag the **bin-width slider**.
# * **Table** — for every active scan × reflection, the **dominant χ slice** (the
#   χ bin covering the most device area) as `(χ, % device area)`, two rows per scan
#   ((001) and (111)), ranked by % device area. Widen the slider and the slices —
#   and this table — recompute live.
#
# Data source per scan: `Labels/<scan>/gaussian_shapes_3x3.json` → `kept[]`
# (linked shapes), device-area denominator: `Metadata/<scan>/grid_mapping_3x3.json`
# → `n_bins`. Run as a notebook (`# %%` cells) or `python feature_appearance_compare.py`
# (needs an interactive matplotlib backend for the widgets).

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- the six focus scans: local WSL project trees (fast disk; mirror /mnt/z) ---
# θ (deg) and samy (mm) are per-scan experiment metadata (not stored in the
# project files) — edit here if a value is corrected. dχ/dθ lives in DCHI_DTHETA.
SCAN_CONFIG = [
    {"proj": "/home/takaji/179-201",        "scan": "Scan_0179", "exp": "No_DI_Yes_GB", "theta": 20.5, "samy": -2.3135},
    {"proj": "/home/takaji/179-201",        "scan": "Scan_0182", "exp": "No_DI_Yes_GB", "theta": 6.0,  "samy": -2.3136},
    {"proj": "/home/takaji/rocking_203_214","scan": "Scan_0203", "exp": "5%_DI_Yes_GB", "theta": 20.5, "samy": -1.61},
    {"proj": "/home/takaji/rocking_203_214","scan": "Scan_0207", "exp": "5%_DI_Yes_GB", "theta": 5.5,  "samy": -1.61},
    {"proj": "/home/takaji/215-226",        "scan": "Scan_0215", "exp": "5%_DI_No_GB",  "theta": 20.5, "samy": -1.61},
    {"proj": "/home/takaji/215-226",        "scan": "Scan_0218", "exp": "5%_DI_No_GB",  "theta": 6.0,  "samy": -1.61},
]

CATALOG_NAME = "gaussian_shapes_3x3.json"          # shapes catalog present for all six
GRID_MAPPING = "grid_mapping_3x3.json"             # device-area denominator (total grid bins)
REFLECTIONS  = ["(001)", "(111)"]                  # the two reflections the table splits on

# χ-normalization mode: "simple" -> χ − θ (rigid rotation, recommended);
# "tilt_rate" -> χ − (dχ/dθ)·θ using DCHI_DTHETA (only if you trust the rates).
NORM_MODE = "simple"

# Per-reflection lattice-tilt rate dχ/dθ (deg χ per deg θ), used only when
# NORM_MODE == "tilt_rate". Defaults are the median chi_tilt_rate from the 203–214
# rocking-curve fits — noisy (beam-damaged series), so treat as a starting point.
# Anything not listed uses DCHI_DTHETA_DEFAULT.
DCHI_DTHETA = {
    "(001)": -0.025, "(111)": -1.292, "(002)": 0.587,
    "(011)": -1.201, "(012)": -0.080, "ITO": 0.371, "PbI2": 1.350,
}
DCHI_DTHETA_DEFAULT = 0.056                          # overall median across all reflections

CHI_BIN_WIDTH = 5           # initial χ-slice width (deg); the slider overrides this live
FIG_DIR = Path("report_figures")
FIG_DIR.mkdir(exist_ok=True)
FIG_PATH = FIG_DIR / "feature_appearance_compare.png"

# one colour per scan; reflection is encoded by line style (solid=001, dashed=111)
SCAN_COLORS = ["#3b6fb6", "#8fb3e0", "#c4432b", "#e39b8f", "#6aab4d", "#b4d69a"]
REFL_STYLE = {"(001)": "-", "(111)": "--"}


def norm_refl(label: str) -> str:
    """Canonicalize a reflection label so config '001' matches data '(001)'."""
    s = str(label).strip()
    core = s.strip("()")
    return f"({core})" if core.isdigit() else s


REFLECTIONS_N = [norm_refl(r) for r in REFLECTIONS]


def dchi_dtheta(refl: str) -> float:
    """dχ/dθ for the chosen mode: 1.0 for the simple rigid-rotation norm, else the
    per-reflection measured tilt rate."""
    if NORM_MODE == "simple":
        return 1.0
    return DCHI_DTHETA.get(refl, DCHI_DTHETA_DEFAULT)


NORM_LABEL = ("normalized χ  =  χ − θ   (deg)" if NORM_MODE == "simple"
              else "normalized χ  =  χ − (dχ/dθ)·θ   (deg)")


def wrap180(x):
    """Wrap angle(s) in degrees into (−180, 180]."""
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


# %% [markdown]
# ## Load: pool the six scans into one table
# Each shape contributes its orientation `chi_deg` and spatial footprint `n_bins`;
# we attach the experiment label, θ, the θ-normalized χ, and the scan's total grid
# bins (device-area denominator). Uses the app's catalog loader when importable,
# else reads `kept[]` directly so the script runs standalone.

# %%
try:
    from xrd_app.core.catalogs import load_features_any

    def _kept(path):
        kept, _filtered = load_features_any(str(path))
        return kept
except Exception:
    def _kept(path):
        return json.load(open(path)).get("kept", [])


def _device_bins(proj: Path, scan: str) -> int:
    """Total spatial bins in the scan grid = the % -of-device-area denominator."""
    gm = proj / "Metadata" / scan / GRID_MAPPING
    try:
        return int(json.load(open(gm))["n_bins"])
    except Exception as e:
        print(f"  WARN {scan}: no {GRID_MAPPING} ({e}); device area unknown")
        return 0


def load_scan(cfg: dict) -> list:
    proj = Path(cfg["proj"])
    scan = cfg["scan"]
    path = proj / "Labels" / scan / CATALOG_NAME
    if not path.exists():
        print(f"  MISSING {scan}: {path}")
        return []
    total_bins = _device_bins(proj, scan)
    rows = []
    for f in _kept(path):
        refl = norm_refl(f.get("reflection", "?"))
        if refl not in REFLECTIONS_N:
            continue
        chi = f.get("chi_deg", np.nan)
        n_bins = f.get("n_bins", np.nan)
        chi_norm = float(wrap180(chi - dchi_dtheta(refl) * cfg["theta"])) if np.isfinite(chi) else np.nan
        rows.append({
            "scan": scan, "exp": cfg["exp"], "theta": cfg["theta"], "samy": cfg["samy"],
            "reflection": refl, "n_bins": n_bins, "chi": chi, "chi_norm": chi_norm,
            "total_bins": total_bins,
            "label": f"{cfg['exp']} θ{cfg['theta']:g} ({scan.split('_')[-1]})",
        })
    print(f"  {scan}: {len(rows):>4d} shapes in {REFLECTIONS_N}  "
          f"(device={total_bins} bins)  <- {path.name}")
    return rows


print("Loading six focus scans:")
ROWS = []
for _cfg in SCAN_CONFIG:
    ROWS.extend(load_scan(_cfg))
if not ROWS:
    raise SystemExit("No shapes loaded — check the SCAN_CONFIG paths.")

SCAN_KEYS = [c["scan"] for c in SCAN_CONFIG]
SCAN_META = {c["scan"]: c for c in SCAN_CONFIG}
COLOR_OF = {c["scan"]: SCAN_COLORS[i % len(SCAN_COLORS)] for i, c in enumerate(SCAN_CONFIG)}


def _rows_for(scan, refl):
    return [r for r in ROWS if r["scan"] == scan and r["reflection"] == refl
            and np.isfinite(r["chi_norm"]) and np.isfinite(r["n_bins"])]


# %% [markdown]
# ## Circular framing (shared with the app's Orientation Map)
# χ is periodic, so a fixed −180…180 axis tears a cluster straddling the wrap into
# two clumps. Frame on the pooled data's minimal enclosing arc instead.

# %%
def _circular_frame(chis):
    """Return (wrap_fn, lo, hi): map χ through wrap_fn, then bin/plot over [lo, hi]."""
    v = np.sort(np.asarray(chis, float))
    v = v[np.isfinite(v)]
    if v.size < 2:
        c = float(v[0]) if v.size else 0.0
        return (lambda x: x), c, c
    gaps = np.diff(v)
    wrap_gap = 360.0 - (v[-1] - v[0])
    if wrap_gap >= gaps.max():
        return (lambda x: x), float(v[0]), float(v[-1])
    k = int(np.argmax(gaps))
    thr = (v[k] + v[k + 1]) / 2.0
    return (lambda x: x + 360.0 if x < thr else x), float(v[k + 1]), float(v[k] + 360.0)


def _edges(active_scans, bin_width):
    """Shared χ_norm bin edges framed on all active scans' pooled orientations."""
    pooled = [r["chi_norm"] for r in ROWS if r["scan"] in active_scans
              and np.isfinite(r["chi_norm"])]
    if not pooled:
        return None, None
    wrap, lo, hi = _circular_frame(pooled)
    edges = np.arange(lo - bin_width, hi + 2 * bin_width, bin_width)
    return wrap, edges


def _area_fraction(scan, refl, wrap, edges):
    """Fraction of the scan's device area covered per χ slice (Σ n_bins ÷ total)."""
    rr = _rows_for(scan, refl)
    total_bins = rr[0]["total_bins"] if rr else 0
    if not rr or not total_bins:
        return np.zeros(len(edges) - 1)
    chi = np.array([wrap(r["chi_norm"]) for r in rr], float)
    w = np.array([r["n_bins"] for r in rr], float)
    counts, _ = np.histogram(chi, bins=edges, weights=w)
    return counts / total_bins


def dominant_slice(scan, refl, wrap, edges):
    """(χ_center_deg, area_fraction) of the fullest χ slice for a scan×reflection."""
    frac = _area_fraction(scan, refl, wrap, edges)
    if frac.size == 0 or frac.max() <= 0:
        return None, 0.0
    k = int(np.argmax(frac))
    center = wrap180((edges[k] + edges[k + 1]) / 2.0)
    return float(center), float(frac[k])


# %% [markdown]
# ## Interactive window: histogram + connected table + bin-width slider
# Toggle scans, drag the slider; both the χ-area histogram and the dominant-slice
# table redraw together. A snapshot is saved to `report_figures/`.

# %%
from matplotlib.widgets import CheckButtons, Slider
from matplotlib.ticker import FuncFormatter

active_scans = set(SCAN_KEYS)
bin_width = float(CHI_BIN_WIDTH)

fig = plt.figure(figsize=(14, 8.5))
ax_hist = fig.add_axes([0.07, 0.56, 0.68, 0.38])
ax_table = fig.add_axes([0.07, 0.05, 0.68, 0.40]); ax_table.axis("off")
ax_check = fig.add_axes([0.79, 0.56, 0.19, 0.38]); ax_check.set_title("scans", fontsize=9)
ax_slider = fig.add_axes([0.81, 0.30, 0.15, 0.03])

check = CheckButtons(ax_check, [SCAN_META[s]["scan"].split("_")[-1] + "  " + SCAN_META[s]["exp"]
                                for s in SCAN_KEYS], [True] * len(SCAN_KEYS))
slider = Slider(ax_slider, "χ bin (°)", 1, 30, valinit=CHI_BIN_WIDTH, valstep=1)


def draw_histogram(wrap, edges):
    ax_hist.clear()
    ax_hist.set_xlabel(NORM_LABEL)
    ax_hist.set_ylabel("fraction of device area")
    ax_hist.grid(True, axis="y", alpha=0.25)
    title = ", ".join(sorted(s.split("_")[-1] for s in active_scans)) or "none"
    ax_hist.set_title(f"θ-normalized χ distribution — scans {title}")
    if edges is None:
        ax_hist.text(0.5, 0.5, "No shapes for the selected scans",
                     transform=ax_hist.transAxes, ha="center", va="center")
        return
    centers = (edges[:-1] + edges[1:]) / 2.0
    ymax = 0.0
    for scan in SCAN_KEYS:
        if scan not in active_scans:
            continue
        for refl in REFLECTIONS_N:
            frac = _area_fraction(scan, refl, wrap, edges)
            if frac.max() <= 0:
                continue
            ymax = max(ymax, frac.max())
            m = SCAN_META[scan]
            ax_hist.step(centers, frac, where="mid",
                         color=COLOR_OF[scan], linestyle=REFL_STYLE.get(refl, "-"),
                         linewidth=1.8,
                         label=f"{m['exp']} θ{m['theta']:g} {refl}")
    ax_hist.set_xlim(edges[0], edges[-1])
    ax_hist.set_ylim(0, max(ymax * 1.15, 0.02))
    # label ticks as wrapped χ so the framed (possibly >180) axis reads naturally
    ax_hist.xaxis.set_major_formatter(FuncFormatter(lambda t, _p: f"{wrap180(t):.0f}"))
    ax_hist.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)


def draw_table(wrap, edges):
    ax_table.clear(); ax_table.axis("off")
    cols = ["independent var", "scan", "θ", "reflection",
            "dominant χ (°)", "% device area"]
    body = []
    if edges is not None:
        for scan in active_scans:
            m = SCAN_META[scan]
            for refl in REFLECTIONS_N:
                chi_c, frac = dominant_slice(scan, refl, wrap, edges)
                if chi_c is None:
                    continue
                body.append([m["exp"], scan.split("_")[-1], f"{m['theta']:g}", refl,
                             f"{chi_c:+.1f}", f"{frac * 100:.2f}%", frac])
    body.sort(key=lambda r: r[-1], reverse=True)          # rank by area of bins
    body = [r[:-1] for r in body]
    if not body:
        ax_table.text(0.5, 0.5, "No shapes for the selected scans",
                      ha="center", va="center", transform=ax_table.transAxes)
        return
    tbl = ax_table.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    for j in range(len(cols)):                             # header styling
        tbl[0, j].set_facecolor("#eeeeee")
        tbl[0, j].set_text_props(weight="bold")
    for i, r in enumerate(body, start=1):                 # tint by experiment colour
        for j, scan in enumerate(SCAN_KEYS):
            if r[1] == scan.split("_")[-1]:
                c = COLOR_OF[scan]
                tbl[i, 1].set_text_props(color=c)
    ax_table.set_title(f"Dominant χ slice per scan×reflection  "
                       f"(χ bin = {bin_width:g}°, ranked by device area)",
                       fontsize=10, pad=10)


def redraw():
    wrap, edges = _edges(active_scans, bin_width)
    draw_histogram(wrap, edges)
    draw_table(wrap, edges)
    fig.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
    fig.canvas.draw_idle()


def _on_check(label):
    scan = "Scan_" + label.split()[0]
    active_scans.symmetric_difference_update({scan})
    redraw()


def _on_slide(val):
    global bin_width
    bin_width = float(val)
    redraw()


check.on_clicked(_on_check)
slider.on_changed(_on_slide)
redraw()

import matplotlib as _mpl
if _mpl.get_backend().lower().endswith("agg"):
    print(f"\nNOTE: non-interactive backend ({_mpl.get_backend()}); widgets won't "
          f"respond. Saved a snapshot to {FIG_PATH} instead.")
else:
    print(f"\nInteractive window open. Snapshot mirrored to {FIG_PATH}.")
plt.show()
