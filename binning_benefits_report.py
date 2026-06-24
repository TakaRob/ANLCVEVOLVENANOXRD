# %% [markdown]
'''
# The benefits of spatial binning for nano-XRD peak detection

**Scan 0203 · ISN 26-ID-C · perovskite/halide thin film**

A *binned* frame is the **sum of N×N neighbouring scan-position frames**
(`xrd_app/core/io.build_bins`). Summing raises the coherent Bragg **signal ~N²**
while shot noise grows only **~N**, so the per-bin **SNR scales ~N**. Weak peaks
that sit *below* the detector's SNR threshold at 1×1 should cross it once binned —
raising recall (the quantity this project chases; see `CLAUDE.md`'s mean-F2 note).
The price is **spatial resolution** (the device map gets ×N coarser, small grains
merge) and a different **runtime** profile.

## Hypothesis

> Increasing bin size raises per-bin Bragg SNR ~linearly, increasing the number of
> detected peaks/features (recall) — **until** resolution loss and peak merging cause
> diminishing returns, predicting an **intermediate sweet spot**.

Sub-hypotheses, each tested by a section below:

| # | Claim | Section |
|---|---|---|
| H1 | in-band signal ∝ ~N², background σ ∝ ~N ⇒ SNR ∝ ~N | C |
| H2 | more peaks/features found at higher N, driven by weak near-threshold peaks | D, E |
| H3 | radial-median background subtraction leaves lower *relative* residual σ at higher N | C |
| H4 | device-map effective pitch degrades ×N; features span fewer bins | F |
| H5 | detection time ∝ n_bins ∝ 1/N²; bin-build cost ~constant | G |

## How to run

Run cell by cell (VSCode "Run Cell" / Jupyter). Each `# %%` is a cell; each
triple-quoted block is a markdown note. **Edit two things and re-run:**

1. The **CONFIG** cell — point `PROJECT_ROOT` at your project (default: TakaProject).
2. The **ALGORITHM REGISTRY** cell — pick the peak/shape (or combined) algorithm per
   bin size, exactly like choosing one in the GUI. Change a name, re-run the
   *"Ensure data"* cell, and every figure downstream updates from the new results.

The engine is the **`xrd-app` CLI** (matches the "CLI is the engine" rule). Data
is generated *into the project* and cached — the first run is heavy (1×1 detects
over every scan position), reruns are fast.
'''

# %%
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ──────────────────────────────────────────────────────────
BASE = Path("/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo")
if not BASE.exists():
    BASE = Path.cwd()

# Default project + scan. To analyse a different project, create it once and
# point PROJECT_ROOT at it, e.g. (run in a terminal, then edit the path below):
#   !xrd-app init --name MyProject --scan-number 203 --root /path/to/MyProject
#   !xrd-app scan-detect --scans-dir <raw dir> --root /path/to/MyProject
#   !xrd-app link --tth tth.tiff --reflections reflections.py --root /path/to/MyProject
PROJECT_ROOT = BASE / "TakaTest" / "TakaProject"
SCAN = "Scan_0203"
BIN_SIZES = [1, 2, 3, 4, 5]      # the binning sweep
FORCE = False                    # True = rebuild/redetect even if artifacts exist
SAVE_FIGS = True                 # also write PNGs into <project>/Figures/
# ─────────────────────────────────────────────────────────────────────

# Make xrd_app importable for the read/plot helpers (CLI does the heavy lifting).
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from xrd_app.config import DataManager          # noqa: E402
from xrd_app.core import io as xio              # noqa: E402

DM = DataManager(str(PROJECT_ROOT), scan=SCAN)
FIG_DIR = DM.figures_dir
FIG_DIR.mkdir(parents=True, exist_ok=True)

# `xrd-app` if on PATH, else run the module directly.
XRD = ["xrd-app"] if shutil.which("xrd-app") else [sys.executable, "-m", "xrd_app.cli"]

print(f"Project : {PROJECT_ROOT}")
print(f"Scan    : {SCAN}")
print(f"Bins    : {BIN_SIZES}")
print(f"Engine  : {' '.join(XRD)}")
print(f"Figures : {FIG_DIR}")
print(f"tth map : {DM.tth_map()}  exists={Path(DM.tth_map()).exists()}")


def show(fig, name):
    """Display a figure and (optionally) save it to the project's Figures/ dir."""
    if SAVE_FIGS:
        fig.savefig(FIG_DIR / name, dpi=140, bbox_inches="tight")
    plt.show()


def run_xrd(args, label):
    """Run an `xrd-app` subcommand, timed. Returns (seconds, ok, tail_of_output)."""
    cmd = XRD + args + ["--root", str(PROJECT_ROOT)]
    print(f"  $ {' '.join(str(a) for a in cmd[len(XRD):])}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-3:])
    print(f"    [{label}] {'OK' if ok else 'FAILED'} in {dt:.1f}s — {tail}")
    return dt, ok, out


# %% [markdown]
'''
## Algorithm registry — the GUI-style selector

One row per bin size. This is where you choose, per bin, *which* algorithm runs —
the notebook equivalent of the GUI's algorithm dropdown. Re-run the **Ensure data**
cell after editing.

- `kind="peak"` → two-stage: `peaks` (detector) then `shapes` (cross-bin link +
  gaussian filter). `detector` is a bundled peak-algorithm name, `shape` a shape
  algorithm (`gaussian`).
- `kind="combined"` → one per-frame pass (`run-combined`); `detector` is a combined
  algorithm name. Produces features only (no per-bin peak list).

List what's available with `!xrd-app detectors --kind peak` (or `--kind shape` /
`--kind combined`). Shape options include `gaussian` (default) and **`gaussian_deskew`**
— a skew-aware linker that corrects each scan's serpentine column backlash to reduce the
1×1 horizontal-slice fragmentation (set `shape="gaussian_deskew"` below). If accuracy
looks low for a new bin size (e.g. 2×2, 4×4), freeze a tuned variant and use its name here:

    !xrd-app save-algorithm --base 5x5_tophat_band_adaptive_snr \\
        --sensitivity 3.0 --bin-size 2 --name tophat_b2 --root <project>
'''

# %%
# Edit these. `detector` must be a name from `xrd-app detectors`.
_DEFAULT_PEAK = "5x5_tophat_band_adaptive_snr"   # best general bundled detector
ALGOS = {
    1: dict(kind="peak", detector=_DEFAULT_PEAK, snr=4.0, shape="gaussian", link=5),
    2: dict(kind="peak", detector=_DEFAULT_PEAK, snr=4.0, shape="gaussian", link=5),
    3: dict(kind="peak", detector=_DEFAULT_PEAK, snr=4.0, shape="gaussian", link=5),
    4: dict(kind="peak", detector=_DEFAULT_PEAK, snr=4.0, shape="gaussian", link=5),
    5: dict(kind="peak", detector=_DEFAULT_PEAK, snr=4.0, shape="gaussian", link=5),
}

print("Bin   kind      detector / snr / shape")
for bs in BIN_SIZES:
    a = ALGOS[bs]
    if a["kind"] == "peak":
        print(f"{bs}x{bs}  peak      {a['detector']}  snr={a['snr']}  shape={a['shape']}")
    else:
        print(f"{bs}x{bs}  combined  {a['detector']}")


# %% [markdown]
'''
## Ensure data — build bins + run detection per bin size (timed)

For every bin size this builds the binned HDF5 (if missing) and runs the selected
algorithm, recording wall-clock time for the **runtime** section. Artifacts land in
`<project>/Binned/` and `<project>/Labels/Scan_0203/`. Set `FORCE=True` above to
regenerate. **1×1 is the slow one** (detection over every scan position).
'''

# %%
def peaks_file(bs):
    return DM.peaks_json(ALGOS[bs]["detector"], bs)


def catalog_file(bs):
    return DM.labels_dir() / f"feature_catalog_{bs}x{bs}.json"


TIMINGS = {bs: dict(build=0.0, peaks=0.0, shapes=0.0) for bs in BIN_SIZES}

for bs in BIN_SIZES:
    a = ALGOS[bs]
    print(f"\n=== {bs}x{bs} ({a['kind']}) ===")

    # 1) bins (grid + binned h5)
    h5 = DM.binned_h5(bs)
    gm = DM.grid_mapping(bin_size=bs)
    if FORCE or not Path(h5).exists() or not Path(gm).exists():
        dt, ok, _ = run_xrd(["make-bins", "--scan", SCAN, "--bin-size", str(bs)], "build")
        TIMINGS[bs]["build"] = dt
    else:
        print(f"  bins exist: {Path(h5).name} (skip)")

    # 2) detection
    if a["kind"] == "peak":
        pj = peaks_file(bs)
        if FORCE or not Path(pj).exists():
            dt, ok, _ = run_xrd(["peaks", "--scan", SCAN, "--bin-size", str(bs),
                                 "--algorithm", a["detector"], "--snr", str(a["snr"])], "peaks")
            TIMINGS[bs]["peaks"] = dt
        else:
            print(f"  peaks exist: {Path(pj).name} (skip)")

        sj = DM.shapes_json(a["shape"], bs)
        if FORCE or not Path(sj).exists():
            dt, ok, _ = run_xrd(["shapes", "--scan", SCAN, "--bin-size", str(bs),
                                 "--algorithm", a["shape"], "--peak-algo", a["detector"],
                                 "--link-tolerance", str(a["link"])], "shapes")
            TIMINGS[bs]["shapes"] = dt
        else:
            print(f"  shapes exist: {Path(sj).name} (skip)")
    else:  # combined
        cj = catalog_file(bs)
        if FORCE or not Path(cj).exists():
            dt, ok, _ = run_xrd(["run-combined", "--scan", SCAN, "--bin-size", str(bs),
                                 "--algorithm", a["detector"]], "combined")
            TIMINGS[bs]["peaks"] = dt
        else:
            print(f"  combined exists: {Path(cj).name} (skip)")

print("\nDone. Timings (s):")
for bs in BIN_SIZES:
    t = TIMINGS[bs]
    print(f"  {bs}x{bs}: build={t['build']:.1f}  detect={t['peaks']:.1f}  shapes={t['shapes']:.1f}")


# %% [markdown]
'''
## Load artifacts

Pull the per-bin peaks and the kept features (shapes) into memory, plus the 2θ map
and reflection bands used for every physics overlay.
'''

# %%
TTH = xio.load_tth_map(DM.tth_map())
DEGS, DEG_LABELS = xio.load_reflections(DM.reflections())
# unique reflection bands (label -> 2θ)
BANDS = {}
for lbl, deg in zip(DEG_LABELS, DEGS):
    BANDS.setdefault(lbl, float(deg))
print(f"tth map: {TTH.shape}  range {TTH.min():.2f}–{TTH.max():.2f}°")
print(f"reflections: {', '.join(f'{k}@{v:.2f}' for k, v in BANDS.items())}")

DATA = {}   # bs -> dict(peaks_by_bin, features, n_rows, n_cols, n_bins, kind)
for bs in BIN_SIZES:
    d = dict(kind=ALGOS[bs]["kind"], peaks_by_bin={}, features=[], n_rows=0, n_cols=0, n_bins=0)
    gm = DM.grid_mapping(bin_size=bs)
    if Path(gm).exists():
        g = json.load(open(gm))
        d["n_rows"], d["n_cols"] = g.get("n_bin_rows", 0), g.get("n_bin_cols", 0)
        d["n_bins"] = len(g.get("bins", {}))
    pj = peaks_file(bs)
    if Path(pj).exists():
        d["peaks_by_bin"] = json.load(open(pj)).get("peaks_by_bin", {})
    # Authoritative kept features = the shapes JSON's "kept" list. The legacy
    # feature_catalog_*.json can be stale/overwritten by other tools, so only use
    # it as a fallback (and for combined algos, which write only the catalog).
    sj = DM.shapes_json(ALGOS[bs]["shape"], bs) if ALGOS[bs]["kind"] == "peak" else None
    if sj and Path(sj).exists():
        d["features"] = json.load(open(sj)).get("kept", [])
    elif Path(catalog_file(bs)).exists():
        d["features"] = json.load(open(catalog_file(bs)))
    DATA[bs] = d
    npk = sum(len(v) for v in d["peaks_by_bin"].values())
    print(f"  {bs}x{bs}: grid {d['n_rows']}x{d['n_cols']} "
          f"({d['n_bins']} occupied bins) · {npk} peaks · {len(d['features'])} features")

# Bin sizes that actually have results, for the analysis cells.
READY = [bs for bs in BIN_SIZES if DATA[bs]["features"] or DATA[bs]["peaks_by_bin"]]


# %% [markdown]
'''
## A — The raw picture

Left: the full detector, summed over all bins. **"Counts" = X-ray photon counts per
detector pixel**, accumulated over every frame in the scan (raw intensity, no
normalisation) — this is where signal *should* live, so the reflection bands are
ringed. Right: the brightest single bin's detector patch at each bin size. A bin sums
N×N neighbouring frames, so its **counts scale ~N²** — that growth is exactly the SNR
lever binning gives us (coherent signal ∝ N², shot noise ∝ N ⇒ SNR ∝ N).
'''

# %%
# Summing every bin = summing every frame, so the global image is ~the same for any
# bin size. Use the coarsest available (fewest datasets) for speed.
_src_bs = next((bs for bs in [5, 4, 3, 2, 1] if Path(DM.binned_h5(bs)).exists()
                or bs in READY), READY[0] if READY else 3)
_cache = FIG_DIR / "summed_detector.npy"
if _cache.exists() and not FORCE:
    summed = np.load(_cache)
else:
    src = xio.open_bin_source(DM, _src_bs, scan=SCAN)
    summed = src.sum_all()
    src.close()
    np.save(_cache, summed)
print(f"summed detector image: {summed.shape} (from {_src_bs}x{_src_bs} source)")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ax = axes[0]
vmax = np.percentile(summed, 99.5)
ax.imshow(summed, cmap="inferno", vmax=vmax, origin="upper")
# ring the reflection bands as iso-2θ contours (only bands within the detector's 2θ range)
_levels = sorted(v for v in BANDS.values() if TTH.min() < v < TTH.max())
if _levels:
    ax.contour(TTH, levels=_levels, colors="cyan", linewidths=0.5, alpha=0.6)
ax.set_title(f"Summed detector (all bins) — reflection bands ringed\n{SCAN}")
ax.set_xlabel("detector_x (px)"); ax.set_ylabel("detector_y (px)")

# same-patch zoom at a few bin sizes: pick the brightest pixel as the patch centre
yc, xc = np.unravel_index(np.argmax(summed), summed.shape)
ax = axes[1]
sample_sizes = [bs for bs in [1, 3, 5] if bs in READY] or READY[:3]
half = 40
for bs in sample_sizes:
    src = xio.open_bin_source(DM, bs, scan=SCAN)
    # the bin containing (yc,xc) in scan space is unknown; instead show the
    # detector-patch brightness from that bin grid's brightest bin image.
    keys = src.keys()
    # brightest bin by integrated counts in the patch (sample up to 200 bins)
    best, bestval = None, -1
    for k in keys[:: max(1, len(keys) // 200)]:
        im = src.image(k)
        if im is None:
            continue
        v = im[max(0, yc - half):yc + half, max(0, xc - half):xc + half].sum()
        if v > bestval:
            best, bestval = im, v
    if best is not None:
        prof = best[max(0, yc - half):yc + half, max(0, xc - half):xc + half].sum(axis=0)
        ax.plot(prof, label=f"{bs}x{bs} (peak={best.max():.0f})")
    src.close()
ax.set_title("Brightest bin's detector patch — counts grow ~N² with binning")
ax.set_xlabel("column offset in patch (px)")
ax.set_ylabel("detector counts (photons, summed over the bin's frames)")
ax.legend()
show(fig, "A_raw_picture.png")


# %% [markdown]
'''
## B — Intensity over the radial coordinate (2θ)

The azimuthally-averaged radial profile shows where diffracted intensity sits in 2θ;
the dashed lines are the known reflection angles. The **physics check** (per
`CLAUDE.md`): detected peaks must fall *in* these bands — points off-band are a bug,
not a discovery. The lower panel histograms each bin level's detected-peak 2θ to
confirm they land on the bands.
'''

# %%
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
centers, profile = xio.radial_profile(summed, TTH, n_bins=600)
ax = axes[0]
ax.semilogy(centers, np.maximum(profile, 1e-3), color="#222", lw=1.2)
for lbl, deg in BANDS.items():
    ax.axvline(deg, color="crimson", ls="--", lw=0.8, alpha=0.7)
    ax.text(deg, ax.get_ylim()[1], lbl, rotation=90, va="top", ha="right",
            fontsize=8, color="crimson")
ax.set_ylabel("mean intensity (log)")
ax.set_title("Azimuthally-averaged radial profile (whole detector) + reflection bands")

ax = axes[1]
for bs in READY:
    tths = []
    for peaks in DATA[bs]["peaks_by_bin"].values():
        for p in peaks:
            x, y = int(p.get("x", -1)), int(p.get("y", -1))
            if 0 <= y < TTH.shape[0] and 0 <= x < TTH.shape[1]:
                tths.append(TTH[y, x])
    if not tths and DATA[bs]["features"]:   # combined: use feature positions
        for f in DATA[bs]["features"]:
            x, y = int(f.get("detector_x", -1)), int(f.get("detector_y", -1))
            if 0 <= y < TTH.shape[0] and 0 <= x < TTH.shape[1]:
                tths.append(TTH[y, x])
    if tths:
        ax.hist(tths, bins=120, histtype="step", lw=1.4, label=f"{bs}x{bs} (n={len(tths)})")
for deg in BANDS.values():
    ax.axvline(deg, color="crimson", ls="--", lw=0.6, alpha=0.5)
ax.set_xlabel("2θ (°)"); ax.set_ylabel("detected peaks")
ax.set_title("Detected-peak 2θ distribution per bin size (should pile on the bands)")
ax.legend()
show(fig, "B_radial_intensity.png")


# %% [markdown]
'''
## C — Noise reduction & signal/noise scaling (H1, H3)

For a sample of bins at each level we apply a radial-median background subtraction
(the same idea the detectors use) and measure, **inside the reflection bands**:

- **signal** — the 99.5th-percentile cleaned intensity (the bright Bragg pixels),
- **noise σ** — robust background spread `1.4826·MAD` of the cleaned in-band pixels.

If binning works as theorised, signal climbs ~N², noise ~N, and **SNR = signal/σ ~N**.
Curves are normalised to the 1×1 (or smallest available) point and compared to the
N and N² references.

> **Caveat:** this samples *random* occupied bins, most of which are background — so it
> largely measures how the *background* scales (≈N), and the apparent SNR can look flat.
> A grain smaller than the bin doesn't fill it, so its signal grows slower than N² while
> noise still grows ~N. For the SNR gain *where features actually are*, see **Section E2**.
'''

# %%
def _radial_median_subtract(image, tth_map, bin_width=0.05):
    """Self-contained radial-median background subtraction (detector-agnostic)."""
    flat_t = tth_map.ravel()
    edges = np.arange(float(tth_map.min()), float(tth_map.max()) + bin_width, bin_width)
    idx = np.clip(np.digitize(flat_t, edges) - 1, 0, len(edges) - 2)
    flat_i = image.ravel()
    med = np.zeros(len(edges) - 1)
    order = np.argsort(idx)
    si, sv = idx[order], flat_i[order]
    bnd = np.searchsorted(si, np.arange(len(med) + 1))
    for i in range(len(med)):
        lo, hi = int(bnd[i]), int(bnd[i + 1])
        if hi > lo:
            med[i] = np.median(sv[lo:hi])
    from scipy.ndimage import uniform_filter1d
    med = uniform_filter1d(med, size=min(15, len(med)))
    return image - med[idx].reshape(image.shape)


# in-band pixel mask (any reflection band, ±0.4°)
band_mask = np.zeros(TTH.shape, dtype=bool)
for deg in BANDS.values():
    band_mask |= np.abs(TTH - deg) <= 0.4

SAMPLE = 60   # bins sampled per level (keeps it fast; 1×1 raw is slow per image)
sig_by_bs, noise_by_bs = {}, {}
for bs in READY:
    src = xio.open_bin_source(DM, bs, scan=SCAN)
    keys = src.keys()
    step = max(1, len(keys) // SAMPLE)
    sigs, noises = [], []
    for k in keys[::step][:SAMPLE]:
        im = src.image(k)
        if im is None:
            continue
        cleaned = _radial_median_subtract(im, TTH)
        vals = cleaned[band_mask]
        if vals.size < 100:
            continue
        sigs.append(np.percentile(vals, 99.5))
        mad = np.median(np.abs(vals - np.median(vals)))
        noises.append(1.4826 * mad if mad > 0 else vals.std())
    src.close()
    sig_by_bs[bs] = float(np.median(sigs)) if sigs else np.nan
    noise_by_bs[bs] = float(np.median(noises)) if noises else np.nan
    print(f"  {bs}x{bs}: signal≈{sig_by_bs[bs]:.1f}  noiseσ≈{noise_by_bs[bs]:.2f}  "
          f"SNR≈{sig_by_bs[bs] / noise_by_bs[bs]:.1f}  (n={len(sigs)} bins)")

ns = np.array(READY, float)
n0 = ns.min()
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sig = np.array([sig_by_bs[b] for b in READY])
noi = np.array([noise_by_bs[b] for b in READY])
snr = sig / noi
for ax, y, name, ref in [
    (axes[0], sig / sig[0], "signal", (ns / n0) ** 2),
    (axes[1], noi / noi[0], "noise σ", ns / n0),
    (axes[2], snr / snr[0], "SNR", ns / n0),
]:
    ax.plot(ns, y, "o-", lw=2, label=name + " (measured)")
    ax.plot(ns, ref, "k--", lw=1, label="∝N² ref" if name == "signal" else "∝N ref")
    ax.set_xlabel("bin size N"); ax.set_ylabel(f"{name} / {name}(N={int(n0)})")
    ax.set_title(name + " vs bin size"); ax.legend()
fig.suptitle("H1/H3 — signal ~N², noise ~N ⇒ SNR ~N (normalised to smallest bin)")
show(fig, "C_signal_noise_scaling.png")


# %% [markdown]
'''
## D — How many peaks / features do we find? (H2)

The headline result. **Total raw peaks** and **kept features** (peaks that survive
cross-bin linking + the gaussian-profile filter) versus bin size. Higher binning lifts
weak peaks over threshold, but coarser binning also means fewer bins — the net is the
empirical question. The per-reflection breakdown shows *where* the gains land.
'''

# %%
tot_peaks = {bs: sum(len(v) for v in DATA[bs]["peaks_by_bin"].values()) for bs in READY}
tot_feats = {bs: len(DATA[bs]["features"]) for bs in READY}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ax = axes[0]
ax.plot(READY, [tot_peaks[b] for b in READY], "o-", label="raw peaks")
ax.plot(READY, [tot_feats[b] for b in READY], "s-", label="kept features")
ax.set_xlabel("bin size N"); ax.set_ylabel("count")
ax.set_title("Total peaks & features found"); ax.legend()

ax = axes[1]
perbin = [tot_peaks[b] / max(1, DATA[b]["n_bins"]) for b in READY]
ax.plot(READY, perbin, "o-", color="#8c564b")
ax.set_xlabel("bin size N"); ax.set_ylabel("peaks per occupied bin")
ax.set_title("Peak density per bin (signal concentration)")

ax = axes[2]
refs = sorted({f["reflection"] for b in READY for f in DATA[b]["features"]})
width = 0.8 / max(1, len(READY))
for i, bs in enumerate(READY):
    counts = [sum(1 for f in DATA[bs]["features"] if f["reflection"] == r) for r in refs]
    ax.bar(np.arange(len(refs)) + i * width, counts, width, label=f"{bs}x{bs}")
ax.set_xticks(np.arange(len(refs)) + 0.4)
ax.set_xticklabels(refs, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("features"); ax.set_title("Features per reflection, by bin size"); ax.legend(fontsize=8)
show(fig, "D_peaks_found.png")


# %% [markdown]
'''
## D2 — Feature counts by reflection (small multiples + stacked)

Left: one panel per reflection, feature count vs bin size — so you can see which
reflections drive the bin-size trend. Right: the same numbers as a **stacked bar**,
one bar per bin size split by reflection, showing total and composition together.
'''

# %%
refs_all = sorted({f["reflection"] for b in READY for f in DATA[b]["features"]})
fcount = {r: [sum(1 for f in DATA[b]["features"] if f["reflection"] == r) for b in READY]
          for r in refs_all}

ncol = min(4, max(1, len(refs_all)))
nrow = int(np.ceil(len(refs_all) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow), squeeze=False)
for i, r in enumerate(refs_all):
    ax = axes[i // ncol][i % ncol]
    ax.bar([str(b) for b in READY], fcount[r], color="#4363d8")
    ax.set_title(r, fontsize=9)
    ax.set_xlabel("bin N", fontsize=8); ax.set_ylabel("features", fontsize=8)
for j in range(len(refs_all), nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
fig.suptitle("Feature count vs bin size — per reflection")
show(fig, "D2_features_by_reflection.png")

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(READY))
cmap = plt.get_cmap("tab20")
for i, r in enumerate(refs_all):
    vals = np.array(fcount[r], float)
    ax.bar([f"{b}x{b}" for b in READY], vals, bottom=bottom, label=r, color=cmap(i % 20))
    bottom += vals
for x, tot in enumerate(bottom):
    ax.text(x, tot, f"{int(tot)}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("bin size"); ax.set_ylabel("features (stacked by reflection)")
ax.set_title("Total features by bin size, stacked by reflection")
ax.legend(ncol=2, fontsize=8)
show(fig, "D2_features_stacked.png")


# %% [markdown]
'''
## D3 — Feature fragmentation at fine binning (the 1×1 "horizontal slices")

A feature that is one clean blob at coarse bins shows up at 1×1 as **horizontal slices
with gaps** — which inflates the 1×1 feature count. We quantify it from each feature's
`spatial_extent` (the scan bins it occupies):

- **row span / col span** of the bounding box,
- **gap fraction** = 1 − n_bins / (row_span × col_span) — holes inside the footprint,
- **single-row fraction** = share of multi-bin features that span exactly one scan row.

**Mechanism.** Linking (`ShapeAlgorithms/gaussian.py:link_peaks`) merges detections in
8-connected neighbour bins **only if** their detector (x,y) agree within `link_tolerance`
(5 px). The scan is a serpentine raster — alternate rows are scanned in opposite
directions, so any stage backlash offsets the registration between adjacent rows. At
1×1 (one frame per bin) two things break cross-row links: (a) weak single-frame signal
makes detection intermittent (gaps), and (b) the single-frame centroid jitters by a few
px and the row-to-row offset exceeds 5 px — so only *within-row* (horizontal) links
survive. Binning sums N² frames, stabilising signal and washing out the offset, so the
blob re-merges. Conclusion: part of the extra 1×1 features is real weak-peak recovery,
part is **over-segmentation** of single physical features.
'''

# %%
def _spans(feat):
    rc = [k.split("_") for k in feat.get("spatial_extent", []) if "_" in k]
    rs = [int(a) for a, _ in rc]; cs = [int(b) for _, b in rc]
    if not rs:
        return 0, 0, 0, 0.0
    rspan, cspan = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    gap = 1 - len(rc) / (rspan * cspan) if rspan * cspan else 0.0
    return len(rc), rspan, cspan, gap

frag_bs = [b for b in READY if DATA[b]["features"]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for b in frag_bs:
    nb = [f["n_bins"] for f in DATA[b]["features"]]
    ax.hist(np.clip(nb, 1, 20), bins=range(1, 22), histtype="step", lw=1.6, label=f"{b}x{b}")
ax.set_xlabel("feature footprint (n_bins)"); ax.set_ylabel("features")
ax.set_title("Footprint size (1×1 skewed to tiny features)"); ax.legend()

ax = axes[1]
srf = []
for b in frag_bs:
    multi = [_spans(f) for f in DATA[b]["features"] if f["n_bins"] >= 2]
    srf.append(float(np.mean([1.0 if m[1] == 1 else 0.0 for m in multi])) if multi else 0.0)
ax.bar([f"{b}x{b}" for b in frag_bs], srf, color="#d62728")
ax.set_ylabel("fraction of multi-bin features spanning 1 row")
ax.set_title("Horizontal-slice fraction (higher = more slicing)")

ax = axes[2]
for b in frag_bs:
    gaps = [_spans(f)[3] for f in DATA[b]["features"] if f["n_bins"] >= 2]
    if gaps:
        ax.hist(gaps, bins=20, histtype="step", lw=1.6, density=True, label=f"{b}x{b}")
ax.set_xlabel("gap fraction inside footprint"); ax.set_ylabel("density")
ax.set_title("Holes in the footprint"); ax.legend()
show(fig, "D3_fragmentation.png")
print("single-row fraction by bin:", {b: round(s, 2) for b, s in zip(frag_bs, srf)})

# Footprint map: each cluster of points is one feature, for the dominant reflection.
_refs = [f["reflection"] for b in frag_bs for f in DATA[b]["features"]]
dom_ref = max(set(_refs), key=_refs.count) if _refs else None
comp = [b for b in [1, 3] if b in frag_bs] or frag_bs[:2]
if dom_ref and len(comp) >= 2:
    fig, axes = plt.subplots(1, len(comp), figsize=(6.5 * len(comp), 6), squeeze=False)
    for ax, b in zip(axes[0], comp):
        feats = [f for f in DATA[b]["features"] if f["reflection"] == dom_ref]
        for i, f in enumerate(feats):
            rc = [k.split("_") for k in f.get("spatial_extent", []) if "_" in k]
            rr = [int(a) for a, _ in rc]; cc = [int(x) for _, x in rc]
            ax.scatter(cc, rr, s=10, color=plt.get_cmap("tab20")(i % 20))
        ax.set_title(f"{b}x{b}: {len(feats)} '{dom_ref}' features")
        ax.set_xlabel("col (scan x)"); ax.set_ylabel("row (scan y)"); ax.invert_yaxis()
    fig.suptitle(f"Feature footprints ('{dom_ref}') — each colour is one feature; "
                 f"1×1 shows horizontal slices")
    show(fig, "D3_footprint_map.png")


# %% [markdown]
'''
## D4 — Deskewed linking: does fixing registration merge the slices?

The slicing comes from `link_peaks` joining bins by their **inferred (row, col) lattice**
(`ShapeAlgorithms/gaussian.py`), which misregisters across serpentine rows. Here we
re-link the *same* peaks but by each bin's **true physical (X, Y) stage position**
(from the position CSV) — a 2-D deskew. Two detections link if their bins are within
~1.6× the natural scan pitch **and** their detector (x, y) agree within 5 px. We compare
the single-row ("slice") fraction and kept-feature count, standard vs deskewed, for
1×1 / 2×2 / 3×3.

Finding (this dataset): deskew cuts the **1×1** slice fraction sharply (≈53%→33%) and
merges slices into fewer features — it recovers the ~38% of detections whose same-grain
partner sat one row away but outside the lattice window. For **2×2 and 3×3 it gives no
benefit** (≈neutral to slightly worse): binning has already averaged out the serpentine
backlash, so the lattice linking is already correct and re-linking by *approximate* mean
position can't beat it. **So deskew is a 1×1-specific remedy** — from 2×2 up, just bin.
What deskew *cannot* fix at any level is the ~45% of positions with no detection at all
(SNR dropouts) — those need binning or a lower threshold.
'''

# %%
import csv as _csv
from scipy.spatial import cKDTree

# physical (X, Y) per frame, once
_X, _Y = [], []
with open(DM.position_csv()) as _f:
    for _r in _csv.DictReader(_f):
        _X.append(float(_r["X_Position"])); _Y.append(float(_r["Y_Position"]))
_X, _Y = np.array(_X), np.array(_Y)
_REF = {l: round(d, 5) for d, l in zip(DEG_LABELS, DEGS)}

def _bin_pos(bs):
    gm = json.load(open(DM.grid_mapping(bin_size=bs)))["bins"]
    pos = {}
    for bk, mem in gm.items():
        idx = [m for m in mem if m < len(_X)]
        if idx:
            pos[bk] = (float(np.mean(_X[idx])), float(np.mean(_Y[idx])))
    return pos

def _link_deskew(pbb, pos, link_tol=5, space_mult=1.6):
    """Re-link peaks by true physical (X,Y) position instead of the inferred (row,col)
    lattice. Spatial neighbours = bins within space_mult × the natural scan pitch
    (median nearest-neighbour spacing); they link if their detector (x,y) also agree
    within link_tol px. This is a deliberately simple, robust prototype — a production
    deskew would calibrate per-axis and correct the serpentine backlash explicitly."""
    from collections import defaultdict
    keys = [k for k in pbb if k in pos]
    P = np.array([pos[k] for k in keys])
    tree = cKDTree(P)
    d, _ = tree.query(P, k=2)
    space_tol = space_mult * float(np.median(d[:, 1]))   # 1.6 × natural pitch
    nodes, nbk = [], defaultdict(list)
    for ki, k in enumerate(keys):
        r, c = map(int, k.split("_"))
        for pi, p in enumerate(pbb[k]):
            nbk[ki].append(len(nodes)); nodes.append((k, pi, r, c, p["x"], p["y"], p))
    par = list(range(len(nodes)))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    for ki, kj in tree.query_pairs(space_tol):
        for a in nbk[ki]:
            xa, ya = nodes[a][4], nodes[a][5]
            for b in nbk[kj]:
                if (xa - nodes[b][4]) ** 2 + (ya - nodes[b][5]) ** 2 <= link_tol * link_tol:
                    union(a, b)
    comp = defaultdict(list)
    for i in range(len(nodes)):
        comp[find(i)].append(i)
    return [[nodes[i] for i in idxs] for idxs in comp.values()]

def _onerow(kept):
    m = []
    for ftr in kept:
        rs = [int(k.split("_")[0]) for k in ftr.get("spatial_extent", []) if "_" in k]
        if len(rs) >= 2:
            m.append(1.0 if max(rs) - min(rs) == 0 else 0.0)
    return 100 * float(np.mean(m)) if m else 0.0

from xrd_app.ShapeAlgorithms import gaussian as _G
desk_bs = [b for b in [1, 2, 3] if b in READY]
rows = []
for bs in desk_bs:
    pbb = DATA[bs]["peaks_by_bin"]
    if not pbb:
        continue
    nr, nc = DATA[bs]["n_rows"], DATA[bs]["n_cols"]
    f_std = _G.link_peaks(pbb, nr, nc, 5)
    k_std, _ = _G.characterize_features(f_std, tth_map=TTH, ref_tth_map=_REF)
    f_dsk = _link_deskew(pbb, _bin_pos(bs), 5)
    k_dsk, _ = _G.characterize_features(f_dsk, tth_map=TTH, ref_tth_map=_REF)
    rows.append((bs, len(k_std), _onerow(k_std), len(k_dsk), _onerow(k_dsk)))
    print(f"{bs}x{bs}: standard kept={len(k_std)} 1-row={_onerow(k_std):.0f}%  |  "
          f"deskew kept={len(k_dsk)} 1-row={_onerow(k_dsk):.0f}%")

if rows:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(rows)); w = 0.38
    lbl = [f"{r[0]}x{r[0]}" for r in rows]
    ax = axes[0]
    ax.bar(x - w/2, [r[2] for r in rows], w, label="standard (lattice)", color="#d62728")
    ax.bar(x + w/2, [r[4] for r in rows], w, label="deskew (true X,Y)", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylabel("single-row feature fraction (%)")
    ax.set_title("Slicing: deskew should drop 1×1 sharply"); ax.legend()
    ax = axes[1]
    ax.bar(x - w/2, [r[1] for r in rows], w, label="standard", color="#d62728")
    ax.bar(x + w/2, [r[3] for r in rows], w, label="deskew", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylabel("kept features")
    ax.set_title("Kept-feature count (fewer, merged = healthier)"); ax.legend()
    show(fig, "D4_deskew_compare.png")


# %% [markdown]
'''
## E — Peak intensity & SNR distributions (H2)

If binning recovers *weak* peaks, the SNR distribution should shift right with N and
the count of **marginal** peaks (SNR just over threshold) should rise. Needs per-peak
SNR, so this uses the `kind="peak"` results.
'''

# %%
peak_bs = [bs for bs in READY if DATA[bs]["peaks_by_bin"]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for bs in peak_bs:
    snrs = [p.get("snr", np.nan) for v in DATA[bs]["peaks_by_bin"].values() for p in v]
    snrs = [s for s in snrs if np.isfinite(s)]
    if snrs:
        ax.hist(np.clip(snrs, 0, 60), bins=60, histtype="step", lw=1.4,
                density=True, label=f"{bs}x{bs}")
ax.set_xlabel("peak SNR"); ax.set_ylabel("density")
ax.set_title("SNR distribution (shifts right ⇒ stronger signal)"); ax.legend()

ax = axes[1]
for bs in peak_bs:
    ints = [p.get("integrated_intensity", np.nan)
            for v in DATA[bs]["peaks_by_bin"].values() for p in v]
    ints = [i for i in ints if np.isfinite(i) and i > 0]
    if ints:
        ax.hist(np.log10(ints), bins=60, histtype="step", lw=1.4, density=True, label=f"{bs}x{bs}")
ax.set_xlabel("log10 integrated intensity"); ax.set_ylabel("density")
ax.set_title("Integrated-intensity distribution"); ax.legend()

ax = axes[2]
thr = {bs: ALGOS[bs]["snr"] for bs in peak_bs}
marg = []
for bs in peak_bs:
    t = thr[bs]
    snrs = [p.get("snr", 0) for v in DATA[bs]["peaks_by_bin"].values() for p in v]
    marg.append(sum(1 for s in snrs if t <= s < 1.5 * t))
ax.bar([f"{b}x{b}" for b in peak_bs], marg, color="#2ca02c")
ax.set_ylabel("marginal peaks (thr ≤ SNR < 1.5·thr)")
ax.set_title("Near-threshold peaks recovered")
show(fig, "E_intensity_snr.png")


# %% [markdown]
'''
## E2 — Per-feature SNR & intensity vs bin size (the fair SNR test)

Section C sampled random bins (mostly background) and looked ~flat. Here we look only at
**detected features**: median peak SNR, the strong-peak tail (90th percentile), and
median feature intensity. This is where binning's SNR benefit actually shows up — 1×1
sits barely above the detection threshold, binned features are far stronger.
'''

# %%
xb = [b for b in READY if DATA[b]["peaks_by_bin"] or DATA[b]["features"]]
msnr, p90, mint, mfeat = [], [], [], []
for b in xb:
    snr = [p.get("snr") for v in DATA[b]["peaks_by_bin"].values() for p in v
           if p.get("snr") is not None]
    msnr.append(np.median(snr) if snr else np.nan)
    p90.append(np.percentile(snr, 90) if snr else np.nan)
    pint = [f.get("peak_intensity") for f in DATA[b]["features"]
            if f.get("peak_intensity") is not None]
    mint.append(np.median(pint) if pint else np.nan)
    fs = [f.get("mean_snr") for f in DATA[b]["features"] if f.get("mean_snr") is not None]
    mfeat.append(np.median(fs) if fs else np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.plot(xb, msnr, "o-", label="peak SNR (median)")
ax.plot(xb, mfeat, "s-", label="feature mean SNR (median)")
ax.axhline(4.0, color="grey", ls=":", lw=1, label="detection threshold")
ax.set_xlabel("bin size N"); ax.set_ylabel("SNR")
ax.set_title("SNR at detections rises with binning"); ax.legend()
ax = axes[1]
ax.plot(xb, p90, "^-", color="#d62728", label="peak SNR (90th pct)")
ax.plot(xb, mint, "D-", color="#2ca02c", label="feature intensity (median)")
ax.set_xlabel("bin size N"); ax.set_ylabel("value")
ax.set_title("Strong-peak SNR tail & feature intensity grow"); ax.legend()
show(fig, "E2_feature_snr.png")
print("median peak SNR by bin:", {b: round(s, 1) for b, s in zip(xb, msnr)})


# %% [markdown]
'''
## F — The resolution trade-off (H4)

Binning costs spatial resolution: the device map is painted on an N× coarser grid, so
fine structure blurs. Below, the same reflection's intensity map at each bin size, plus
the effective pitch (in scan steps) and the number of resolvable device cells.
'''

# %%
def build_device_grids(features, n_rows, n_cols, metric="intensity"):
    """Inlined from gui/device_map.py (numpy-only; avoids the Qt import)."""
    grids = {}
    for ref in sorted({f["reflection"] for f in features}):
        grid = np.full((n_rows, n_cols), np.nan)
        for feat in (f for f in features if f["reflection"] == ref):
            for bk, entry in feat.get("intensity_profile", {}).items():
                parts = bk.split("_")
                if len(parts) != 2:
                    continue
                r, c = int(parts[0]), int(parts[1])
                if 0 <= r < n_rows and 0 <= c < n_cols and isinstance(entry, dict):
                    val = entry.get("integrated", entry.get("intensity", 0))
                    grid[r, c] = np.nanmax([grid[r, c], val])
        grids[ref] = grid
    return grids


map_bs = [bs for bs in READY if DATA[bs]["features"] and DATA[bs]["n_rows"]]
# choose the reflection with the most features for a fair side-by-side
all_refs = [f["reflection"] for bs in map_bs for f in DATA[bs]["features"]]
target_ref = max(set(all_refs), key=all_refs.count) if all_refs else None

if target_ref and map_bs:
    fig, axes = plt.subplots(1, len(map_bs), figsize=(4 * len(map_bs), 4.2), squeeze=False)
    for ax, bs in zip(axes[0], map_bs):
        grids = build_device_grids(DATA[bs]["features"], DATA[bs]["n_rows"], DATA[bs]["n_cols"])
        g = grids.get(target_ref, np.full((DATA[bs]["n_rows"], DATA[bs]["n_cols"]), np.nan))
        im = ax.imshow(g, cmap="viridis", origin="upper", aspect="equal")
        ax.set_title(f"{bs}x{bs} · {DATA[bs]['n_rows']}x{DATA[bs]['n_cols']} grid")
        ax.set_xlabel("col (scan x)"); ax.set_ylabel("row (scan y)")
    fig.suptitle(f"Device map — '{target_ref}' intensity, coarsening with bin size")
    show(fig, "F_resolution_maps.png")

print("\nResolution summary:")
print("  bin   grid (r x c)   occupied cells   pitch (scan steps)")
for bs in map_bs:
    print(f"  {bs}x{bs}   {DATA[bs]['n_rows']:>3} x {DATA[bs]['n_cols']:>3}      "
          f"{DATA[bs]['n_bins']:>6}            {bs}")


# %% [markdown]
'''
## G — Runtime (H5)

Detection time should fall ~1/N² (fewer bins → fewer detector calls), while the
one-off bin-build cost is roughly constant (same raw frames either way). Only stages
actually run this session are timed; cached stages show 0.
'''

# %%
timed = [bs for bs in BIN_SIZES if any(TIMINGS[bs].values())]
if timed:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    build = [TIMINGS[b]["build"] for b in timed]
    detect = [TIMINGS[b]["peaks"] for b in timed]
    shape = [TIMINGS[b]["shapes"] for b in timed]
    x = np.arange(len(timed))
    ax.bar(x, build, label="build bins")
    ax.bar(x, detect, bottom=build, label="detect (peaks)")
    ax.bar(x, shape, bottom=np.array(build) + np.array(detect), label="shapes")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}x{b}" for b in timed])
    ax.set_ylabel("seconds"); ax.set_title("Wall-clock per stage"); ax.legend()

    ax = axes[1]
    nb = [DATA[b]["n_bins"] for b in timed]
    ax.plot(nb, detect, "o-")
    for b, x_, y_ in zip(timed, nb, detect):
        ax.annotate(f"{b}x{b}", (x_, y_), fontsize=8)
    ax.set_xlabel("n bins detected"); ax.set_ylabel("detect time (s)")
    ax.set_title("Detection time scales with bin count")
    show(fig, "G_runtime.png")
else:
    print("Nothing timed this session (all cached). Set FORCE=True to re-time.")


# %% [markdown]
'''
## H — Synthesis: where is the sweet spot?

Three axes, normalised to [0,1] across the bin sweep:

- **recall proxy** — features found (higher is better),
- **resolution** — occupied cells, i.e. spatial detail retained (higher is better),
- **speed** — inverse detection cost ∝ N² (higher is better).

The recommended bin size maximises recall × resolution (the project chases recall,
but a map you can't resolve is useless). Speed is shown as context.
'''

# %%
def norm(d):
    vals = np.array([d[b] for b in READY], float)
    rng = vals.max() - vals.min()
    return {b: (d[b] - vals.min()) / rng if rng else 1.0 for b in READY}

recall = norm({b: len(DATA[b]["features"]) for b in READY})
resolution = norm({b: DATA[b]["n_bins"] for b in READY})
speed = norm({b: float(b) ** 2 for b in READY})        # ∝ 1/n_bins ∝ N²
score = {b: recall[b] * (0.5 + 0.5 * resolution[b]) for b in READY}  # recall-led, resolution-weighted

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(READY, [recall[b] for b in READY], "o-", label="recall (features)")
ax.plot(READY, [resolution[b] for b in READY], "s-", label="resolution (cells)")
ax.plot(READY, [speed[b] for b in READY], "^--", label="speed (∝N²)", alpha=0.6)
ax.plot(READY, [score[b] for b in READY], "D-", color="k", lw=2.5, label="combined score")
best = max(READY, key=lambda b: score[b])
ax.axvline(best, color="crimson", ls=":", lw=2)
ax.text(best, 1.01, f"  sweet spot ≈ {best}x{best}", color="crimson", va="bottom")
ax.set_xlabel("bin size N"); ax.set_ylabel("normalised score [0–1]")
ax.set_title("Binning trade-off — recall vs resolution vs speed"); ax.legend()
show(fig, "H_synthesis.png")
print(f"Recommended bin size (recall × resolution): {best}x{best}")


# %% [markdown]
'''
## Summary — hypothesis vs finding
'''

# %%
print(f"{'='*70}\nBINNING REPORT — {SCAN}\n{'='*70}")
print(f"Bin sizes analysed: {READY}\n")
print(f"{'N':>3}  {'peaks':>7}  {'feats':>6}  {'cells':>6}  {'signal':>7}  {'noiseσ':>7}  {'SNR':>5}")
for bs in READY:
    s = sig_by_bs.get(bs, np.nan); no = noise_by_bs.get(bs, np.nan)
    print(f"{bs:>3}  {tot_peaks.get(bs,0):>7}  {len(DATA[bs]['features']):>6}  "
          f"{DATA[bs]['n_bins']:>6}  {s:>7.1f}  {no:>7.2f}  {(s/no if no else 0):>5.1f}")

print("\nHypotheses:")
def trend(d):
    v = [d[b] for b in READY if np.isfinite(d.get(b, np.nan))]
    return "rises with N" if len(v) > 1 and v[-1] > v[0] else "flat/falls with N"
print(f"  H1 signal ~N² : signal {trend(sig_by_bs)}")
print(f"  H1 noise  ~N  : noise  {trend(noise_by_bs)}")
print(f"  H2 more found : features {trend({b: len(DATA[b]['features']) for b in READY})}")
print(f"  H4 resolution : occupied cells {trend({b: DATA[b]['n_bins'] for b in READY})} (fewer = coarser)")
print(f"  Sweet spot    : {best}x{best}")
print("\nFigures saved to:", FIG_DIR)


# %% [markdown]
'''
## Optional — accuracy vs manual labels

If you have hand-labelled ground truth (`Labels/<scan>/bin_annotations_NxN.json`, made
in the GUI's View/Label tab), this estimates recall = (detected near a label) / (labels).
Only the bin sizes you've actually labelled will report. This is the honest
recall number; everything above is counts/intensity.
'''

# %%
def _load_label_points(path):
    """Best-effort extract (detector_x, detector_y) ground-truth points."""
    try:
        obj = json.load(open(path))
    except Exception:
        return []
    pts = []
    items = obj.values() if isinstance(obj, dict) else obj
    for it in items:
        for p in (it if isinstance(it, list) else [it]):
            if isinstance(p, dict):
                x, y = p.get("x", p.get("detector_x")), p.get("y", p.get("detector_y"))
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
    return pts

print("Accuracy vs manual labels (recall within 8 px):")
found_any = False
for bs in READY:
    lbl_path = DM.labels_dir() / f"bin_annotations_{bs}x{bs}.json"
    if not lbl_path.exists():
        continue
    found_any = True
    truth = _load_label_points(lbl_path)
    det = [(int(p.get("x", -1)), int(p.get("y", -1)))
           for v in DATA[bs]["peaks_by_bin"].values() for p in v]
    det += [(int(f.get("detector_x", -1)), int(f.get("detector_y", -1)))
            for f in DATA[bs]["features"]]
    if not truth:
        print(f"  {bs}x{bs}: label file has no usable points")
        continue
    det = np.array(det, float)
    hits = 0
    for tx, ty in truth:
        if det.size and np.min((det[:, 0] - tx) ** 2 + (det[:, 1] - ty) ** 2) <= 64:
            hits += 1
    print(f"  {bs}x{bs}: recall {hits}/{len(truth)} = {hits/len(truth):.2f}")
if not found_any:
    print("  (no bin_annotations_*.json found — label bins in the GUI to enable this)")


# %% [markdown]
'''
## I — New 1×1 combined detector vs the 5×5-peak-detector shapes (both on 1×1 data)

A direct spatial comparison of two detectors run on the **same 1×1 data**:

- **left** — the detector built in this session, **`1x1_global_perframe_uf_voigt`**
  (the global one-pass per-frame combined algorithm), and
- **right** — the standard two-stage pipeline, the **`5x5_tophat_band_adaptive_snr`**
  peak detector + `gaussian` shapes (the report's `DATA[1]`).

Same device view as cell **F**, on the full 156×221 (1×1) grid for both. The combined
detector emits per-frame *points* with no per-bin intensity, so to compare like with
like both panels map **features per bin** (occupancy) rather than intensity. This
shows where each algorithm places features across the device — and the 1×1 serpentine
horizontal-slicing of §D3 in both.
'''

# %%
def _device_grid_count(features, n_rows, n_cols, ref):
    """Per-bin feature COUNT for one reflection — an occupancy device map that
    works for both intensity-bearing shapes and point-only (combined) features.
    Bins come from intensity_profile keys, else spatial_extent, else the center bin."""
    grid = np.zeros((n_rows, n_cols))
    for f in features:
        if f["reflection"] != ref:
            continue
        bins = list(f.get("intensity_profile", {}).keys()) or \
            f.get("spatial_extent", []) or [f.get("center_bin", "")]
        for bk in bins:
            parts = str(bk).split("_")
            if len(parts) != 2:
                continue
            try:
                r, c = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if 0 <= r < n_rows and 0 <= c < n_cols:
                grid[r, c] += 1
    grid[grid == 0] = np.nan
    return grid


# 1×1 grid dims (both panels share this resolution).
_g1 = json.load(open(DM.grid_mapping(bin_size=1)))
_R, _C = _g1["n_bin_rows"], _g1["n_bin_cols"]

# Left: our new combined catalog (1×1) — not in the registry, so read it directly.
_COMBINED = "1x1_global_perframe_uf_voigt"
_cj = DM.labels_dir() / f"{_COMBINED}_combined_1x1.json"
_our = json.load(open(_cj))["features"] if Path(_cj).exists() else []

# Right: the 5×5-peak-detector shapes on 1×1 data = the report's DATA[1] (gaussian
# shapes from 5x5_tophat_band_adaptive_snr at bin 1), with a direct fallback.
_d1 = DATA.get(1, {})
_shp = _d1.get("features", [])
if not _shp:
    _sj = DM.shapes_json("gaussian", 1)
    _shp = json.load(open(_sj)).get("kept", []) if Path(_sj).exists() else []
_peakdet = ALGOS.get(1, {}).get("detector", "5x5_tophat_band_adaptive_snr")

if _our and _shp:
    _common = {f["reflection"] for f in _our} & {f["reflection"] for f in _shp}
    if _common:
        _cnt = {r: sum(1 for f in _our if f["reflection"] == r) for r in _common}
        _tgt = max(_common, key=lambda r: _cnt[r])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, feats, title in (
            (axes[0], _our, f"1×1 combined · {_COMBINED}"),
            (axes[1], _shp, f"1×1 shapes · {_peakdet} + gaussian"),
        ):
            g = _device_grid_count(feats, _R, _C, _tgt)
            im = ax.imshow(g, cmap="viridis", origin="upper", aspect="equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="features / bin")
            n = sum(1 for f in feats if f["reflection"] == _tgt)
            ax.set_title(f"{title}\n{_R}×{_C} grid · '{_tgt}' ({n} features)")
            ax.set_xlabel("col (scan x)"); ax.set_ylabel("row (scan y)")
        fig.suptitle(f"Device map (1×1) — new combined detector vs "
                     f"{_peakdet} shapes ('{_tgt}')")
        show(fig, "I_combined_vs_shapes_1x1_device.png")
    else:
        print("No reflection shared by the combined catalog and the 1×1 shapes.")
else:
    print("Missing inputs for the I comparison:")
    print(f"  {_cj.name}: {'ok' if _our else 'MISSING'} ({len(_our)} features)")
    print(f"  1×1 shapes (DATA[1]/gaussian_shapes_1x1): {len(_shp)} features")
    print("  Run the 'Ensure data' cell (bin 1) + run-combined for the 1×1 algo.")


# %% [markdown]
'''
End of report. Edit the CONFIG and ALGORITHM REGISTRY cells and re-run to compare
detectors, bin sizes, or projects.
'''
