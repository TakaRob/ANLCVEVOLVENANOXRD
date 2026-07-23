# %% [markdown]
# # 1×1 (tightened) and 2×2 novel-feature sets — two coordinate systems compared
#
# Follow-on to `feature_brightness_size_candidates.py`. Two changes:
# 1. **2×2 novel set** — 2×2 kept features not covered (region+reflection) by any
#    coarser bin (3×3/4×4/5×5).
# 2. **1×1 tightened** — now excludes features also seen at **2×2 or 3×3**, not just
#    4×4/5×5. A 1×1 feature survives only if *no* coarser bin detected it.
#
# Run for **both** catalog coordinate systems so we can see how the deskew changes
# the dropped population:
# * `preHybrid156x221`  — 156×221 grid (the earlier analysis).
# * `perrowOffset151x235` — per-row-offset deskew, 151 rows = file-per-row layout.
#
# Detector crops are rendered from `xrd_1x1_bins.h5` by summing each feature's
# constituent frames, routed through the **rawgrid** map (the labeling the h5 was
# built with) — so crops are correct for *either* coordinate system even though the
# h5 itself is on neither.

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import spearmanr

LABELS = Path("TakaTest/TakaProject/Labels/Scan_0203")
META = Path("TakaTest/TakaProject/Metadata/Scan_0203")
H5_1x1 = Path("/home/takaji/xrd_1x1_bins.h5")
FIG_DIR = Path("report_figures"); FIG_DIR.mkdir(exist_ok=True)

SYSTEMS = ["preHybrid156x221", "perrowOffset151x235"]
# (target bin size, coarser bins to exclude against, short tag)
ANALYSES = [(1, [2, 3, 4, 5], "1x1_tight"), (2, [3, 4, 5], "2x2")]
COVERAGE_MAX_BINS = None   # None: big substrate/powder rings count as coverage.
CROP = 45

# %%
# --- catalogs, coverage, containment ----------------------------------------
def catalog(n, sys):
    return json.load(open(LABELS / f"feature_catalog_{n}x{n}_{sys}.json"))


def coverage(feats):
    """(reflection, coarse_row, coarse_col) cells occupied by coarse features."""
    cells = set()
    for f in feats:
        if COVERAGE_MAX_BINS is not None and f["n_bins"] > COVERAGE_MAX_BINS:
            continue
        for bk in f["spatial_extent"]:
            R, C = map(int, bk.split("_"))
            cells.add((f["reflection"], R, C))
    return cells


def is_covered(f, covN, T, N):
    """Does target-level-T feature f fall inside a same-reflection level-N region?"""
    refl = f["reflection"]
    for bk in f["spatial_extent"]:
        r, c = map(int, bk.split("_"))
        r0, c0, r1, c1 = T * r, T * c, T * r + T, T * c + T   # 1×1-unit span
        for Rn in range(r0 // N, (r1 - 1) // N + 1):
            for Cn in range(c0 // N, (c1 - 1) // N + 1):
                if (refl, Rn, Cn) in covN:
                    return True
    return False


def split_novel(target, coarse, sys):
    cov = {n: coverage(catalog(n, sys)) for n in coarse}
    only, also = [], []
    for f in catalog(target, sys):
        covered = any(is_covered(f, cov[n], target, n) for n in coarse)
        (also if covered else only).append(f)
    return only, also

# %% [markdown]
# ## Count comparison
# How many features each level uniquely contributes, in each coordinate system.

# %%
def arr(fs, k):
    return np.array([f[k] for f in fs], float) if fs else np.array([])

results = {}
print(f"{'analysis':<10} {'system':<20} {'total':>6} {'novel':>6} {'also':>6} "
      f"{'medSNR':>7} {'med_nb':>7} {'medI':>6}")
for target, coarse, tag in ANALYSES:
    for sys in SYSTEMS:
        only, also = split_novel(target, coarse, sys)
        results[(tag, sys)] = (only, also)
        print(f"{tag:<10} {sys:<20} {len(only)+len(also):>6} {len(only):>6} {len(also):>6} "
              f"{np.median(arr(only,'mean_snr')):>7.1f} "
              f"{np.median(arr(only,'n_bins')):>7.1f} {np.median(arr(only,'peak_intensity')):>6.1f}")

# %% [markdown]
# ## Brightness / SNR vs size — novel vs also-coarse, both systems
# One row per analysis, columns = coordinate systems. Red = novel (this level
# only), blue = also found coarser. Red hugging the small/faint corner = binning is
# dropping small dim features; how that cloud shifts between systems is the deskew
# effect.

# %%
for target, coarse, tag in ANALYSES:
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(6.4 * len(SYSTEMS), 5), sharey=True)
    for ax, sys in zip(np.atleast_1d(axes), SYSTEMS):
        only, also = results[(tag, sys)]
        rho, _ = spearmanr(arr(catalog(target, sys), "n_bins"),
                           arr(catalog(target, sys), "peak_intensity"))
        ax.scatter(arr(also, "n_bins"), arr(also, "peak_intensity"), s=12, alpha=0.4,
                   color="#3b6fb6", label=f"also coarser (n={len(also)})")
        ax.scatter(arr(only, "n_bins"), arr(only, "peak_intensity"), s=14, alpha=0.5,
                   color="#c4432b", label=f"{target}×{target}-only (n={len(only)})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("feature size (n_bins)")
        ax.set_title(f"{sys}\nSpearman ρ(size,I)={rho:.2f}")
        ax.legend(loc="lower right", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("peak intensity")
    fig.suptitle(f"{tag}: brightness vs size — novel = excludes bins {coarse}", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"compare_{tag}_brightness_size.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Detector-crop galleries (real vs noise)
# Frame-routed through rawgrid, so correct for both systems. Default = lowest-SNR
# novel features (the left-hand end). A real Bragg peak = compact centered hotspot;
# noise = speckle with nothing at the cyan mark.

# %%
# frame -> h5 key (the rawgrid labeling the h5 was built with)
_raw = json.load(open(META / "grid_mapping_1x1_rawgrid.json"))["bins"]
FRAME2KEY = {}
for bk, fr in _raw.items():
    for t in (fr if isinstance(fr, list) else [fr]):
        FRAME2KEY[t] = bk

_gm_cache = {}
def grid_bins(T, sys):
    key = (T, sys)
    if key not in _gm_cache:
        _gm_cache[key] = json.load(open(META / f"grid_mapping_{T}x{T}_{sys}.json"))["bins"]
    return _gm_cache[key]


def feature_image(h5, f, T, sys):
    """Sum the feature's center-bin frames from the h5, via rawgrid routing."""
    fr = grid_bins(T, sys).get(f["center_bin"], [])
    fr = fr if isinstance(fr, list) else [fr]
    acc = None
    for t in fr:
        k = FRAME2KEY.get(t)
        if k and k in h5:
            d = h5[k][...]
            acc = d if acc is None else acc + d
    return acc


def gallery(features, T, sys, title, fname, ncols=8, nrows=5):
    sel = features[: ncols * nrows]
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows))
    axes = np.atleast_2d(axes).ravel()
    with h5py.File(H5_1x1, "r") as h5:
        for ax, f in zip(axes, sel):
            img = feature_image(h5, f, T, sys)
            x, y = int(f["detector_x"]), int(f["detector_y"])
            if img is None:
                ax.axis("off"); continue
            r0, r1 = max(0, y - CROP), min(img.shape[0], y + CROP)
            c0, c1 = max(0, x - CROP), min(img.shape[1], x + CROP)
            crop = img[r0:r1, c0:c1]
            ax.imshow(crop, cmap="inferno", origin="lower",
                      vmin=np.percentile(crop, 20), vmax=max(np.percentile(crop, 99.5), 1))
            ax.plot(x - c0, y - r0, "+", color="cyan", ms=9, mew=1.3)
            ax.set_title(f"{f['reflection']} S{f['mean_snr']:.0f} "
                         f"I{f['peak_intensity']:.0f} {f['n_bins']}b", fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[len(sel):]:
            ax.axis("off")
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140, bbox_inches="tight")
    plt.show()


for target, coarse, tag in ANALYSES:
    for sys in SYSTEMS:
        only, _ = results[(tag, sys)]
        low = sorted(only, key=lambda f: f["mean_snr"])
        gallery(low, target, sys,
                f"{tag} / {sys} — lowest-SNR novel features (real or noise?)",
                f"gallery_{tag}_{sys}_lowsnr.png")

# %% [markdown]
# ## Export novel subsets for the feature viewer (optional)

# %%
for target, coarse, tag in ANALYSES:
    for sys in SYSTEMS:
        only, _ = results[(tag, sys)]
        out = LABELS / f"feature_catalog_{tag}_{sys}_NOVEL.json"
        json.dump(only, open(out, "w"))
        print(f"wrote {len(only):4d} -> {out.name}")
print("\nFigures:")
for p in sorted(FIG_DIR.glob("compare_*.png")) + sorted(FIG_DIR.glob("gallery_*x*_*_lowsnr.png")):
    print("  ", p.name)

# %% [markdown]
# ## Bright end of the 2×2-only set (perrowOffset) — sorted by peak intensity
# NOTE: `mean_snr` is **not** a brightness proxy here — the (112) band has
# near-zero local background, so faint (112) peaks (I≈5) score SNR≈55. Honest
# brightness is `peak_intensity`, so we sort by that. These are the features 2×2
# uniquely keeps that actually carry counts.

# %%
BRIGHT_SYS = "perrowOffset151x235"
only2, _ = results[("2x2", BRIGHT_SYS)]
bright2 = sorted(only2, key=lambda f: -f["peak_intensity"])
gallery(bright2, 2, BRIGHT_SYS,
        f"2×2-only / {BRIGHT_SYS} — brightest first (by peak_intensity)",
        f"gallery_2x2_{BRIGHT_SYS}_bright.png")

# %% [markdown]
# ## Same-spot triptych: 1×1 → 2×2 → 5×5
# For each bright 2×2-only feature, the *same detector window* at three bin sizes,
# rendered by summing that location's frames: **1×1** = the single best constituent
# frame, **2×2** = the 4-frame sum (what was detected), **5×5** = the ~25-frame sum
# of the 5×5 cell covering the same spot. Watch the peak emerge from single-frame
# noise at 2×2, then dilute/merge into the larger 5×5 cell. Per-panel contrast;
# the number after each label is the local max count.

# %%
def sum_frames(h5, frames):
    frames = frames if isinstance(frames, list) else [frames]
    acc = None
    for t in frames:
        k = FRAME2KEY.get(t)
        if k and k in h5:
            d = h5[k][...]
            acc = d if acc is None else acc + d
    return acc


def best_single_frame(h5, frames, x, y):
    frames = frames if isinstance(frames, list) else [frames]
    best, bestval = None, -1
    for t in frames:
        k = FRAME2KEY.get(t)
        if k and k in h5:
            d = h5[k][...]
            v = d[max(0, y - 8):y + 8, max(0, x - 8):x + 8].max()
            if v > bestval:
                best, bestval = d, v
    return best


def frames_5x5_at(feat2, sys):
    """5×5 cell frames covering a 2×2 feature's location."""
    R2, C2 = map(int, feat2["center_bin"].split("_"))
    R5, C5 = (2 * R2) // 5, (2 * C2) // 5
    return grid_bins(5, sys).get(f"{R5}_{C5}", [])


def triptych(features, sys, fname, n=6):
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.4 * n))
    with h5py.File(H5_1x1, "r") as h5:
        for i, f in enumerate(features[:n]):
            x, y = int(f["detector_x"]), int(f["detector_y"])
            f2 = grid_bins(2, sys).get(f["center_bin"], [])
            imgs = [("1×1", best_single_frame(h5, f2, x, y)),
                    ("2×2", sum_frames(h5, f2)),
                    ("5×5", sum_frames(h5, frames_5x5_at(f, sys)))]
            for j, (lab, img) in enumerate(imgs):
                ax = axes[i, j]
                if img is None:
                    ax.axis("off"); continue
                r0, r1 = max(0, y - CROP), min(img.shape[0], y + CROP)
                c0, c1 = max(0, x - CROP), min(img.shape[1], x + CROP)
                crop = img[r0:r1, c0:c1]
                ax.imshow(crop, cmap="inferno", origin="lower",
                          vmin=np.percentile(crop, 20),
                          vmax=max(np.percentile(crop, 99.5), 1))
                ax.plot(x - c0, y - r0, "+", color="cyan", ms=9, mew=1.3)
                ax.set_title(f"{lab}  max{crop.max():.0f}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[i, 0].set_ylabel(f"{f['reflection']}\nI={f['peak_intensity']:.0f}", fontsize=8)
    fig.suptitle(f"Same spot at 1×1 / 2×2 / 5×5 — bright 2×2-only ({sys})", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140, bbox_inches="tight")
    plt.show()


triptych(bright2, BRIGHT_SYS, f"triptych_2x2only_{BRIGHT_SYS}.png", n=6)
print("wrote bright gallery + triptych")
