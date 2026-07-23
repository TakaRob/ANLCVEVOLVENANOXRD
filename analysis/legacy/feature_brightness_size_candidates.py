# %% [markdown]
# # Brightness vs size, and the features only 1×1 detects — Scan_0203
#
# All bin sizes find the big, bright features and agree on their shape. The
# question is what binning **throws away**: the low-SNR end that only the 1×1 grid
# detects. Are those real small/faint Bragg features, or noise that summing
# correctly rejects?
#
# This notebook:
# 1. **brightness vs size** scatter — does bright ⇒ large feature?
# 2. isolates the **1×1-only** features (kept at 1×1 but absent at 4×4 *and* 5×5),
# 3. renders **detector-image crops** of those candidates straight from the binned
#    HDF5 so you can judge real-vs-noise without opening the GUI (a contact sheet
#    instead of clicking each one in `feature_viewer.py`),
# 4. exports the 1×1-only subset as a feature-catalog JSON in case you *do* want to
#    load just those into the viewer.

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

LABELS = Path("TakaTest/TakaProject/Labels/Scan_0203")
H5_1x1 = Path("/home/takaji/xrd_1x1_bins.h5")     # binned images, keyed by "row_col"
FIG_DIR = Path("report_figures"); FIG_DIR.mkdir(exist_ok=True)

CROP = 45          # half-width of detector crop (px) for the galleries
COARSE_BINS = [4, 5]   # "cut out everything these bins also found"
COVERAGE_MAX_BINS = None  # None = a coarse feature of any size covers its region
                          # (so a substrate/powder ring suppresses 1x1 fragments on it);
                          # set e.g. 50 to ignore giant rings and keep more candidates.

# %% [markdown]
# ## Matching: region + reflection containment (not detector position)
# A single physical grain often **fragments into several 1×1 features** (the
# serpentine over-segmentation), and the fragments sit at slightly different
# detector positions. Matching by detector position therefore leaves the offset
# fragments wrongly in the "1×1-only" pile. Instead: a 1×1 feature is **covered**
# if *any* of its 1×1 bins falls inside the scan region of a coarse (4×4/5×5)
# feature **of the same reflection**. A 4×4 bin `R_C` covers 1×1 bins
# `[4R,4R+4) × [4C,4C+4)`, so a 1×1 bin `(r,c)` maps to coarse cell `(r//N, c//N)`.
# "1×1-only" = covered by neither 4×4 nor 5×5.

# %%
def load_catalog(n):
    return json.load(open(LABELS / f"feature_catalog_{n}x{n}.json"))


def coverage(coarse_feats):
    """set of (reflection, coarse_row, coarse_col) cells occupied by coarse features."""
    cells = set()
    for f in coarse_feats:
        if COVERAGE_MAX_BINS is not None and f["n_bins"] > COVERAGE_MAX_BINS:
            continue
        for bk in f["spatial_extent"]:
            R, C = map(int, bk.split("_"))
            cells.add((f["reflection"], R, C))
    return cells


def is_covered(f1, cov, N):
    refl = f1["reflection"]
    for bk in f1["spatial_extent"]:
        r, c = map(int, bk.split("_"))
        if (refl, r // N, c // N) in cov:
            return True
    return False


cat = {n: load_catalog(n) for n in [1] + COARSE_BINS}
cov = {n: coverage(cat[n]) for n in COARSE_BINS}

only, also = [], []
for f in cat[1]:
    covered = any(is_covered(f, cov[n], n) for n in COARSE_BINS)
    (also if covered else only).append(f)

print(f"1x1 kept features            : {len(cat[1])}")
print(f"  also in 4x4/5x5 region     : {len(also)}")
print(f"  1x1-ONLY (truly novel)     : {len(only)}")
print(f"  (detector-bucket method previously gave 1349 — fragments inflated it)")


def arr(fs, k):
    return np.array([f[k] for f in fs], float)


for lbl, fs in [("1x1-only", only), ("also-coarse", also)]:
    print(f"  {lbl:12} medSNR={np.median(arr(fs,'mean_snr')):5.1f}  "
          f"med n_bins={np.median(arr(fs,'n_bins')):4.1f}  "
          f"med intensity={np.median(arr(fs,'peak_intensity')):5.1f}")

# %% [markdown]
# ## 1. Brightness vs size — does bright ⇒ large feature?
# x = feature size (`n_bins`, the spatial footprint), y = brightness
# (`peak_intensity`). Blue = also seen in coarse bins, red = 1×1-only. If the red
# cloud sits in the **small + faint** corner, binning is dropping exactly the
# population we suspected. Spearman ρ quantifies the bright⇔large link.

# %%
from scipy.stats import spearmanr

nb_all = arr(cat[1], "n_bins")
I_all = arr(cat[1], "peak_intensity")
snr_all = arr(cat[1], "mean_snr")
rho_I, _ = spearmanr(nb_all, I_all)
rho_s, _ = spearmanr(nb_all, snr_all)
print(f"Spearman rho (n_bins vs intensity) = {rho_I:.2f}")
print(f"Spearman rho (n_bins vs SNR)       = {rho_s:.2f}")

fig, (axI, axS) = plt.subplots(1, 2, figsize=(13, 5.2))
jit = lambda a: a + np.random.default_rng(0).uniform(-0.25, 0.25, size=a.shape)
for ax, ydata, ylab, rho in [(axI, "peak_intensity", "peak intensity", rho_I),
                             (axS, "mean_snr", "mean SNR", rho_s)]:
    ax.scatter(jit(arr(also, "n_bins")), arr(also, ydata), s=10, alpha=0.4,
               color="#3b6fb6", label=f"also in {COARSE_BINS} (n={len(also)})")
    ax.scatter(jit(arr(only, "n_bins")), arr(only, ydata), s=10, alpha=0.4,
               color="#c4432b", label=f"1×1-only (n={len(only)})")
    ax.set_xlabel("feature size  (n_bins)"); ax.set_ylabel(ylab)
    ax.set_yscale("log"); ax.set_title(f"{ylab} vs size   (Spearman ρ={rho:.2f})")
    ax.legend()
fig.suptitle("Brightness / SNR vs feature size — Scan_0203 (1×1 features)", y=1.0)
fig.tight_layout()
fig.savefig(FIG_DIR / "brightness_vs_size.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Where the 1×1-only features live (SNR vs size density)
# Same split as a 2-D histogram so the overlap doesn't hide density. The 1×1-only
# mass should hug the bottom-left (small, low-SNR). The handful of red points that
# are bright-but-tiny (top-left) are the "rare high-peak small features" worth a
# direct look.

# %%
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(jit(arr(also, "n_bins")), arr(also, "mean_snr"), s=12, alpha=0.5,
           color="#3b6fb6", label="also in coarse")
ax.scatter(jit(arr(only, "n_bins")), arr(only, "mean_snr"), s=12, alpha=0.5,
           color="#c4432b", label="1×1-only")
ax.axhline(np.median(arr(only, "mean_snr")), color="#c4432b", ls=":", lw=1)
ax.set_xlabel("feature size (n_bins)"); ax.set_ylabel("mean SNR"); ax.set_yscale("log")
ax.set_title("1×1-only features cluster small & low-SNR"); ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "oneonly_snr_vs_size.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Real or noise? — detector crops of the 1×1-only candidates
# Pulls each candidate's bin image from `xrd_1x1_bins.h5` and crops a window around
# its detector position. A real Bragg peak = a compact, centered hotspot above a
# smooth background; noise = a speckle with nothing coherent at the marked spot.
# Default gallery = the **lowest-SNR** 1×1-only features (the left-hand end you'd
# scroll to in the viewer).

# %%
def render_gallery(features, title, fname, ncols=8, nrows=6, contrast=99.5):
    sel = features[: ncols * nrows]
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows))
    axes = np.atleast_2d(axes).ravel()
    with h5py.File(H5_1x1, "r") as h5:
        for ax, f in zip(axes, sel):
            bk = f["center_bin"]
            x, y = int(f["detector_x"]), int(f["detector_y"])   # x=col, y=row
            try:
                img = h5[bk][...]
            except KeyError:
                ax.axis("off"); continue
            r0, r1 = max(0, y - CROP), min(img.shape[0], y + CROP)
            c0, c1 = max(0, x - CROP), min(img.shape[1], x + CROP)
            crop = img[r0:r1, c0:c1]
            vmax = np.percentile(crop, contrast) if crop.size else 1
            ax.imshow(crop, cmap="inferno", vmin=np.percentile(crop, 20), vmax=max(vmax, 1),
                      origin="lower")
            ax.plot(x - c0, y - r0, "+", color="cyan", ms=10, mew=1.5)
            ax.set_title(f"{f['reflection']} SNR{f['mean_snr']:.0f}\n"
                         f"I{f['peak_intensity']:.0f} {f['n_bins']}b", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes[len(sel):]:
            ax.axis("off")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140, bbox_inches="tight")
    plt.show()


low_snr = sorted(only, key=lambda f: f["mean_snr"])
render_gallery(low_snr, "1×1-only features — lowest SNR (real Bragg peak or noise?)",
               "gallery_1x1only_lowest_snr.png")

# %% [markdown]
# ## 3b. The rare bright-but-tiny 1×1-only features
# Top-left of the scatter: high intensity, ≤2 bins. These are the ones most likely
# to be *genuine* small grains that binning blurred away — worth a separate look.

# %%
tiny_bright = sorted([f for f in only if f["n_bins"] <= 2],
                     key=lambda f: -f["peak_intensity"])
render_gallery(tiny_bright, "1×1-only & tiny (≤2 bins) — brightest first",
               "gallery_1x1only_tiny_bright.png")

# %% [markdown]
# ## 4. Export the 1×1-only subset for the feature viewer (optional)
# If you want to click through these in `feature_viewer.py` instead of / in
# addition to the contact sheets, this writes them in the same catalog format.

# %%
out = LABELS / "feature_catalog_1x1_ONLY.json"
json.dump(only, open(out, "w"))
print(f"wrote {len(only)} features -> {out}")
print("\nSaved figures:")
for p in sorted(FIG_DIR.glob("*.png")):
    print("  ", p.name)
