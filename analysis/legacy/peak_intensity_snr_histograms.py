# %% [markdown]
# # Detected-peak intensity & SNR histograms by bin size — Scan_0203
#
# Goal: show how much **brighter / higher-SNR** detected peaks get as the spatial
# bin (radius) grows, and whether smaller bins pick up **extraneous near-noise
# detections** (peaks piling up at the low end, just above the SNR = 4 threshold).
#
# One detector (`5x5_tophat_band_adaptive_snr`, snr=4) was run at every bin size,
# so the only thing that changes between curves is the binning. Data source:
# `Labels/Scan_0203/..._peaks_NxN.json` → `peaks_by_bin`.
#
# Run as a notebook (VS Code / Jupytext `# %%` cells) or `python peak_intensity_snr_histograms.py`.

# %%
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- config -----------------------------------------------------------------
LABELS_DIR = Path("TakaTest/TakaProject/Labels/Scan_0203")
DETECTOR = "5x5_tophat_band_adaptive_snr"
BIN_SIZES = [1, 2, 3, 4, 5]
INTENSITY_FIELD = "integrated_intensity"   # or "peak_val" / "cleaned_intensity"
SNR_THRESHOLD = 4.0                         # detector's snr cut-off (peaks below don't exist)
NEAR_NOISE_SNR = 6.0                        # "barely above threshold" band: SNR in [4, 6)
FIG_DIR = Path("report_figures")
FIG_DIR.mkdir(exist_ok=True)

# colour per bin size (small=cool/noisy, large=warm/bright)
COLORS = {1: "#3b6fb6", 2: "#46a0a0", 3: "#6aab4d", 4: "#e0902b", 5: "#c4432b"}


def peaks_path(n: int) -> Path:
    return LABELS_DIR / f"{DETECTOR}_peaks_{n}x{n}.json"


# %%
# --- load: one record array of (intensity, snr, label) per bin size ----------
def load_peaks(n: int):
    d = json.load(open(peaks_path(n)))
    inten, snr, labels = [], [], []
    for bin_key, plist in d["peaks_by_bin"].items():
        for pk in plist:
            inten.append(pk.get(INTENSITY_FIELD, np.nan))
            snr.append(pk.get("snr", np.nan))
            labels.append(pk.get("label", "?"))
    return {
        "intensity": np.asarray(inten, float),
        "snr": np.asarray(snr, float),
        "labels": np.asarray(labels),
        "snr_thr": d.get("snr", SNR_THRESHOLD),
        "n_bins": len(d["peaks_by_bin"]),
    }


data = {}
for n in BIN_SIZES:
    if peaks_path(n).exists():
        data[n] = load_peaks(n)
        print(f"{n}x{n}: {data[n]['intensity'].size:>7d} peaks  "
              f"over {data[n]['n_bins']:>6d} occupied bins  (snr_thr={data[n]['snr_thr']})")
    else:
        print(f"{n}x{n}: MISSING -> {peaks_path(n)}")

BINS_PRESENT = list(data.keys())

# %% [markdown]
# ## Summary statistics
# Medians and percentiles per bin size. Watch the **median SNR** and **median
# intensity** climb with bin radius (more frames summed → higher SNR), and the
# **near-noise fraction** (SNR in [4, 6)) shrink — that fraction is the likely
# "extraneous bins" the small bins over-detect.

# %%
def pct(a, q):
    return np.nanpercentile(a, q) if a.size else np.nan

print(f"{'bin':>4} {'peaks':>8} {'med_SNR':>8} {'p90_SNR':>8} "
      f"{'med_I':>9} {'p90_I':>10} {'near-noise%':>11}")
summary = {}
for n in BINS_PRESENT:
    s, i = data[n]["snr"], data[n]["intensity"]
    near = np.mean((s >= SNR_THRESHOLD) & (s < NEAR_NOISE_SNR)) * 100 if s.size else np.nan
    summary[n] = dict(med_snr=pct(s, 50), p90_snr=pct(s, 90),
                      med_i=pct(i, 50), p90_i=pct(i, 90), near=near, n=s.size)
    print(f"{n}x{n:<2} {s.size:>8d} {summary[n]['med_snr']:>8.1f} {summary[n]['p90_snr']:>8.1f} "
          f"{summary[n]['med_i']:>9.0f} {summary[n]['p90_i']:>10.0f} {near:>10.1f}%")

# %% [markdown]
# ## SNR histograms by bin size
# Log-x because SNR spans a few × to hundreds. Dashed line = detection threshold
# (SNR = 4); shaded band = near-noise zone [4, 6). Top row is **absolute counts**
# (how many peaks total — small bins detect far more), bottom row is **normalized
# density** (shape comparison — does the distribution shift right with binning?).

# %%
def log_edges(arrays, lo_floor, nbins=50):
    vals = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in arrays])
    lo = max(lo_floor, np.nanpercentile(vals, 0.5))
    hi = np.nanpercentile(vals, 99.8)
    return np.logspace(np.log10(lo), np.log10(hi), nbins)

snr_edges = log_edges([data[n]["snr"] for n in BINS_PRESENT], lo_floor=SNR_THRESHOLD * 0.9)

fig, axes = plt.subplots(2, len(BINS_PRESENT), figsize=(3.0 * len(BINS_PRESENT), 6),
                         sharex=True)
for col, n in enumerate(BINS_PRESENT):
    s = data[n]["snr"]
    s = s[np.isfinite(s) & (s > 0)]
    for row, density in enumerate((False, True)):
        ax = axes[row, col]
        ax.hist(s, bins=snr_edges, density=density, color=COLORS[n],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(SNR_THRESHOLD, color="k", ls="--", lw=1)
        ax.axvspan(SNR_THRESHOLD, NEAR_NOISE_SNR, color="grey", alpha=0.15)
        ax.set_xscale("log")
        if row == 0:
            ax.set_title(f"{n}x{n}\nmed SNR={summary[n]['med_snr']:.1f}")
        if col == 0:
            ax.set_ylabel("count" if row == 0 else "density")
        ax.set_xlabel("SNR")
fig.suptitle("Detected-peak SNR distribution by bin size — Scan_0203", y=1.0)
fig.tight_layout()
fig.savefig(FIG_DIR / "snr_hist_by_binsize.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Intensity histograms by bin size
# Same layout for integrated intensity. The rightward march of the bulk with
# bigger bins = "detected peaks are brighter per bin." A tall spike hugging the
# left edge at 1×1 that thins out by 3×3/5×5 = the near-noise detections.

# %%
i_edges = log_edges([data[n]["intensity"] for n in BINS_PRESENT], lo_floor=1.0)

fig, axes = plt.subplots(2, len(BINS_PRESENT), figsize=(3.0 * len(BINS_PRESENT), 6),
                         sharex=True)
for col, n in enumerate(BINS_PRESENT):
    i = data[n]["intensity"]
    i = i[np.isfinite(i) & (i > 0)]
    for row, density in enumerate((False, True)):
        ax = axes[row, col]
        ax.hist(i, bins=i_edges, density=density, color=COLORS[n],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(i), color="k", ls=":", lw=1)
        ax.set_xscale("log")
        if row == 0:
            ax.set_title(f"{n}x{n}\nmed I={summary[n]['med_i']:.0f}")
        if col == 0:
            ax.set_ylabel("count" if row == 0 else "density")
        ax.set_xlabel(INTENSITY_FIELD)
fig.suptitle(f"Detected-peak {INTENSITY_FIELD} distribution by bin size — Scan_0203", y=1.0)
fig.tight_layout()
fig.savefig(FIG_DIR / "intensity_hist_by_binsize.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Overlaid density (all bin sizes on one axis)
# The cleanest single picture of the *shift*: normalized SNR and intensity curves
# stacked on shared axes. If binning genuinely raises peak quality, each larger
# bin's curve sits to the right of the smaller one.

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
for n in BINS_PRESENT:
    s = data[n]["snr"]; s = s[np.isfinite(s) & (s > 0)]
    i = data[n]["intensity"]; i = i[np.isfinite(i) & (i > 0)]
    axL.hist(s, bins=snr_edges, density=True, histtype="step", lw=2,
             color=COLORS[n], label=f"{n}x{n}")
    axR.hist(i, bins=i_edges, density=True, histtype="step", lw=2,
             color=COLORS[n], label=f"{n}x{n}")
for ax, lbl, name in ((axL, "SNR", "SNR"), (axR, INTENSITY_FIELD, "intensity")):
    ax.set_xscale("log"); ax.set_xlabel(lbl); ax.set_ylabel("density"); ax.legend(title="bin")
axL.axvline(SNR_THRESHOLD, color="k", ls="--", lw=1)
axL.axvspan(SNR_THRESHOLD, NEAR_NOISE_SNR, color="grey", alpha=0.15)
axL.set_title("SNR — normalized, by bin size")
axR.set_title(f"{INTENSITY_FIELD} — normalized, by bin size")
fig.tight_layout()
fig.savefig(FIG_DIR / "overlaid_density_by_binsize.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Near-noise / "extraneous" detections vs bin size
# Two views: the **fraction** of peaks sitting in the near-noise band [SNR 4, 6)
# (left) and the **absolute count** of those peaks (right). Smaller bins should
# show both a higher fraction and a far larger absolute pile of marginal
# detections — the over-segmentation / noise the report flags at 1×1.

# %%
ns = BINS_PRESENT
frac = [summary[n]["near"] for n in ns]
near_counts = [int(np.sum((data[n]["snr"] >= SNR_THRESHOLD) &
                          (data[n]["snr"] < NEAR_NOISE_SNR))) for n in ns]
tot_counts = [summary[n]["n"] for n in ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.bar([f"{n}x{n}" for n in ns], frac, color=[COLORS[n] for n in ns])
ax1.set_ylabel("% of peaks with SNR in [4, 6)")
ax1.set_title("Near-noise fraction vs bin size")
for x, v in enumerate(frac):
    ax1.text(x, v, f"{v:.0f}%", ha="center", va="bottom")

ax2.bar([f"{n}x{n}" for n in ns], tot_counts, color="lightgrey", label="all peaks")
ax2.bar([f"{n}x{n}" for n in ns], near_counts, color=[COLORS[n] for n in ns],
        label="near-noise [4,6)")
ax2.set_yscale("log"); ax2.set_ylabel("peak count (log)")
ax2.set_title("Absolute peak counts: total vs near-noise"); ax2.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "near_noise_vs_binsize.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Median + p90 trend vs bin size
# The histograms compressed into trend lines. Median tracks the *typical* peak;
# p90 tracks the *bright* tail. Expectation if SNR grew cleanly as √(frames)=N is
# the dotted grey line. Note the **median is non-monotonic** (2×2 dips below the
# theory line oddly, 3×3↔4×4 plateau) while the **p90 climbs steadily** — i.e.
# the bright peaks keep improving with radius even where the median stalls.

# %%
ns = BINS_PRESENT
med_snr = [summary[n]["med_snr"] for n in ns]
p90_snr = [summary[n]["p90_snr"] for n in ns]
med_i = [summary[n]["med_i"] for n in ns]
p90_i = [summary[n]["p90_i"] for n in ns]
theory = [med_snr[0] * n for n in ns]  # SNR ~ N, anchored at 1x1 median

fig, (axS, axI) = plt.subplots(1, 2, figsize=(13, 5))
axS.plot(ns, med_snr, "o-", lw=2, color="#3b6fb6", label="median SNR")
axS.plot(ns, p90_snr, "s-", lw=2, color="#c4432b", label="p90 SNR")
axS.plot(ns, theory, ":", color="grey", label="√N expectation (∝ bin size)")
axS.set_xlabel("bin size (radius)"); axS.set_ylabel("SNR"); axS.set_xticks(ns)
axS.set_title("SNR vs bin size — median stalls, p90 climbs"); axS.legend()

axI.plot(ns, med_i, "o-", lw=2, color="#3b6fb6", label="median intensity")
axI.plot(ns, p90_i, "s-", lw=2, color="#c4432b", label="p90 intensity")
axI.set_xlabel("bin size (radius)"); axI.set_ylabel(INTENSITY_FIELD); axI.set_xticks(ns)
axI.set_yscale("log")
axI.set_title(f"{INTENSITY_FIELD} vs bin size"); axI.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "trend_median_p90_vs_binsize.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Matched-feature SNR vs radius (survivorship-free)
# The fair test: track the **same physical feature** across every bin size, so the
# changing population can't fake a trend. A feature is keyed by (coarse 5×5 scan
# cell, reflection label, detector position bucketed at 40 px); at each bin size we
# take that feature's brightest detection. Only features detected at **all** bin
# sizes are kept. Faint grey lines = individual features; thick line = median
# growth. If binning genuinely raises SNR per feature, the median rises
# monotonically here even though the per-bin medians (above) do not.

# %%
DET_TOL = 40.0      # detector-position bucket (px) — matches the 40px detection tolerance
CELL = max(BINS_PRESENT)  # use the coarsest grid as the common physical scan cell


def feature_map(n):
    """key -> (best_snr, best_intensity) for one bin size."""
    d = json.load(open(peaks_path(n)))
    out = {}
    for bk, plist in d["peaks_by_bin"].items():
        r, c = map(int, bk.split("_"))
        pr, pc = n * r + n / 2.0, n * c + n / 2.0      # 1×1-unit center of the bin
        cell = (int(pr // CELL), int(pc // CELL))
        for pk in plist:
            key = (cell[0], cell[1], pk.get("label", "?"),
                   round(pk["x"] / DET_TOL), round(pk["y"] / DET_TOL))
            s = pk.get("snr", np.nan)
            if key not in out or s > out[key][0]:
                out[key] = (s, pk.get(INTENSITY_FIELD, np.nan))
    return out


fmaps = {n: feature_map(n) for n in BINS_PRESENT}
common = set.intersection(*[set(fmaps[n]) for n in BINS_PRESENT])
print(f"features detected at all bin sizes: {len(common)}")

snr_curves = np.array([[fmaps[n][k][0] for n in BINS_PRESENT] for k in common])
int_curves = np.array([[fmaps[n][k][1] for n in BINS_PRESENT] for k in common])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
# subsample grey spaghetti so the plot stays legible
rng = np.random.default_rng(0)
idx = rng.choice(len(common), size=min(400, len(common)), replace=False)
for row in snr_curves[idx]:
    axA.plot(BINS_PRESENT, row, color="grey", alpha=0.06, lw=0.6)
axA.plot(BINS_PRESENT, np.median(snr_curves, axis=0), "o-", color="#c4432b", lw=3,
         label="median (matched features)")
axA.plot(BINS_PRESENT, [np.median(snr_curves[:, 0]) * n for n in BINS_PRESENT], ":",
         color="k", label="√N expectation")
axA.set_xlabel("bin size (radius)"); axA.set_ylabel("feature SNR (best detection)")
axA.set_yscale("log"); axA.set_xticks(BINS_PRESENT)
axA.set_title(f"Matched-feature SNR vs radius (n={len(common)})"); axA.legend()

for row in int_curves[idx]:
    axB.plot(BINS_PRESENT, row, color="grey", alpha=0.06, lw=0.6)
axB.plot(BINS_PRESENT, np.median(int_curves, axis=0), "o-", color="#3b6fb6", lw=3,
         label="median (matched features)")
axB.set_xlabel("bin size (radius)"); axB.set_ylabel(f"feature {INTENSITY_FIELD}")
axB.set_yscale("log"); axB.set_xticks(BINS_PRESENT)
axB.set_title("Matched-feature intensity vs radius"); axB.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "matched_feature_snr_vs_radius.png", dpi=150, bbox_inches="tight")
plt.show()

# matched-feature median table (the survivorship-free version of the §summary)
print(f"\n{'bin':>4} {'med_SNR':>9} {'med_I':>10}  (matched features only)")
for j, n in enumerate(BINS_PRESENT):
    print(f"{n}x{n:<2} {np.median(snr_curves[:, j]):>9.1f} {np.median(int_curves[:, j]):>10.0f}")

# %%
print("\nSaved figures to", FIG_DIR.resolve())
for p in sorted(FIG_DIR.glob("*.png")):
    print("  ", p.name)
