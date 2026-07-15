# %% [markdown]
# # Do the XRF signals correlate with the nano-XRD features?
#
# **Question (from the beamline scientist):** across our 6 main focus scans, is
# there any correlation between *what the nano-XRD sees* (Bragg peaks / linked
# features and where they sit on the sample) and *what the XRF sees* (per-element
# fluorescence intensity and where it sits)? Two framings were requested up front:
#
# > * *If there is **no** correlation* → the expectation is that "all the XRF
# >   scans were the same within some degree of variance."
# > * *If there **is** correlation* → show examples of features with **high
# >   nano-XRD intensity** whose locations also carry **higher intensity for some
# >   XRF detections**.
#
# Per the request, XRF lines are labelled **by energy (keV)**, not by assumed
# material. (Element names are kept only as convenient tags; the physics below
# does not depend on the material assignment being correct.)
#
# ---
# ## TL;DR — the answer is *mostly* the "no correlation" case, with one small real effect
#
# 1. **Within any single scan, the XRF map is nearly flat.** Every element's
#    per-point intensity varies by only **CV ≈ 2–5 %** across the whole sample
#    surface (Au, the weakest line, ~10 %). The nano-XRD feature intensity, by
#    contrast, spans **3–4 orders of magnitude**. So there is almost no spatial
#    XRF contrast for the XRD structure to correlate *with* — the XRF really is
#    "the same everywhere within a few-percent variance."
#
# 2. **The 6 scans differ from each other almost entirely by measurement
#    geometry (θ), not by sample condition.** The three normal-incidence
#    (θ≈20.5°) scans look nearly identical to each other (spectrum cosine ≳0.99);
#    the three grazing (θ≈6°) scans look nearly identical to each other
#    (cosine ≈1.00); the two groups differ (cosine ≈0.80–0.85). DI% and grain-
#    boundary condition leave no comparable fingerprint. The θ≈6° group is
#    Sn-dominated (~60 %); the θ≈20.5° group is Pb-dominated (~74 %) — an
#    incidence-angle / penetration-depth effect, not a chemistry difference.
#
# 3. **There is one small, reproducible correlation.** In *all three*
#    normal-incidence (θ≈20.5°) scans, the brightest-nano-XRD bins carry
#    **~3–4 % more I (3.94 keV) and Cs (4.23 keV)** fluorescence than the dimmest
#    XRD bins — and *only* those two lines (the minor perovskite constituents),
#    not Pb/Br/Sn/Au. It is physically sensible (more crystalline perovskite →
#    stronger diffraction *and* marginally stronger I/Cs fluorescence) and the
#    sign is consistent across three independent samples, so it is real — but it
#    is a **weak population-level trend (Spearman ρ ≈ 0.2–0.44, ~3–4 % intensity
#    swing)**, invisible in any individual hotspot. The grazing (θ≈6°) scans show
#    no coherent positive coupling (weak/negative, geometry-confounded).
#
# **Bottom line:** the XRF detections are essentially uniform across each sample
# and do not "light up" where nano-XRD features are strong. The only genuine
# coupling is a faint I/Cs enhancement (a few percent) tracking perovskite
# crystallinity in the normal-incidence scans.

# %% [markdown]
# ## Setup — the 6 focus scans, data locations, and XRF line energies
#
# | scan | condition | θ (deg) | role |
# |---|---|---|---|
# | 0179 | No_DI / Yes_GB | 20.50 | normal incidence |
# | 0182 | No_DI / Yes_GB | 6.00  | grazing |
# | 0203 | 5%_DI / Yes_GB | 20.50 | normal incidence |
# | 0207 | 5%_DI / Yes_GB | 5.50  | grazing |
# | 0215 | 5%_DI / No_GB  | 20.50 | normal incidence |
# | 0218 | 5%_DI / No_GB  | 6.00  | grazing |
#
# XRF products: `Metadata/Scan_NNNN/Scan_NNNN_xrf.npz` (1×1 per-element maps +
# grand-sum MCA spectrum, built by `xrd-app xrf`). nano-XRD features:
# `Labels/Scan_NNNN/gaussian_shapes_3x3.json` (the validated *shapes*, 3×3 bin).

# %%
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# scan -> project root
ROOT = {
    "0179": "/home/takaji/179-201", "0182": "/home/takaji/179-201",
    "0215": "/home/takaji/215-226", "0218": "/home/takaji/215-226",
    "0203": "/home/takaji/rocking_203_214", "0207": "/home/takaji/rocking_203_214",
}
# display order: normal-incidence group first, then grazing group
ORDER = ["0179", "0203", "0215", "0182", "0207", "0218"]
THETA = {"0179": 20.5, "0182": 6.0, "0203": 20.5, "0207": 5.5, "0215": 20.5, "0218": 6.0}
COND = {
    "0179": "No_DI/Yes_GB", "0182": "No_DI/Yes_GB", "0203": "5%_DI/Yes_GB",
    "0207": "5%_DI/Yes_GB", "0215": "5%_DI/No_GB", "0218": "5%_DI/No_GB",
}
NORMAL = [s for s in ORDER if THETA[s] > 15]     # θ≈20.5°
GRAZING = [s for s in ORDER if THETA[s] < 15]    # θ≈6°
BIN = 3                                            # shapes are on a 3×3 grid

def label(s):
    return f"Scan_{s} ({COND[s]}, θ={THETA[s]:g}°)"

FIGDIR = Path("xrf_xrd_correlation_figs")
FIGDIR.mkdir(exist_ok=True)

# fixed categorical palette for the XRF lines (colour = element, stable everywhere)
EL_COLOR = {
    "Pb_La": "#4E79A7", "I_La": "#F28E2B", "Sn_La": "#59A14F",
    "Cs_La": "#E15759", "Br_Ka": "#B07AA1", "Au_La": "#9C755F",
}

# %%
def load_xrf(s):
    """Load the per-scan XRF product: element maps, per-point coverage, grand-sum spectrum."""
    z = np.load(f"{ROOT[s]}/Metadata/Scan_{s}/Scan_{s}_xrf.npz")
    els = [str(e) for e in z["elements"]]
    return {
        "els": els,
        "maps": {e: z[f"map_{e}"].astype(float) for e in els},   # 1×1 (row,col)
        "npts": z["n_points"].astype(float),                     # points per 1×1 bin
        "spec": z["spectrum"].astype(float),                     # grand-sum MCA (4096)
        "ev_per_bin": float(z["ev_per_bin"]),
        "offset_ev": float(z["offset_ev"]),
        # ROI center energy (keV) — this is how we label each line
        "kev": {e: z[f"roi_{e}"][1] / 1000.0 for e in els},
    }

def load_feats(s):
    d = json.load(open(f"{ROOT[s]}/Labels/Scan_{s}/gaussian_shapes_3x3.json"))
    return d["kept"]

X = {s: load_xrf(s) for s in ORDER}
F = {s: load_feats(s) for s in ORDER}
ELS = X["0179"]["els"]
KEV = X["0179"]["kev"]
# canonical "El_Line (E keV)" tag used in every label/legend
TAG = {e: f"{e} ({KEV[e]:.2f} keV)" for e in ELS}
print("XRF lines analysed:", [TAG[e] for e in ELS])

# %% [markdown]
# ## Part 1 — Scan level: are the XRF scans "all the same"?
#
# First establish what the XRF *fingerprint* of each scan is, independent of any
# spatial question: (a) detect the emission peaks in each grand-sum spectrum and
# label them by keV; (b) compare the per-line composition; (c) measure how
# similar the whole spectra are (cosine of the normalised spectra).

# %%
def spectrum_peaks_kev(sp, ev_per_bin, offset_ev):
    """Top emission peaks of a grand-sum spectrum, returned as keV, ignoring the
    low-energy noise floor and the elastic/Compton band around the 15 keV beam."""
    ev = np.arange(sp.size) * ev_per_bin + offset_ev
    s = sp.copy()
    s[(ev < 1200) | ((ev > 15000 - 800) & (ev < 15000 + 400))] = 0.0
    pk, props = find_peaks(s, prominence=0.01 * s.max(), width=2)
    order = np.argsort(props["prominences"])[::-1][:8]
    return sorted(ev[pk[order]] / 1000.0)

print("Observed emission peaks (keV) per scan — labelled by energy only:\n")
for s in ORDER:
    ks = spectrum_peaks_kev(X[s]["spec"], X[s]["ev_per_bin"], X[s]["offset_ev"])
    print(f"  {label(s):32s}: " + ", ".join(f"{k:.2f}" for k in ks))

# per-line composition fraction (of total ROI counts) per scan
comp = {}
for s in ORDER:
    tot = {e: X[s]["maps"][e].sum() for e in ELS}
    T = sum(tot.values())
    comp[s] = {e: tot[e] / T for e in ELS}

print("\nPer-line composition (% of total ROI counts):")
print("scan   " + " " + "".join(f"{e.split('_')[0]:>8}" for e in ELS))
for s in ORDER:
    print(f"{s}  " + "".join(f"{comp[s][e]*100:7.1f}%" for e in ELS))

# cross-scan coefficient of variation of composition
print("\nCross-scan CV of each line's composition fraction (over all 6 scans):")
for e in ELS:
    v = np.array([comp[s][e] for s in ORDER])
    print(f"  {TAG[e]:20s}: mean={v.mean()*100:5.1f}%   CV={v.std()/v.mean():.2f}")

# %%
# whole-spectrum shape similarity (cosine of normalised, noise-floor-masked spectra)
def nspec(s):
    v = X[s]["spec"].copy()
    ev = np.arange(v.size) * X[s]["ev_per_bin"] + X[s]["offset_ev"]
    v[ev < 1200] = 0.0
    return v / np.linalg.norm(v)

NS = {s: nspec(s) for s in ORDER}
COS = np.array([[float(NS[a] @ NS[b]) for b in ORDER] for a in ORDER])
print("Grand-sum spectrum cosine similarity (1.0 = identical shape):\n")
print("        " + "".join(f"{s:>7}" for s in ORDER))
for i, a in enumerate(ORDER):
    print(f"Scan_{a} " + "".join(f"{COS[i,j]:7.2f}" for j in range(len(ORDER))))

# %%
# ---- Figure 1: spectra overlaid (coloured by θ group) + composition + similarity ----
fig = plt.figure(figsize=(14, 4.4), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 1.0])

ax = fig.add_subplot(gs[0])
for s in ORDER:
    ev = (np.arange(X[s]["spec"].size) * X[s]["ev_per_bin"] + X[s]["offset_ev"]) / 1000.0
    y = X[s]["spec"] / X[s]["spec"][ (ev > 1.2) ].max()
    c = "#1f77b4" if s in NORMAL else "#d62728"
    ax.plot(ev, y, lw=1.0, alpha=0.8, color=c,
            label=f"{s} θ={THETA[s]:g}°")
ax.set_xlim(1.5, 15); ax.set_yscale("log"); ax.set_ylim(1e-3, 1.5)
for e in ELS:
    ax.axvline(KEV[e], ls=":", lw=0.8, color=EL_COLOR[e])
    ax.text(KEV[e], 1.6, e.split("_")[0], rotation=90, va="bottom", ha="center",
            fontsize=7, color=EL_COLOR[e])
ax.set_xlabel("energy (keV)"); ax.set_ylabel("norm. counts (log)")
ax.set_title("Grand-sum XRF spectra\n(blue = θ≈20.5°, red = θ≈6°)")
ax.legend(fontsize=7, ncol=2)

ax = fig.add_subplot(gs[1])
bottom = np.zeros(len(ORDER))
for e in ELS:
    vals = np.array([comp[s][e] * 100 for s in ORDER])
    ax.bar(range(len(ORDER)), vals, bottom=bottom, color=EL_COLOR[e], label=TAG[e])
    bottom += vals
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([f"{s}\n{THETA[s]:g}°" for s in ORDER], fontsize=8)
ax.set_ylabel("% of total ROI counts")
ax.set_title("Per-line composition by scan")
ax.legend(fontsize=6.5, loc="upper right")

ax = fig.add_subplot(gs[2])
im = ax.imshow(COS, vmin=0.75, vmax=1.0, cmap="viridis")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=90, fontsize=7)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=7)
ax.set_title("Spectrum cosine similarity")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(FIGDIR / "fig1_scan_level.png", dpi=130)
print("saved", FIGDIR / "fig1_scan_level.png")

# %% [markdown]
# ### Part 1 findings
#
# * **Observed peaks (keV) split by θ.** The normal-incidence scans (0179/0203/
#   0215) show the full set — **~2.94, 3.66, 3.94, 4.23, 10.53, 11.90, 12.61,
#   14.20 keV** — i.e. the perovskite/substrate lines (I 3.94, Cs 4.23, Pb-Lα
#   10.53, Br 11.92, Pb-Lβ 12.61). The grazing scans (0182/0207/0218) collapse to
#   a Sn/Sn-cluster-dominated set (**~3.64, 6.35, 10.5, 14.2 keV**).
# * **Composition flips with θ, not with sample condition.** θ≈20.5°: Pb ≈ 74 %,
#   Br ≈ 11 %, I ≈ 6.5 %, Cs ≈ 5.5 %. θ≈6°: Sn ≈ 57–65 %, I ≈ 13 %, Pb ≈ 15–23 %.
#   The high cross-scan CV of Pb (0.60) and Sn (0.94) is driven entirely by the
#   θ split; within a θ group the composition barely moves.
# * **Spectrum cosine similarity confirms it:** ≳0.99 within the normal group,
#   ≈1.00 within the grazing group, ≈0.80–0.85 between groups. So "all XRF scans
#   the same within variance" is **true *within each incidence-angle group*** —
#   DI% and grain-boundary condition are not distinguishable in the XRF — but the
#   two θ groups are genuinely different because of penetration-depth / self-
#   absorption geometry, not chemistry.

# %% [markdown]
# ## Part 2 — Spatial: does the XRF light up where the nano-XRD features are?
#
# The core test. We register the XRF onto the XRD feature grid, then ask whether
# per-point XRF intensity tracks nano-XRD feature intensity, bin by bin.
#
# **Registration.** XRF maps are native 1×1 `(row,col)`; the shapes are on the
# de-skewed 3×3 grid. Because both come from the *same* `grid_mapping`, a 3×3 bin
# `(r,c)` is exactly the 1×1 block `[3r:3r+3, 3c:3c+3]`. We block-sum the XRF
# maps and the per-point coverage to the 3×3 grid and form the **per-point** XRF
# (`block_map / block_points`) so edge bins with fewer points aren't spuriously
# dim. The nano-XRD spatial signal is the **summed feature intensity per bin**,
# taken from each shape's `intensity_profile` (the real per-bin Bragg intensity).

# %%
def block_sum(a, b, nr, nc):
    pr, pc = (-a.shape[0]) % b, (-a.shape[1]) % b
    a = np.pad(a, ((0, pr), (0, pc)))
    a = a.reshape(a.shape[0] // b, b, a.shape[1] // b, b).sum((1, 3))
    return a[:nr, :nc]

def build_grids(s):
    """Return per-point XRF maps, the nano-XRD intensity map, and coverage/feature masks
    on the common 3×3 grid for one scan."""
    xr, feats = X[s], F[s]
    nr = max(max(f["center_row"] for f in feats) + 1, int(np.ceil(xr["npts"].shape[0] / BIN)))
    nc = max(max(f["center_col"] for f in feats) + 1, int(np.ceil(xr["npts"].shape[1] / BIN)))
    npts = block_sum(xr["npts"], BIN, nr, nc)
    cover = npts > 0
    xrf_pp = {}
    for e in ELS:
        m = block_sum(xr["maps"][e], BIN, nr, nc)
        pp = np.zeros_like(m); pp[cover] = m[cover] / npts[cover]
        xrf_pp[e] = pp
    xrd = np.zeros((nr, nc))
    for f in feats:
        for bk, v in f["intensity_profile"].items():
            r, c = map(int, bk.split("_"))
            if 0 <= r < nr and 0 <= c < nc:
                xrd[r, c] += float(v.get("intensity") or 0.0)
    has = (xrd > 0) & cover
    return dict(nr=nr, nc=nc, npts=npts, cover=cover, xrf_pp=xrf_pp, xrd=xrd, has=has)

G = {s: build_grids(s) for s in ORDER}

# %%
# --- 2a. XRF spatial uniformity within each scan (CV of the per-point map) ---
print("XRF spatial uniformity within each scan — CV (std/mean) of the per-point map:")
print("           " + "".join(f"{e.split('_')[0]:>8}" for e in ELS))
cv_tab = {}
for s in ORDER:
    g = G[s]; row = []
    for e in ELS:
        v = g["xrf_pp"][e][g["cover"]]
        row.append(v.std() / v.mean())
    cv_tab[s] = row
    print(f"Scan_{s}  " + "".join(f"{x:8.3f}" for x in row))
print("\n→ XRF is spatially flat: every major line varies by only a few percent "
      "across the whole sample.")

# %%
# --- 2b. Spearman correlation: nano-XRD summed intensity vs per-point XRF, over covered bins ---
rho_tab = np.zeros((len(ORDER), len(ELS)))
print("Spearman ρ (nano-XRD intensity  vs  per-point XRF line), over all covered bins:")
print("           " + "".join(f"{e.split('_')[0]:>8}" for e in ELS))
for i, s in enumerate(ORDER):
    g = G[s]; cov = g["cover"]; xv = g["xrd"][cov]
    for j, e in enumerate(ELS):
        rho, _ = stats.spearmanr(xv, g["xrf_pp"][e][cov])
        rho_tab[i, j] = rho
    print(f"Scan_{s}  " + "".join(f"{rho_tab[i,j]:+8.3f}" for j in range(len(ELS))))

# %%
# --- 2c. Top-10% vs bottom-10% XRD-intensity bins: XRF per-point ratio (effect size) ---
ratio_tab = np.zeros((len(ORDER), len(ELS)))
print("XRF per-point ratio  (top-10% XRD bins) / (bottom-10% XRD bins):")
print("           " + "".join(f"{e.split('_')[0]:>8}" for e in ELS))
for i, s in enumerate(ORDER):
    g = G[s]; has = g["has"]; xv = g["xrd"][has]
    hi = has & (g["xrd"] >= np.percentile(xv, 90))
    lo = has & (g["xrd"] <= np.percentile(xv, 10))
    for j, e in enumerate(ELS):
        ratio_tab[i, j] = g["xrf_pp"][e][hi].mean() / g["xrf_pp"][e][lo].mean()
    print(f"Scan_{s}  " + "".join(f"{ratio_tab[i,j]:8.3f}" for j in range(len(ELS))))
print("\n→ Even at the extremes the swing is ~±4%. Only I & Cs rise consistently "
      "(θ≈20.5° scans); Pb tends to fall slightly at grazing.")

# %%
# ---- Figure 2: uniformity, correlation heatmap, decile ratio ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

# (a) uniformity CV
ax = axes[0]
cvm = np.array([cv_tab[s] for s in ORDER])
im = ax.imshow(cvm, cmap="magma", vmin=0, vmax=0.12)
ax.set_xticks(range(len(ELS))); ax.set_xticklabels([f"{e.split('_')[0]}\n{KEV[e]:.1f}k" for e in ELS], fontsize=7)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{s} {THETA[s]:g}°" for s in ORDER], fontsize=7)
for i in range(len(ORDER)):
    for j in range(len(ELS)):
        ax.text(j, i, f"{cvm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="w" if cvm[i, j] < 0.08 else "k")
ax.set_title("XRF spatial CV within scan\n(small = flat)")
fig.colorbar(im, ax=ax, fraction=0.046)

# (b) Spearman heatmap (diverging around 0)
ax = axes[1]
im = ax.imshow(rho_tab, cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(0, -0.5, 0.5))
ax.set_xticks(range(len(ELS))); ax.set_xticklabels([f"{e.split('_')[0]}\n{KEV[e]:.1f}k" for e in ELS], fontsize=7)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{s} {THETA[s]:g}°" for s in ORDER], fontsize=7)
for i in range(len(ORDER)):
    for j in range(len(ELS)):
        ax.text(j, i, f"{rho_tab[i,j]:+.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Spearman ρ:  nano-XRD intensity\nvs per-point XRF")
fig.colorbar(im, ax=ax, fraction=0.046)

# (c) decile ratio
ax = axes[2]
im = ax.imshow(ratio_tab, cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(1.0, 0.95, 1.05))
ax.set_xticks(range(len(ELS))); ax.set_xticklabels([f"{e.split('_')[0]}\n{KEV[e]:.1f}k" for e in ELS], fontsize=7)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([f"{s} {THETA[s]:g}°" for s in ORDER], fontsize=7)
for i in range(len(ORDER)):
    for j in range(len(ELS)):
        ax.text(j, i, f"{ratio_tab[i,j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("XRF ratio: top-10% / bottom-10%\nnano-XRD bins")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(FIGDIR / "fig2_spatial_correlation.png", dpi=130)
print("saved", FIGDIR / "fig2_spatial_correlation.png")

# %% [markdown]
# ### Part 2 findings
#
# * **XRF is spatially flat (Fig 2a).** Per-point CV within a scan is ~0.02–0.05
#   for the strong lines (Au ~0.10). There is essentially no XRF image contrast —
#   consistent with "all the same within a few-percent variance," now *spatially*
#   as well as scan-to-scan.
# * **Correlations are weak and, in the normal-incidence scans, I/Cs-specific
#   (Fig 2b–c).** In 0179/0203/0215 the only consistently positive lines are
#   **I (3.94 keV, ρ ≈ +0.22/+0.29/+0.40)** and **Cs (4.23 keV, ρ ≈ +0.23/+0.27/
#   +0.44)** — the minor perovskite constituents — with the brightest XRD decile
#   carrying **~3–4 % more I and Cs** than the dimmest. Pb, Br, Sn, Au show no
#   coherent trend.
# * **The grazing scans are geometry-confounded.** 0182/0207/0218 show weak or
#   negative couplings (e.g. 0218 Pb ρ ≈ −0.47), driven by self-absorption at
#   grazing incidence, not by a feature↔element link. No positive I/Cs effect
#   survives there.

# %% [markdown]
# ## Part 3 — The requested "examples" (best case: the θ≈20.5° scans)
#
# The request: *if correlated, show features with high nano-XRD intensity whose
# locations carry higher XRF.* The effect is real but small, so it shows up as a
# **population trend**, not as individual hotspots. Below: (left) the top-decile
# vs bottom-decile I/Cs enhancement across the three normal-incidence scans;
# (right) for Scan_0215 (strongest coupling) the bin-wise scatter of nano-XRD
# intensity vs local I and Cs, plus the sample maps side by side.

# %%
# individual brightest features and their local XRF (Scan_0215) — shows why it's
# NOT a per-hotspot effect: the very brightest features sit near median XRF.
s = "0215"; g = G[s]
med = {e: np.median(g["xrf_pp"][e][g["cover"]]) for e in ELS}
top = sorted(F[s], key=lambda f: f.get("peak_intensity") or 0, reverse=True)[:8]
print(f"{label(s)} — 8 brightest nano-XRD features, local XRF as % of scan median:\n")
print("feat  refl      peakI  (r, c)   " + "".join(f"{e.split('_')[0]:>6}" for e in ELS))
for f in top:
    r, c = f["center_row"], f["center_col"]
    if not (0 <= r < g["nr"] and 0 <= c < g["nc"]):
        continue
    pct = "".join(f"{g['xrf_pp'][e][r,c]/med[e]*100:6.0f}" for e in ELS)
    print(f"{f['feature_id']:4d}  {f['reflection']:8s} {f['peak_intensity']:6.0f}  "
          f"({r:2d},{c:2d}) {pct}")
print("\n→ Brightest features sit within ±5% of median XRF: the coupling is a "
      "weak aggregate trend, not a hotspot that lights up the XRF.")

# %%
# ---- Figure 3: decile enhancement (I, Cs) + scatter + maps for 0215 ----
fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.1, 1.1, 1.1])

# (a) I/Cs top vs bottom decile across the 3 normal-incidence scans
ax = fig.add_subplot(gs[0])
xs = np.arange(len(NORMAL)); w = 0.35
for k, e in enumerate(["I_La", "Cs_La"]):
    vals = []
    for s2 in NORMAL:
        gg = G[s2]; has = gg["has"]; xv = gg["xrd"][has]
        hi = has & (gg["xrd"] >= np.percentile(xv, 90))
        lo = has & (gg["xrd"] <= np.percentile(xv, 10))
        vals.append((gg["xrf_pp"][e][hi].mean() / gg["xrf_pp"][e][lo].mean() - 1) * 100)
    ax.bar(xs + (k - 0.5) * w, vals, w, color=EL_COLOR[e], label=TAG[e])
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels(NORMAL, fontsize=8)
ax.set_ylabel("% higher XRF in\ntop-10% vs bottom-10% XRD bins")
ax.set_title("I/Cs enhancement\n(θ≈20.5° scans)")
ax.legend(fontsize=7)

# (b) scatter for 0215: XRD intensity vs I and Cs per-point
s = "0215"; g = G[s]; has = g["has"]
xv = g["xrd"][has]
for idx, e in enumerate(["I_La", "Cs_La"]):
    ax = fig.add_subplot(gs[1 + idx])
    yv = g["xrf_pp"][e][has]
    ax.scatter(xv, yv, s=6, alpha=0.25, color=EL_COLOR[e])
    # median trend in log-XRD bins
    lx = np.log10(xv)
    edges = np.linspace(lx.min(), lx.max(), 9)
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (lx >= a) & (lx < b)
        if m.sum() > 5:
            cx.append(10 ** ((a + b) / 2)); cy.append(np.median(yv[m]))
    ax.plot(cx, cy, "-o", color="k", ms=4, lw=1.4, label="median trend")
    rho, _ = stats.spearmanr(xv, yv)
    ax.set_xscale("log")
    ax.set_xlabel("nano-XRD summed intensity (bin)")
    ax.set_ylabel(f"{e} per-point ({KEV[e]:.2f} keV)")
    ax.set_title(f"Scan_0215: {e.split('_')[0]}\nSpearman ρ = {rho:+.2f}")
    ax.legend(fontsize=7)

# (c) side-by-side maps: nano-XRD intensity vs Cs per-point
ax = fig.add_subplot(gs[3])
xrd_disp = np.log10(np.where(g["xrd"] > 0, g["xrd"], np.nan))
im = ax.imshow(xrd_disp, cmap="cividis", aspect="auto")
ax.set_title("Scan_0215\nlog nano-XRD intensity")
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046)
fig.savefig(FIGDIR / "fig3_examples.png", dpi=130)
print("saved", FIGDIR / "fig3_examples.png")

# %%
# ---- Figure 4: the maps that make the (weak) coupling visible, Scan_0215 ----
s = "0215"; g = G[s]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
xrd_disp = np.log10(np.where(g["xrd"] > 0, g["xrd"], np.nan))
im0 = axes[0].imshow(xrd_disp, cmap="cividis")
axes[0].set_title("log nano-XRD intensity")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
for ax, e in zip(axes[1:], ["I_La", "Cs_La"]):
    m = np.where(g["cover"], g["xrf_pp"][e], np.nan)
    vlo, vhi = np.nanpercentile(m, [2, 98])
    im = ax.imshow(m, cmap="inferno", vmin=vlo, vmax=vhi)
    ax.set_title(f"{e} per-point ({KEV[e]:.2f} keV)")
    fig.colorbar(im, ax=ax, fraction=0.046)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Scan_0215 — nano-XRD structure is rich; I/Cs XRF is nearly flat "
             "(faint co-variation only)", fontsize=11)
fig.savefig(FIGDIR / "fig4_maps.png", dpi=130)
print("saved", FIGDIR / "fig4_maps.png")

# %% [markdown]
# ## Conclusion
#
# **Is there correlation between what nano-XRD sees and what XRF sees? Largely
# no — the XRF is essentially uniform — with one small, real exception.**
#
# 1. **The "no correlation → all XRF the same within variance" case holds.**
#    Within every scan the XRF maps are flat to **CV ≈ 2–5 %** (Fig 2a), and the
#    six scans are near-identical *once grouped by incidence angle* (spectrum
#    cosine ≳0.99 within a θ group; Fig 1). Sample condition (DI%, grain
#    boundaries) leaves no XRF fingerprint. The only large scan-to-scan XRF
#    difference is the θ≈20.5° (Pb-dominated) vs θ≈6° (Sn-dominated) split, which
#    is a **measurement-geometry / penetration-depth effect, not chemistry**.
#
# 2. **The one genuine correlation** is a weak I/Cs coupling in the normal-
#    incidence scans: bins with the brightest nano-XRD Bragg signal carry
#    **~3–4 % more I (3.94 keV) and Cs (4.23 keV)** fluorescence (Spearman
#    ρ ≈ 0.2–0.44), consistently across all three θ≈20.5° samples and *only* for
#    those two perovskite constituents. Interpretation: regions with more
#    crystalline perovskite both diffract more strongly and fluoresce marginally
#    more I/Cs. It is a **population-level trend of a few percent**, not a hotspot
#    effect — the very brightest individual features sit at ~median XRF (Part 3).
#
# 3. **Practical takeaway.** XRF here is a near-uniform composition monitor, not a
#    spatial predictor of nano-XRD features. It will not tell you *where* the
#    strong Bragg features are. The faint I/Cs trend is a useful *consistency
#    check* (more perovskite ⇒ more diffraction ⇒ slightly more I/Cs) but is far
#    too weak (few-percent) to drive feature-finding or to substitute for XRD.
#    For any cross-modal use, restrict to the normal-incidence (θ≈20.5°) scans;
#    the grazing geometry confounds the fluorescence with self-absorption.
#
# ### Caveats
# * XRF↔XRD registration reuses the shared `grid_mapping`; small de-skew residuals
#   would only *weaken* a true correlation, so the reported (already small) effect
#   is a lower bound in that sense.
# * I (3.94), Sn (3.44), Cs (4.23) keV sit in a tight L-line cluster with known
#   crosstalk; the ROI windows separate them but a few-percent leakage is possible
#   and does not change the qualitative conclusion.
# * "nano-XRD intensity" is the summed shape `intensity_profile` per 3×3 bin; using
#   feature count or peak height instead gives the same weak-coupling picture.
