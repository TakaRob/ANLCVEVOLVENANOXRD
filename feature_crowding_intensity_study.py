# %% [markdown]
# # Does feature crowding change intensity? (6-scan study)
#
# **Question (from the researcher):** when more features of the *same angle*
# overlap/coexist, does that *raise* the overall intensity, or the opposite —
# does packing more features into one scan *lower* the intensity of each?
#
# Two competing hypotheses were posed:
#
# 1. **Competition / shared budget** — more features visible in a scan *decrease*
#    the intensity of the others (a fixed diffracted-intensity budget split more ways).
# 2. **Independence** — distinct features don't overlap or interfere much, so
#    intensity is set by each grain locally and is ~independent of how many other
#    features are around.
#
# **What we measure.** We use the verified *features* (shapes) from
# `gaussian_shapes_3x3.json` for the six focus scans (179, 182, 203, 207, 215, 218).
# Each feature carries `peak_intensity` (brightest bin, background-subtracted),
# a per-bin `intensity_profile` (from which we sum `integrated` counts),
# `reflection` (which 2θ band / phase), `chi_deg` (azimuthal angle on the Debye
# ring), and its `center_row/col` on the sample grid.
#
# **"Same angle" is operationalised two ways:**
# - **crystallographic angle** = same `reflection` (same 2θ band) *and* similar
#   `chi_deg` — i.e. grains pointing the same way. This is the literal "overlapping
#   points of the same angle."
# - **spatial neighbourhood** = features sitting close on the sample surface,
#   which is the other thing "more features in a scan" could mean (local density).
#
# **Confounders we control for.** Intensity is wildly right-skewed (median ≈ 33
# counts, max ≈ 58 000) and depends strongly on *which* reflection it is (a bright
# ITO substrate spot vs. a faint perovskite (112)). So we (a) work in log10
# intensity and (b) measure **relative brightness** = a feature's log-intensity
# minus the median log-intensity of its own `(scan, reflection)` group. Relative
# brightness asks "is this feature bright *for its kind, in its scan*?" — which is
# the fair quantity to correlate against crowding.

# %%
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # headless (WSL, no X display for a data question)
import matplotlib.pyplot as plt
from pathlib import Path

# The six focus scans — 3x3 gaussian shapes (full intensity/chi/reflection payload).
SCANS = {
    "Scan_0179": "/home/takaji/179-201/Labels/Scan_0179/gaussian_shapes_3x3.json",
    "Scan_0182": "/home/takaji/179-201/Labels/Scan_0182/gaussian_shapes_3x3.json",
    "Scan_0203": "/home/takaji/rocking_203_214/Labels/Scan_0203/gaussian_shapes_3x3.json",
    "Scan_0207": "/home/takaji/rocking_203_214/Labels/Scan_0207/gaussian_shapes_3x3.json",
    "Scan_0215": "/home/takaji/215-226/Labels/Scan_0215/gaussian_shapes_3x3.json",
    "Scan_0218": "/home/takaji/215-226/Labels/Scan_0218/gaussian_shapes_3x3.json",
}

FIGDIR = Path(__file__).resolve().parent / "figures_crowding_intensity"
FIGDIR.mkdir(exist_ok=True)


def load_features(scans):
    """One row per verified feature across all scans."""
    rows = []
    for scan, path in scans.items():
        kept = json.load(open(path))["kept"]
        for f in kept:
            pi = f.get("peak_intensity")
            if pi is None or pi <= 0:
                continue  # combined/1x1 outputs have null intensities; skip defensively
            ip = f.get("intensity_profile", {}) or {}
            integ = sum((v.get("integrated") or 0) for v in ip.values())
            rows.append(dict(
                scan=scan,
                reflection=f["reflection"],
                chi_deg=f.get("chi_deg"),
                ref_tth=f.get("ref_tth"),
                n_bins=f["n_bins"],
                peak_intensity=pi,
                integrated_total=integ,
                center_row=f["center_row"],
                center_col=f["center_col"],
            ))
    return pd.DataFrame(rows)


df = load_features(SCANS)
df["logI"] = np.log10(df["peak_intensity"])
# Relative brightness: remove the scan x reflection median (control for phase & scan).
df["grp"] = df["scan"] + "|" + df["reflection"]
df["rel_bright"] = df["logI"] - df.groupby("grp")["logI"].transform("median")
print(f"Loaded {len(df)} features across {df['scan'].nunique()} scans")
print(df.groupby("scan").size())

# %% [markdown]
# ## Intensity is skewed and phase-driven — why we work in relative brightness
#
# Before any crowding test, note the scale of the confounders:
# `peak_intensity` spans ~4 orders of magnitude and its **median (~33) is far
# below its mean (~200–530)** — a handful of very bright spots dominate any raw
# average. And the median intensity differs by reflection (substrate ITO vs.
# faint perovskite orders). A raw "mean intensity vs. count" comparison would just
# be re-measuring *which phases happened to be present*, not competition. The
# `rel_bright` residual removes both, so the crowding correlations below are
# apples-to-apples.

# %%
# Sanity: per-reflection median intensity varies a lot (the confounder we remove).
ref_med = (df.groupby("reflection")["peak_intensity"].median()
             .sort_values(ascending=False))
print("Median peak_intensity by reflection:")
print(ref_med.round(1))

# %% [markdown]
# ## Analysis A — Cross-scan: does a busier scan mean dimmer features?
#
# The bluntest version of the question: across the six scans, does a *higher total
# feature count* go with *lower typical intensity* (competition) or not?
# We correlate feature count against (i) the median log peak-intensity and
# (ii) the total integrated counts summed over all features in the scan.

# %%
scan_stats = df.groupby("scan").agg(
    n_features=("peak_intensity", "size"),
    median_logI=("logI", "median"),
    median_peakI=("peak_intensity", "median"),
    total_integrated=("integrated_total", "sum"),
).reset_index()

n = scan_stats["n_features"].values
sp_med = stats.spearmanr(n, scan_stats["median_logI"])
sp_tot = stats.spearmanr(n, scan_stats["total_integrated"])
print(scan_stats.to_string(index=False))
print(f"\nSpearman  n_features vs median logI     : rho={sp_med.statistic:+.2f}  p={sp_med.pvalue:.3f}")
print(f"Spearman  n_features vs total integrated: rho={sp_tot.statistic:+.2f}  p={sp_tot.pvalue:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(n, scan_stats["median_peakI"], s=60, color="#3b6ea5")
for _, r in scan_stats.iterrows():
    ax[0].annotate(r["scan"].replace("Scan_0", ""), (r["n_features"], r["median_peakI"]),
                   fontsize=8, xytext=(3, 3), textcoords="offset points")
ax[0].set_xlabel("features in scan (n_kept)")
ax[0].set_ylabel("median peak_intensity")
ax[0].set_title(f"Typical brightness vs. scan busyness\nSpearman rho={sp_med.statistic:+.2f} (p={sp_med.pvalue:.2f})")

ax[1].scatter(n, scan_stats["total_integrated"] / 1e6, s=60, color="#a5533b")
for _, r in scan_stats.iterrows():
    ax[1].annotate(r["scan"].replace("Scan_0", ""),
                   (r["n_features"], r["total_integrated"] / 1e6),
                   fontsize=8, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("features in scan (n_kept)")
ax[1].set_ylabel("total integrated counts (millions)")
ax[1].set_title(f"Total diffracted signal vs. count\nSpearman rho={sp_tot.statistic:+.2f} (p={sp_tot.pvalue:.2f})")
fig.tight_layout()
fig.savefig(FIGDIR / "A_cross_scan.png", dpi=120)
print("saved", FIGDIR / "A_cross_scan.png")

# %% [markdown]
# ### Conclusion A
#
# **No competition signal at the scan level, and the opposite of a shared budget.**
#
# - Feature count vs. *median* brightness: **Spearman rho = -0.43 (p = 0.40)** —
#   a weak, non-significant negative lean. With only 6 scans this is noise, not a
#   fixed-budget effect.
# - Feature count vs. *total* integrated signal: **rho = +0.60 (p = 0.21)** —
#   busier scans carry *more* total diffracted signal, not less. If features were
#   splitting a fixed pie, total signal would be flat and per-feature intensity
#   would fall; neither happens.
#
# So at the coarsest level the data already lean toward **hypothesis 2
# (independence)**: adding features adds signal, it doesn't dilute it. (Caveat:
# n = 6; treat A as directional, and let the within-scan tests below carry the
# statistical weight.)

# %% [markdown]
# ## Analysis B — Same-angle crowding (the core test)
#
# This is the literal question: *for each feature, how many other features share
# its angle, and is it dimmer when that angle is crowded?*
#
# For every feature we count "co-oriented neighbours" = other features **in the
# same scan, same reflection, with `chi_deg` within ±Δχ**. Then we correlate that
# crowding count against **relative brightness** (`rel_bright`, which already has
# the scan and reflection effects removed). A strong negative correlation would
# mean competition; ~zero means independence.

# %%
def same_angle_crowd(df, dchi):
    out = np.full(len(df), np.nan)
    pos = {s: i for i, s in enumerate(df.index)}  # not used; keep index-safe below
    for scan, sd in df.groupby("scan"):
        chi = sd["chi_deg"].values
        refl = sd["reflection"].values
        idx = sd.index.values
        for k in range(len(sd)):
            if chi[k] is None or (isinstance(chi[k], float) and np.isnan(chi[k])):
                continue
            same = (refl == refl[k]) & (np.abs(chi - chi[k]) <= dchi)
            out[df.index.get_loc(idx[k])] = int(same.sum() - 1)
    return out


for dchi in (5, 10, 15):
    df[f"crowd_chi{dchi}"] = same_angle_crowd(df, dchi)

for dchi in (5, 10, 15):
    col = f"crowd_chi{dchi}"
    sub = df.dropna(subset=[col])
    sp = stats.spearmanr(sub[col], sub["rel_bright"])
    print(f"Delta_chi=+/-{dchi:2d} deg : co-oriented neighbours "
          f"mean={sub[col].mean():5.1f} (0-{int(sub[col].max())}) | "
          f"Spearman(crowd, rel_bright) rho={sp.statistic:+.3f}  p={sp.pvalue:.3f}")

# Plot binned relative brightness vs same-angle crowding (Delta_chi = 10 deg).
sub = df.dropna(subset=["crowd_chi10"]).copy()
bins = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, 1e9]
labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
sub["cbin"] = pd.cut(sub["crowd_chi10"], bins=bins, labels=labels)
grp = sub.groupby("cbin", observed=True)["rel_bright"]
fig, ax = plt.subplots(figsize=(7, 4.5))
data = [sub.loc[sub["cbin"] == lb, "rel_bright"].values for lb in labels]
ax.boxplot(data, tick_labels=labels, showfliers=False)
ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_xlabel("co-oriented neighbours (same reflection, chi within +/-10 deg)")
ax.set_ylabel("relative brightness  [log10, vs scan x reflection median]")
sp10 = stats.spearmanr(sub["crowd_chi10"], sub["rel_bright"])
ax.set_title(f"Same-angle crowding vs. brightness\nSpearman rho={sp10.statistic:+.3f} (p={sp10.pvalue:.3f}), "
             f"r^2={sp10.statistic**2*100:.2f}%")
fig.tight_layout()
fig.savefig(FIGDIR / "B_same_angle_crowd.png", dpi=120)
print("saved", FIGDIR / "B_same_angle_crowd.png")

# %% [markdown]
# ### Conclusion B
#
# **First: co-oriented crowding is the *norm*, not the exception.** Averaged over
# the six scans a feature has **~11 same-reflection neighbours within ±5° of chi**
# (up to ~45), and ~18 within ±10°. Features of a given reflection cluster tightly
# in chi — that's crystallographic **texture** (a preferred orientation), so
# "overlapping points of the same angle" happens all the time.
#
# **Second, and decisively: crowding barely touches brightness.**
#
# | Δχ | Spearman(crowd, rel_bright) | p | variance explained |
# |----|------|------|------|
# | ±5°  | **-0.067** | 0.011 | ~0.4% |
# | ±10° | **-0.053** | 0.044 | ~0.3% |
# | ±15° | **-0.043** | 0.10  | ~0.2% |
#
# The correlation is **negative but vanishingly small** — statistically detectable
# at ±5–10° only because there are ~1460 features, but it explains **well under
# 1%** of the brightness variation. The boxplot is flat: median relative
# brightness sits on ~0 across every crowding bucket from 0 to 21+ neighbours.
#
# Physically, a real "shared budget" would give a steep negative slope
# (rho closer to -0.5). What we see instead is essentially independence, with a
# whisper of a negative trend that is more plausibly a **detection artefact** than
# physics: near a very bright spot, the local SNR floor rises and adjacent faint
# co-oriented features are slightly harder to call, nudging the residual down.
# There is **no evidence that same-angle features rob each other of intensity.**

# %% [markdown]
# ## Analysis C — Spatial crowding: do features dim when packed close on the sample?
#
# The other reading of "more features in a scan" is *local density on the sample
# surface*. For each feature we count how many other features fall within R bins
# of its `center_row/col`, and correlate that with relative brightness.

# %%
def spatial_density(df, R):
    dens = np.zeros(len(df), dtype=int)
    for scan, sd in df.groupby("scan"):
        r = sd["center_row"].values
        c = sd["center_col"].values
        idx = sd.index.values
        for k in range(len(sd)):
            d2 = (r - r[k]) ** 2 + (c - c[k]) ** 2
            dens[df.index.get_loc(idx[k])] = int(((d2 <= R * R) & (d2 > 0)).sum())
    return dens


for R in (3, 5, 8):
    df[f"dens{R}"] = spatial_density(df, R)
    sp = stats.spearmanr(df[f"dens{R}"], df["rel_bright"])
    print(f"R={R} bins: neighbours mean={df[f'dens{R}'].mean():5.2f} | "
          f"Spearman(density, rel_bright) rho={sp.statistic:+.3f}  p={sp.pvalue:.3f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df["dens5"] + np.random.default_rng(0).normal(0, 0.08, len(df)),
           df["rel_bright"], s=6, alpha=0.25, color="#3b6ea5")
ax.axhline(0, color="grey", ls="--", lw=1)
sp5 = stats.spearmanr(df["dens5"], df["rel_bright"])
ax.set_xlabel("neighbouring features within 5 bins on the sample")
ax.set_ylabel("relative brightness  [log10]")
ax.set_title(f"Spatial density vs. brightness\nSpearman rho={sp5.statistic:+.3f} (p={sp5.pvalue:.2f}) - no relationship")
fig.tight_layout()
fig.savefig(FIGDIR / "C_spatial_density.png", dpi=120)
print("saved", FIGDIR / "C_spatial_density.png")

# %% [markdown]
# ### Conclusion C
#
# **Spatial crowding has zero effect on brightness.** Spearman correlations are
# rho = +0.02 (R=3), -0.00 (R=5), -0.01 (R=8), all non-significant (p ≥ 0.38).
# Features packed close together on the sample are neither brighter nor dimmer
# than isolated ones. This is the cleanest independence result of the three —
# consistent with each grain diffracting on its own, and with the beam sampling
# one position at a time (spatial neighbours are literally different frames, so
# there is no physical mechanism for them to share intensity).

# %% [markdown]
# ## Analysis D — Summary table across crowding buckets
#
# One table pulling B and C together: median relative brightness and median raw
# peak intensity as crowding increases. Flat columns = independence.

# %%
sub = df.dropna(subset=["crowd_chi10"]).copy()
sub["cbin"] = pd.cut(sub["crowd_chi10"], bins=[-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, 1e9],
                     labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
summary = sub.groupby("cbin", observed=True).agg(
    n=("peak_intensity", "size"),
    median_rel_bright=("rel_bright", "median"),
    median_peakI=("peak_intensity", "median"),
).reset_index()
print("Same-angle crowding buckets (Delta_chi = 10 deg):")
print(summary.to_string(index=False))

# %% [markdown]
# ## Overall conclusion
#
# **The data support hypothesis 2: distinct features are essentially independent —
# they do not overlap or interfere enough to steal intensity from one another.**
#
# Across the six focus scans (179, 182, 203, 207, 215, 218; 1 463 verified
# features, 3×3 gaussian shapes):
#
# 1. **No shared-budget effect between scans.** Busier scans do *not* have dimmer
#    features; if anything they carry *more* total diffracted signal
#    (rho = +0.60). Adding features adds signal.
# 2. **Same-angle crowding does not lower intensity.** Even though co-oriented
#    features are extremely common (texture: ~11–18 same-reflection neighbours
#    within a few degrees of chi), the correlation between crowding and relative
#    brightness is rho ≈ -0.05 to -0.07 — negative but explaining **<1%** of the
#    variance. Not the steep negative slope a real competition would produce.
# 3. **Spatial density is irrelevant.** rho ≈ 0 at every radius.
#
# **So neither of the researcher's phrasings of "competition" holds up.** More
# features in a scan do *not* meaningfully dim the others, and more overlapping
# points of the same angle do *not* boost a single feature's intensity either.
# Each feature's intensity is set locally (grain size, illuminated volume,
# structure factor for that reflection) and is effectively independent of how many
# neighbours — in angle or in space — surround it.
#
# **The one caveat / whisper of a signal:** the tiny, marginally-significant
# *negative* same-angle correlation (strongest at ±5°, p = 0.01). It is far too
# weak to be physical competition; the more likely cause is a **detection floor
# effect** — a bright spot raises the local SNR threshold so faint co-oriented
# neighbours read slightly lower. If you want to chase it, the test is to redo
# this on the 1×1 territory catalogs (finer angular resolution) and check whether
# the effect survives when detection SNR is held fixed. But for the physics
# question as asked, the answer is: **features don't fight over intensity.**
#
# **Physics-sanity notes / limitations.**
# - Uses `peak_intensity` (background-subtracted, brightest bin) and summed
#   `integrated` counts — both from the same 3×3 gaussian pipeline, so cross-scan
#   comparisons are fair (same detector/SNR lineage; verify with `xrd-app lineage`).
# - `chi_deg` depends on an *estimated* beam centre, so absolute chi is
#   approximate; the analysis only uses chi *differences* within a scan, which are
#   robust.
# - Results are for 3×3 binning. 1×1 over-segments and combined runs have null
#   intensities, so 3×3 is the right granularity for an intensity question.
# - Figures written to `figures_crowding_intensity/`.
