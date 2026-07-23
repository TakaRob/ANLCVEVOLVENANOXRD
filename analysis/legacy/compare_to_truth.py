# %% [markdown]
'''
# Compare fast 1×1 shapes against the territorial source-of-truth

**Scan 0203 · ISN 26-ID-C**

The territorial (cell-model) catalog bins by **true (X, Y) stage positions**, so
it is immune to the serpentine/backlash registration error that skews the N×N
grid (see `xrd_app/core/territory.py`). That makes it the **source of truth** for
tuning the fast post-processing (skew / backlash fixing) applied to the 1×1 grid
catalog.

This script matches a fast 1×1 shapes catalog (optionally skew-fixed) to the
territorial truth **in a common true-(X, Y) + detector-pixel space at full 1×1
resolution** — that is how we "match resolution to the 1×1 data": every feature,
truth or fast, is projected to physical position before matching, so the coarser
territories do not bias the comparison. It reports recall / precision / mean F2
(recall-weighted, per `CLAUDE.md`) plus the residual spatial-offset vector — the
skew the fast algorithm still has. A better skew fix → higher recall/F2 and a
smaller offset.

## How to run

Edit the **CONFIG** cell, then run the file (`python3 compare_to_truth.py`) or
cell-by-cell. Generate the truth first if you have not::

    xrd-app territory-grid --target-size 9
    xrd-app bin    --bin-size 1 --variant territory
    xrd-app peaks  --bin-size 1 --variant territory
    xrd-app shapes --bin-size 1 --variant territory --algorithm territory
'''

# %%
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xrd_app.config import DataManager          # noqa: E402
from xrd_app.core import io                      # noqa: E402

# %% [markdown]
# ## CONFIG — point these at your project / scan, then re-run.

# %%
PROJECT_ROOT = "TakaTest/TakaProject"  # project tree (Raw/ Binned/ Metadata/ Labels/)
SCAN = None                      # None → the project's configured scan

# Fast 1×1 shapes catalogs to score against truth. Compare several (e.g. the
# current catalog vs a skew-fixed one) — each becomes a row in the report.
# None → auto-discover the plain 1×1 shapes for the scan.
FAST_SHAPES = [
    "gaussian_shapes_1x1.json",            # grid linking (shapes --grid-link) — the old default
    "territory_shapes_1x1_coord.json",     # gridless coordinate linking (shapes --bin-size 1, new default)
    "gaussian_shapes_1x1_faithful.json",   # faithful re-grid (lattice snap to true Y) — for reference
]

# center_bin keys of a re-gridded catalog index ITS grid, not the default one.
# Map each such catalog to its grid variant so its (X,Y) projection is correct;
# anything not listed uses the default 1×1 grid.
FAST_GRIDS = {
    "gaussian_shapes_1x1_faithful.json": "faithful",
}

DET_TOL = 8.0     # max detector-pixel distance for a match (≈ link tolerance)
XY_TOL = None     # max true-(X,Y) distance for a match; None → 3 × territory step

# %% [markdown]
# ## Loaders


# %%
def _kept(path):
    with open(path) as f:
        return json.load(f).get("kept", [])


def _resolve(dm, scan):
    """Resolve truth shapes + territorial mapping, the 1×1 grid mapping, positions."""
    truth = dm.shapes_json("territory", 1, scan, variant="territory")
    if not Path(truth).exists():
        hits = sorted(dm.labels_dir(scan).glob("*_shapes*territory*.json"))
        truth = hits[-1] if hits else truth
    terr_gm = dm.grid_mapping(bin_size=1, scan=scan, variant="territory")
    grid1 = dm.grid_mapping(bin_size=1, scan=scan)
    pos = dm.position_csv(scan=scan)
    return Path(truth), Path(terr_gm), Path(grid1), Path(pos)


def _territory_centroids(terr_gm):
    """{territory_key: (X, Y)} from the territorial grid mapping's polygon block."""
    with open(terr_gm) as f:
        terr = json.load(f).get("territories", {})
    return {k: tuple(v["centroid_xy"]) for k, v in terr.items()
            if v.get("centroid_xy")}


def _fast_catalogs(dm, scan):
    if FAST_SHAPES:
        return [Path(p) if Path(p).is_absolute() else dm.labels_dir(scan) / p
                for p in FAST_SHAPES]
    # Auto: plain (non-territory) 1×1 shapes catalogs.
    ldir = dm.labels_dir(scan)
    return [p for p in sorted(ldir.glob("*_shapes_1x1*.json"))
            if "territory" not in p.name]


def _bin_xy(grid1, pos_csv):
    """Map every 1×1 bin key '<r>_<c>' → its mean true (X, Y) from positions."""
    gm = io.load_grid_mapping(grid1)
    n_total = gm.get("n_total_frames") or len(gm.get("frame_map", []))
    fx, fy = io.load_positions_xy(pos_csv, n_total)
    fx, fy = io._interp_nan(fx), io._interp_nan(fy)
    out = {}
    for key, frames in gm["bins"].items():
        fr = [i for i in frames if i < n_total]
        if fr:
            out[key] = (float(np.mean(fx[fr])), float(np.mean(fy[fr])))
    return out


def _truth_points(feats, terr_centroids):
    """Truth features → (reflection, det, X, Y). The feature's physical position
    is its brightest territory's (center_bin) centroid from the mapping."""
    pts = []
    for f in feats:
        xy = terr_centroids.get(f.get("center_bin"))
        pts.append({
            "reflection": f.get("reflection"),
            "det": (f.get("detector_x"), f.get("detector_y")),
            "xy": xy, "n_bins": f.get("n_bins"), "chi": f.get("chi_deg"),
        })
    return pts


def _fast_points(feats, bin_xy):
    pts = []
    for f in feats:
        cb = f.get("center_bin")
        xy = bin_xy.get(cb)
        pts.append({
            "reflection": f.get("reflection"),
            "det": (f.get("detector_x"), f.get("detector_y")),
            "xy": xy, "n_bins": f.get("n_bins"), "chi": f.get("chi_deg"),
        })
    return pts


# %% [markdown]
# ## Matching + metrics


# %%
def _match(truth, fast, det_tol, xy_tol):
    """Greedy one-to-one match by reflection + detector proximity.

    Builds all candidate (truth, fast) pairs within tolerance, sorts by detector
    distance, and assigns greedily. Returns (pairs, offsets) where offsets are
    fast_xy − truth_xy for pairs that have both XY (the residual skew).
    """
    cand = []
    for i, t in enumerate(truth):
        for j, fz in enumerate(fast):
            if t["reflection"] != fz["reflection"]:
                continue
            if None in t["det"] or None in fz["det"]:
                continue
            dd = np.hypot(t["det"][0] - fz["det"][0], t["det"][1] - fz["det"][1])
            if dd > det_tol:
                continue
            if xy_tol is not None and t["xy"] and fz["xy"]:
                if np.hypot(t["xy"][0] - fz["xy"][0], t["xy"][1] - fz["xy"][1]) > xy_tol:
                    continue
            cand.append((dd, i, j))
    cand.sort()
    used_t, used_f, pairs = set(), set(), []
    for dd, i, j in cand:
        if i in used_t or j in used_f:
            continue
        used_t.add(i); used_f.add(j); pairs.append((i, j, dd))
    offsets = []
    for i, j, _ in pairs:
        if truth[i]["xy"] and fast[j]["xy"]:
            offsets.append((fast[j]["xy"][0] - truth[i]["xy"][0],
                            fast[j]["xy"][1] - truth[i]["xy"][1]))
    return pairs, offsets


def _match_footprint(truth, fast, det_tol, xy_tol):
    """Many-to-one (footprint) match: a coarse truth shape may absorb several fine
    fast shapes without penalty. Removes the 1-to-1 count cap on precision.

    Returns (n_truth_covered, n_fast_on_target, offsets) — a truth shape is
    *covered* if ≥1 fast shape of the same reflection lands within tolerance; a
    fast shape is *on-target* if it lands on some truth shape. ``offsets`` are
    fast_xy − nearest-truth_xy over all on-target fast shapes (the residual skew,
    now uncapped by the match count).
    """
    covered = [False] * len(truth)
    on_target = 0
    offsets = []
    for fz in fast:
        if None in fz["det"]:
            continue
        best, best_i = None, None
        for i, t in enumerate(truth):
            if t["reflection"] != fz["reflection"] or None in t["det"]:
                continue
            dd = np.hypot(t["det"][0] - fz["det"][0], t["det"][1] - fz["det"][1])
            if dd > det_tol:
                continue
            if xy_tol is not None and t["xy"] and fz["xy"]:
                if np.hypot(t["xy"][0] - fz["xy"][0], t["xy"][1] - fz["xy"][1]) > xy_tol:
                    continue
            if best is None or dd < best:
                best, best_i = dd, i
        if best_i is not None:
            on_target += 1
            covered[best_i] = True
            if truth[best_i]["xy"] and fz["xy"]:
                offsets.append((fz["xy"][0] - truth[best_i]["xy"][0],
                                fz["xy"][1] - truth[best_i]["xy"][1]))
    return sum(covered), on_target, offsets


def _metrics(n_truth, n_fast, n_match):
    recall = n_match / n_truth if n_truth else 0.0
    prec = n_match / n_fast if n_fast else 0.0
    f2 = (5 * prec * recall / (4 * prec + recall)) if (4 * prec + recall) else 0.0
    return recall, prec, f2


def _f2(recall, prec):
    return (5 * prec * recall / (4 * prec + recall)) if (4 * prec + recall) else 0.0


# %% [markdown]
# ## Run


# %%
def main():
    dm = DataManager(PROJECT_ROOT, scan=SCAN)
    truth_path, terr_gm, grid1, pos = _resolve(dm, SCAN)
    if not truth_path.exists():
        sys.exit(f"No territorial truth at {truth_path}. Build it first (see header).")
    if not terr_gm.exists():
        sys.exit(f"Need the territorial grid mapping ({terr_gm}) for centroids.")
    if not grid1.exists() or not pos.exists():
        sys.exit(f"Need the 1×1 grid mapping ({grid1}) and positions ({pos}).")

    step = json.loads(terr_gm.read_text()).get("step")
    xy_tol = XY_TOL if XY_TOL is not None else (3 * step if step else None)

    truth = _truth_points(_kept(truth_path), _territory_centroids(terr_gm))
    fasts = _fast_catalogs(dm, SCAN)
    if not fasts:
        sys.exit("No fast 1×1 shapes catalogs found (set FAST_SHAPES).")

    # (X,Y) projection per grid variant — re-gridded catalogs index their own grid.
    bin_xy_cache = {}
    def bin_xy_for(variant):
        if variant not in bin_xy_cache:
            g = dm.grid_mapping(bin_size=1, variant=variant) if variant else grid1
            bin_xy_cache[variant] = _bin_xy(g, pos) if Path(g).exists() else {}
        return bin_xy_cache[variant]

    print(f"\nTruth: {truth_path.name}  ({len(truth)} shapes)")
    print(f"Match: detector ≤ {DET_TOL}px"
          + (f", XY ≤ {xy_tol:.3g}" if xy_tol else "")
          + "    [1:1 = greedy one-to-one;  fp = many-to-one footprint]\n")
    hdr = (f"{'fast catalog':38s} {'n':>5s} | {'R(1:1)':>6s} {'F2(1:1)':>7s} | "
           f"{'R(fp)':>6s} {'P(fp)':>6s} {'F2(fp)':>6s} {'|offset|':>9s}")
    print(hdr); print("-" * len(hdr))
    for fp in fasts:
        if not fp.exists():
            print(f"{fp.name:38s}   (missing)")
            continue
        variant = FAST_GRIDS.get(fp.name)
        fast = _fast_points(_kept(fp), bin_xy_for(variant))
        n = len(fast)
        # 1-to-1 (original ruler)
        pairs, _ = _match(truth, fast, DET_TOL, xy_tol)
        r11, _, f211 = _metrics(len(truth), n, len(pairs))
        # many-to-one footprint (count-confound removed)
        cov, on_t, offs = _match_footprint(truth, fast, DET_TOL, xy_tol)
        rfp = cov / len(truth) if truth else 0.0
        pfp = on_t / n if n else 0.0
        f2fp = _f2(rfp, pfp)
        mag = np.hypot(np.mean([o[0] for o in offs]),
                       np.mean([o[1] for o in offs])) if offs else float("nan")
        print(f"{fp.name:38s} {n:5d} | {r11:6.3f} {f211:7.3f} | "
              f"{rfp:6.3f} {pfp:6.3f} {f2fp:6.3f} {mag:9.4f}")
    print("\nfp metrics remove the count cap (a coarse truth shape may absorb several\n"
          "fine fast shapes). Higher R(fp)/F2(fp) + smaller |offset| ⇒ better skew fix.")


if __name__ == "__main__":
    main()
