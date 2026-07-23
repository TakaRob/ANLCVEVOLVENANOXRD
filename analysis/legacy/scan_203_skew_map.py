# %% [markdown]
'''
# Scan 203 — bin-occupancy & detector-space skew

Goal: see *why* bins like `3_68`, `3_70`, `3_72` (and the empty `4_73`) show up far
from everything else, and how features #35/#36 end up looking isolated.

Run cell-by-cell (VSCode "Run Cell" / Jupyter). Each `# %%` is a cell, each
triple-quoted block is a markdown note.

## Pick your dataset
Set `DATASET` in the config cell below:
- `"3x3"` -> coarse binned grid (52 x 74) + feature overlays.
- `"raw"` -> **every individual scan point** (1x1, 156 x 221). Same occupancy
  graph, just all the points. (Other bin sizes `"4x4"`, `"5x5"` also work.)

Key finding either way:
- The scan is **not** a clean rectangle. The bulk of every row stops early.
- **One row is a 1-bin-tall "whisker"** that runs far past the rest -> the skew.
- Features #35/#36 (3x3) sit on that whisker, which is why they look detached.
'''

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ─── CONFIG ──────────────────────────────────────────────────────────
# "raw" = 1x1 (all individual points) | "3x3" | "4x4" | "5x5"
DATASET = "3x3"
# ─────────────────────────────────────────────────────────────────────

BASE = Path("/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo")
if not BASE.exists():
    BASE = Path.cwd()

SCAN = "Scan_0203"
BIN_SIZE = 1 if DATASET == "raw" else int(DATASET.split("x")[0])

GRID_MAP = BASE / "data" / "holdout" / SCAN / f"grid_mapping_{BIN_SIZE}x{BIN_SIZE}.json"
CATALOG  = BASE / "results" / SCAN / f"feature_catalog_{BIN_SIZE}x{BIN_SIZE}.json"
HAS_CATALOG = CATALOG.exists()

# 3x3 bins to call out (only meaningful when DATASET == "3x3")
TARGET_BINS = ["3_68", "3_70", "3_72", "4_73"]

print(f"DATASET = {DATASET}  (bin size {BIN_SIZE})")
print("grid map :", GRID_MAP.exists(), GRID_MAP.name)
print("catalog  :", HAS_CATALOG, CATALOG.name if HAS_CATALOG else "(none for this bin size)")

# %% [markdown]
'''
## Load the occupancy grid

`grid_mapping_{n}x{n}.json` has a `bins` dict keyed by `"row_col"`. A key present
= that point was actually scanned / has data. We build a boolean occupancy grid.
'''

# %%
gm = json.load(open(GRID_MAP))
N_ROWS, N_COLS = gm["n_bin_rows"], gm["n_bin_cols"]

occ = np.zeros((N_ROWS, N_COLS), dtype=bool)
for k in gm["bins"]:
    r, c = map(int, k.split("_"))
    occ[r, c] = True

# rightmost occupied column per row + the "whisker" row (largest overshoot)
max_col = np.array([np.where(occ[r])[0].max() if occ[r].any() else -1
                    for r in range(N_ROWS)])
whisker_row = int(np.argmax(max_col))

print(f"grid = {N_ROWS} x {N_COLS} = {N_ROWS*N_COLS} cells")
print(f"occupied = {occ.sum()}  ({100*occ.sum()/occ.size:.1f}% filled)")
print(f"whisker row = {whisker_row}  (reaches col {max_col[whisker_row]}, "
      f"vs median row edge {int(np.median(max_col[max_col>=0]))})")

# %% [markdown]
'''
## Map 1 — occupancy (all points), whisker highlighted

Grey = scanned, white = empty. The auto-detected whisker row is tinted orange.
With `DATASET="raw"` this shows every individual scan point.
'''

# %%
fig, ax = plt.subplots(figsize=(13, 9))
ax.imshow(occ, cmap=ListedColormap(["white", "#bcbcbc"]),
          origin="upper", interpolation="nearest", aspect="equal")

# tint the whisker row where occupied
row_mask = np.zeros_like(occ, dtype=float)
row_mask[whisker_row, :] = occ[whisker_row, :]
ax.imshow(np.ma.masked_where(row_mask == 0, row_mask),
          cmap=ListedColormap(["#f0a030"]), origin="upper",
          interpolation="nearest", aspect="equal", alpha=0.55)

# called-out bins (only when on the 3x3 grid those labels refer to)
if DATASET == "3x3":
    for bk in TARGET_BINS:
        r, c = map(int, bk.split("_"))
        present = occ[r, c]
        ax.scatter([c], [r], s=140, marker="s", facecolors="none",
                   edgecolors=("lime" if present else "red"), linewidths=2.2, zorder=5)
        ax.annotate(bk, (c, r), textcoords="offset points", xytext=(6, -10),
                    color=("green" if present else "red"), fontsize=9, weight="bold")

ax.set_title(f"Scan 203 occupancy — DATASET={DATASET} ({N_ROWS}x{N_COLS})\n"
             f"grey=data, white=empty, orange=whisker row {whisker_row}")
ax.set_xlabel("col (scan X)")
ax.set_ylabel("row (scan Y)")
ax.grid(True, color="#dddddd", linewidth=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
'''
## Map 2 — rightmost data per row (the skew, quantified)

One bar per row. Almost every row stops near the same column; the whisker row
(orange) overshoots far past the rest. That spike *is* the skew.
'''

# %%
fig, ax = plt.subplots(figsize=(11, 7))
rows = np.arange(N_ROWS)
ax.barh(rows, np.where(max_col >= 0, max_col, 0), color="#7aa6c2")
ax.barh([whisker_row], [max_col[whisker_row]], color="#f0a030")
med = int(np.median(max_col[max_col >= 0]))
ax.axvline(med, color="grey", ls="--", lw=1, label=f"median row edge (col {med})")
ax.set_xlabel("max occupied column in row")
ax.set_ylabel("row")
ax.set_title(f"Rightmost data per row — whisker row {whisker_row} (orange) overshoots")
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
'''
## Map 3 — feature centers over occupancy  (3x3 only)

Only the 3x3 catalog exists, so this cell runs when `DATASET="3x3"`. #35/#36 sit
out on the whisker with no neighbours above/below.
'''

# %%
if HAS_CATALOG:
    cat = json.load(open(CATALOG))
    fx = [f["center_col"] for f in cat]
    fy = [f["center_row"] for f in cat]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(occ, cmap=ListedColormap(["white", "#e3e3e3"]),
              origin="upper", interpolation="nearest", aspect="equal")
    ax.scatter(fx, fy, s=12, c="#1f77b4", label="feature centers")
    for fid in (34, 35, 36):
        f = next((g for g in cat if g.get("feature_id") == fid), None)
        if f:
            ax.scatter([f["center_col"]], [f["center_row"]], s=160, marker="*",
                       c="red", zorder=6)
            ax.annotate(f"#{fid} {f['reflection']}", (f["center_col"], f["center_row"]),
                        textcoords="offset points", xytext=(6, 4),
                        color="red", fontsize=9, weight="bold")
    ax.set_title(f"{len(cat)} feature centers (red stars = #34/#35/#36)")
    ax.set_xlabel("col (scan X)")
    ax.set_ylabel("row (scan Y)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
else:
    print(f"No feature catalog for {DATASET} — skipping feature overlay.")

# %% [markdown]
'''
## Map 4 — detector-space scatter  (3x3 only)

Where features land on the detector (pixels), coloured by reflection. Spread here
is by diffraction angle, not sample skew.
'''

# %%
if HAS_CATALOG:
    cat = json.load(open(CATALOG))
    dx = np.array([f["detector_x"] for f in cat], float)
    dy = np.array([f["detector_y"] for f in cat], float)
    refs = [f["reflection"] for f in cat]
    uref = sorted(set(refs))
    cmap = plt.get_cmap("tab10")
    rcolor = {r: cmap(i % 10) for i, r in enumerate(uref)}

    fig, ax = plt.subplots(figsize=(10, 10))
    for r in uref:
        m = [i for i, rr in enumerate(refs) if rr == r]
        ax.scatter(dx[m], dy[m], s=18, color=rcolor[r], label=r)
    for fid in (35, 36):
        f = next((g for g in cat if g.get("feature_id") == fid), None)
        if f:
            ax.scatter([f["detector_x"]], [f["detector_y"]], s=180, marker="*",
                       edgecolors="k", facecolors="none", linewidths=1.6, zorder=6)
            ax.annotate(f"#{fid}", (f["detector_x"], f["detector_y"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_title("Feature detector-space positions (colour = reflection)")
    ax.set_xlabel("detector_x (px)")
    ax.set_ylabel("detector_y (px)")
    ax.invert_yaxis()
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
else:
    print(f"No feature catalog for {DATASET} — skipping detector scatter.")

# %% [markdown]
'''
## Summary

- Set `DATASET="raw"` (top) to see the occupancy graph with **every scan point**
  (1x1, 156x221); `DATASET="3x3"` for the coarse binned view + feature overlays.
- Either way one row is a 1-bin-tall **whisker** that overshoots the rest = the skew.
- `4_73` (3x3) is empty; the only far-right column bin is `3_73`.
- Features #35/#36 sit on that whisker, which is why they look isolated.
'''
