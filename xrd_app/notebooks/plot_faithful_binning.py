"""Draw a toy example of coordinate-faithful gridding and spatial binning."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

from xrd_app.core import io  # noqa: E402


def main():
    n_acq_rows = 6
    samples_per_row = 13
    bin_size = 3
    rng = np.random.default_rng(7)

    # File index is the exact slow-axis row. The fast axis is acquired in a
    # serpentine path with a small alternate-row offset (stage backlash).
    slow_positions = np.linspace(0.0, 5.0, n_acq_rows)
    fast_positions = np.linspace(-5.0, 5.0, samples_per_row)
    frame_x = []
    frame_y = []
    frame_map = []
    for row, slow in enumerate(slow_positions):
        fast = fast_positions if row % 2 == 0 else fast_positions[::-1]
        backlash = 0.22 if row % 2 else -0.08
        for within_file, position in enumerate(fast):
            frame_x.append(slow + rng.normal(0.0, 0.025))
            frame_y.append(position + backlash + rng.normal(0.0, 0.045))
            frame_map.append([row, within_file])

    frame_x = np.asarray(frame_x)
    frame_y = np.asarray(frame_y)
    grid_row, grid_col, n_rows, n_cols = io.assign_grid_coordinate_faithful(
        frame_x, frame_y, frame_map, column_mode="square", log=print
    )
    grid_to_frames = {}
    for frame, (row, col) in enumerate(zip(grid_row, grid_col)):
        grid_to_frames.setdefault((int(row), int(col)), []).append(frame)
    bins, n_bin_rows, n_bin_cols = io.build_bin_mapping(
        n_rows, n_cols, bin_size, grid_to_frames
    )

    fast_lo, fast_hi = np.percentile(frame_y, [0.2, 99.8])
    colors = plt.colormaps["tab20"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)

    ax = axes[0]
    for row in range(n_acq_rows):
        take = np.asarray([fm[0] == row for fm in frame_map])
        order = np.flatnonzero(take)
        ax.plot(frame_y[take], frame_x[take], "-", color="#7a8793", lw=1.3)
        ax.scatter(frame_y[take], frame_x[take], s=30, color=colors(row), zorder=3)
        start = order[0]
        ax.annotate(
            f"start {start}", (frame_y[start], frame_x[start]),
            xytext=(4, 8), textcoords="offset points", fontsize=8,
        )
    ax.axvline(fast_lo, color="#c73e1d", ls="--", lw=1.8, label="0.2 percentile (min)")
    ax.axvline(fast_hi, color="#1f6f8b", ls="--", lw=1.8, label="99.8 percentile (max)")
    ax.scatter(frame_y[0], frame_x[0], marker="*", s=180, color="black", zorder=5)
    ax.set(
        title="1. True stage positions",
        xlabel="Fast-axis position (col axis)",
        ylabel="Slow-axis position (row axis)",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.5, -0.19,
        "Acquisition starts at the star and snakes row by row.\n"
        "Robust min/max ignore extreme 0.2% tails.",
        transform=ax.transAxes, fontsize=9, ha="center", va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )

    ax = axes[1]
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="#d5d9dc", lw=0.8, zorder=0)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="#d5d9dc", lw=0.8, zorder=0)
    for row, col in zip(grid_row, grid_col):
        ax.scatter(col, row, s=42, color=colors(row), edgecolor="white", lw=0.5)
    for (row, col), frames in grid_to_frames.items():
        if len(frames) > 1:
            ax.text(col, row, str(len(frames)), ha="center", va="center", fontsize=7)
    ax.annotate(
        "grid origin (row=0, col=0)", xy=(0, 0), xytext=(1.2, -0.85),
        arrowprops={"arrowstyle": "->", "color": "black"}, fontsize=9,
    )
    ax.text(
        0.98, 0.02, "Number 2 = two frames snapped into the same grid cell",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )
    ax.set(
        title=f"2. Faithful square grid: {n_rows} rows x {n_cols} cols",
        xlabel="grid_col (robust min -> 0, max -> n_cols - 1)",
        ylabel="grid_row (exact file/acquisition row)",
        xlim=(-1.0, n_cols - 0.35), ylim=(-1.15, n_rows - 0.35),
        xticks=range(n_cols), yticks=range(n_rows), aspect="equal",
    )
    ax.invert_yaxis()

    ax = axes[2]
    for br in range(n_bin_rows):
        for bc in range(n_bin_cols):
            key = f"{br}_{bc}"
            face = colors((br * n_bin_cols + bc) % 20)
            width = min(bin_size, n_cols - bc * bin_size)
            height = min(bin_size, n_rows - br * bin_size)
            rect = Rectangle(
                (bc * bin_size - 0.5, br * bin_size - 0.5), width, height,
                facecolor=face, edgecolor="black", alpha=0.28, lw=2,
            )
            ax.add_patch(rect)
            if key in bins:
                ax.text(
                    bc * bin_size - 0.5 + width / 2,
                    br * bin_size - 0.5 + height / 2,
                    f"bin {key}\n{len(bins[key])} frames",
                    ha="center", va="center", fontsize=9, weight="bold",
                )
    for x in np.arange(-0.5, n_cols, 1):
        ax.axvline(x, color="white", lw=0.8, zorder=0)
    for y in np.arange(-0.5, n_rows, 1):
        ax.axhline(y, color="white", lw=0.8, zorder=0)
    ax.set(
        title=f"3. {bin_size} x {bin_size} spatial bins",
        xlabel=f"bin_col = grid_col // {bin_size}",
        ylabel=f"bin_row = grid_row // {bin_size}",
        xlim=(-0.55, n_cols - 0.45), ylim=(-0.55, n_rows - 0.45),
        xticks=range(n_cols), yticks=range(n_rows), aspect="equal",
    )
    ax.invert_yaxis()

    fig.suptitle(
        "xrd-app assign_grid_coordinate_faithful: positions -> grid cells -> bins",
        fontsize=16, weight="bold",
    )
    output = Path(__file__).with_name("faithful_binning_demo.png")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
