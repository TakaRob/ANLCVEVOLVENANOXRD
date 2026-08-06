"""Build Sarah-style landscape PDF reports from existing XRD result artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import DataManager
from . import (
    catalogs, detector_display, device_maps, io, processing, reflection_sum,
    reflections,
)


@dataclass(frozen=True)
class ReportTarget:
    scan: str
    bin_size: int
    catalog: Optional[Path] = None


@dataclass(frozen=True)
class ReportOptions:
    summed_images: bool = True
    all_reflections: bool = True
    features_by_reflection: bool = True
    top_features: bool = True
    top_count: int = 5
    top_scope: str = "reflection"
    allow_more_than_five: bool = False
    source_images: bool = True
    roi_images: bool = False
    calculate_rois: bool = False
    territory_maps: bool = False

    def validated(self):
        if self.top_count < 1:
            raise ValueError("Top feature count must be at least one")
        if self.top_scope not in ("reflection", "total"):
            raise ValueError("Top feature scope must be 'reflection' or 'total'")
        if self.top_count > 5 and not self.allow_more_than_five:
            raise ValueError("Top feature count is capped at five unless override is enabled")
        return self


def _detector_image(dm, scan):
    path = reflection_sum.sum_path(dm, scan)
    if not path.exists():
        raise FileNotFoundError(
            f"Summed detector image not found: {path}. Run xrd-app reflection-sum first.")
    with np.load(path) as saved:
        image = np.asarray(saved["image"], dtype=np.float32)
        max_bins = int(saved["max_bins"]) if "max_bins" in saved.files else 0
    return image, max_bins


def _show_detector(axis, image, title, *, tth_map=None, noise_reduction=False):
    display = detector_display.prepare(
        image, tth_map=tth_map, noise_reduction=noise_reduction, log_scale=True)
    low, high = detector_display.auto_levels(display)
    artist = axis.imshow(
        display, origin="upper", cmap="inferno", vmin=low, vmax=high)
    axis.set_title(title)
    axis.set_xlabel("Detector x / column (pixels)")
    axis.set_ylabel("Detector y / row (pixels)")
    return artist


def _overlay_reflections(axis, tth_map, reflection_set, names=None):
    wanted = set(names) if names is not None else None
    colors = ("#55d6ff", "#f7d154", "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe")
    for index, reflection in enumerate(reflection_set):
        name = str(reflection.get("name", "?"))
        if wanted is not None and name not in wanted:
            continue
        level = float(reflection["two_theta"])
        axis.contour(tth_map, levels=[level], colors=[colors[index % len(colors)]],
                     linewidths=1.0)
        mask = np.isfinite(tth_map) & (np.abs(tth_map - level) < 0.03)
        ys, xs = np.nonzero(mask)
        if xs.size:
            point = xs.size // 2
            axis.text(xs[point], ys[point], name, color=colors[index % len(colors)],
                      fontsize=7, bbox={"facecolor": "black", "alpha": 0.45, "pad": 1})


def _show_grid(axis, grid, title):
    artist = axis.imshow(grid, origin="upper", cmap="magma", interpolation="nearest",
                         aspect="equal")
    axis.set_title(title)
    axis.set_xlabel("Spatial bin column")
    axis.set_ylabel("Spatial bin row")
    return artist


def _show_segmentation(axis, masks, title):
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    shape = next(iter(masks.values())).shape
    rgba = np.zeros((*shape, 4), dtype=float)
    handles = []
    for index, (reflection, mask) in enumerate(masks.items()):
        color = to_rgb(device_maps.REFLECTION_PALETTE[
            index % len(device_maps.REFLECTION_PALETTE)])
        pastel = tuple(channel * 0.35 + 0.65 for channel in color)
        rgba[mask, :3] = pastel
        rgba[mask, 3] = 1.0
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.2)
        handles.append(Patch(facecolor=pastel, edgecolor=color, label=reflection))
    axis.imshow(rgba, origin="upper", interpolation="nearest", aspect="equal")
    axis.set_title(title)
    axis.set_xlabel("Spatial bin column")
    axis.set_ylabel("Spatial bin row")
    if handles:
        axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                    borderaxespad=0, fontsize=8, title="Reflection")


def _feature_size(feature):
    return int(feature.get("n_bins") or len(feature.get("intensity_profile") or {}) or 1)


def _profile_grid(feature, n_rows, n_cols):
    return device_maps.build_device_grids(
        [feature], n_rows, n_cols, metric="intensity").get(
            feature.get("reflection"), np.full((n_rows, n_cols), np.nan))


def _error_page(pdf, scan, section, error):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.333, 7.5), constrained_layout=True)
    figure.suptitle(f"{scan} | {section}", fontsize=20, fontweight="bold")
    figure.text(0.08, 0.65, "Section unavailable", fontsize=18, color="#9b1c1c")
    figure.text(0.08, 0.53, f"{type(error).__name__}: {error}", fontsize=11, wrap=True)
    figure.text(0.08, 0.12,
                "Report generation continued so available scans and sections are retained.",
                fontsize=10, color="#666666")
    pdf.savefig(figure)
    plt.close(figure)


def _page(pdf, title, draw):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(13.333, 7.5), constrained_layout=True)
    figure.suptitle(title, fontsize=18, fontweight="bold")
    draw(figure)
    pdf.savefig(figure)
    plt.close(figure)


def _load_target(dm, target):
    catalog = Path(target.catalog) if target.catalog else catalogs.default_feature_source(
        dm.labels_dir(target.scan), target.bin_size)
    if catalog is None:
        raise FileNotFoundError(
            f"No shape/combined catalog for {target.scan} at "
            f"{target.bin_size}x{target.bin_size}")
    features, _ = catalogs.load_features_any(catalog)
    sources = catalogs.resolve_catalog_sources(
        dm, catalog, bin_size=target.bin_size, scan=target.scan)
    grid = io.load_grid_mapping(sources.grid_mapping)
    return catalog, features, sources, grid


def _sum_page(pdf, dm, target):
    image, max_bins = _detector_image(dm, target.scan)
    tth_map = io.load_tth_map(dm.tth_map(scan=target.scan))

    def draw(figure):
        axis = figure.add_subplot(111)
        artist = _show_detector(
            axis, image, "Full scan detector sum | radial background removed",
            tth_map=tth_map, noise_reduction=True)
        figure.colorbar(artist, ax=axis, label="log(1 + summed detector counts)")
        if max_bins:
            axis.text(0.01, 0.01, f"PREVIEW SUM: first {max_bins} bins", color="white",
                      transform=axis.transAxes, bbox={"facecolor": "#9b1c1c", "alpha": 0.8})

    _page(pdf, f"{target.scan} | Summed detector image", draw)


def _feature_pages(pdf, dm, target, *, split_reflections):
    catalog, features, _sources, grid = _load_target(dm, target)
    image, _ = _detector_image(dm, target.scan)
    tth_map = io.load_tth_map(dm.tth_map(scan=target.scan))
    reflection_set = reflections.read_json(dm.reflections(scan=target.scan))
    n_rows, n_cols = int(grid["n_bin_rows"]), int(grid["n_bin_cols"])
    grids = device_maps.build_device_grids(
        features, n_rows, n_cols, metric="intensity")
    masks = device_maps.feature_masks(features, n_rows, n_cols)
    groups = sorted(grids) if split_reflections else [None]
    for reflection in groups:
        names = [reflection] if reflection is not None else None
        device = grids[reflection] if reflection is not None else None
        label = reflection or "All reflections"

        def draw(figure, names=names, device=device, label=label):
            left, right = figure.subplots(1, 2)
            det = _show_detector(
                left, image, f"Detector sum | {label} | radial background removed",
                tth_map=tth_map, noise_reduction=True)
            _overlay_reflections(left, tth_map, reflection_set, names)
            figure.colorbar(det, ax=left, fraction=0.046, label="log summed counts")
            if device is None:
                _show_segmentation(
                    right, masks, "Crystal segmentation | transparent reflection view")
            else:
                mapped = _show_grid(right, device, f"Device intensity map | {label}")
                figure.colorbar(
                    mapped, ax=right, fraction=0.046, label="Integrated intensity")

        _page(pdf, f"{target.scan} | {label} | {target.bin_size}x{target.bin_size} | "
              f"{catalog.name}", draw)


def _rank_top_features(features, count, scope):
    """Return ``(reflection, rank, feature)`` entries in report-page order."""
    rank_key = lambda feature: (
        _feature_size(feature), float(feature.get("peak_intensity") or 0))
    if scope == "total":
        selected = sorted(features, key=rank_key, reverse=True)[:count]
        overall_rank = {id(feature): rank for rank, feature in enumerate(selected, 1)}
        return [
            (reflection, overall_rank[id(feature)], feature)
            for reflection in sorted({feature.get("reflection", "unknown")
                                      for feature in selected})
            for feature in selected
            if feature.get("reflection", "unknown") == reflection
        ]
    return [
        (reflection, rank, feature)
        for reflection in sorted({feature.get("reflection", "unknown")
                                  for feature in features})
        for rank, feature in enumerate(sorted(
            (feature for feature in features
             if feature.get("reflection", "unknown") == reflection),
            key=rank_key, reverse=True)[:count], 1)
    ]


def _top_feature_pages(pdf, dm, target, options):
    catalog, features, sources, grid = _load_target(dm, target)
    n_rows, n_cols = int(grid["n_bin_rows"]), int(grid["n_bin_cols"])
    source = io.open_bin_source(
        dm, target.bin_size, scan=target.scan, grid_mapping=sources.grid_mapping,
        variant=sources.variant)
    try:
        ranked = _rank_top_features(
            features, options.top_count, options.top_scope)
        for reflection, rank, feature in ranked:
            source_image = source.image(feature.get("center_bin")) if options.source_images else None
            feature_grid = _profile_grid(feature, n_rows, n_cols)

            def draw(figure, feature=feature, source_image=source_image,
                     feature_grid=feature_grid, rank=rank, reflection=reflection):
                if source_image is None:
                    axis = figure.add_subplot(111)
                    mapped = _show_grid(axis, feature_grid, "Feature intensity profile")
                    figure.colorbar(mapped, ax=axis, label="Integrated intensity")
                else:
                    left, right = figure.subplots(1, 2)
                    det = _show_detector(left, source_image,
                                         f"Source bin {feature.get('center_bin', '?')}")
                    x, y = feature.get("detector_x"), feature.get("detector_y")
                    if x is not None and y is not None:
                        left.plot(x, y, "+", color="cyan", markersize=12,
                                  markeredgewidth=1.5)
                    figure.colorbar(det, ax=left, fraction=0.046, label="log counts")
                    mapped = _show_grid(right, feature_grid, "Feature intensity profile")
                    figure.colorbar(mapped, ax=right, fraction=0.046,
                                    label="Integrated intensity")
                figure.text(0.5, 0.015,
                            f"feature #{feature.get('feature_id', '?')} | "
                            f"size {_feature_size(feature)} bins | "
                            f"peak {float(feature.get('peak_intensity') or 0):.4g}",
                            ha="center", fontsize=9)

            scope_label = "overall" if options.top_scope == "total" else "reflection"
            _page(pdf, f"{target.scan} | {reflection} | Top {scope_label} by size "
                  f"#{rank} | {catalog.name}", draw)
    finally:
        source.close()


def _roi_features(dm, target, calculate, top_count):
    from . import roi_catalog, roi_map

    paths = roi_catalog.discover(dm.labels_dir(target.scan), target.bin_size)
    features = []
    for path in paths:
        features.extend(roi_catalog.load(path).get("features") or [])
    if features or not calculate:
        return features

    _catalog, source_features, sources, grid = _load_target(dm, target)
    selected = sorted(source_features, key=_feature_size, reverse=True)[:top_count]
    detector, _ = _detector_image(dm, target.scan)
    rois = []
    for feature in selected:
        roi = roi_map.auto_roi_from_click(
            detector, int(feature["detector_x"]), int(feature["detector_y"]))
        if roi is not None:
            rois.append((feature, roi))
    if not rois:
        raise ValueError("No saved ROI catalogs or top-feature detector peaks are available")
    source = io.open_bin_source(
        dm, target.bin_size, scan=target.scan, grid_mapping=sources.grid_mapping,
        variant=sources.variant)
    try:
        sampled = roi_map.sample_rois(
            source, [roi for _feature, roi in rois], grid_mapping=grid,
            metric="integrated", fast=False, log=lambda _message: None)
    finally:
        source.close()
    tth_map = io.load_tth_map(dm.tth_map(scan=target.scan))
    beam_center = processing.estimate_beam_center(tth_map)
    return [roi_map.to_shape_feature(
        result, reflection=f"{feature.get('reflection', 'peak')} #{feature.get('feature_id', '?')}",
        feature_id=index, tth_map=tth_map, beam_center=beam_center)
        for index, ((feature, _roi), result) in enumerate(zip(rois, sampled), 1)]


def _roi_pages(pdf, dm, target, options):
    from matplotlib.patches import Rectangle

    features = _roi_features(dm, target, options.calculate_rois, options.top_count)
    if not features:
        raise FileNotFoundError(
            "No saved ROI catalog. Enable on-demand ROI calculation to use top features.")
    detector, _ = _detector_image(dm, target.scan)
    tth_map = io.load_tth_map(dm.tth_map(scan=target.scan))
    for feature in features:
        roi = feature.get("manual_roi") or {}
        n_rows = int(feature.get("n_bin_rows") or 0)
        n_cols = int(feature.get("n_bin_cols") or 0)
        feature_grid = _profile_grid(feature, n_rows, n_cols)

        def draw(figure, feature=feature, roi=roi, feature_grid=feature_grid):
            left, right = figure.subplots(1, 2)
            det = _show_detector(
                left, detector,
                "Detector sum and selected ROI | radial background removed",
                tth_map=tth_map, noise_reduction=True)
            if roi:
                left.add_patch(Rectangle(
                    (roi["x0"], roi["y0"]), roi["x1"] - roi["x0"],
                    roi["y1"] - roi["y0"], fill=False, edgecolor="cyan", linewidth=1.5))
            figure.colorbar(det, ax=left, fraction=0.046, label="log summed counts")
            mapped = _show_grid(right, feature_grid, "ROI integrated-intensity map")
            figure.colorbar(mapped, ax=right, fraction=0.046, label="Integrated intensity")

        _page(pdf, f"{target.scan} | ROI | {feature.get('reflection', 'manual ROI')}", draw)


def _territory_pages(pdf, dm, target):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    import matplotlib.pyplot as plt

    grid_path = dm.grid_mapping(bin_size=1, variant="territory", scan=target.scan)
    if not grid_path.exists():
        raise FileNotFoundError(f"Territorial grid not found: {grid_path}")
    grid = io.load_grid_mapping(grid_path)
    territories = grid.get("territories") or {}
    candidates = [path for path in catalogs.feature_sources(dm.labels_dir(target.scan), 1)
                  if catalogs.catalog_variant(path, dm.labels_dir(target.scan)) == "territory"]
    if not candidates:
        raise FileNotFoundError("No territorial shape catalog found")
    features, _ = catalogs.load_features_any(candidates[-1])
    polygons = {key: Polygon(info["polygon"], closed=True)
                for key, info in territories.items() if len(info.get("polygon") or []) >= 3}
    for reflection in sorted({feature.get("reflection", "unknown") for feature in features}):
        values = device_maps.territory_intensities(features, reflection=reflection)

        def draw(figure, reflection=reflection, values=values):
            axis = figure.add_subplot(111)
            keys = [key for key in polygons if key in values]
            collection = PatchCollection([polygons[key] for key in keys], cmap="magma",
                                         edgecolor="#333333", linewidth=0.25)
            collection.set_array(np.log1p([values[key] for key in keys]))
            axis.add_collection(collection)
            axis.autoscale_view()
            axis.set_aspect("equal")
            axis.invert_yaxis()
            axis.set_title(f"Territorial intensity map | {reflection}")
            axis.set_xlabel("Stage X")
            axis.set_ylabel("Stage Y")
            figure.colorbar(collection, ax=axis, label="log(1 + peak intensity)")

        _page(pdf, f"{target.scan} | Territorial map | {reflection}", draw)


def generate_pdf(project_root, targets, output, options=None, *, preview=False,
                 log: Callable[[str], None] = print):
    """Generate a resilient multi-page report and return an execution summary."""
    from matplotlib.backends.backend_pdf import PdfPages

    options = (options or ReportOptions()).validated()
    targets = list(targets)
    if preview:
        targets = targets[:1]
    if not targets:
        raise ValueError("Select at least one scan for the report")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dm = DataManager(project_root)
    failures = []
    pages = 0
    sections = [
        ("Summed detector image", options.summed_images, _sum_page),
        ("All reflections", options.all_reflections,
         lambda pdf, manager, target: _feature_pages(
             pdf, manager, target, split_reflections=False)),
        ("Features by reflection", options.features_by_reflection,
         lambda pdf, manager, target: _feature_pages(
             pdf, manager, target, split_reflections=True)),
        ("Top features", options.top_features,
         lambda pdf, manager, target: _top_feature_pages(
             pdf, manager, target, options)),
        ("ROI images", options.roi_images,
         lambda pdf, manager, target: _roi_pages(pdf, manager, target, options)),
        ("Territorial maps", options.territory_maps, _territory_pages),
    ]
    with PdfPages(output) as pdf:
        for target in targets:
            for section, enabled, render in sections:
                if not enabled:
                    continue
                try:
                    render(pdf, dm, target)
                    log(f"[{target.scan}] {section}: ready")
                except Exception as error:
                    failures.append({"scan": target.scan, "section": section,
                                     "error": f"{type(error).__name__}: {error}"})
                    _error_page(pdf, target.scan, section, error)
                    log(f"[{target.scan}] {section}: unavailable: {error}")
        pages = pdf.get_pagecount()
    return {"path": output, "targets": len(targets), "pages": pages,
            "failures": failures, "preview": bool(preview)}
