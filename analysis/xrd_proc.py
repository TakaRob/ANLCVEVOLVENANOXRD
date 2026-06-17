import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os 

from scipy import ndimage as ndi
from mictools.roi_utils import Roi
from mictools.process_data import mesh_detector_data

def save_roi_outputs_h5(h5_path, scan, xrd_sum, all_rois, X, Y, v_mask, roi_names):

    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('xrd_sum', data=np.asarray(xrd_sum), compression='gzip')
        hf.create_dataset('X', data=np.asarray(X), compression='gzip')
        hf.create_dataset('Y', data=np.asarray(Y), compression='gzip')
        hf.create_dataset('v_mask', data=np.asarray(v_mask), compression='gzip')

        roi_name_arr = np.asarray(roi_names, dtype='S')
        hf.create_dataset('roi_names', data=roi_name_arr)

        rois_grp = hf.create_group('all_rois')
        for label, boxes in all_rois.items():
            label_key = str(label).replace('/', '_')
            if len(boxes) == 0:
                arr = np.empty((0, 4), dtype=np.int32)
            else:
                arr = np.asarray(boxes, dtype=np.int32)
            rois_grp.create_dataset(label_key, data=arr)

    print(f'Saved ROI outputs to: {h5_path}')
    return h5_path

def load_roi_outputs_h5(h5_path):
    """
    Load ROI outputs saved by the HDF5 export cell.

    Returns
    -------
    xrd_sum, all_rois, roi_names, X, Y, v_mask
    """
    import h5py
    import numpy as np

    all_rois = {}
    with h5py.File(h5_path, 'r') as hf:
        xrd_sum = hf['xrd_sum'][()]
        X = hf['X'][()]
        Y = hf['Y'][()]
        v_mask = hf['v_mask'][()]

        roi_raw = hf['roi_names'][()]
        roi_names = [r.decode('utf-8') if isinstance(r, (bytes, np.bytes_)) else str(r) for r in roi_raw]

        rois_grp = hf['all_rois']
        for label in rois_grp.keys():
            arr = rois_grp[label][()]
            all_rois[label] = [tuple(map(int, row)) for row in arr.tolist()]

    return xrd_sum, all_rois, roi_names, X, Y, v_mask


def plot_meshed_data(X, Y, Z, ax = None, fig = None, title=None, **kwargs):

    if ax is None or fig is None:
        fig, ax = plt.subplots()

    pcm = ax.pcolormesh(X, Y, Z, shading='auto', **kwargs)
    fig.colorbar(pcm, ax=ax)

    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(title)

    plt.show()

def detect_hotspot_rois(
    xrd_sum,
    tth,
    degs,
    deg_labels,
    target_labels,
    line_tol,
    hotspot_percentile=99.0,
    min_pixels=4,
    pad=6,
    ignore_y_range=None,
    ignore_edge_rows=0,
    ignore_edge_cols=0,
    tth_mask=None,
    show_plot=True,
    figsize=(8, 8),
    cmap='inferno',
    vmin_percentile=50,
    vmax_percentile=98,
    savefig = False,
    figname = None,
):
    label_to_deg = {lab: deg for lab, deg in zip(deg_labels, degs)}
    all_rois = {}

    ax = None
    overlay_mask = tth_mask
    if overlay_mask is None:
        overlay_mask = np.zeros_like(tth, dtype=float)
        for d in degs:
            overlay_mask[np.abs(tth - d) < line_tol] = 1.0
        overlay_mask[overlay_mask == 0] = np.nan

    if show_plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(
            xrd_sum,
            cmap=cmap,
            vmax=np.percentile(xrd_sum, hotspot_percentile),
            vmin=np.percentile(xrd_sum, vmin_percentile),
        )
        cmap = plt.get_cmap('gray_r')
        cmap.set_bad('white', alpha = 0)
        ax.imshow(overlay_mask, cmap=cmap, alpha=0.4)

    for lab in target_labels:
        if lab not in label_to_deg:
            print(f'Skipping {lab}: not found in deg_labels')
            continue

        d = label_to_deg[lab]
        line_mask = np.abs(tth - d) < line_tol

        if np.count_nonzero(line_mask) == 0:
            print(f'Skipping {lab}: empty line mask')
            all_rois[lab] = []
            continue

        line_vals = xrd_sum[line_mask]
        thr = np.percentile(line_vals, hotspot_percentile)
        hotspot_mask = line_mask & (xrd_sum >= thr)

        if ignore_y_range is not None:
            y0, y1 = ignore_y_range
            y0 = int(np.clip(y0, 0, xrd_sum.shape[0]))
            y1 = int(np.clip(y1, 0, xrd_sum.shape[0]))
            if y1 > y0:
                hotspot_mask[y0:y1, :] = False

        if ignore_edge_rows > 0:
            n = int(min(ignore_edge_rows, xrd_sum.shape[0] // 2))
            hotspot_mask[:n, :] = False
            hotspot_mask[-n:, :] = False

        if ignore_edge_cols > 0:
            m = int(min(ignore_edge_cols, xrd_sum.shape[1] // 2))
            hotspot_mask[:, :m] = False
            hotspot_mask[:, -m:] = False

        cc, n_comp = ndi.label(hotspot_mask)
        rois = []
        for comp_id in range(1, n_comp + 1):
            ys, xs = np.where(cc == comp_id)
            if ys.size < min_pixels:
                continue

            ry0 = max(int(ys.min()) - pad, 0)
            ry1 = min(int(ys.max()) + pad + 1, xrd_sum.shape[0])
            rx0 = max(int(xs.min()) - pad, 0)
            rx1 = min(int(xs.max()) + pad + 1, xrd_sum.shape[1])
            rois.append((ry0, ry1, rx0, rx1))

        rois = sorted(rois, key=lambda r: (r[0], r[2]))
        dedup = []
        for r in rois:
            keep = True
            for q in dedup:
                if r[0] >= q[0] and r[1] <= q[1] and r[2] >= q[2] and r[3] <= q[3]:
                    keep = False
                    break
            if keep:
                dedup.append(r)

        all_rois[lab] = dedup

        if show_plot and ax is not None:
            for j, (y0, y1, x0, x1) in enumerate(dedup, start=1):
                rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='cyan', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x0, max(0, y0 - 6), f'{lab}-{j}', color='cyan', fontsize=8, va='bottom', ha='left')

    if show_plot and ax is not None:
        ax.set_title('Detected hotspot ROIs on selected lines')
        plt.show()

    if savefig:
        fig.savefig(figname, dpi = 300)

    return all_rois, overlay_mask


def build_rois_from_detected(all_rois, target_labels=None, prefix='roi_xrd'):
    """Convert detected ROI boxes to Roi objects.

    all_rois format: {label: [(y_start, y_end, x_start, x_end), ...]}
    """
    roi_dict = {}
    labels = target_labels if target_labels is not None else list(all_rois.keys())

    for lab in labels:
        boxes = all_rois.get(lab, [])
        if len(boxes) > 0:
            lab_clean = lab.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')

            for j, (y0, y1, x0, x1) in enumerate(boxes):
                roi_name = f'{prefix}_{lab_clean}_{j}'
                roi_dict[roi_name] = Roi(y0, y1, x0, x1, name=roi_name)

    return roi_dict


def get_detected_rois_on_scan(
    sc,
    roi_dict,
    detector='xrd',
    threshold_percentile=96,
    show_each=True,
    each_cmap='Blues',
    overlay_colors=None,
    legend_loc='center left',
    legend_bbox=(1.02, 0.5),
):
    roi_names = list(roi_dict.keys())
    if len(roi_names) == 0:
        print('No ROIs to plot.')
        return None, []

    roi_value_list = []
    X = Y = None

    for n in roi_names:
        r = roi_dict[n]
        try:
            X, Y, Z = mesh_detector_data(sc, detector, roi=r)
            roi_value_list.append((n, Z))
            print(f"Done processing {n=}")
        except Exception:
            print(f'Not able to process {n}')

    if len(roi_value_list) == 0:
        print('No ROI maps were generated.')
        return None, []

    v_mask = None
    valid_roi_names = []

    for i, (name, v) in enumerate(roi_value_list):
        v_clip = v.copy()
        threshold = np.percentile(v_clip, threshold_percentile)
        v_clip[v_clip <= threshold] = 0
        v_clip[v_clip > threshold] = i + 1

        if v_mask is None:
            v_mask = np.zeros_like(v_clip)

        # Keep one fixed class value per ROI in overlay.
        v_mask[v_clip > 0] = i + 1
        valid_roi_names.append(name)

        if show_each:
            plot_meshed_data(X, Y, v_clip, cmap=each_cmap, title=f'scan: {sc} | {name}')

    return X, Y, v_mask, valid_roi_names


def plot_v_mask_overlay(
    X,
    Y,
    v_mask,
    roi_names=None,
    target_labels=None,
    overlay_colors=None,
    legend_loc='center left',
    legend_bbox=(1.02, 0.5),
    title='ROI overlay (fixed colors)',
    savefig = False,
    figname = None
):
    """Plot a precomputed ROI class mask with fixed colors and legend.

    v_mask is expected to contain 0 for background and 1..N for ROI classes.
    If target_labels is provided, colors are assigned by matching ROI names to labels.
    """
    if v_mask is None:
        print('v_mask is None; nothing to plot.')
        return

    n_roi = int(np.nanmax(v_mask))
    if n_roi <= 0:
        print('v_mask has no ROI classes > 0.')
        return

    if roi_names is None:
        roi_names = [f'ROI_{i}' for i in range(1, n_roi + 1)]
    else:
        roi_names = list(roi_names)[:n_roi]
        if len(roi_names) < n_roi:
            roi_names += [f'ROI_{i}' for i in range(len(roi_names) + 1, n_roi + 1)]

    if target_labels is not None and len(target_labels) > 0:
        target_labels = list(target_labels)
        if overlay_colors is None:
            overlay_colors = [plt.cm.tab10(i % 10) for i in range(len(target_labels))]
        overlay_colors = list(overlay_colors)

        if len(overlay_colors) < len(target_labels):
            extra = len(target_labels) - len(overlay_colors)
            overlay_colors += [plt.cm.tab10((i + len(overlay_colors)) % 10) for i in range(extra)]

        def _norm_label(s):
            return ''.join(ch for ch in str(s).lower() if ch.isalnum())

        norm_targets = [_norm_label(t) for t in target_labels]
        roi_color_map = []
        roi_group_idx = []

        for rn in roi_names:
            rn_norm = _norm_label(rn)
            found = None
            for j, tnorm in enumerate(norm_targets):
                if tnorm and tnorm in rn_norm:
                    found = j
                    break
            roi_group_idx.append(found)
            if found is None:
                roi_color_map.append((0.7, 0.7, 0.7, 1.0))
            else:
                roi_color_map.append(overlay_colors[found])

        cmap = mcolors.ListedColormap([(0, 0, 0, 0)] + roi_color_map)
        bounds = np.arange(0, n_roi + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.pcolormesh(X, Y, v_mask, shading='auto', cmap=cmap, norm=norm)

        handles = []
        used_groups = sorted({g for g in roi_group_idx if g is not None})
        for g in used_groups:
            handles.append(patches.Patch(facecolor=overlay_colors[g], edgecolor='none', label=target_labels[g]))
        if any(g is None for g in roi_group_idx):
            handles.append(patches.Patch(facecolor=(0.7, 0.7, 0.7, 1.0), edgecolor='none', label='Unmatched'))

        if handles:
            ax.legend(handles=handles, title='ROIs', loc=legend_loc, bbox_to_anchor=legend_bbox)
    else:
        if overlay_colors is None:
            overlay_colors = [plt.cm.tab20(i % 20) for i in range(n_roi)]
        overlay_colors = list(overlay_colors)[:n_roi]
        if len(overlay_colors) < n_roi:
            extra = n_roi - len(overlay_colors)
            overlay_colors += [plt.cm.tab20((i + len(overlay_colors)) % 20) for i in range(extra)]

        cmap = mcolors.ListedColormap([(0, 0, 0, 0)] + overlay_colors)
        bounds = np.arange(0, n_roi + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.pcolormesh(X, Y, v_mask, shading='auto', cmap=cmap, norm=norm)

        handles = [
            patches.Patch(facecolor=overlay_colors[i], edgecolor='none', label=roi_names[i])
            for i in range(n_roi)
        ]
        ax.legend(handles=handles, title='ROIs', loc=legend_loc, bbox_to_anchor=legend_bbox)

    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(title)
    plt.show()

    if savefig:
        fig.savefig(figname, dpi = 300)




# # Example usage
# target_labels = ['(012)', '(002)', '(011)', '(001)']
# all_rois, tth_mask = detect_hotspot_rois(
#     xrd_sum=xrd_sum,
#     tth=tth,
#     degs=degs,
#     deg_labels=deg_labels,
#     target_labels=target_labels,
#     line_tol=0.04,
#     hotspot_percentile=99,
#     min_pixels=4,
#     pad=13,
#     ignore_y_range=(510, 560),
#     ignore_edge_rows=2,
    # ignore_edge_cols=2,
#     # tth_mask=tth_mask,
#     show_plot=True,
# )

# if 'tth' in globals() and 'degs' in globals():
#     has_custom_labels = 'deg_labels' in globals() and len(deg_labels) > 0

#     for i, d in enumerate(degs):
#         if i == 0:
#             ref_row_idx = 500
#         elif i == 1:
#             ref_row_idx = 320
#         elif i == 2:
#             ref_row_idx = 200
#         else:
#             ref_row_idx = 0

#         ref_row_idx = int(np.clip(ref_row_idx, 0, tth.shape[0] - 1))
#         ref_row = tth[ref_row_idx, :]

#         x_idx = int(np.argmin(np.abs(ref_row - d)))
#         y_idx = int(np.argmin(np.abs(tth[:, x_idx] - d)))

#         if np.abs(ref_row[x_idx] - d) > max(err * 3, 0.05):
#             continue

#         if has_custom_labels and i < len(deg_labels):
#             label_txt = str(deg_labels[i])
#         else:
#             label_txt = str(8 + i)

#         ax.text(
#             x_idx + 40,
#             y_idx - 10,
#             label_txt,
#             color='white',
#             fontsize=9,
#             fontweight='bold',
#             ha='center',
#             va='center',
#             rotation=90,
#             rotation_mode='anchor',
#             bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1),
#         )
# else:
#     print('Run the cells defining tth/degs/err first to add arc labels.')