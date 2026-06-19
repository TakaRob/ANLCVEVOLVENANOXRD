"""
Multi-Radius Concentric Stack Detection + Per-Frame Union-Find + Voigt Verification.

MATERIALLY NEW APPROACH: "Concentric Stack Detection"
Instead of detecting in a single sum-stack (dilutes small grains) or per-frame only
(too noisy), this candidate detects peaks in MULTIPLE concentric stacks (r=1 and r=3)
PLUS the full-radius stack. Small-grain peaks appear in the tight r=1 stack but are
diluted in the full stack. Large grains are captured by the full stack.

Combined with per-frame detection at SNR=3.0 and Union-Find spatial linking, this
creates a more diverse candidate pool that can catch peaks missed by either approach
alone.

Key innovations vs prior candidates:
1. Multi-radius concentric stacking (r=1,3) plus full radius for diverse detection
2. Stack detections are added as center-bin detections, then linked with per-frame
   detections via Union-Find to create features
3. Position refinement on cleaned sum-stack
4. Tuned verification to balance recall vs precision for F2 optimization
"""

import argparse, csv, importlib.util, json
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, scipy.ndimage as ndi

BINS_H5_DEFAULT = "/home/takaji/xrd_1x1_bins.h5"

def block_reduce_mean(arr, factor):
    if factor <= 1: return arr
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    arr = arr[:h2, :w2]
    return arr.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))

def get_neighbor_bins(cr, cc, nr, nc, radius=5):
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = cr + dr, cc + dc
            if 0 <= r < nr and 0 <= c < nc:
                out.append((r, c))
    return out

def precompute_tth(tth_map, bin_width=0.05):
    tth_min, tth_max = float(tth_map.min()), float(tth_map.max())
    edges = np.arange(tth_min, tth_max + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)
    flat = tth_map.ravel()
    indices = np.digitize(flat, edges) - 1
    indices = np.clip(indices, 0, n_bins - 1).astype(np.int32)
    counts = np.bincount(indices, minlength=n_bins)[:n_bins]
    order = np.argsort(indices)
    sorted_idx = indices[order]
    boundaries = np.searchsorted(sorted_idx, np.arange(n_bins + 1))
    return {'edges': edges, 'centers': centers, 'n_bins': n_bins,
            'indices': indices, 'counts': counts, 'order': order,
            'boundaries': boundaries, 'valid_mask': counts > 50}

def compute_radial_bg(image, tth_data):
    img_flat = image.ravel()
    sorted_vals = img_flat[tth_data['order']]
    n = tth_data['n_bins']
    boundaries = tth_data['boundaries']
    profile = np.zeros(n)
    for i in range(n):
        lo, hi = int(boundaries[i]), int(boundaries[i + 1])
        if hi > lo:
            profile[i] = np.median(sorted_vals[lo:hi])
    if n > 5:
        profile = ndi.uniform_filter1d(profile, size=min(15, n))
    return profile[tth_data['indices']].reshape(image.shape)

def build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=0.5):
    bands = {}
    for label, deg_val in zip(deg_labels, degs):
        mask = np.abs(tth_map - deg_val) <= tth_tolerance
        if label in bands:
            bands[label] = bands[label] | mask
        else:
            bands[label] = mask
    return bands

def fast_tophat(image, size=13):
    eroded = ndi.minimum_filter(image, size=size)
    opened = ndi.maximum_filter(eroded, size=size)
    return image - opened

def precompute_band_data(band_masks, ignore_edge=3, shape=(1062, 1028)):
    rows, cols = shape
    edge_mask = np.ones((rows, cols), dtype=bool)
    if ignore_edge > 0:
        edge_mask[:ignore_edge, :] = False
        edge_mask[-ignore_edge:, :] = False
        edge_mask[:, :ignore_edge] = False
        edge_mask[:, -ignore_edge:] = False
    band_data = {}
    for label, bm in band_masks.items():
        valid = bm & edge_mask
        flat_valid = np.flatnonzero(valid)
        if len(flat_valid) < 100:
            continue
        band_data[label] = {'valid_mask': valid, 'flat_valid': flat_valid,
                            'n_valid': len(flat_valid)}
    return band_data

def detect_in_image(tophat_img, cleaned_img, band_data,
                    min_pixels=2, max_pixels=300, min_compactness=0.03,
                    max_detections=80, snr_threshold=3.5):
    """Peak detection on a (stack or single-frame) image."""
    rows, cols = tophat_img.shape
    tophat_flat = tophat_img.ravel()
    all_peaks = []
    for label, bd in band_data.items():
        fv = bd['flat_valid']
        vals = tophat_flat[fv]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        sigma = mad * 1.4826 if mad > 0 else np.std(vals)
        if sigma <= 0:
            continue
        threshold = med + snr_threshold * sigma
        valid_mask = bd['valid_mask']
        hotspot = np.zeros((rows, cols), dtype=bool)
        hotspot[valid_mask] = tophat_img[valid_mask] > threshold
        cc, n_comp = ndi.label(hotspot)
        if n_comp == 0:
            continue
        comp_sizes = np.bincount(cc.ravel(), minlength=n_comp + 1)
        valid_comps = np.where((comp_sizes >= min_pixels) & (comp_sizes <= max_pixels))[0]
        valid_comps = valid_comps[valid_comps > 0]
        if len(valid_comps) == 0:
            continue
        flat_idx = np.flatnonzero(cc)
        comp_of_pixel = cc.ravel()[flat_idx]
        keep_mask = np.isin(comp_of_pixel, valid_comps)
        if not np.any(keep_mask):
            continue
        flat_idx = flat_idx[keep_mask]
        comp_of_pixel = comp_of_pixel[keep_mask]
        ys_flat = flat_idx // cols
        xs_flat = flat_idx % cols
        sort_order = np.argsort(comp_of_pixel, kind="stable")
        ys_flat = ys_flat[sort_order]
        xs_flat = xs_flat[sort_order]
        comp_sorted = comp_of_pixel[sort_order]
        uniq_comps, seg_start_idx = np.unique(comp_sorted, return_index=True)
        seg_ends = np.append(seg_start_idx[1:], len(comp_sorted))
        for i in range(len(uniq_comps)):
            lo = seg_start_idx[i]; hi = seg_ends[i]
            npix = hi - lo
            ys = ys_flat[lo:hi]; xs = xs_flat[lo:hi]
            h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
            if h == 0 or w == 0: continue
            aspect = min(h, w) / max(h, w)
            fill = npix / (h * w)
            if aspect * fill < min_compactness: continue
            weights = np.maximum(cleaned_img[ys, xs], 0)
            ws = weights.sum()
            if ws > 0:
                cx = int(round(np.sum(xs * weights) / ws))
                cy = int(round(np.sum(ys * weights) / ws))
            else:
                cx = int(np.mean(xs)); cy = int(np.mean(ys))
            pv = np.max(tophat_img[ys, xs])
            snr = (pv - med) / sigma if sigma > 0 else 0
            all_peaks.append({'x': cx, 'y': cy, 'label': label,
                              'npix': npix, 'snr': snr, 'compactness': aspect*fill})
    if not all_peaks: return []
    all_peaks.sort(key=lambda p: p['snr'], reverse=True)
    kept = []
    for p in all_peaks:
        if not any((p['x']-e['x'])**2+(p['y']-e['y'])**2 < 196 for e in kept):
            kept.append(p)
    return kept[:max_detections] if len(kept) > max_detections else kept

# ── Union-Find ──────────────────────────────────────────

def link_peaks_union_find(all_detections, link_tolerance=5):
    nodes = []
    for bk, peaks in all_detections.items():
        r, c = int(bk.split("_")[0]), int(bk.split("_")[1])
        for pi, p in enumerate(peaks):
            nodes.append((bk, pi, r, c, p['x'], p['y'], p))
    if not nodes: return []
    n = len(nodes)
    parent = list(range(n)); rank = [0] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if rank[ra] < rank[rb]: ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]: rank[ra] += 1
    spatial = defaultdict(list)
    for idx, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        spatial[(r, c)].append(idx)
    lt_sq = link_tolerance * link_tolerance
    for idx, (bk, pi, r, c, x, y, p) in enumerate(nodes):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                key = (r + dr, c + dc)
                if key not in spatial: continue
                for nidx in spatial[key]:
                    nx, ny = nodes[nidx][4], nodes[nidx][5]
                    if (x - nx)**2 + (y - ny)**2 <= lt_sq:
                        union(idx, nidx)
    components = defaultdict(list)
    for idx in range(n): components[find(idx)].append(idx)
    return [[nodes[i] for i in mi] for mi in components.values()]

# ── Voigt verification ──────────────────────────────────

def pseudo_voigt(r, amplitude, sigma, gamma):
    fg = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0))
    fl = 2.0 * gamma
    f = (fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
         + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5)**0.2
    if f < 1e-10: return np.zeros_like(r)
    eta = 1.36603*(fl/f) - 0.47719*(fl/f)**2 + 0.11116*(fl/f)**3
    eta = np.clip(eta, 0, 1)
    return amplitude * (eta / (1.0 + (r/(gamma+1e-10))**2) +
                        (1 - eta) * np.exp(-r**2/(2.0*sigma**2+1e-10)))

def probe_feature_net(px, py, raw_frames, bg_per_frame, aperture=4):
    rows, cols = bg_per_frame.shape
    y0 = max(0, py-aperture); y1 = min(rows, py+aperture+1)
    x0 = max(0, px-aperture); x1 = min(cols, px+aperture+1)
    bg_sum = float(np.sum(bg_per_frame[y0:y1, x0:x1]))
    net = {}
    for bk, frame in raw_frames.items():
        net[bk] = float(np.sum(frame[y0:y1, x0:x1])) - bg_sum
    return net

def verify_voigt(net_by_bin, center_bin, voigt_r2_thresh=0.20,
                 min_signal_bins=2, monotonic_thresh=0.60,
                 center_percentile=0.12, io_ratio_thresh=1.15,
                 max_width=8.0, cv_flat_thresh=0.10):
    bks = list(net_by_bin.keys())
    vals = np.array([net_by_bin[bk] for bk in bks])
    if len(vals) < 3: return False, {}, "few"
    mx = np.max(vals)
    if mx <= 0: return False, {}, "neg"
    norm = vals / (mx + 1e-10)
    if center_bin not in net_by_bin: return False, {}, "no center"
    ci = bks.index(center_bin); cn = norm[ci]
    pctl = float(np.mean(norm <= cn))
    if pctl < center_percentile: return False, {}, "weak"
    ra = np.array([int(bk.split("_")[0]) for bk in bks], dtype=float)
    ca = np.array([int(bk.split("_")[1]) for bk in bks], dtype=float)
    wn = np.maximum(norm, 0); ws = np.sum(wn)
    if ws > 0: pr = np.sum(ra*wn)/ws; pc = np.sum(ca*wn)/ws
    else: imax = int(np.argmax(norm)); pr, pc = ra[imax], ca[imax]
    distances = np.sqrt((ra-pr)**2 + (ca-pc)**2)
    mn = np.median(norm)
    ne = np.median(np.abs(norm-mn)) * 1.4826
    st = max(0.08, mn + 1.5*ne) if ne > 0 else 0.08
    n_sig = int(np.sum(norm > st))
    if n_sig < min_signal_bins: return False, {}, "few sig"
    cv = np.std(norm) / (np.mean(norm)+1e-10)
    if cv < cv_flat_thresh: return False, {}, "flat"
    im = distances <= 2.5; om = distances > 2.5; ior = 0.0
    if np.sum(im) >= 2 and np.sum(om) >= 3:
        ior = np.mean(norm[im]) / (np.mean(norm[om])+1e-10)
        if ior < io_ratio_thresh: return False, {}, "flat io"
    if n_sig <= 2:
        sm = norm > st; sd = distances[sm]
        if len(sd) > 0 and np.max(sd) <= 3.5 and cn > st * 0.3:
            return True, {'n_signal': n_sig, 'io_ratio': ior}, "compact"
        return False, {}, "spread"
    from scipy.optimize import curve_fit
    def voff(r, a, s, g, o): return o + pseudo_voigt(r, a, s, g)
    try:
        rm = np.max(distances) + 1
        fw = np.maximum(norm, 0.05)
        popt, _ = curve_fit(voff, distances, norm,
                           p0=[1.0, rm/3, rm/3, 0.0],
                           bounds=([0.1,0.1,0.01,-0.3],[2.0,rm*2,rm*2,0.7]),
                           sigma=1.0/fw, maxfev=600)
        fitted = voff(distances, *popt)
        res = norm - fitted
        ss_res = np.sum((res*fw)**2)
        ss_tot = np.sum(((norm-np.mean(norm))*fw)**2)
        r2 = 1 - ss_res/(ss_tot+1e-10)
        params = {'r2': float(r2), 'n_signal': n_sig, 'sigma': float(popt[1]),
                  'gamma': float(popt[2]), 'cv': float(cv), 'io_ratio': float(ior)}
        if popt[1] > max_width and popt[2] > max_width: return False, params, "wide"
        et = voigt_r2_thresh
        if n_sig <= 4: et = max(0.10, voigt_r2_thresh - 0.10)
        if r2 > et: return True, params, f"R2={r2:.3f}"
        nc, nt = 0, 0
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if abs(distances[i]-distances[j]) < 0.1: continue
                nt += 1
                if (distances[i]<distances[j]) == (norm[i]>norm[j]): nc += 1
        em = monotonic_thresh if n_sig >= 5 else max(0.50, monotonic_thresh-0.05)
        if nt > 3 and nc/nt > em:
            params['mono'] = nc/nt; return True, params, "mono"
        return False, params, f"R2={r2:.3f}"
    except (RuntimeError, ValueError):
        nc, nt = 0, 0
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if abs(distances[i]-distances[j]) < 0.1: continue
                nt += 1
                if (distances[i]<distances[j]) == (norm[i]>norm[j]): nc += 1
        if nt > 3 and nc/nt > monotonic_thresh:
            return True, {'mono': nc/nt, 'n_signal': n_sig}, "mono"
        return False, {}, "failed"

# ── Full pipeline ─────────────────────────────────────────────────

def run_full_pipeline(center_bin, bins_h5_path, tth_map, degs, deg_labels,
                      grid_mapping, spatial_radius=5, downsample=1,
                      perframe_snr=3.0, perframe_tophat_size=13,
                      stack_snr=3.5, tight_stack_radius=1, mid_stack_radius=3,
                      tight_snr=3.0, mid_snr=3.0,
                      max_det_per_frame=30, max_det_stack=80,
                      tth_tolerance=0.5, min_compactness=0.03,
                      link_tolerance=5, min_feature_bins=2,
                      voigt_r2_thresh=0.20, min_signal_bins=2,
                      monotonic_thresh=0.60, center_percentile=0.15,
                      io_ratio_thresh=1.20, max_width=8.0,
                      cv_flat_thresh=0.10, aperture=4,
                      dedup_dist=20, refinement_radius=5):
    downsample = max(1, int(downsample))
    if downsample > 1:
        tth_map = block_reduce_mean(tth_map, downsample)
    tth_data = precompute_tth(tth_map)
    bands = build_tth_band_masks(tth_map, degs, deg_labels, tth_tolerance=tth_tolerance)
    
    eff_lt = max(1.0, link_tolerance/downsample) if downsample > 1 else link_tolerance
    eff_th = max(3, int(round(perframe_tophat_size/downsample))) if downsample > 1 else perframe_tophat_size
    eff_ap = max(2, aperture//downsample) if downsample > 1 else aperture
    
    cr = int(center_bin.split("_")[0]); cc_col = int(center_bin.split("_")[1])
    nr = grid_mapping['n_bin_rows']; nc = grid_mapping['n_bin_cols']
    nps = get_neighbor_bins(cr, cc_col, nr, nc, radius=spatial_radius)
    nks = [f"{r}_{c}" for r, c in nps]
    
    # Load all frames and compute distances
    raw_frames = {}
    frame_dists = {}
    sum_stack = np.zeros(tth_map.shape, dtype=np.float64)
    tight_stack = np.zeros(tth_map.shape, dtype=np.float64)
    mid_stack = np.zeros(tth_map.shape, dtype=np.float64)
    n_tight = 0; n_mid = 0
    with h5py.File(bins_h5_path, "r") as h5f:
        for bk in nks:
            if bk not in h5f: continue
            f = np.clip(h5f[bk][:].astype(np.float64), 0, 1e9)
            if downsample > 1: f = block_reduce_mean(f, downsample)
            raw_frames[bk] = f
            rr, cc2 = int(bk.split("_")[0]), int(bk.split("_")[1])
            d = max(abs(rr - cr), abs(cc2 - cc_col))
            frame_dists[bk] = d
            sum_stack += f
            if d <= tight_stack_radius: tight_stack += f; n_tight += 1
            if d <= mid_stack_radius: mid_stack += f; n_mid += 1
    if not raw_frames: return {}
    nf = len(raw_frames)
    
    # Background from full sum-stack
    bg_total = compute_radial_bg(sum_stack, tth_data)
    bg_pf = bg_total / nf
    
    # Pre-compute band data
    bd = precompute_band_data(bands, ignore_edge=3, shape=tth_map.shape)
    
    all_det = {}
    
    # ── Stage 1a: Mid stack detection (r=3, ~49 frames) ──
    # Small grains appear more strongly in mid stack vs full stack
    if n_mid >= 5:
        mid_bg = compute_radial_bg(mid_stack, tth_data)
        mid_cleaned = mid_stack - mid_bg
        mid_tophat = np.zeros_like(mid_cleaned)
        for sz in [5, 7, 11, 15]:
            esz = max(3, sz//downsample) if downsample > 1 else sz
            np.maximum(mid_tophat, fast_tophat(mid_cleaned, size=esz), out=mid_tophat)
        mid_peaks = detect_in_image(mid_tophat, mid_cleaned, bd,
                                     max_pixels=400, max_detections=max_det_stack,
                                     min_compactness=min_compactness,
                                     snr_threshold=mid_snr)
        if mid_peaks:
            if center_bin not in all_det: all_det[center_bin] = []
            ex = all_det[center_bin]
            for p in mid_peaks:
                if not any((p['x']-e['x'])**2+(p['y']-e['y'])**2 < 196 for e in ex):
                    ex.append(p)
    
    # ── Stage 1c: Full stack detection ──
    ss_cleaned = sum_stack - bg_total
    ss_tophat = np.zeros_like(ss_cleaned)
    for sz in [5, 7, 11, 15]:
        esz = max(3, sz//downsample) if downsample > 1 else sz
        np.maximum(ss_tophat, fast_tophat(ss_cleaned, size=esz), out=ss_tophat)
    full_peaks = detect_in_image(ss_tophat, ss_cleaned, bd,
                                  max_pixels=400, max_detections=max_det_stack,
                                  min_compactness=min_compactness,
                                  snr_threshold=stack_snr)
    if full_peaks:
        if center_bin not in all_det: all_det[center_bin] = []
        ex = all_det[center_bin]
        for p in full_peaks:
            if not any((p['x']-e['x'])**2+(p['y']-e['y'])**2 < 196 for e in ex):
                ex.append(p)
    
    # ── Stage 1d: Per-frame detection (only near frames for speed) ──
    # Only do per-frame detection on frames within mid_stack_radius
    # Distant frames rarely contribute to features that include center bin
    perframe_radius = mid_stack_radius  # typically 3 -> ~49 frames
    for bk, frame in raw_frames.items():
        if frame_dists[bk] > perframe_radius: continue
        cleaned = frame - bg_pf
        tophat = fast_tophat(cleaned, size=eff_th)
        peaks = detect_in_image(tophat, cleaned, bd,
                               min_compactness=min_compactness,
                               max_detections=max_det_per_frame,
                               snr_threshold=perframe_snr)
        if peaks:
            if bk not in all_det: all_det[bk] = []
            ex = all_det[bk]
            for p in peaks:
                if not any((p['x']-e['x'])**2+(p['y']-e['y'])**2 < 196 for e in ex):
                    ex.append(p)
    
    # ── Stage 2: Union-Find linking ──
    features = link_peaks_union_find(all_det, link_tolerance=eff_lt)
    
    # ── Stage 3: Voigt verification ──
    validated = {}; vpos = []; dsq = dedup_dist * dedup_dist
    
    for members in features:
        bins_in = set(m[0] for m in members)
        if center_bin not in bins_in: continue
        if len(bins_in) < min_feature_bins: continue
        
        xs = np.array([m[4] for m in members], dtype=float)
        ys = np.array([m[5] for m in members], dtype=float)
        snrs = np.array([m[6].get('snr', 0) for m in members])
        wgts = np.maximum(snrs, 0.1); ws = wgts.sum()
        px = int(round(np.sum(xs*wgts)/ws)) if ws > 0 else int(np.mean(xs))
        py = int(round(np.sum(ys*wgts)/ws)) if ws > 0 else int(np.mean(ys))
        
        net = probe_feature_net(px, py, raw_frames, bg_pf, aperture=eff_ap)
        ok, params, reason = verify_voigt(net, center_bin,
            voigt_r2_thresh=voigt_r2_thresh, min_signal_bins=min_signal_bins,
            monotonic_thresh=monotonic_thresh, center_percentile=center_percentile,
            io_ratio_thresh=io_ratio_thresh, max_width=max_width,
            cv_flat_thresh=cv_flat_thresh)
        if not ok: continue
        
        cm = [m for m in members if m[0] == center_bin]
        if cm:
            best = max(cm, key=lambda m: m[6].get('snr', 0))
            bx, by = best[4], best[5]; label = best[6]['label']
        else:
            bx, by = px, py; label = members[0][6]['label']
        
        # Position refinement on cleaned sum-stack
        if refinement_radius > 0:
            ref_img = ss_cleaned
            r = refinement_radius
            y0 = max(0, by-r); y1 = min(ref_img.shape[0], by+r+1)
            x0 = max(0, bx-r); x1 = min(ref_img.shape[1], bx+r+1)
            patch = ref_img[y0:y1, x0:x1]
            if patch.max() > 0:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                w = np.maximum(patch, 0); wsv = w.sum()
                if wsv > 0:
                    nx = int(round(np.sum(xx*w)/wsv))
                    ny = int(round(np.sum(yy*w)/wsv))
                    if abs(nx-bx) <= 7 and abs(ny-by) <= 7: bx, by = nx, ny
        
        if downsample > 1:
            bx = bx*downsample + downsample//2
            by = by*downsample + downsample//2
        
        quality = params.get('r2', params.get('mono', 0))
        is_dup = False
        for vi, (vx, vy, vq) in enumerate(vpos):
            if (bx-vx)**2+(by-vy)**2 < dsq:
                if quality <= vq: is_dup = True; break
                else:
                    vpos[vi] = (bx, by, quality)
                    for lb in list(validated.keys()):
                        validated[lb] = [p for p in validated[lb]
                                        if (p[0]-vx)**2+(p[1]-vy)**2 >= dsq]
                        if not validated[lb]: del validated[lb]
                    break
        if not is_dup:
            vpos.append((bx, by, quality))
            validated.setdefault(label, []).append([bx, by])
    
    return validated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--center-bin", required=True)
    parser.add_argument("--bins-h5", default=BINS_H5_DEFAULT)
    parser.add_argument("--two-theta", default="tth.tiff")
    parser.add_argument("--reflections", default="reflections.py")
    parser.add_argument("--grid-mapping", default="grid_mapping.json")
    parser.add_argument("--output", default="detections.csv")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--spatial-radius", type=int, default=5)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args()
    
    spec = importlib.util.spec_from_file_location("reflections", args.reflections)
    ref_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref_mod)
    import tifffile
    tth_map = tifffile.imread(args.two_theta)
    with open(args.grid_mapping) as f: grid_mapping = json.load(f)
    
    validated = run_full_pipeline(args.center_bin, args.bins_h5, tth_map,
                                  ref_mod.degs, ref_mod.deg_labels, grid_mapping,
                                  spatial_radius=args.spatial_radius,
                                  downsample=args.downsample)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reflection", "x", "y"])
        for label, pts in validated.items():
            for x, y in pts: writer.writerow([label, x, y])
    total = sum(len(pts) for pts in validated.values())
    print(f"Detected {total} spatially-validated peaks at center bin {args.center_bin}")
    
    if args.labels:
        with open(args.labels) as f: labels = json.load(f)
        labels = {k: v for k, v in labels.items() if not k.startswith("__")}
        gt_pts = []
        for lab, pts in labels.items():
            for pt in pts: gt_pts.append(pt)
        det_pts = []
        for lab, pts in validated.items():
            for pt in pts: det_pts.append(pt)
        matched_gt = set(); matched_det = set()
        for gi, gp in enumerate(gt_pts):
            best_d2 = 1600; best_di = -1
            for di, dp in enumerate(det_pts):
                if di in matched_det: continue
                d2 = (gp[0]-dp[0])**2 + (gp[1]-dp[1])**2
                if d2 < best_d2: best_d2 = d2; best_di = di
            if best_di >= 0 and best_d2 < 1600:
                matched_gt.add(gi); matched_det.add(best_di)
        tp = len(matched_gt); fp = len(det_pts) - tp; fn = len(gt_pts) - tp
        p = tp/(tp+fp) if tp+fp > 0 else 0
        r = tp/(tp+fn) if tp+fn > 0 else 0
        f1 = 2*p*r/(p+r) if p+r > 0 else 0
        f2 = 5*p*r/(4*p+r) if 4*p+r > 0 else 0
        print(f"P={p:.3f} R={r:.3f} F1={f1:.3f} F2={f2:.3f}")

if __name__ == "__main__":
    main()
