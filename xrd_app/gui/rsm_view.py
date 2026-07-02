"""Reciprocal-space (RSM) view — 2D projections + 3D volume + per-grain cloud.

Render-only face over ``core/rsm.py`` and ``core/studies.py``:

* a **Study selector** lists every rocking-study result set found under the
  project (``core.studies``) so you can switch between analyses (e.g. 3×3 vs
  1×1, or different scan subsets) without editing paths;
* the **2D view** shows a max-intensity projection of the fused 3D volume
  (``<study>/rsm.npz`` from ``xrd-app rsm``) on true q-axes (1/Å), overlaid with
  the per-grain **feature cloud** (``<study>/qspace/<scan>_features_q.csv`` from
  ``xrd-app qspace``);
* the **3D view** renders the whole ``I(qx,qy,qz)`` volume as an alpha-composited
  cloud (pyqtgraph OpenGL), with an **adjustable resolution** (downsample the
  grid for speed) and an opacity control, plus the 3D feature cloud, the
  **reflection rings** (concentric |Q| shells the cloud sits on, drawn as "CDs")
  and optional **per-scan rings** (one circle around each scan's features).

All data prep lives in ``core.rsm`` / ``core.studies``; this module only draws.
Missing artifacts (or a missing OpenGL stack) degrade to an inline hint, never a
crash.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import reflections as refl_io
from ..core import rsm as rsm_core
from ..core import studies as studies_core
from .palette import ARC_COLORS, _get_cmap

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# WSL2/WSLg quirk: Qt draws the GL scene through GLX, but PyOpenGL auto-selects
# the EGL platform whenever WAYLAND_DISPLAY is set. The mismatch makes every GL
# item fail to paint with "Attempt to retrieve context when no valid context"
# (PyOpenGL queries eglGetCurrentContext while the live context is GLX). Pin
# PyOpenGL to GLX so it reads the same context Qt created — must happen before
# ``pyqtgraph.opengl`` (which imports OpenGL). setdefault lets a user override.
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

# 3D needs PyOpenGL; import lazily-guarded so the 2D view works without it.
try:
    import pyqtgraph.opengl as gl
    _HAVE_GL = True
except Exception:  # pragma: no cover - depends on the GL stack being installed
    gl = None
    _HAVE_GL = False

# The three orthogonal planes: label -> (proj key, q-index for x, y).
_PLANES = {
    "qx–qy (top)":  ("qx_qy", 0, 1),
    "qx–qz (side)": ("qx_qz", 0, 2),
    "qy–qz (front)": ("qy_qz", 1, 2),
}
_AXIS_LABELS = {0: "qx", 1: "qy", 2: "qz"}
_COLOR_MODES = ["reflection", "θ", "intensity"]
# 3D grid downsample targets (voxels per longest axis).
_RES_CHOICES = [("32³ (fast)", 32), ("48³", 48), ("64³", 64),
                ("96³", 96), ("128³ (full)", 128)]
_GL_SCALE = 10.0  # normalized cube edge for the 3D scene (q-aspect preserved)


class RSMView(QWidget):
    """Self-contained tab widget; reads study artifacts under ``project_root``."""

    def __init__(self, project_root=".", scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.scan = scan
        # Active study dir (defaults to <root>/Study; the selector repoints it).
        self._study = self.project_root / "Study"
        self._rsm_path = self._study / "rsm.npz"
        self._qspace_dir = self._study / "qspace"
        self._rsm = None
        self._cloud = None
        self._shells = []          # [(name, |Q|)] reflection shells, sorted by |Q|
        self._refl_cmap = {}       # reflection name -> QColor (shared 3D cloud/rings)
        self._colorbar = None
        self._gl_items = []

        self._build_ui()
        self._populate_studies()
        self._reload()

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Row 1 — study selector + view mode.
        top = QHBoxLayout()
        top.addWidget(QLabel("Study:"))
        self.study_cb = QComboBox()
        self.study_cb.setMinimumWidth(240)
        self.study_cb.currentIndexChanged.connect(self._on_study_changed)
        top.addWidget(self.study_cb)

        top.addSpacing(16)
        top.addWidget(QLabel("View:"))
        self.view_cb = QComboBox()
        self.view_cb.addItems(["2D projection", "3D volume"])
        if not _HAVE_GL:
            self.view_cb.setItemData(
                1, "3D needs PyOpenGL (pip install 'xrd-app[gl]')", Qt.ToolTipRole)
        self.view_cb.currentIndexChanged.connect(self._on_view_changed)
        top.addWidget(self.view_cb)

        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload)
        top.addWidget(reload_btn)

        top.addStretch(1)
        self.status = QLabel("")
        self.status.setStyleSheet("color: #888;")
        top.addWidget(self.status)
        root.addLayout(top)

        # Row 2 — controls that apply to the current view (rebuilt per mode).
        self.bar2d = self._build_2d_bar()
        self.bar3d = self._build_3d_bar()
        root.addWidget(self.bar2d)
        root.addWidget(self.bar3d)
        self.bar3d.setVisible(False)

        # Stacked canvas: page 0 = 2D pyqtgraph, page 1 = 3D GL (or hint).
        self.stack = QStackedWidget()
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot()
        self.plot.setAspectLocked(False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.legend = None
        self.stack.addWidget(self.glw)

        self.gl_view = self._make_gl_widget()
        self.stack.addWidget(self.gl_view)
        root.addWidget(self.stack, 1)

    def _build_2d_bar(self) -> QWidget:
        w = QWidget()
        bar = QHBoxLayout(w)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(QLabel("Plane:"))
        self.plane_cb = QComboBox()
        self.plane_cb.addItems(list(_PLANES))
        self.plane_cb.currentIndexChanged.connect(self._refresh)
        bar.addWidget(self.plane_cb)

        self.heat_chk = QCheckBox("RSM heatmap")
        self.heat_chk.setChecked(True)
        self.heat_chk.stateChanged.connect(self._refresh)
        bar.addWidget(self.heat_chk)

        self.log_chk = QCheckBox("log I")
        self.log_chk.setChecked(True)
        self.log_chk.stateChanged.connect(self._refresh)
        bar.addWidget(self.log_chk)

        self.cloud_chk = QCheckBox("Feature cloud")
        self.cloud_chk.setChecked(True)
        self.cloud_chk.stateChanged.connect(self._refresh)
        bar.addWidget(self.cloud_chk)

        bar.addWidget(QLabel("Color by:"))
        self.color_cb = QComboBox()
        self.color_cb.addItems(_COLOR_MODES)
        self.color_cb.currentIndexChanged.connect(self._refresh)
        bar.addWidget(self.color_cb)
        bar.addStretch(1)
        return w

    def _build_3d_bar(self) -> QWidget:
        w = QWidget()
        bar = QHBoxLayout(w)
        bar.setContentsMargins(0, 0, 0, 0)

        self.vol_chk = QCheckBox("Volume")
        self.vol_chk.setChecked(True)
        self.vol_chk.stateChanged.connect(self._refresh_3d)
        bar.addWidget(self.vol_chk)

        self.log3d_chk = QCheckBox("log I")
        self.log3d_chk.setChecked(True)
        self.log3d_chk.stateChanged.connect(self._refresh_3d)
        bar.addWidget(self.log3d_chk)

        bar.addWidget(QLabel("Resolution:"))
        self.res_cb = QComboBox()
        for label, n in _RES_CHOICES:
            self.res_cb.addItem(label, n)
        self.res_cb.setCurrentIndex(2)  # 64³ default
        self.res_cb.currentIndexChanged.connect(self._refresh_3d)
        bar.addWidget(self.res_cb)

        bar.addWidget(QLabel("Opacity:"))
        self.opacity = QSlider(Qt.Horizontal)
        self.opacity.setRange(1, 100)
        self.opacity.setValue(40)
        self.opacity.setMaximumWidth(120)
        self.opacity.sliderReleased.connect(self._refresh_3d)
        bar.addWidget(self.opacity)

        self.cloud3d_chk = QCheckBox("Feature cloud")
        self.cloud3d_chk.setChecked(True)
        self.cloud3d_chk.stateChanged.connect(self._refresh_3d)
        bar.addWidget(self.cloud3d_chk)

        # Reflection shells: concentric |Q| rings ("CDs") the cloud sits on.
        self.shells_chk = QCheckBox("Reflection rings")
        self.shells_chk.setChecked(True)
        self.shells_chk.setToolTip(
            "Concentric |Q| = 4π·sin(2θ/2)/λ shells from the reflection set — "
            "a Debye–Scherrer ring per reflection that features cluster onto.")
        self.shells_chk.stateChanged.connect(self._refresh_3d)
        bar.addWidget(self.shells_chk)

        # Per-scan rings: one circle outline around each scan's feature group.
        self.scanrings_chk = QCheckBox("Per-scan rings")
        self.scanrings_chk.setChecked(False)
        self.scanrings_chk.setToolTip(
            "One circle outline around each scan's features (centroid + spread) "
            "at that scan's mean qz — the θ series as stacked discs.")
        self.scanrings_chk.stateChanged.connect(self._refresh_3d)
        bar.addWidget(self.scanrings_chk)
        bar.addStretch(1)
        return w

    def _make_gl_widget(self):
        """A GLViewWidget when OpenGL is available, else an explanatory label."""
        if _HAVE_GL:
            try:
                view = gl.GLViewWidget()
                view.setCameraPosition(distance=_GL_SCALE * 2.2)
                view.setBackgroundColor(pg.mkColor(15, 15, 20))
                grid = gl.GLGridItem()
                grid.setSize(_GL_SCALE, _GL_SCALE)
                grid.setSpacing(_GL_SCALE / 10, _GL_SCALE / 10)
                view.addItem(grid)
                self._gl_grid = grid
                return view
            except Exception as e:  # pragma: no cover
                return self._gl_hint(f"OpenGL init failed: {e}")
        return self._gl_hint(
            "3D view needs PyOpenGL.\n\nInstall it with:\n"
            "    pip install 'xrd-app[gl]'\n(or: pip install PyOpenGL)")

    def _gl_hint(self, text) -> QWidget:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color:#aaa; padding:24px;")
        return lbl

    # ---- study selector --------------------------------------------------
    def _populate_studies(self):
        """Fill the study dropdown from discovery; keep the current one selected."""
        self.study_cb.blockSignals(True)
        self.study_cb.clear()
        try:
            found = studies_core.list_studies(self.project_root)
        except Exception:
            found = []
        cur = str(self._study.resolve())
        sel = 0
        if found:
            for i, e in enumerate(found):
                desc = studies_core.describe(e)
                label = e["name"] + (f"   ({desc})" if desc else "")
                self.study_cb.addItem(label, e["abs_path"])
                if e["abs_path"] == cur:
                    sel = i
        else:
            # No discovered studies — offer the conventional Study/ path.
            self.study_cb.addItem("Study (default)", str(self._study))
        self.study_cb.setCurrentIndex(sel)
        # Adopt whatever ended up selected as the active study.
        data = self.study_cb.currentData()
        if data:
            self._set_study(Path(data))
        self.study_cb.blockSignals(False)

    def _set_study(self, study_dir: Path):
        self._study = Path(study_dir)
        self._rsm_path = self._study / "rsm.npz"
        self._qspace_dir = self._study / "qspace"

    def _on_study_changed(self, _idx):
        data = self.study_cb.currentData()
        if data:
            self._set_study(Path(data))
            self._reload()

    # ---- data ------------------------------------------------------------
    def _reload(self):
        self._populate_studies_if_needed()
        try:
            self._rsm = rsm_core.load_rsm(self._rsm_path) if self._rsm_path.exists() else None
        except Exception as e:
            self._rsm = None
            self.status.setText(f"rsm.npz error: {e}")
        try:
            self._cloud = (rsm_core.load_feature_cloud(self._qspace_dir)
                           if self._qspace_dir.is_dir() else None)
        except Exception as e:
            self._cloud = None
            self.status.setText(f"features error: {e}")
        self._load_shells()
        self._refresh_current()

    def _load_shells(self):
        """Reflection |Q| shells from the project's reflection set (best-effort).

        Uses the same resolution as the rest of the app (per-scan Metadata →
        project → bundled default) via ``DataManager.reflections_json`` +
        ``core.reflections.read_json``, so ring radii match the 2θ overlays
        elsewhere. Failures leave the ring list empty (rings just don't draw).
        """
        try:
            dm = DataManager(self.project_root, scan=self.scan)
            refls = refl_io.read_json(dm.reflections_json(self.scan))
            if not refls:  # fresh project with no saved set → bundled default
                refls = refl_io.default_reflections()
            self._shells = rsm_core.reflection_shells(refls)
        except Exception:
            self._shells = []
        self._refl_cmap = {
            name: QColor(ARC_COLORS[k % len(ARC_COLORS)])
            for k, (name, _q) in enumerate(self._shells)
        }

    def _populate_studies_if_needed(self):
        # Refresh the list on an explicit Reload so newly-run studies appear.
        sender = self.sender()
        if isinstance(sender, QPushButton):
            self._populate_studies()

    # ---- view switching --------------------------------------------------
    def _on_view_changed(self, idx):
        is_3d = (idx == 1)
        self.bar2d.setVisible(not is_3d)
        self.bar3d.setVisible(is_3d)
        self.stack.setCurrentIndex(1 if is_3d else 0)
        self._refresh_current()

    def _refresh_current(self):
        if self.view_cb.currentIndex() == 1:
            self._refresh_3d()
        else:
            self._refresh()

    # ---- 2D rendering ----------------------------------------------------
    def _clear(self):
        self.plot.clear()
        if self.legend is not None:
            try:
                self.legend.scene().removeItem(self.legend)
            except Exception:
                pass
            self.legend = None
        if self._colorbar is not None:
            try:
                self.glw.removeItem(self._colorbar)
            except Exception:
                pass
            self._colorbar = None

    def _refresh(self):
        self._clear()
        have_rsm = self._rsm is not None
        have_cloud = self._cloud is not None and self._cloud["n"] > 0
        if not have_rsm and not have_cloud:
            self.status.setText(
                "No RSM data. Run `xrd-app qspace` then `xrd-app rsm` "
                "(writes <study>/rsm.npz + <study>/qspace/*_features_q.csv).")
            return

        plane_label = self.plane_cb.currentText()
        proj_key, ax_x, ax_y = _PLANES[plane_label]
        self.plot.setLabel("bottom", f"{_AXIS_LABELS[ax_x]} (1/Å)")
        self.plot.setLabel("left", f"{_AXIS_LABELS[ax_y]} (1/Å)")

        if have_rsm and self.heat_chk.isChecked():
            self._draw_heatmap(proj_key, ax_x, ax_y)
        if have_cloud and self.cloud_chk.isChecked():
            self._draw_cloud(ax_x, ax_y)
        self._set_status(have_rsm, have_cloud)

    def _set_status(self, have_rsm, have_cloud):
        bits = []
        if have_rsm:
            bits.append(f"{len(self._rsm['scans'])} scans fused")
            if self._rsm.get("volume") is not None:
                bits.append("×".join(str(n) for n in self._rsm["volume"].shape) + " grid")
        if have_cloud:
            bits.append(f"{self._cloud['n']} features")
        self.status.setText("  •  ".join(bits))

    def _draw_heatmap(self, proj_key, ax_x, ax_y):
        proj = np.asarray(self._rsm["proj"][proj_key], dtype=np.float64)
        edges = self._rsm["edges"]
        ex, ey = edges[ax_x], edges[ax_y]
        img = proj.copy()
        if self.log_chk.isChecked():
            img = np.log1p(np.clip(img, 0, None))
        item = pg.ImageItem()
        item.setImage(img.T, autoLevels=False)
        finite = img[np.isfinite(img)]
        if finite.size:
            lo = float(np.percentile(finite, 1.0))
            hi = float(np.percentile(finite, 99.5))
            if hi <= lo:
                hi = lo + 1.0
            item.setLevels((lo, hi))
        try:
            item.setLookupTable(_get_cmap("inferno").getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass
        x0, y0 = float(ex[0]), float(ey[0])
        item.setRect(pg.QtCore.QRectF(x0, y0, float(ex[-1] - x0), float(ey[-1] - y0)))
        item.setZValue(-10)
        self.plot.addItem(item)

    def _draw_cloud(self, ax_x, ax_y):
        c = self._cloud
        coords = {0: c["qx"], 1: c["qy"], 2: c["qz"]}
        xs, ys = coords[ax_x], coords[ax_y]
        good = np.isfinite(xs) & np.isfinite(ys)
        mode = self.color_cb.currentText()

        if mode == "reflection":
            self.legend = self.plot.addLegend(offset=(-10, 10))
            refls = [c["reflection"][i] for i in range(c["n"]) if good[i]]
            uniq = sorted(set(refls))
            for k, ref in enumerate(uniq):
                sel = good & np.array([c["reflection"][i] == ref for i in range(c["n"])])
                if not np.any(sel):
                    continue
                col = QColor(ARC_COLORS[k % len(ARC_COLORS)])
                sp = pg.ScatterPlotItem(
                    x=xs[sel], y=ys[sel], size=7, pen=pg.mkPen("k", width=0.3),
                    brush=pg.mkBrush(col), name=(ref or "?"))
                self.plot.addItem(sp)
            return

        vals = c["theta"].copy() if mode == "θ" else c["intensity"].copy()
        sel = good & np.isfinite(vals)
        if not np.any(sel):
            return
        v = vals[sel]
        if mode == "intensity":
            v = np.log1p(np.clip(v, 0, None))
        vmin, vmax = float(np.min(v)), float(np.max(v))
        norm = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
        cmap = _get_cmap("viridis")
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        idx = np.clip((norm * 255).astype(int), 0, 255)
        brushes = [pg.mkBrush(int(lut[i][0]), int(lut[i][1]), int(lut[i][2])) for i in idx]
        sp = pg.ScatterPlotItem(
            x=xs[sel], y=ys[sel], size=7, pen=pg.mkPen("k", width=0.3), brush=brushes)
        self.plot.addItem(sp)
        try:
            bar = pg.ColorBarItem(values=(vmin, vmax), colorMap=cmap,
                                  label=("log I" if mode == "intensity" else "θ (°)"))
            self.glw.addItem(bar)
            self._colorbar = bar
        except Exception:
            self._colorbar = None

    # ---- 3D rendering ----------------------------------------------------
    def _clear_gl(self):
        if not _HAVE_GL or not hasattr(self.gl_view, "removeItem"):
            return
        for it in self._gl_items:
            try:
                self.gl_view.removeItem(it)
            except Exception:
                pass
        self._gl_items = []

    def _refresh_3d(self):
        if not _HAVE_GL or not hasattr(self.gl_view, "removeItem"):
            return
        self._clear_gl()
        have_rsm = self._rsm is not None and self._rsm.get("volume") is not None
        have_cloud = self._cloud is not None and self._cloud["n"] > 0
        if not have_rsm and not have_cloud:
            self.status.setText(
                "No 3D data. Run `xrd-app rsm` (volume) / `xrd-app qspace` "
                "(feature cloud) for this study.")
            return

        edges = self._rsm["edges"] if (self._rsm is not None) else self._cloud_edges()
        qmins = np.array([float(e[0]) for e in edges])
        spans = np.array([float(e[-1] - e[0]) for e in edges])
        span_max = float(spans.max()) or 1.0
        s = _GL_SCALE / span_max  # single scale → physical q-aspect preserved

        if have_rsm and self.vol_chk.isChecked():
            self._draw_volume(edges, qmins, spans, s)
        if have_cloud and self.cloud3d_chk.isChecked():
            self._draw_cloud_3d(qmins, s)
        if self._shells and self.shells_chk.isChecked():
            self._draw_shells_3d(qmins, s, edges)
        if have_cloud and self.scanrings_chk.isChecked():
            self._draw_scan_rings_3d(qmins, s)

        # Centre the camera on the box.
        try:
            centre = pg.Vector(*(spans * s / 2.0))
            self.gl_view.opts["center"] = centre
            self.gl_view.update()
        except Exception:
            pass
        self._set_status(have_rsm, have_cloud)

    def _cloud_edges(self):
        """Fallback q-edges from the feature cloud when there is no volume."""
        c = self._cloud
        out = []
        for a in ("qx", "qy", "qz"):
            v = c[a][np.isfinite(c[a])]
            lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            out.append(np.array([lo, hi]))
        return out

    def _draw_volume(self, edges, qmins, spans, s):
        vol = np.asarray(self._rsm["volume"], dtype=np.float32)
        target = int(self.res_cb.currentData() or 64)
        stride = max(1, int(np.ceil(max(vol.shape) / target)))
        vol = vol[::stride, ::stride, ::stride]
        rgba = self._volume_rgba(vol)
        item = gl.GLVolumeItem(rgba, sliceDensity=1, smooth=True,
                               glOptions="translucent")
        # voxel size in normalized units = (q span / n_voxels) * s
        vsize = (spans / np.array(vol.shape, dtype=float)) * s
        item.scale(*vsize)  # voxel index → normalized q (origin already at 0)
        self.gl_view.addItem(item)
        self._gl_items.append(item)

    def _volume_rgba(self, vol):
        """Map an intensity volume to an (nx,ny,nz,4) uint8 RGBA field.

        Colour = inferno LUT of the (optionally log) normalized intensity; alpha
        rises with intensity (empty/low voxels stay transparent so the bright
        blobs read as a cloud rather than a solid block). The Opacity slider
        scales the overall alpha.
        """
        v = vol.astype(np.float32)
        if self.log3d_chk.isChecked():
            v = np.log1p(np.clip(v, 0, None))
        finite = v[np.isfinite(v)]
        if finite.size:
            lo = float(np.percentile(finite, 50.0))
            hi = float(np.percentile(finite, 99.5))
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        try:
            lut = _get_cmap("inferno").getLookupTable(0.0, 1.0, 256)
        except Exception:
            lut = (np.linspace(0, 255, 256)[:, None] * np.ones(3)).astype(np.uint8)
        idx = np.clip((norm * 255).astype(np.int32), 0, 255)
        rgba = np.zeros(vol.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = lut[idx][..., :3]
        amax = self.opacity.value() / 100.0 * 255.0
        alpha = (norm ** 2) * amax
        alpha[norm < 0.05] = 0.0  # suppress background haze
        rgba[..., 3] = alpha.astype(np.uint8)
        return rgba

    def _draw_cloud_3d(self, qmins, s):
        c = self._cloud
        q = np.stack([c["qx"], c["qy"], c["qz"]], axis=1)
        good = np.all(np.isfinite(q), axis=1)
        if not np.any(good):
            return
        pos = (q[good] - qmins) * s
        refls = [c["reflection"][i] for i in range(c["n"]) if good[i]]
        uniq = sorted(set(refls))
        # Share the shell colour map so a reflection's points match its ring.
        cmap = {r: (self._refl_cmap.get(r) or QColor(ARC_COLORS[k % len(ARC_COLORS)]))
                for k, r in enumerate(uniq)}
        colors = np.array([
            [cmap[r].redF(), cmap[r].greenF(), cmap[r].blueF(), 1.0] for r in refls],
            dtype=np.float32)
        sp = gl.GLScatterPlotItem(pos=pos.astype(np.float32), color=colors,
                                  size=6.0, pxMode=True)
        sp.setGLOptions("translucent")
        self.gl_view.addItem(sp)
        self._gl_items.append(sp)

    def _add_ring(self, pts, qmins, s, qcolor, width=2.0):
        """Draw one closed circle (real-q ``pts``) into the normalized GL box."""
        posn = ((pts - qmins) * s).astype(np.float32)
        col = (qcolor.redF(), qcolor.greenF(), qcolor.blueF(), 0.9)
        line = gl.GLLinePlotItem(pos=posn, color=col, width=width,
                                 antialias=True, mode="line_strip")
        line.setGLOptions("translucent")
        self.gl_view.addItem(line)
        self._gl_items.append(line)

    def _draw_shells_3d(self, qmins, s, edges):
        """Concentric reflection |Q| shells as horizontal ring slices ("CDs").

        Each shell is the sphere ``|Q| = const`` (centred at the reciprocal-space
        origin) sliced by a horizontal plane ``qz = z``; that slice is a circle of
        radius ``√(|Q|² − z²)`` centred on the qz axis. Each reflection is sliced
        at the mean qz of *its own* detected features (so the ring threads that
        cluster), falling back to the global mean qz for reflections with no
        features. Shells that don't reach their plane (``|Q| < |z|``) are skipped.
        """
        refl_z, z0 = self._shell_planes(edges)
        for k, (name, q) in enumerate(self._shells):
            z = refl_z.get(name, z0)
            r2 = q * q - z * z
            if r2 <= 0.0:
                continue
            pts = rsm_core.ring_points(np.sqrt(r2), n=180, plane="xy",
                                       offset=(0.0, 0.0, z))
            col = self._refl_cmap.get(name) or QColor(ARC_COLORS[k % len(ARC_COLORS)])
            self._add_ring(pts, qmins, s, col, width=2.0)

    def _shell_planes(self, edges):
        """(per-reflection mean qz, global mean qz) for slicing the shells."""
        refl_z = {}
        if self._cloud is not None and self._cloud["n"] > 0:
            c = self._cloud
            qz = c["qz"]
            fin = np.isfinite(qz)
            z0 = float(np.mean(qz[fin])) if np.any(fin) else 0.0
            names = np.array(c["reflection"], dtype=object)
            for name in set(c["reflection"]):
                sel = fin & (names == name)
                if np.any(sel):
                    refl_z[name] = float(np.mean(qz[sel]))
        else:
            z0 = 0.5 * (float(edges[2][0]) + float(edges[2][-1]))
        return refl_z, z0

    def _draw_scan_rings_3d(self, qmins, s):
        """One circle outline around each scan's feature group (the θ stack).

        For every scan, the ring is centred on that scan's feature centroid and
        sized to enclose most of them (90th-percentile in-plane spread), laid in
        the qx–qy plane at the scan's mean qz — so the θ series reads as a stack
        of discs. Coloured by θ (viridis) when known, else cycled.
        """
        c = self._cloud
        q = np.stack([c["qx"], c["qy"], c["qz"]], axis=1)
        good = np.all(np.isfinite(q), axis=1)
        scans = [c["scan"][i] for i in range(c["n"])]
        uniq = sorted(set(s_ for i, s_ in enumerate(scans) if good[i]))
        if not uniq:
            return
        thetas = c["theta"]
        tvals = thetas[np.isfinite(thetas)]
        tmin, tmax = (float(tvals.min()), float(tvals.max())) if tvals.size else (0.0, 1.0)
        cmap = _get_cmap("viridis")
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        for k, scan in enumerate(uniq):
            sel = good & np.array([s_ == scan for s_ in scans])
            if not np.any(sel):
                continue
            pts_q = q[sel]
            cx, cy, cz = (float(np.mean(pts_q[:, j])) for j in range(3))
            dist = np.hypot(pts_q[:, 0] - cx, pts_q[:, 1] - cy)
            radius = float(np.percentile(dist, 90.0)) if dist.size else 0.0
            if radius <= 0.0:
                radius = float(np.max(dist)) if dist.size else 0.0
            if radius <= 0.0:
                continue
            ring = rsm_core.ring_points(radius, n=160, plane="xy",
                                        offset=(cx, cy, cz))
            th = thetas[sel]
            th = th[np.isfinite(th)]
            if th.size and tmax > tmin:
                frac = (float(np.mean(th)) - tmin) / (tmax - tmin)
            else:
                frac = k / max(1, len(uniq) - 1)
            rgb = lut[int(np.clip(frac * 255, 0, 255))]
            col = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            self._add_ring(ring, qmins, s, col, width=2.5)
