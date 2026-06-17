#!/usr/bin/env python3
"""
XRD detector frame labeling GUI for scan 203.

Browse raw diffraction patterns from Scan_0203/XRD/ and annotate them
with bounding boxes (Bragg peaks, diffuse scattering, defect signatures, etc.).

Features:
  - Lazy-load 1062x1028 detector frames from H5 files (no full scan in RAM)
  - Navigate frames: file selector + frame slider, arrow keys, global index
  - Draw, resize, move, delete bounding boxes with class labels
  - Per-frame annotations with copy-to-all-frames for stable ROIs
  - Contrast adjustment (percentile sliders + presets + log scale)
  - Colormap selector with reverse toggle
  - Coordinate + intensity readout on hover
  - Undo/redo for all annotation operations
  - Export/import annotations as JSON (includes frame metadata)
  - Keyboard shortcuts for fast labeling

Usage:
    python labeling_gui.py                          # auto-detect paths
    python labeling_gui.py /path/to/Scan_0203/XRD   # explicit XRD dir
"""

import sys
import os
import json
import copy
from pathlib import Path
from datetime import datetime

import warnings
import numpy as np
import h5py
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.patches import Rectangle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFileDialog, QGroupBox,
    QCheckBox, QInputDialog, QMessageBox, QSplitter,
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu,
    QSpinBox, QDoubleSpinBox, QGridLayout, QSizePolicy,
    QColorDialog,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QKeySequence


COLORMAPS = [
    "inferno", "viridis", "plasma", "magma", "cividis",
    "hot", "coolwarm", "gray", "jet", "turbo",
    "cubehelix", "bone", "copper", "pink", "seismic",
]

DEFAULT_LABELS = [
    "bragg_peak", "diffuse_scattering", "detector_artifact",
    "beam_stop", "powder_ring", "satellite_peak",
    "defect_streak", "background", "unknown",
]

LABEL_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


class BoundingBox:
    def __init__(self, x0, y0, x1, y1, label="unknown", color="#e6194b"):
        self.x0 = min(x0, x1)
        self.y0 = min(y0, y1)
        self.x1 = max(x0, x1)
        self.y1 = max(y0, y1)
        self.label = label
        self.color = color

    def contains(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def edge_at(self, x, y, tol=5.0):
        if abs(x - self.x0) < tol and self.y0 - tol <= y <= self.y1 + tol:
            return "left"
        if abs(x - self.x1) < tol and self.y0 - tol <= y <= self.y1 + tol:
            return "right"
        if abs(y - self.y0) < tol and self.x0 - tol <= x <= self.x1 + tol:
            return "top"
        if abs(y - self.y1) < tol and self.x0 - tol <= x <= self.x1 + tol:
            return "bottom"
        return None

    def to_dict(self):
        return {
            "x0": float(self.x0), "y0": float(self.y0),
            "x1": float(self.x1), "y1": float(self.y1),
            "label": self.label, "color": self.color,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["x0"], d["y0"], d["x1"], d["y1"],
                   d["label"], d.get("color", "#e6194b"))


class XRDFrameLoader:
    """Lazy-loads detector frames from Scan_0203/XRD/ H5 files."""

    def __init__(self, xrd_dir):
        self.xrd_dir = Path(xrd_dir)
        self.h5_files = sorted(self.xrd_dir.glob("scan_0203_*.h5"))
        if not self.h5_files:
            raise FileNotFoundError(f"No H5 files found in {xrd_dir}")

        self.num_files = len(self.h5_files)
        with h5py.File(self.h5_files[0], "r") as f:
            shape = f["entry/data/data"].shape
            self.frames_per_file = shape[0]
            self.frame_height = shape[1]
            self.frame_width = shape[2]

        self.total_frames = self.num_files * self.frames_per_file
        self._cache_file_idx = None
        self._cache_handle = None

    def get_frame(self, file_idx, frame_idx):
        if file_idx < 0 or file_idx >= self.num_files:
            return None
        if frame_idx < 0 or frame_idx >= self.frames_per_file:
            return None

        if self._cache_file_idx != file_idx:
            if self._cache_handle is not None:
                self._cache_handle.close()
            self._cache_handle = h5py.File(self.h5_files[file_idx], "r")
            self._cache_file_idx = file_idx

        return self._cache_handle["entry/data/data"][frame_idx].astype(np.float64)

    def global_to_local(self, global_idx):
        file_idx = global_idx // self.frames_per_file
        frame_idx = global_idx % self.frames_per_file
        return file_idx, frame_idx

    def local_to_global(self, file_idx, frame_idx):
        return file_idx * self.frames_per_file + frame_idx

    def get_filename(self, file_idx):
        return self.h5_files[file_idx].name

    def close(self):
        if self._cache_handle is not None:
            self._cache_handle.close()
            self._cache_handle = None
            self._cache_file_idx = None


class AnnotationCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas with bounding box interaction."""

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 8))
        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.06)
        super().__init__(self.fig)
        self.setParent(parent)

        self.image_data = None
        self.im = None
        self.cbar = None
        self.cmap = "inferno"
        self.vmin = None
        self.vmax = None

        self.boxes = []
        self.box_patches = []
        self.box_labels_text = []
        self.selected_box = None

        self.current_label = "unknown"
        self.current_color = "#e6194b"

        self.drawing = False
        self.moving = False
        self.resizing = False
        self.resize_edge = None
        self.draw_start = None
        self.move_offset = None
        self.temp_rect = None

        self.undo_stack = []
        self.redo_stack = []

        self._on_hover_callback = None
        self._on_box_change_callback = None

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_hover_callback(self, cb):
        self._on_hover_callback = cb

    def set_box_change_callback(self, cb):
        self._on_box_change_callback = cb

    def display_image(self, data, cmap=None, vmin=None, vmax=None, title=None):
        self.image_data = data
        if cmap:
            self.cmap = cmap
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax

        finite = data[np.isfinite(data)]
        if self.vmin is None:
            self.vmin = float(np.percentile(finite, 2)) if len(finite) else 0
        if self.vmax is None:
            self.vmax = float(np.percentile(finite, 98)) if len(finite) else 1

        if self.im is None:
            self.im = self.ax.imshow(
                data,
                origin="upper",
                aspect="equal",
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
                interpolation="nearest",
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, shrink=0.75, pad=0.01)
            self.ax.set_xlabel("Detector X (px)")
            self.ax.set_ylabel("Detector Y (px)")
        else:
            self.im.set_data(data)
            self.im.set_clim(self.vmin, self.vmax)
            self.im.set_cmap(self.cmap)

        if title:
            self.ax.set_title(title, fontsize=10)

        self._redraw_boxes()
        self.draw_idle()

    def update_contrast(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        if self.im is not None:
            self.im.set_clim(vmin, vmax)
            self.draw_idle()

    def update_cmap(self, cmap):
        self.cmap = cmap
        if self.im is not None:
            self.im.set_cmap(cmap)
            self.draw_idle()

    def _save_state(self):
        state = [copy.deepcopy(b.to_dict()) for b in self.boxes]
        self.undo_stack.append(state)
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append([copy.deepcopy(b.to_dict()) for b in self.boxes])
        state = self.undo_stack.pop()
        self.boxes = [BoundingBox.from_dict(d) for d in state]
        self.selected_box = None
        self._redraw_boxes()
        self.draw_idle()
        if self._on_box_change_callback:
            self._on_box_change_callback()

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append([copy.deepcopy(b.to_dict()) for b in self.boxes])
        state = self.redo_stack.pop()
        self.boxes = [BoundingBox.from_dict(d) for d in state]
        self.selected_box = None
        self._redraw_boxes()
        self.draw_idle()
        if self._on_box_change_callback:
            self._on_box_change_callback()

    def delete_selected(self):
        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            self._save_state()
            self.boxes.pop(self.selected_box)
            self.selected_box = None
            self._redraw_boxes()
            self.draw_idle()
            if self._on_box_change_callback:
                self._on_box_change_callback()

    def _redraw_boxes(self):
        for p in self.box_patches:
            try:
                p.remove()
            except Exception:
                pass
        for t in self.box_labels_text:
            try:
                t.remove()
            except Exception:
                pass
        self.box_patches.clear()
        self.box_labels_text.clear()

        for i, box in enumerate(self.boxes):
            lw = 2.5 if i == self.selected_box else 1.5
            ls = "-" if i == self.selected_box else "--"
            rect = Rectangle(
                (box.x0, box.y0),
                box.x1 - box.x0,
                box.y1 - box.y0,
                linewidth=lw,
                linestyle=ls,
                edgecolor=box.color,
                facecolor="none",
            )
            self.ax.add_patch(rect)
            self.box_patches.append(rect)

            txt = self.ax.text(
                box.x0, box.y0,
                f" {box.label}",
                color=box.color,
                fontsize=8,
                fontweight="bold" if i == self.selected_box else "normal",
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6),
            )
            self.box_labels_text.append(txt)

    def _update_patch_positions(self):
        for i, box in enumerate(self.boxes):
            if i < len(self.box_patches):
                self.box_patches[i].set_xy((box.x0, box.y0))
                self.box_patches[i].set_width(box.x1 - box.x0)
                self.box_patches[i].set_height(box.y1 - box.y0)
            if i < len(self.box_labels_text):
                self.box_labels_text[i].set_position((box.x0, box.y0))
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        toolbar = self.parent().findChild(NavigationToolbar2QT) if self.parent() else None
        if toolbar and toolbar.mode:
            return

        x, y = event.xdata, event.ydata
        tol = 5.0

        if self.selected_box is not None and 0 <= self.selected_box < len(self.boxes):
            edge = self.boxes[self.selected_box].edge_at(x, y, tol)
            if edge:
                self._save_state()
                self.resizing = True
                self.resize_edge = edge
                return

        for i, box in enumerate(self.boxes):
            edge = box.edge_at(x, y, tol)
            if edge:
                self._save_state()
                self.selected_box = i
                self.resizing = True
                self.resize_edge = edge
                self._redraw_boxes()
                self.draw_idle()
                if self._on_box_change_callback:
                    self._on_box_change_callback()
                return

        for i, box in enumerate(self.boxes):
            if box.contains(x, y):
                self._save_state()
                self.selected_box = i
                self.moving = True
                self.move_offset = (x - box.x0, y - box.y0)
                self._redraw_boxes()
                self.draw_idle()
                if self._on_box_change_callback:
                    self._on_box_change_callback()
                return

        self._save_state()
        self.drawing = True
        self.draw_start = (x, y)
        self.selected_box = None
        self._redraw_boxes()
        self.draw_idle()

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            if self._on_hover_callback:
                self._on_hover_callback(None, None, None)
            return

        x, y = event.xdata, event.ydata

        if self._on_hover_callback and self.image_data is not None:
            col, row = int(round(x)), int(round(y))
            if 0 <= row < self.image_data.shape[0] and 0 <= col < self.image_data.shape[1]:
                val = self.image_data[row, col]
                self._on_hover_callback(col, row, val)
            else:
                self._on_hover_callback(col, row, None)

        if self.drawing and self.draw_start:
            if self.temp_rect:
                try:
                    self.temp_rect.remove()
                except Exception:
                    pass
            x0, y0 = self.draw_start
            self.temp_rect = Rectangle(
                (min(x0, x), min(y0, y)),
                abs(x - x0), abs(y - y0),
                linewidth=2, edgecolor=self.current_color,
                facecolor="none", linestyle=":",
            )
            self.ax.add_patch(self.temp_rect)
            self.draw_idle()

        elif self.moving and self.selected_box is not None:
            box = self.boxes[self.selected_box]
            dx = x - self.move_offset[0] - box.x0
            dy = y - self.move_offset[1] - box.y0
            box.x0 += dx
            box.x1 += dx
            box.y0 += dy
            box.y1 += dy
            self._update_patch_positions()

        elif self.resizing and self.selected_box is not None:
            box = self.boxes[self.selected_box]
            if self.resize_edge == "left":
                box.x0 = x
            elif self.resize_edge == "right":
                box.x1 = x
            elif self.resize_edge == "top":
                box.y0 = y
            elif self.resize_edge == "bottom":
                box.y1 = y

            if box.x0 > box.x1:
                box.x0, box.x1 = box.x1, box.x0
                self.resize_edge = {"left": "right", "right": "left"}.get(self.resize_edge, self.resize_edge)
            if box.y0 > box.y1:
                box.y0, box.y1 = box.y1, box.y0
                self.resize_edge = {"top": "bottom", "bottom": "top"}.get(self.resize_edge, self.resize_edge)

            self._update_patch_positions()

    def _on_release(self, event):
        if event.button != 1:
            return

        if self.drawing and self.draw_start:
            if self.temp_rect:
                try:
                    self.temp_rect.remove()
                except Exception:
                    pass
                self.temp_rect = None

            if event.inaxes == self.ax:
                x0, y0 = self.draw_start
                x1, y1 = event.xdata, event.ydata
                if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
                    box = BoundingBox(x0, y0, x1, y1,
                                     self.current_label, self.current_color)
                    self.boxes.append(box)
                    self.selected_box = len(self.boxes) - 1
                    self._redraw_boxes()
                    if self._on_box_change_callback:
                        self._on_box_change_callback()
                else:
                    if self.undo_stack:
                        self.undo_stack.pop()

            self.drawing = False
            self.draw_start = None
            self.draw_idle()

        elif self.moving:
            self.moving = False
            self.move_offset = None
            if self._on_box_change_callback:
                self._on_box_change_callback()

        elif self.resizing:
            self.resizing = False
            self.resize_edge = None
            if self._on_box_change_callback:
                self._on_box_change_callback()


class LabelingGUI(QMainWindow):
    def __init__(self, xrd_dir):
        super().__init__()
        self.setWindowTitle("XRD Detector Frame Labeling — Scan 203")
        self.setGeometry(80, 40, 1550, 950)

        self.loader = XRDFrameLoader(xrd_dir)
        self.xrd_dir = xrd_dir
        self.labels = list(DEFAULT_LABELS)
        self.label_colors = dict(zip(DEFAULT_LABELS, LABEL_COLORS))

        self.current_file_idx = 0
        self.current_frame_idx = 0

        # Per-frame annotations: global_idx -> list of box dicts
        self.frame_annotations = {}

        self._build_ui()

        self.canvas.set_hover_callback(self._on_hover)
        self.canvas.set_box_change_callback(self._on_boxes_changed)
        self._on_label_changed(self.label_combo.currentText())

        QTimer.singleShot(100, self._load_and_display)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---- Left: canvas ----
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = AnnotationCanvas(left_panel)
        self.toolbar = NavigationToolbar2QT(self.canvas, left_panel)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("padding: 4px; font-family: monospace;")
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left_panel)

        # ---- Right: controls ----
        right_panel = QWidget()
        right_panel.setMaximumWidth(380)
        right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(right_panel)

        # --- Frame navigation ---
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QGridLayout()

        nav_layout.addWidget(QLabel("File:"), 0, 0)
        self.file_spin = QSpinBox()
        self.file_spin.setRange(1, self.loader.num_files)
        self.file_spin.setValue(1)
        self.file_spin.valueChanged.connect(self._on_file_changed)
        nav_layout.addWidget(self.file_spin, 0, 1)

        self.file_name_label = QLabel(self.loader.get_filename(0))
        self.file_name_label.setStyleSheet("font-size: 10px; color: #aaa;")
        nav_layout.addWidget(self.file_name_label, 0, 2)

        nav_layout.addWidget(QLabel("Frame:"), 1, 0)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, self.loader.frames_per_file - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        nav_layout.addWidget(self.frame_slider, 1, 1, 1, 2)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, self.loader.frames_per_file - 1)
        self.frame_spin.setValue(0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        nav_layout.addWidget(self.frame_spin, 1, 3)

        nav_btn_layout = QHBoxLayout()
        prev_btn = QPushButton("<< Prev")
        prev_btn.clicked.connect(self._prev_frame)
        nav_btn_layout.addWidget(prev_btn)

        next_btn = QPushButton("Next >>")
        next_btn.clicked.connect(self._next_frame)
        nav_btn_layout.addWidget(next_btn)

        self.global_label = QLabel("Global: 0 / 0")
        self.global_label.setStyleSheet("font-size: 10px; color: #aaa;")
        nav_btn_layout.addWidget(self.global_label)
        nav_layout.addLayout(nav_btn_layout, 2, 0, 1, 4)

        nav_group.setLayout(nav_layout)
        right_layout.addWidget(nav_group)

        # --- Colormap ---
        cmap_group = QGroupBox("Colormap")
        cmap_layout = QVBoxLayout()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText("inferno")
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_layout.addWidget(self.cmap_combo)

        self.reverse_cmap_cb = QCheckBox("Reverse colormap")
        self.reverse_cmap_cb.toggled.connect(self._on_cmap_changed)
        cmap_layout.addWidget(self.reverse_cmap_cb)
        cmap_group.setLayout(cmap_layout)
        right_layout.addWidget(cmap_group)

        # --- Contrast ---
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QGridLayout()

        contrast_layout.addWidget(QLabel("Min %:"), 0, 0)
        self.vmin_slider = QSlider(Qt.Horizontal)
        self.vmin_slider.setRange(0, 1000)
        self.vmin_slider.setValue(20)
        self.vmin_slider.valueChanged.connect(self._on_vmin_slider)
        contrast_layout.addWidget(self.vmin_slider, 0, 1)
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(0, 100)
        self.vmin_spin.setValue(2.0)
        self.vmin_spin.setSingleStep(0.5)
        self.vmin_spin.setDecimals(1)
        self.vmin_spin.setFixedWidth(65)
        self.vmin_spin.valueChanged.connect(self._on_vmin_spin)
        contrast_layout.addWidget(self.vmin_spin, 0, 2)

        contrast_layout.addWidget(QLabel("Max %:"), 1, 0)
        self.vmax_slider = QSlider(Qt.Horizontal)
        self.vmax_slider.setRange(0, 1000)
        self.vmax_slider.setValue(995)
        self.vmax_slider.valueChanged.connect(self._on_vmax_slider)
        contrast_layout.addWidget(self.vmax_slider, 1, 1)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(0, 100)
        self.vmax_spin.setValue(99.5)
        self.vmax_spin.setSingleStep(0.5)
        self.vmax_spin.setDecimals(1)
        self.vmax_spin.setFixedWidth(65)
        self.vmax_spin.valueChanged.connect(self._on_vmax_spin)
        contrast_layout.addWidget(self.vmax_spin, 1, 2)

        preset_layout = QHBoxLayout()
        for name, lo, hi in [("Full", 0, 100), ("Auto", 2, 98),
                              ("Tight", 10, 95), ("High", 50, 99.9)]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, l=lo, h=hi: self._set_contrast_preset(l, h))
            preset_layout.addWidget(btn)
        contrast_layout.addLayout(preset_layout, 2, 0, 1, 3)

        self.log_scale_cb = QCheckBox("Log scale")
        self.log_scale_cb.toggled.connect(self._load_and_display)
        contrast_layout.addWidget(self.log_scale_cb, 3, 0, 1, 3)

        contrast_group.setLayout(contrast_layout)
        right_layout.addWidget(contrast_group)

        # --- Label selector ---
        label_group = QGroupBox("Annotation Label")
        label_layout = QVBoxLayout()

        self.label_combo = QComboBox()
        self.label_combo.addItems(self.labels)
        self.label_combo.currentTextChanged.connect(self._on_label_changed)
        label_layout.addWidget(self.label_combo)

        label_btn_layout = QHBoxLayout()
        add_label_btn = QPushButton("Add Label")
        add_label_btn.clicked.connect(self._add_label)
        label_btn_layout.addWidget(add_label_btn)
        color_btn = QPushButton("Color")
        color_btn.clicked.connect(self._pick_label_color)
        label_btn_layout.addWidget(color_btn)
        label_layout.addLayout(label_btn_layout)

        label_group.setLayout(label_layout)
        right_layout.addWidget(label_group)

        # --- Bounding box list ---
        bbox_group = QGroupBox("Bounding Boxes (this frame)")
        bbox_layout = QVBoxLayout()

        self.bbox_list = QListWidget()
        self.bbox_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.bbox_list.currentRowChanged.connect(self._on_bbox_list_select)
        self.bbox_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.bbox_list.customContextMenuRequested.connect(self._bbox_context_menu)
        bbox_layout.addWidget(self.bbox_list)

        bbox_btn_layout = QHBoxLayout()
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_selected_box)
        bbox_btn_layout.addWidget(del_btn)

        relabel_btn = QPushButton("Relabel")
        relabel_btn.clicked.connect(self._relabel_selected_box)
        bbox_btn_layout.addWidget(relabel_btn)
        bbox_layout.addLayout(bbox_btn_layout)

        bbox_btn_layout2 = QHBoxLayout()
        copy_all_btn = QPushButton("Copy to All Frames")
        copy_all_btn.clicked.connect(self._copy_boxes_to_all)
        bbox_btn_layout2.addWidget(copy_all_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_all_boxes)
        bbox_btn_layout2.addWidget(clear_btn)
        bbox_layout.addLayout(bbox_btn_layout2)

        self.ann_count_label = QLabel("0 frames annotated")
        self.ann_count_label.setStyleSheet("font-size: 10px; color: #aaa;")
        bbox_layout.addWidget(self.ann_count_label)

        bbox_group.setLayout(bbox_layout)
        right_layout.addWidget(bbox_group)

        # --- I/O ---
        io_group = QGroupBox("Save / Load")
        io_layout = QHBoxLayout()
        save_btn = QPushButton("Export JSON")
        save_btn.clicked.connect(self._export_annotations)
        io_layout.addWidget(save_btn)
        load_btn = QPushButton("Import JSON")
        load_btn.clicked.connect(self._import_annotations)
        io_layout.addWidget(load_btn)
        io_group.setLayout(io_layout)
        right_layout.addWidget(io_group)

        right_layout.addStretch()

        # --- Shortcuts ---
        shortcuts_group = QGroupBox("Shortcuts")
        sc_layout = QVBoxLayout()
        sc_text = QLabel(
            "Left/A: Prev frame\n"
            "Right/D: Next frame\n"
            "Ctrl+Z: Undo   Ctrl+Y: Redo\n"
            "Del: Delete box   Esc: Deselect\n"
            "Ctrl+S: Save   1-9: Quick label"
        )
        sc_text.setStyleSheet("font-size: 11px; color: #888;")
        sc_layout.addWidget(sc_text)
        shortcuts_group.setLayout(sc_layout)
        right_layout.addWidget(shortcuts_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([1150, 380])

    # ---- Frame navigation ----

    def _save_current_boxes(self):
        gidx = self.loader.local_to_global(self.current_file_idx, self.current_frame_idx)
        if self.canvas.boxes:
            self.frame_annotations[gidx] = [b.to_dict() for b in self.canvas.boxes]
        else:
            self.frame_annotations.pop(gidx, None)
        self._update_ann_count()

    def _load_boxes_for_frame(self):
        gidx = self.loader.local_to_global(self.current_file_idx, self.current_frame_idx)
        saved = self.frame_annotations.get(gidx, [])
        self.canvas.boxes = [BoundingBox.from_dict(d) for d in saved]
        self.canvas.selected_box = None
        self.canvas.undo_stack.clear()
        self.canvas.redo_stack.clear()

    def _load_and_display(self):
        frame = self.loader.get_frame(self.current_file_idx, self.current_frame_idx)
        if frame is None:
            return

        if self.log_scale_cb.isChecked():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                frame = np.where(frame > 0, np.log10(frame), np.nan)

        finite = frame[np.isfinite(frame)]
        if len(finite) == 0:
            return

        lo_pct = self.vmin_spin.value()
        hi_pct = self.vmax_spin.value()
        vmin = float(np.percentile(finite, lo_pct))
        vmax = float(np.percentile(finite, hi_pct))

        cmap = self.cmap_combo.currentText()
        if self.reverse_cmap_cb.isChecked():
            cmap = cmap + "_r"

        gidx = self.loader.local_to_global(self.current_file_idx, self.current_frame_idx)
        title = (f"File {self.current_file_idx + 1}/{self.loader.num_files}  "
                 f"Frame {self.current_frame_idx}/{self.loader.frames_per_file - 1}  "
                 f"[global {gidx}]")

        self.canvas.vmin = None
        self.canvas.vmax = None
        self.canvas.display_image(frame, cmap, vmin, vmax, title)

        self._load_boxes_for_frame()
        self.canvas._redraw_boxes()
        self.canvas.draw_idle()
        self._refresh_bbox_list()

        self.global_label.setText(
            f"Global: {gidx} / {self.loader.total_frames - 1}")
        self.file_name_label.setText(self.loader.get_filename(self.current_file_idx))

    def _navigate_to(self, file_idx, frame_idx):
        self._save_current_boxes()
        self.current_file_idx = file_idx
        self.current_frame_idx = frame_idx

        self.file_spin.blockSignals(True)
        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.file_spin.setValue(file_idx + 1)
        self.frame_slider.setValue(frame_idx)
        self.frame_spin.setValue(frame_idx)
        self.file_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)

        self._load_and_display()

    def _on_file_changed(self, val):
        self._navigate_to(val - 1, self.current_frame_idx)

    def _on_frame_slider_changed(self, val):
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(val)
        self.frame_spin.blockSignals(False)
        self._navigate_to(self.current_file_idx, val)

    def _on_frame_spin_changed(self, val):
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(val)
        self.frame_slider.blockSignals(False)
        self._navigate_to(self.current_file_idx, val)

    def _prev_frame(self):
        if self.current_frame_idx > 0:
            self._navigate_to(self.current_file_idx, self.current_frame_idx - 1)
        elif self.current_file_idx > 0:
            self._navigate_to(self.current_file_idx - 1, self.loader.frames_per_file - 1)

    def _next_frame(self):
        if self.current_frame_idx < self.loader.frames_per_file - 1:
            self._navigate_to(self.current_file_idx, self.current_frame_idx + 1)
        elif self.current_file_idx < self.loader.num_files - 1:
            self._navigate_to(self.current_file_idx + 1, 0)

    # ---- Key handling ----

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Undo):
            self.canvas.undo()
            self._refresh_bbox_list()
        elif event.matches(QKeySequence.Redo):
            self.canvas.redo()
            self._refresh_bbox_list()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected_box()
        elif event.key() == Qt.Key_Escape:
            self.canvas.selected_box = None
            self.canvas._redraw_boxes()
            self.canvas.draw_idle()
            self._refresh_bbox_list()
        elif event.matches(QKeySequence.Save):
            self._export_annotations()
        elif event.key() in (Qt.Key_Left, Qt.Key_A):
            self._prev_frame()
        elif event.key() in (Qt.Key_Right, Qt.Key_D):
            self._next_frame()
        elif Qt.Key_1 <= event.key() <= Qt.Key_9:
            idx = event.key() - Qt.Key_1
            if idx < len(self.labels):
                self.label_combo.setCurrentIndex(idx)
        else:
            super().keyPressEvent(event)

    # ---- Colormap / contrast ----

    def _on_cmap_changed(self, *args):
        cmap = self.cmap_combo.currentText()
        if self.reverse_cmap_cb.isChecked():
            cmap = cmap + "_r"
        self.canvas.update_cmap(cmap)

    def _on_vmin_slider(self, val):
        pct = val / 10.0
        self.vmin_spin.blockSignals(True)
        self.vmin_spin.setValue(pct)
        self.vmin_spin.blockSignals(False)
        self._on_contrast_changed()

    def _on_vmin_spin(self, val):
        self.vmin_slider.blockSignals(True)
        self.vmin_slider.setValue(int(val * 10))
        self.vmin_slider.blockSignals(False)
        self._on_contrast_changed()

    def _on_vmax_slider(self, val):
        pct = val / 10.0
        self.vmax_spin.blockSignals(True)
        self.vmax_spin.setValue(pct)
        self.vmax_spin.blockSignals(False)
        self._on_contrast_changed()

    def _on_vmax_spin(self, val):
        self.vmax_slider.blockSignals(True)
        self.vmax_slider.setValue(int(val * 10))
        self.vmax_slider.blockSignals(False)
        self._on_contrast_changed()

    def _on_contrast_changed(self):
        frame = self.loader.get_frame(self.current_file_idx, self.current_frame_idx)
        if frame is None:
            return
        if self.log_scale_cb.isChecked():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                frame = np.where(frame > 0, np.log10(frame), np.nan)
        finite = frame[np.isfinite(frame)]
        if len(finite) == 0:
            return
        vmin = float(np.percentile(finite, self.vmin_spin.value()))
        vmax = float(np.percentile(finite, self.vmax_spin.value()))
        self.canvas.update_contrast(vmin, vmax)

    def _set_contrast_preset(self, lo, hi):
        self.vmin_slider.blockSignals(True)
        self.vmax_slider.blockSignals(True)
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setValue(lo)
        self.vmax_spin.setValue(hi)
        self.vmin_slider.setValue(int(lo * 10))
        self.vmax_slider.setValue(int(hi * 10))
        self.vmin_slider.blockSignals(False)
        self.vmax_slider.blockSignals(False)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)
        self._on_contrast_changed()

    # ---- Labels ----

    def _on_label_changed(self, label):
        color = self.label_colors.get(label, "#e6194b")
        self.canvas.current_label = label
        self.canvas.current_color = color

    def _add_label(self):
        text, ok = QInputDialog.getText(self, "New Label", "Label name:")
        if ok and text.strip():
            name = text.strip().replace(" ", "_").lower()
            if name not in self.labels:
                self.labels.append(name)
                color = LABEL_COLORS[len(self.labels) % len(LABEL_COLORS)]
                self.label_colors[name] = color
                self.label_combo.addItem(name)
            self.label_combo.setCurrentText(name)

    def _pick_label_color(self):
        label = self.label_combo.currentText()
        current = QColor(self.label_colors.get(label, "#e6194b"))
        color = QColorDialog.getColor(current, self, "Pick label color")
        if color.isValid():
            self.label_colors[label] = color.name()
            self.canvas.current_color = color.name()

    # ---- Hover ----

    def _on_hover(self, x, y, val):
        if x is None:
            self.status_label.setText("Ready")
        else:
            val_str = f"{val:.2f}" if val is not None else "N/A"
            self.status_label.setText(
                f"Pixel: ({x}, {y})  |  Intensity: {val_str}  |  "
                f"File {self.current_file_idx + 1} Frame {self.current_frame_idx}")

    # ---- Bounding box management ----

    def _on_boxes_changed(self):
        self._refresh_bbox_list()

    def _refresh_bbox_list(self):
        self.bbox_list.blockSignals(True)
        self.bbox_list.clear()
        for i, box in enumerate(self.canvas.boxes):
            w = abs(box.x1 - box.x0)
            h = abs(box.y1 - box.y0)
            item = QListWidgetItem(f"[{i}] {box.label}  ({w:.0f}x{h:.0f} px)")
            item.setForeground(QColor(box.color))
            self.bbox_list.addItem(item)
        if (self.canvas.selected_box is not None
                and 0 <= self.canvas.selected_box < self.bbox_list.count()):
            self.bbox_list.setCurrentRow(self.canvas.selected_box)
        self.bbox_list.blockSignals(False)

    def _update_ann_count(self):
        n = sum(1 for v in self.frame_annotations.values() if v)
        self.ann_count_label.setText(f"{n} frames annotated")

    def _on_bbox_list_select(self, row):
        if row < 0:
            return
        self.canvas.selected_box = row
        self.canvas._redraw_boxes()
        self.canvas.draw_idle()

    def _bbox_context_menu(self, pos):
        item = self.bbox_list.itemAt(pos)
        if item is None:
            return
        row = self.bbox_list.row(item)
        menu = QMenu()
        delete_action = menu.addAction("Delete")
        relabel_action = menu.addAction("Relabel")
        action = menu.exec_(self.bbox_list.mapToGlobal(pos))
        if action == delete_action:
            self.canvas.selected_box = row
            self._delete_selected_box()
        elif action == relabel_action:
            self.canvas.selected_box = row
            self._relabel_selected_box()

    def _delete_selected_box(self):
        self.canvas.delete_selected()
        self._refresh_bbox_list()

    def _relabel_selected_box(self):
        idx = self.canvas.selected_box
        if idx is None or idx < 0 or idx >= len(self.canvas.boxes):
            return
        label, ok = QInputDialog.getItem(
            self, "Relabel", "Select label:", self.labels, 0, False)
        if ok:
            self.canvas._save_state()
            self.canvas.boxes[idx].label = label
            self.canvas.boxes[idx].color = self.label_colors.get(label, "#e6194b")
            self.canvas._redraw_boxes()
            self.canvas.draw_idle()
            self._refresh_bbox_list()

    def _copy_boxes_to_all(self):
        if not self.canvas.boxes:
            return
        reply = QMessageBox.question(
            self, "Copy Boxes",
            f"Copy {len(self.canvas.boxes)} box(es) from this frame to ALL "
            f"{self.loader.total_frames} frames?\n\n"
            "This will overwrite existing annotations on other frames.",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        boxes_dicts = [b.to_dict() for b in self.canvas.boxes]
        for gidx in range(self.loader.total_frames):
            self.frame_annotations[gidx] = copy.deepcopy(boxes_dicts)
        self._update_ann_count()
        self.status_label.setText(
            f"Copied {len(boxes_dicts)} boxes to {self.loader.total_frames} frames")

    def _clear_all_boxes(self):
        if not self.canvas.boxes:
            return
        self.canvas._save_state()
        self.canvas.boxes.clear()
        self.canvas.selected_box = None
        self.canvas._redraw_boxes()
        self.canvas.draw_idle()
        self._refresh_bbox_list()

    # ---- I/O ----

    def _export_annotations(self):
        self._save_current_boxes()
        default_path = os.path.join(
            os.path.dirname(self.xrd_dir), "results", "scan203",
            "scan_203_xrd_annotations.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", default_path, "JSON (*.json)")
        if not path:
            return

        ann = {
            "scan": "scan_203",
            "created": datetime.now().isoformat(),
            "xrd_dir": str(self.xrd_dir),
            "num_files": self.loader.num_files,
            "frames_per_file": self.loader.frames_per_file,
            "frame_shape": [self.loader.frame_height, self.loader.frame_width],
            "label_colors": self.label_colors,
            "annotations": {},
        }
        for gidx, boxes in self.frame_annotations.items():
            if boxes:
                file_idx, frame_idx = self.loader.global_to_local(gidx)
                ann["annotations"][str(gidx)] = {
                    "file_idx": file_idx,
                    "frame_idx": frame_idx,
                    "filename": self.loader.get_filename(file_idx),
                    "boxes": boxes,
                }

        with open(path, "w") as f:
            json.dump(ann, f, indent=2)

        n = sum(1 for v in ann["annotations"].values() if v.get("boxes"))
        self.status_label.setText(f"Saved annotations for {n} frames to {path}")

    def _import_annotations(self):
        default_path = os.path.join(
            os.path.dirname(self.xrd_dir), "results", "scan203",
            "scan_203_xrd_annotations.json")
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Annotations", default_path, "JSON (*.json)")
        if not path:
            return

        with open(path, "r") as f:
            ann = json.load(f)

        self.frame_annotations.clear()
        for gidx_str, data in ann.get("annotations", {}).items():
            gidx = int(gidx_str)
            self.frame_annotations[gidx] = data.get("boxes", [])

        for lbl, col in ann.get("label_colors", {}).items():
            self.label_colors[lbl] = col
            if lbl not in self.labels:
                self.labels.append(lbl)
                self.label_combo.addItem(lbl)

        self._load_boxes_for_frame()
        self.canvas._redraw_boxes()
        self.canvas.draw_idle()
        self._refresh_bbox_list()
        self._update_ann_count()

        n = sum(1 for v in self.frame_annotations.values() if v)
        self.status_label.setText(f"Loaded annotations for {n} frames from {path}")

    def closeEvent(self, event):
        self._save_current_boxes()
        self.loader.close()
        super().closeEvent(event)


def main():
    xrd_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "raw_scans", "Scan_0203", "XRD",
    )

    if len(sys.argv) > 1:
        xrd_dir = sys.argv[1]

    xrd_dir = os.path.abspath(xrd_dir)

    if not os.path.isdir(xrd_dir):
        print(f"Error: XRD directory not found: {xrd_dir}")
        print("Usage: python labeling_gui.py [path/to/Scan_0203/XRD]")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, QColor(220, 220, 220))
    palette.setColor(palette.Base, QColor(35, 35, 35))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(palette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(palette.Text, QColor(220, 220, 220))
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, QColor(220, 220, 220))
    palette.setColor(palette.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)

    window = LabelingGUI(xrd_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
