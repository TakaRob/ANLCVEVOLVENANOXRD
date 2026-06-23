"""View/Label tab — wraps the interactive labeling GUI."""

from __future__ import annotations

from ..gui import labeling

TAB_META = {
    "title": "View/Label",
    "order": 30,
    "takes_bin_size": False,
    "scan_dependent": True,
    "general": (
        "Interactive per-bin viewer and Bragg-peak labeler. Navigate bins, toggle "
        "noise reduction, overlay a detector algorithm, adjust sensitivity, and "
        "annotate peaks. Human labels are stored per bin and can be marked "
        "'verified' for use as CVEvolve ground truth."
    ),
}


def make_tab(project_root=".", scan=None, bin_size=3):
    from PyQt5.QtWidgets import (
        QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
    )

    win = labeling.build_window(project_root, scan=scan, bin_size=bin_size)

    def save_algo():
        from .save_algorithm_dialog import SaveAlgorithmDialog
        # Sensitivity, bin size and noise reduction come straight from the live
        # view, so the popup only asks for the base detector + name.
        #
        # The popup reproduces the full "Run on displayed image (filtered/clipped)"
        # transform, so it only needs the base detector + name:
        #  - Noise reduction: the *selected* radial model (gaussian /
        #    split_gaussian / skewed_gaussian / fourier) plus the Strength/Shift
        #    sliders. The saved detector reruns noise_reduction_algorithms.
        #    reduce_noise per image before the base detector's own cleaning.
        #  - Log scale + contrast clip: the log toggle and the contrast
        #    *percentile* bounds (not the absolute vmin/vmax, which are per-image
        #    values that would misapply across bins). The saved detector
        #    recomputes the clip per image. See core.save_algorithm._TEMPLATE.
        sens = win.peak_sens_slider.value() / 10.0
        if win.noise_cb.isChecked():
            nr = win.noise_algo_combo.currentData()
            strength = win.strength_slider.value() / 100.0
            shift = win.shift_slider.value() / 10.0
        else:
            nr, strength, shift = None, 1.0, 0.0
        log_scale = bool(getattr(win.canvas, "log_scale", False))
        clip_lo = win.vmin_slider.value() / 10.0
        clip_hi = win.vmax_slider.value() / 10.0
        SaveAlgorithmDialog(
            project_root, bin_size=win.bin_size, kind="peak",
            sensitivity=sens, noise_reduction=nr, noise_strength=strength,
            noise_shift=shift, log_scale=log_scale, clip_lo_pct=clip_lo,
            clip_hi_pct=clip_hi, compact=True, parent=win).exec_()

    # Build a thin toolbar so we keep a handle on the button for visibility
    # wiring. The "Save Algorithm…" action only makes sense once the user is
    # running detection on the displayed (filtered/clipped) image, so the button
    # is shown only while that checkbox is checked.
    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 0)
    bar = QHBoxLayout()
    save_btn = QPushButton("Save Algorithm…")
    save_btn.clicked.connect(save_algo)
    bar.addWidget(save_btn)
    bar.addStretch()
    lay.addLayout(bar)
    lay.addWidget(win)
    container._embedded_window = win

    cb = getattr(win, "peak_use_filtered_cb", None)
    if cb is not None:
        save_btn.setVisible(cb.isChecked())
        cb.toggled.connect(save_btn.setVisible)
    return container


if __name__ == "__main__":
    from ._standalone import run_standalone
    run_standalone(make_tab, TAB_META["title"])
