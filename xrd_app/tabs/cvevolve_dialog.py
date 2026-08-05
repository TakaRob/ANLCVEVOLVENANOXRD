"""[Use CVEvolve] popup: scaffold a project, split holdout, edit files, run.

The flow, top to bottom:

1. **Create CVEvolve Project** — seed default ``config.yaml`` / ``prompt.md`` /
   ``holdout_test_prompt.md`` (beamline users don't arrive with these).
2. **Pick Development Set** — build a seeded dev/holdout split by bins.
3. **Edit default files** — open each generated file in an embedded editor so the
   user can tweak small things (metric hint, what the model looks for).
4. **Run CVEvolve** — with an optional Hutch toggle. When Hutch is on, the bottom
   area becomes a live ``[SQL table] [model output]`` view.

Every action shells out to the CLI (``cvevolve-init`` / ``build-holdout`` /
``run-cvevolve``) through the embedded console — the CLI is the engine.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import DataManager
from ..core import catalogs
from ._console import JobConsole


class _FileEditorDialog(QDialog):
    """Minimal embedded text editor: load a file, edit, save in place.

    Non-modal, so several files can be open at once, and a font-size spinbox sets
    the editor text size (persisted back to the launcher so the next editor opens
    at the same size). Used for the generated config/prompt files so editing works
    self-contained on WSL/OneDrive (spaced paths) without an external editor.
    """

    def __init__(self, path, parent=None, font_size=11, on_font_change=None):
        super().__init__(parent)
        self.path = Path(path)
        self._on_font_change = on_font_change
        self.setWindowTitle(f"Edit — {self.path.name}")
        self.resize(820, 640)

        lay = QVBoxLayout(self)
        head = QHBoxLayout()
        head.addWidget(QLabel(str(self.path)), 1)
        head.addWidget(QLabel("Font:"))
        self.font_size = QSpinBox()
        self.font_size.setRange(6, 48)
        self.font_size.setValue(int(font_size))
        self.font_size.setSuffix(" pt")
        self.font_size.valueChanged.connect(self._apply_font)
        head.addWidget(self.font_size)
        lay.addLayout(head)

        self.edit = QPlainTextEdit()
        self.edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        lay.addWidget(self.edit, 1)
        self._apply_font(int(font_size))

        self.status = QLabel("")
        row = QHBoxLayout()
        row.addWidget(self.status, 1)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        row.addWidget(save_btn)
        row.addWidget(close_btn)
        lay.addLayout(row)

        self._load()

    def _apply_font(self, size):
        self.edit.setFont(QFont("monospace", int(size)))
        if self._on_font_change:
            self._on_font_change(int(size))

    def _load(self):
        try:
            self.edit.setPlainText(self.path.read_text())
        except OSError as e:
            self.edit.setPlainText("")
            self.status.setText(f"could not read file: {e}")

    def _save(self):
        try:
            self.path.write_text(self.edit.toPlainText())
            self.status.setText("saved")
        except OSError as e:
            self.status.setText(f"save failed: {e}")


class HutchDbView(QWidget):
    """Live, read-only browser over the Hutch SQLite DB.

    A generic table browser (dropdown of tables + a grid) refreshed on a timer —
    robust to whatever schema Hutch writes. Shows a placeholder until the DB file
    appears, then lists tables and the most-recent rows of the selected one.
    """

    _MAX_ROWS = 500

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db_path = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("SQL table:"))
        self.table_pick = QComboBox()
        self.table_pick.currentTextChanged.connect(lambda *_: self._refresh(force=True))
        top.addWidget(self.table_pick, 1)
        self.info = QLabel("")
        top.addWidget(self.info)
        lay.addLayout(top)

        self.grid = QTableWidget()
        self.grid.setEditTriggers(QTableWidget.NoEditTriggers)
        self.grid.setFont(QFont("monospace", 8))
        lay.addWidget(self.grid, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self._refresh)

    # ----- control -----------------------------------------------------
    def set_db(self, path):
        self._db_path = Path(path)
        self.info.setText("waiting for hutch db…")
        self._refresh(force=True)
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        self.stop()
        super().closeEvent(event)

    # ----- internals ---------------------------------------------------
    def _connect(self):
        # Read-only URI so we never block the writer; may raise while the DB is
        # being created or is momentarily locked — callers swallow that.
        uri = f"file:{self._db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=0.5)

    def _tables(self, con):
        cur = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [r[0] for r in cur.fetchall()]

    def _refresh(self, force=False):
        if not self._db_path or not self._db_path.exists():
            self.info.setText("waiting for hutch db…")
            return
        try:
            con = self._connect()
        except sqlite3.Error:
            return
        try:
            tables = self._tables(con)
            if not tables:
                self.info.setText("db present, no tables yet")
                return
            # Keep the dropdown in sync without clobbering the user's selection.
            current = self.table_pick.currentText()
            existing = [self.table_pick.itemText(i) for i in range(self.table_pick.count())]
            if existing != tables:
                self.table_pick.blockSignals(True)
                self.table_pick.clear()
                self.table_pick.addItems(tables)
                if current in tables:
                    self.table_pick.setCurrentText(current)
                self.table_pick.blockSignals(False)
            table = self.table_pick.currentText() or tables[0]
            self._load_table(con, table)
        except sqlite3.Error as e:
            self.info.setText(f"read error: {e}")
        finally:
            con.close()

    def _load_table(self, con, table):
        # Newest rows first when the table has a rowid; fall back otherwise.
        try:
            cur = con.execute(
                f'SELECT * FROM "{table}" ORDER BY rowid DESC LIMIT {self._MAX_ROWS}')
        except sqlite3.Error:
            cur = con.execute(f'SELECT * FROM "{table}" LIMIT {self._MAX_ROWS}')
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        total = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

        self.grid.setColumnCount(len(cols))
        self.grid.setHorizontalHeaderLabels(cols)
        self.grid.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.grid.setItem(r, c, QTableWidgetItem("" if val is None else str(val)))
        self.grid.resizeColumnsToContents()
        shown = f"showing {len(rows)}" + (f" of {total}" if total > len(rows) else "")
        self.info.setText(f"{table}: {shown} rows")


class CVEvolveDialog(QDialog):
    def __init__(self, project_root, scan=None, bin_size=3, parent=None):
        super().__init__(parent)
        self.project_root = str(Path(project_root).resolve())
        self.scan = scan
        self.bin_size = bin_size
        self.dm = DataManager(self.project_root, scan=self.scan)
        self.cvevolve_dir = self.dm.cvevolve_dir
        self.setWindowTitle("Use CVEvolve")
        self.resize(1100, 760)

        lay = QVBoxLayout(self)

        # ---- 1. Create project ------------------------------------------
        create_box = QGroupBox("1. Create CVEvolve Project")
        cbl = QHBoxLayout(create_box)
        cbl.addWidget(QLabel("Name:"))
        self.name = QLineEdit()
        self.name.setPlaceholderText(self.cvevolve_dir.name or "cvevolve")
        cbl.addWidget(self.name, 1)
        create_btn = QPushButton("Create CVEvolve Project")
        create_btn.clicked.connect(self._create_project)
        cbl.addWidget(create_btn)
        lay.addWidget(create_box)

        # ---- 2. Development set -----------------------------------------
        dev_box = QGroupBox("2. Pick Development Set (by bins)")
        dv = QVBoxLayout(dev_box)
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source file:"))
        src_row.addStretch(1)
        rescan_btn = QPushButton("↻ Rescan")
        rescan_btn.clicked.connect(self._refresh_sources)
        src_row.addWidget(rescan_btn)
        dv.addLayout(src_row)

        self.sources = QListWidget()
        self.sources.setMaximumHeight(150)
        dv.addWidget(self.sources)

        form = QFormLayout()
        self.holdout = QDoubleSpinBox()
        self.holdout.setRange(0, 90)
        self.holdout.setValue(20)
        self.holdout.setSuffix(" %")
        form.addRow("Holdout:", self.holdout)

        self.seed = QSpinBox()
        self.seed.setRange(0, 10_000)
        self.seed.setValue(42)
        form.addRow("Seed:", self.seed)

        self.count_lbl = QLabel("(build to see counts)")
        form.addRow("Split:", self.count_lbl)
        dv.addLayout(form)

        build_btn = QPushButton("Build Split")
        build_btn.clicked.connect(self._build)
        dv.addWidget(build_btn)
        lay.addWidget(dev_box)

        # ---- 3. Edit default files --------------------------------------
        edit_box = QGroupBox("3. Edit Default Files")
        ebl = QHBoxLayout(edit_box)
        for label, fname in (("config.yaml", "config.yaml"),
                             ("prompt.md", "prompt.md"),
                             ("holdout_test_prompt.md", "holdout_test_prompt.md")):
            btn = QPushButton(f"Edit {label}")
            btn.clicked.connect(lambda _=False, f=fname: self._edit_file(f))
            ebl.addWidget(btn)
        lay.addWidget(edit_box)

        # ---- 4. Run CVEvolve --------------------------------------------
        run_box = QGroupBox("4. Run CVEvolve")
        rl = QFormLayout(run_box)
        cfg_row = QHBoxLayout()
        self.config = QLineEdit()
        self.config.setPlaceholderText("CVEvolve config.yaml")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_config)
        cfg_row.addWidget(self.config)
        cfg_row.addWidget(browse)
        rl.addRow("Config:", cfg_row)

        self.engine = QComboBox()
        self.engine.addItems(["podman", "docker", "local"])
        rl.addRow("Engine:", self.engine)

        self.argo_key = QLineEdit()
        self.argo_key.setPlaceholderText("Transient; not saved (uses ARGO_API_KEY)")
        rl.addRow("Argo username:", self.argo_key)

        self.hutch = QCheckBox("Enable Hutch tracking (live SQL + model output)")
        rl.addRow("Tracking:", self.hutch)

        run_btn = QPushButton("Run CVEvolve")
        run_btn.clicked.connect(self._run)
        rl.addRow(run_btn)
        lay.addWidget(run_box)

        # ---- Output: [SQL table] | [model output] -----------------------
        self.split = QSplitter(Qt.Horizontal)
        self.hutch_view = HutchDbView()
        self.console = JobConsole()
        self.split.addWidget(self.hutch_view)
        self.split.addWidget(self.console)
        self.hutch_view.hide()  # revealed only for a Hutch-enabled run
        lay.addWidget(self.split, 1)

        # Open editors (non-modal) and the font size they share.
        self._editors = []
        self._editor_font = 11

        # Prefill the config field if a project already exists.
        existing_cfg = self.cvevolve_dir / "config.yaml"
        if existing_cfg.exists():
            self.config.setText(str(existing_cfg))
        self._refresh_sources()

    # ----- helpers ----------------------------------------------------
    def _create_project(self):
        name = self.name.text().strip()
        args = ["cvevolve-init", "--root", self.project_root]
        if name:
            args += ["--name", name]
        self.console.run(args, on_finished=self._on_project_created)

    def _on_project_created(self, code):
        if code == 0:
            cfg = self.cvevolve_dir / "config.yaml"
            if cfg.exists():
                self.config.setText(str(cfg))

    def _edit_file(self, fname):
        path = self.cvevolve_dir / fname
        if not path.exists():
            self.console._append(
                f"[{fname} not found — click 'Create CVEvolve Project' first]\n")
            return
        ed = _FileEditorDialog(path, parent=self, font_size=self._editor_font,
                               on_font_change=self._set_editor_font)
        self._editors.append(ed)  # keep a ref so the non-modal window survives
        ed.show()
        ed.raise_()

    def _set_editor_font(self, size):
        self._editor_font = int(size)

    def _refresh_sources(self, *_):
        """List the actual dev-set source files: verified labels and peaks HDF5.

        (Shapes are intentionally excluded.) Each item carries the ``--source``
        kind and algorithm name that ``build-holdout`` needs.
        """
        self.sources.clear()
        ldir = self.dm.labels_dir(self.scan)
        bs = f"{self.bin_size}x{self.bin_size}"

        verified = ldir / f"bin_annotations_{bs}.json"
        if verified.exists():
            it = QListWidgetItem(f"{verified.name}   (verified labels)")
            it.setData(Qt.UserRole, {"source": "verified", "algorithm": None})
            self.sources.addItem(it)

        if ldir.is_dir():
            for p in catalogs.list_catalogs(ldir, "peaks", self.bin_size):
                info = catalogs.parse_name(p.name) or {}
                if info.get("tag"):
                    continue
                algo = info.get("algo", p.stem)
                it = QListWidgetItem(f"{p.name}   (peaks · {algo})")
                it.setData(Qt.UserRole, {"source": "peaks", "algorithm": algo})
                self.sources.addItem(it)

        if self.sources.count() == 0:
            it = QListWidgetItem(
                f"(no verified/peaks label files for {bs} — label bins or run peaks first)")
            it.setFlags(Qt.NoItemFlags)
            self.sources.addItem(it)
        else:
            self.sources.setCurrentRow(0)

    def _pick_config(self):
        start = str(self.cvevolve_dir) if self.cvevolve_dir.exists() else self.project_root
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CVEvolve config.yaml", start, "YAML (*.yaml *.yml)")
        if path:
            self.config.setText(path)

    def _build(self):
        item = self.sources.currentItem()
        meta = item.data(Qt.UserRole) if item else None
        if not meta:
            self.count_lbl.setText("select a source file first")
            return
        args = ["build-holdout", "--root", self.project_root,
                "--source", meta["source"],
                "--bin-size", str(self.bin_size),
                "--holdout-pct", str(self.holdout.value()),
                "--seed", str(self.seed.value())]
        if self.scan:
            args += ["--scan", str(self.scan)]
        if meta["algorithm"]:
            args += ["--algorithm", meta["algorithm"]]
        self.count_lbl.setText("building…")
        self.console.run(args)

    def _run(self):
        cfg = self.config.text().strip()
        if not cfg:
            self.console._append("[pick a CVEvolve config.yaml first]\n")
            return
        args = ["run-cvevolve", "--root", self.project_root,
                "--config", cfg, "--engine", self.engine.currentText()]
        config_dir = Path(cfg).resolve().parent
        prompt = config_dir / "prompt.md"
        if prompt.exists():
            args += ["--prompt", str(prompt)]
        holdout_prompt = config_dir / "holdout_test_prompt.md"
        if holdout_prompt.exists():
            args += ["--holdout-test-prompt", str(holdout_prompt)]
        if self.hutch.isChecked():
            args += ["--hutch"]
            from ..core.cvevolve_setup import default_hutch_db
            self.hutch_view.set_db(default_hutch_db(cfg))
            self.hutch_view.show()
            self.split.setSizes([self.width() // 2, self.width() // 2])
        else:
            self.hutch_view.stop()
            self.hutch_view.hide()
        key = self.argo_key.text().strip()
        env = {"ARGO_API_KEY": key} if key else None
        self.console.run(args, env=env, on_finished=self._on_run_finished)

    def _on_run_finished(self, code):
        if code != 0:
            return
        cfg = self.config.text().strip()
        args = ["register-cvevolve", "--root", self.project_root,
                "--config", cfg, "--bin-size", str(self.bin_size)]
        self.console.run(
            args, header="[CVEvolve completed; validating and registering the winner]",
            clear=False)

    def closeEvent(self, event):  # noqa: N802 (Qt signature)
        self.hutch_view.stop()
        super().closeEvent(event)
