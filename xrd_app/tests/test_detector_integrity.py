import json
import tomllib
from pathlib import Path

import pytest

from xrd_app.config import DataManager, format_detector_label
from xrd_app.core import cvevolve_results, io, save_algorithm
from xrd_app.core.processing import REQUIRED_DETECTOR_API, load_detector


_VALID_DETECTOR = "\n".join(f"def {name}(*args, **kwargs): pass" for name in REQUIRED_DETECTOR_API)


def _write_project_detector(root, name, *, f1, f2, bin_size=None):
    library = root / "Algorithms" / "PeakAlgorithms"
    library.mkdir(parents=True, exist_ok=True)
    (library / f"{name}.py").write_text(_VALID_DETECTOR)
    catalog = library / "catalog.json"
    data = json.loads(catalog.read_text()) if catalog.exists() else {"detectors": []}
    data["detectors"].append({
        "name": name, "file": f"{name}.py", "role": "detector",
        "bin_size": bin_size, "holdout_f1": f1, "holdout_f2": f2,
    })
    catalog.write_text(json.dumps(data))


def test_catalog_offers_only_production_detector_contract(tmp_path):
    dm = DataManager(tmp_path)

    detectors = dm.list_detectors()

    assert detectors
    for entry in detectors:
        module = io.load_module(entry["_path"])
        assert all(callable(getattr(module, name, None)) for name in REQUIRED_DETECTOR_API)


def test_best_detector_and_label_are_f2_first(tmp_path):
    _write_project_detector(tmp_path, "f1_winner", f1=0.99, f2=0.40)
    _write_project_detector(tmp_path, "f2_winner", f1=0.70, f2=0.80)
    dm = DataManager(tmp_path)

    assert dm.best_detector(3).stem == "f2_winner"
    assert format_detector_label({
        "name": "ranked", "holdout_f1": 0.99, "holdout_f2": 0.80,
    }) == "ranked (F2 0.80)"


def test_detector_resolution_accepts_exact_and_generic_bins(tmp_path):
    _write_project_detector(
        tmp_path, "generic", f1=0.60, f2=0.70, bin_size=None)
    _write_project_detector(
        tmp_path, "exact", f1=0.50, f2=0.80, bin_size="3x3")
    _write_project_detector(
        tmp_path, "wrong", f1=0.99, f2=0.99, bin_size="5x5")
    dm = DataManager(tmp_path)

    assert dm.best_detector(3).stem == "exact"
    assert dm.best_detector(4).stem == "generic"
    assert dm.detector_script("exact", bin_size=3).stem == "exact"
    assert dm.detector_script("generic", bin_size=4).stem == "generic"


def test_detector_resolution_rejects_declared_wrong_size(tmp_path):
    _write_project_detector(
        tmp_path, "only_5x5", f1=0.90, f2=0.90, bin_size="5x5")
    dm = DataManager(tmp_path)
    detector = dm.project_algorithms_dir("peak") / "only_5x5.py"

    with pytest.raises(ValueError, match="declares bin_size '5x5'.*3x3"):
        dm.detector_script("only_5x5", bin_size=3)
    with pytest.raises(ValueError, match="declares bin_size '5x5'.*3x3"):
        dm.detector_script(str(detector), bin_size=3)

    dm.config.data.setdefault("data_sources", {})["detector_script"] = str(detector)
    with pytest.raises(ValueError, match="declares bin_size '5x5'.*3x3"):
        dm.detector_script(bin_size=3)


def test_uncataloged_external_detector_is_generic(tmp_path):
    external = tmp_path / "external.py"
    external.write_text(_VALID_DETECTOR)

    assert DataManager(tmp_path).detector_script(
        str(external), bin_size=4) == external


def test_register_cvevolve_winner_is_project_owned_and_discoverable(tmp_path):
    session = tmp_path / "sessions" / "peak_search"
    reports = session / "reports"
    history = session / "history"
    source = session / "workspace" / "candidates" / "winner"
    reports.mkdir(parents=True)
    history.mkdir()
    source.mkdir(parents=True)
    detector = source / "candidate.py"
    detector.write_text(_VALID_DETECTOR)
    (source / "support.py").write_text("VALUE = 1\n")
    (reports / "best_candidate.py").write_text(_VALID_DETECTOR)
    (reports / "final_summary.json").write_text(json.dumps({
        "best_candidate": {
            "candidate_id": "candidate-1", "candidate_name": "Strong detector",
            "code_file_path": str(detector), "primary_metric_value": 0.82,
        },
    }))
    import sqlite3
    with sqlite3.connect(history / "search_history.sqlite") as connection:
        connection.execute(
            "CREATE TABLE holdout_test_metrics "
            "(id INTEGER PRIMARY KEY, candidate_id TEXT, metric_name TEXT, value REAL)")
        connection.execute(
            "INSERT INTO holdout_test_metrics VALUES (1, 'candidate-1', 'mean_f2', 0.77)")
    config = tmp_path / "cvevolve.yaml"
    config.write_text(
        f"name: peak_search\nworkspace:\n  root_dir: {tmp_path / 'sessions'}\n")

    result = cvevolve_results.register_winner(config, tmp_path, bin_size=3)

    output = tmp_path / "Algorithms" / "PeakAlgorithms" / "Strong_detector"
    assert result["path"] == output / "detector.py"
    assert (output / "support.py").is_file()
    entry = DataManager(tmp_path).load_detector_catalog()["detectors"][-1]
    assert entry["holdout_f2"] == 0.77
    assert entry["bin_size"] == "3x3"
    assert DataManager(tmp_path).resolve_detector_name("Strong_detector", 3) == result["path"]


def test_save_algorithm_uses_project_storage_and_is_discoverable(tmp_path):
    out = save_algorithm.save_algorithm(
        "5x5_tophat_band_adaptive_snr", sensitivity=5.0, bin_size=3,
        name="user_detector", project_root=tmp_path,
    )

    assert out == tmp_path / "Algorithms" / "PeakAlgorithms" / "user_detector.py"
    dm = DataManager(tmp_path)
    assert dm.resolve_detector_name("user_detector") == out
    assert (out.parent / "catalog.json").is_file()
    load_detector(out)


def test_save_algorithm_rejects_incompatible_base(tmp_path):
    incompatible = tmp_path / "research.py"
    incompatible.write_text("def detect_peaks(): pass\n")

    with pytest.raises(TypeError, match="missing:"):
        save_algorithm.save_algorithm(
            str(incompatible), sensitivity=5.0, bin_size=3,
            project_root=tmp_path,
        )


def test_dynamic_import_identity_tracks_path_and_content(tmp_path):
    first = tmp_path / "one" / "detector.py"
    second = tmp_path / "two" / "detector.py"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("VALUE = 1\n")
    second.write_text("VALUE = 2\n")

    mod1 = io.load_module(first)
    mod2 = io.load_module(second)
    first.write_text("VALUE = 3\n")
    mod3 = io.load_module(first)

    assert len({mod1.__name__, mod2.__name__, mod3.__name__}) == 3
    assert (mod1.VALUE, mod2.VALUE, mod3.VALUE) == (1, 2, 3)


def test_tracked_notebooks_are_portable_percent_scripts():
    notebooks = sorted(Path("notebooks").glob("[0-9][0-9]_*.py"))

    assert notebooks
    for notebook in notebooks:
        source = notebook.read_text()
        compile(source, str(notebook), "exec")
        assert "# %%" in source
        assert "/home/" not in source
        assert "/mnt/" not in source
        assert "xrd-app gui" not in source
        assert "xrf-app gui" not in source


def test_packaging_excludes_development_trees_and_keeps_runtime_assets():
    config = tomllib.loads(Path("pyproject.toml").read_text())
    discovery = config["tool"]["setuptools"]["packages"]["find"]
    setuptools = config["tool"]["setuptools"]
    package_data = setuptools["package-data"]["xrd_app"]
    excluded_data = setuptools["exclude-package-data"]["*"]

    assert "xrd_app.tests*" in discovery["exclude"]
    assert discovery["namespaces"] is False
    assert setuptools["include-package-data"] is False
    assert "PeakAlgorithms/catalog.json" in package_data
    assert "PeakAlgorithms/*.py" in package_data
    assert "tests/**/*" in excluded_data
    assert "**/*.pyc" in excluded_data
