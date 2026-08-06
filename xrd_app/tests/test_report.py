from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from xrd_app.cli import main
from xrd_app.config import ProjectConfig, default_config
from xrd_app.core import device_maps
from xrd_app.core.report import ReportOptions, ReportTarget, generate_pdf


def test_device_grid_matches_existing_max_intensity_semantics():
    features = [
        {"reflection": "(001)", "intensity_profile": {
            "0_0": {"integrated": 4.0}, "0_1": {"integrated": 2.0}}},
        {"reflection": "(001)", "intensity_profile": {
            "0_0": {"integrated": 7.0}}},
        {"reflection": "(011)", "intensity_profile": {
            "1_0": {"integrated": 3.0}}},
    ]

    grids = device_maps.build_device_grids(features, 2, 2)

    assert grids["(001)"][0, 0] == 7.0
    assert grids["(001)"][0, 1] == 2.0
    assert np.isnan(grids["(001)"][1, 1])
    assert grids["(011)"][1, 0] == 3.0


def test_report_options_require_explicit_top_five_override():
    with pytest.raises(ValueError, match="capped at five"):
        ReportOptions(top_count=6).validated()

    assert ReportOptions(top_count=6, allow_more_than_five=True).validated().top_count == 6


def test_preview_uses_first_scan_and_continues_missing_data(tmp_path):
    config = ProjectConfig(tmp_path, data=default_config("report-test", tmp_path))
    config.create_tree()
    config.save()
    output = tmp_path / "preview.pdf"
    options = ReportOptions(
        summed_images=True, all_reflections=False, features_by_reflection=False,
        top_features=False)

    result = generate_pdf(
        tmp_path,
        [ReportTarget("Scan_0001", 3), ReportTarget("Scan_0002", 3)],
        output,
        options,
        preview=True,
        log=lambda _message: None,
    )

    assert output.exists() and output.stat().st_size > 0
    assert result["targets"] == 1
    assert result["pages"] == 1
    assert [failure["scan"] for failure in result["failures"]] == ["Scan_0001"]


def test_report_cli_rejects_top_count_over_five_without_override(tmp_path):
    output = tmp_path / "report.pdf"
    result = CliRunner().invoke(main, [
        "report", "--root", str(tmp_path), "--target", "Scan_0001:3",
        "--output", str(output), "--top-count", "6",
    ])

    assert result.exit_code != 0
    assert "capped at five" in result.output
    assert not output.exists()
