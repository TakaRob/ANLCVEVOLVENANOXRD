import pytest
from click.testing import CliRunner

from xrd_app.cli import grid, main, peaks


@pytest.mark.parametrize("command", sorted(main.commands))
def test_every_command_help_succeeds(command):
    result = CliRunner().invoke(main, [command, "--help"])

    assert result.exit_code == 0, result.output


def test_help_lists_workflow_and_defaults():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])
    grid_help = runner.invoke(main, ["grid", "--help"])

    assert result.exit_code == 0
    assert "make-bins" in result.output
    assert "run-pipeline" in result.output
    assert grid_help.exit_code == 0
    assert "[default: 3]" in grid_help.output
    defaults = {param.name: param.default for param in grid.params}
    assert defaults["deskew_method"] == "auto"
    peak_params = {param.name: param for param in peaks.params}
    assert peak_params["workers"].default is None
    assert "--workers" in runner.invoke(main, ["peaks", "--help"]).output


def test_init_requires_name_without_prompting(tmp_path):
    result = CliRunner().invoke(main, ["init", "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "Missing option '--name'" in result.stderr


def test_init_refuses_to_overwrite_existing_project(tmp_path):
    runner = CliRunner()

    first = runner.invoke(
        main, ["init", "--name", "test-project", "--root", str(tmp_path)]
    )
    original = (tmp_path / "config.yaml").read_text()
    second = runner.invoke(
        main, ["init", "--name", "replacement", "--root", str(tmp_path)]
    )

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "refusing to overwrite" in second.stderr
    assert (tmp_path / "config.yaml").read_text() == original


def test_scan_detect_usage_errors_go_to_stderr():
    result = CliRunner().invoke(main, ["scan-detect"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Provide --scans-dir" in result.stderr


def test_scan_detect_rejects_multiple_sources():
    result = CliRunner().invoke(
        main, ["scan-detect", "--scans-dir", "scans", "--scan-file", "scan.h5"]
    )

    assert result.exit_code == 2
    assert "Use only one" in result.stderr


def test_lineage_missing_explicit_target_fails(tmp_path):
    result = CliRunner().invoke(
        main,
        ["lineage", "missing.json", "--root", str(tmp_path), "--scan", "203"],
    )

    assert result.exit_code == 1
    assert "Result JSON not found" in result.stderr
