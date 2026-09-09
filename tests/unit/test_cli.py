"""
Unit tests for saaga CLI commands.
"""
from click.testing import CliRunner
from saaga.cli.main import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "SAAGA" in result.output
    assert "run" in result.output
    assert "bench" in result.output
    assert "download-models" in result.output
    assert "serve" in result.output


def test_cli_download_models_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["download-models", "--list-available"])
    assert result.exit_code == 0
    assert "victim" in result.output
    assert "embedding" in result.output
    assert "translation" in result.output


def test_cli_run_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    assert "--victim-provider" in result.output
    assert "--victim-url" in result.output
    assert "--access-code" in result.output
