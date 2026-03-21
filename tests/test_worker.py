"""Tests for worker CLI commands — retained tests only."""

from unittest.mock import patch

from click.testing import CliRunner

from neurotrace.cli import cli


def test_worker_update_cli_with_change():
    """Test update CLI when changes are pulled."""
    runner = CliRunner()

    with patch("neurotrace.remote.WorkerClient") as mock_cls:
        mock_worker = mock_cls.return_value
        mock_worker.worker_update_stream.return_value = iter([
            {"status": "pulling", "message": "Pulling latest from origin/main..."},
            {"status": "pulled", "message": "Updating abc..def"},
            {"status": "restarting", "message": "Restarting worker in 2 seconds..."},
            {"old_commit": "abc", "new_commit": "def", "changed": True},
        ])

        result = runner.invoke(cli, ["worker", "update", "--remote", "http://localhost:8877"])

        assert result.exit_code == 0
        assert "Updated: abc -> def" in result.output


# ---------------------------------------------------------------------------
# 5. CLI: worker group exists
# ---------------------------------------------------------------------------


def test_worker_group_help():
    """Test worker group has help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["worker", "--help"])
    assert result.exit_code == 0
    assert "Manage the GPU worker" in result.output
    assert "version" in result.output
    assert "update" in result.output
