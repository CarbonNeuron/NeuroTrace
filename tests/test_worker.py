"""Tests for worker CLI commands and remote methods — all mocked, no server required."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from neurotrace.cli import cli

# ---------------------------------------------------------------------------
# 1. RemoteWorker.worker_version()
# ---------------------------------------------------------------------------


def test_remote_worker_version_parse():
    """Mock httpx.Client.get and verify worker_version() parses JSON."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "commit": "a1b2c3d",
            "branch": "main",
            "dirty": False,
            "timestamp": "2026-03-20T16:30:00Z",
            "python": "3.12.1",
            "torch": "2.1.0",
            "device": "privateuseone:1",
            "device_name": "AMD Radeon RX 7800 XT",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "uptime_seconds": 3600,
        }
        mock_client.get.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        info = worker.worker_version()

        assert info["commit"] == "a1b2c3d"
        assert info["branch"] == "main"
        assert info["dirty"] is False
        assert info["uptime_seconds"] == 3600
        assert info["device_name"] == "AMD Radeon RX 7800 XT"
        mock_client.get.assert_called_once_with("http://localhost:8877/version")
        mock_response.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# 2. RemoteWorker.worker_update_stream()
# ---------------------------------------------------------------------------


def test_remote_worker_update_stream_parse():
    """Mock SSE stream for worker_update_stream()."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = [
            'data: {"status": "pulling",'
            ' "message": "Pulling latest from origin/main..."}',
            'data: {"status": "pulled",'
            ' "message": "Already up to date."}',
            'data: {"old_commit": "a1b2c3d",'
            ' "new_commit": "a1b2c3d", "changed": false}',
        ]
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        events = list(worker.worker_update_stream())

        assert len(events) == 3
        assert events[0]["status"] == "pulling"
        assert events[1]["status"] == "pulled"
        assert events[2]["changed"] is False
        assert events[2]["old_commit"] == "a1b2c3d"

        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:8877/update"


# ---------------------------------------------------------------------------
# 3. CLI: worker version
# ---------------------------------------------------------------------------


def test_worker_version_cli():
    """Test the worker version CLI command with mocked remote."""
    runner = CliRunner()

    with patch("neurotrace.remote.WorkerClient") as mock_cls:
        mock_worker = mock_cls.return_value
        mock_worker.worker_version.return_value = {
            "commit": "a1b2c3d",
            "branch": "main",
            "dirty": False,
            "python": "3.12.1",
            "torch": "2.1.0",
            "device": "privateuseone:1",
            "device_name": "AMD Radeon RX 7800 XT",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "uptime_seconds": 7260,
        }

        result = runner.invoke(cli, ["worker", "version", "--remote", "http://localhost:8877"])

        assert result.exit_code == 0
        assert "AMD Radeon RX 7800 XT" in result.output
        assert "a1b2c3d" in result.output
        assert "main" in result.output
        assert "TinyLlama" in result.output
        assert "2h 1m" in result.output
        mock_cls.assert_called_once_with("http://localhost:8877")


def test_worker_version_cli_dirty():
    """Test dirty flag display."""
    runner = CliRunner()

    with patch("neurotrace.remote.WorkerClient") as mock_cls:
        mock_worker = mock_cls.return_value
        mock_worker.worker_version.return_value = {
            "commit": "xyz",
            "branch": "dev",
            "dirty": True,
            "python": "3.12.0",
            "torch": "2.0.0",
            "device": "cpu",
            "device_name": "CPU",
            "model": "test-model",
            "uptime_seconds": 60,
        }

        result = runner.invoke(cli, ["worker", "version", "--remote", "http://localhost:8877"])

        assert result.exit_code == 0
        assert "[dirty]" in result.output


# ---------------------------------------------------------------------------
# 4. CLI: worker update
# ---------------------------------------------------------------------------


def test_worker_update_cli_no_change():
    """Test update CLI when no changes detected."""
    runner = CliRunner()

    with patch("neurotrace.remote.WorkerClient") as mock_cls:
        mock_worker = mock_cls.return_value
        mock_worker.worker_update_stream.return_value = iter([
            {"status": "pulling", "message": "Pulling latest from origin/main..."},
            {"status": "pulled", "message": "Already up to date."},
            {"old_commit": "abc", "new_commit": "abc", "changed": False},
        ])

        result = runner.invoke(cli, ["worker", "update", "--remote", "http://localhost:8877"])

        assert result.exit_code == 0
        assert "Already up to date" in result.output


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
