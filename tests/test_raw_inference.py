"""Tests for raw inference mode — verifying raw/chat flag propagation."""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. RemoteWorker.trace() — raw mode sends prompt + raw: true
# ---------------------------------------------------------------------------

def test_trace_raw_mode_sends_prompt():
    """When raw=True, trace sends {"prompt": ..., "raw": true}."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "prompt": "The capital of Japan is",
            "num_layers": 22,
            "final_token": "Tokyo",
            "final_prob": 0.85,
            "layers": [],
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        result = worker.trace("The capital of Japan is", raw=True)

        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        assert body["prompt"] == "The capital of Japan is"
        assert body["raw"] is True
        assert "messages" not in body
        assert result["final_token"] == "Tokyo"


# ---------------------------------------------------------------------------
# 2. RemoteWorker.trace() — chat mode sends messages
# ---------------------------------------------------------------------------

def test_trace_chat_mode_sends_messages():
    """When raw=False, trace sends {"messages": [...]}."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "prompt": "<formatted>",
            "num_layers": 22,
            "final_token": "The",
            "final_prob": 0.3,
            "layers": [],
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        worker.trace("The capital of Japan is", raw=False)

        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        assert "messages" in body
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "The capital of Japan is"
        assert "prompt" not in body
        assert "raw" not in body


# ---------------------------------------------------------------------------
# 3. RemoteWorker.trace() — default is raw=True
# ---------------------------------------------------------------------------

def test_trace_defaults_to_raw():
    """trace() defaults to raw=True."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "prompt": "test",
            "num_layers": 1,
            "final_token": "x",
            "final_prob": 0.5,
            "layers": [],
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        worker.trace("test prompt")

        body = mock_client.post.call_args[1]["json"]
        assert body["prompt"] == "test prompt"
        assert body["raw"] is True


# ---------------------------------------------------------------------------
# 4. batch_ablate_stream — raw mode
# ---------------------------------------------------------------------------

def test_batch_ablate_raw_mode():
    """batch_ablate_stream sends raw prompt by default."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = []
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        list(worker.batch_ablate_stream("test", num_layers=2))

        body = mock_client.stream.call_args[1]["json"]
        assert body["prompt"] == "test"
        assert body["raw"] is True
        assert "messages" not in body


# ---------------------------------------------------------------------------
# 5. batch_ablate_stream — chat mode
# ---------------------------------------------------------------------------

def test_batch_ablate_chat_mode():
    """batch_ablate_stream with raw=False sends messages."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = []
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        list(worker.batch_ablate_stream("test", num_layers=2, raw=False))

        body = mock_client.stream.call_args[1]["json"]
        assert "messages" in body
        assert "prompt" not in body


# ---------------------------------------------------------------------------
# 6. decompose_stream — raw mode
# ---------------------------------------------------------------------------

def test_decompose_raw_mode():
    """decompose_stream sends raw prompt by default."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = []
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        list(worker.decompose_stream("test", ["Tokyo", "Paris"]))

        body = mock_client.stream.call_args[1]["json"]
        assert body["prompt"] == "test"
        assert body["raw"] is True
        assert body["tokens"] == ["Tokyo", "Paris"]


# ---------------------------------------------------------------------------
# 7. repair_stream — raw mode
# ---------------------------------------------------------------------------

def test_repair_raw_mode():
    """repair_stream sends raw prompt by default."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = []
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        list(worker.repair_stream("test prompt", "Tokyo"))

        body = mock_client.stream.call_args[1]["json"]
        assert body["prompt"] == "test prompt"
        assert body["raw"] is True
        assert body["answer"] == "Tokyo"


# ---------------------------------------------------------------------------
# 8. repair_stream — chat mode
# ---------------------------------------------------------------------------

def test_repair_chat_mode():
    """repair_stream with raw=False sends messages."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = []
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        list(worker.repair_stream("test prompt", "Tokyo", raw=False))

        body = mock_client.stream.call_args[1]["json"]
        assert "messages" in body
        assert "prompt" not in body
        assert body["answer"] == "Tokyo"


# ---------------------------------------------------------------------------
# 9. CLI --raw/--chat mutual exclusion
# ---------------------------------------------------------------------------

def test_cli_raw_chat_mutual_exclusion():
    """CLI raises error when both --raw and --chat are passed."""
    from click.testing import CliRunner

    from neurotrace.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "trace", "--raw", "--chat",
        "--model", "test", "--prompt", "test", "--db", "test.db",
    ])
    assert result.exit_code != 0
    assert "Cannot use both --raw and --chat" in result.output
