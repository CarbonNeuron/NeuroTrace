"""Tests for the RemoteWorker client — all mocked, no server required."""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Health endpoint
# ---------------------------------------------------------------------------

def test_remote_health_parse():
    """Mock httpx.Client.get and verify RemoteWorker.health() parses the JSON."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "device": "cuda:0",
            "device_name": "RTX 4090",
            "num_layers": 22,
        }
        mock_client.get.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        health = worker.health()

        assert health["status"] == "ok"
        assert health["num_layers"] == 22
        assert health["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        mock_client.get.assert_called_once_with("http://localhost:8877/health")
        mock_response.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# 2. batch_ablate_stream builds the correct request body
# ---------------------------------------------------------------------------

def test_remote_batch_ablate_builds_request():
    """Verify the request body contains baseline + one ablation per layer."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        # Set up a streaming context manager that yields no lines
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = []
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        num_layers = 3
        list(worker.batch_ablate_stream(
            "The capital of France is", num_layers=num_layers,
        ))

        mock_client.stream.assert_called_once()
        call_args = mock_client.stream.call_args

        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:8877/batch-ablate"

        body = call_args[1]["json"]
        assert body["prompt"] == "The capital of France is"
        # baseline (no ablation) + one per layer
        assert len(body["ablations"]) == num_layers + 1
        assert body["ablations"][0] == {"zero_mlp_layers": []}
        assert body["ablations"][1] == {"zero_mlp_layers": [0]}
        assert body["ablations"][2] == {"zero_mlp_layers": [1]}
        assert body["ablations"][3] == {"zero_mlp_layers": [2]}


# ---------------------------------------------------------------------------
# 3. batch_ablate_stream parses SSE lines
# ---------------------------------------------------------------------------

def test_remote_batch_ablate_parses_sse_stream():
    """Mock streaming response with SSE lines and verify parsed events."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = [
            'data: {"type":"progress","index":0,'
            '"total":2,"description":"baseline"}',
            'data: {"type":"result","index":0,'
            '"zero_mlp_layers":[],'
            '"final_token":"Paris","final_prob":0.51}',
            'data: {"type":"result","index":1,'
            '"zero_mlp_layers":[0],'
            '"final_token":"London","final_prob":0.30}',
            'data: {"type":"done","total_results":2}',
        ]
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        events = list(worker.batch_ablate_stream("test prompt", num_layers=1))

        assert len(events) == 4
        assert events[0]["type"] == "progress"
        assert events[0]["index"] == 0
        assert events[1]["type"] == "result"
        assert events[1]["final_token"] == "Paris"
        assert events[1]["final_prob"] == pytest.approx(0.51)
        assert events[2]["type"] == "result"
        assert events[2]["zero_mlp_layers"] == [0]
        assert events[3]["type"] == "done"
        assert events[3]["total_results"] == 2


# ---------------------------------------------------------------------------
# 4. finetune_stream parses SSE lines
# ---------------------------------------------------------------------------

def test_remote_finetune_parses_sse_stream():
    """Mock finetune SSE stream and verify parsed events."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_lines.return_value = [
            'data: {"type":"progress","epoch":1,"step":10,"loss":2.15}',
            'data: {"type":"progress","epoch":1,"step":20,"loss":1.80}',
            'data: {"type":"done","adapter_id":"abc123","final_loss":1.42}',
        ]
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        config = {
            "dataset": "math_qa",
            "target_layers": [10, 11],
            "epochs": 1,
            "lr": 2e-4,
        }
        worker = RemoteWorker("http://localhost:8877")
        events = list(worker.finetune_stream(config))

        assert len(events) == 3
        assert events[0]["type"] == "progress"
        assert events[0]["loss"] == pytest.approx(2.15)
        assert events[1]["step"] == 20
        assert events[2]["type"] == "done"
        assert events[2]["adapter_id"] == "abc123"

        # Verify the request was made correctly
        call_args = mock_client.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:8877/finetune"
        assert call_args[1]["json"] == config
        assert call_args[1]["timeout"] == 600.0


# ---------------------------------------------------------------------------
# 5. Connection error propagation
# ---------------------------------------------------------------------------

def test_remote_connection_error():
    """Verify httpx.ConnectError propagates from RemoteWorker.health()."""
    with patch("httpx.Client") as mock_cls:
        import httpx

        mock_client = mock_cls.return_value
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")

        with pytest.raises(httpx.ConnectError, match="Connection refused"):
            worker.health()
