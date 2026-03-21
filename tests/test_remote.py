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

# ---------------------------------------------------------------------------
# 5a. Model config endpoint
# ---------------------------------------------------------------------------

def test_remote_get_model_config():
    """Mock httpx.Client.get and verify get_model_config() parses the JSON."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "num_hidden_layers": 28,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "vocab_size": 128256,
        }
        mock_client.get.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        config = worker.get_model_config()

        assert config["num_hidden_layers"] == 28
        assert config["hidden_size"] == 3072
        mock_client.get.assert_called_once_with(
            "http://localhost:8877/model/config"
        )
        mock_response.raise_for_status.assert_called_once()


# ---------------------------------------------------------------------------
# 5b. Format prompt endpoint
# ---------------------------------------------------------------------------

def test_remote_format_prompt():
    """Mock httpx.Client.post and verify format_prompt() sends messages."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "formatted": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            "num_tokens": 12,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker("http://localhost:8877")
        result = worker.format_prompt([{"role": "user", "content": "Hello"}])

        assert "formatted" in result
        assert result["num_tokens"] == 12
        mock_client.post.assert_called_once_with(
            "http://localhost:8877/format",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )


# ---------------------------------------------------------------------------
# 6. Connection error propagation
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


# ---------------------------------------------------------------------------
# 7. WorkerError on non-200 responses
# ---------------------------------------------------------------------------

def test_worker_error_attributes():
    """WorkerError stores status_code, detail, and endpoint."""
    from neurotrace.remote import WorkerError

    err = WorkerError(422, "Validation failed", "/model/rome-edit")
    assert err.status_code == 422
    assert err.detail == "Validation failed"
    assert err.endpoint == "/model/rome-edit"
    assert "422" in str(err)
    assert "/model/rome-edit" in str(err)


def test_rome_edit_raises_worker_error_on_422():
    """rome_edit() raises WorkerError when the worker returns 422."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "detail": "[{'msg': 'field required', 'type': 'value_error'}]",
        }
        mock_response.text = "validation error"
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient, WorkerError

        worker = WorkerClient("http://localhost:8877")
        with pytest.raises(WorkerError, match="422") as exc_info:
            worker.rome_edit("The capital of France is", "France", "Paris", 14)

        assert exc_info.value.status_code == 422
        assert exc_info.value.endpoint == "/model/rome-edit"


def test_rome_edit_raises_worker_error_on_500():
    """rome_edit() raises WorkerError on 500 with text body."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "Internal Server Error"
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient, WorkerError

        worker = WorkerClient("http://localhost:8877")
        with pytest.raises(WorkerError) as exc_info:
            worker.rome_edit("test", "subj", "target", 5)

        assert exc_info.value.status_code == 500
        assert "Internal Server Error" in exc_info.value.detail


def test_rome_edit_success():
    """rome_edit() returns RomeEditResult on 200."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "edit_id": 1,
            "stack_size": 1,
            "pre_prob": 0.05,
            "post_prob": 0.82,
            "pre_margin": -3.2,
            "post_margin": 1.5,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        worker = WorkerClient("http://localhost:8877")
        result = worker.rome_edit("prompt", "subj", "target", 14)

        assert result.success is True
        assert result.edit_id == 1
        assert result.pre_prob == pytest.approx(0.05)
        assert result.post_prob == pytest.approx(0.82)
