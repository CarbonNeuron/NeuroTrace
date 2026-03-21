"""Tests for WorkerClient v2 inference primitives — all mocked, no server required."""

import base64
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. WorkerClient.forward() — basic forward pass
# ---------------------------------------------------------------------------


def test_forward_basic():
    """Test forward() parses top_tokens and metadata correctly."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "top_tokens": [
                {"token": "Tokyo", "token_id": 23456, "logit": 14.2, "prob": 0.82},
                {"token": "Osaka", "token_id": 34567, "logit": 10.1, "prob": 0.05},
            ],
            "residuals": None,
            "num_layers": 36,
            "vocab_size": 151936,
            "hidden_dim": 2048,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.forward("The capital of Japan is")

        assert result.num_layers == 36
        assert result.vocab_size == 151936
        assert result.hidden_dim == 2048
        assert len(result.top_tokens) == 2
        assert result.top_tokens[0].token == "Tokyo"
        assert result.top_tokens[0].token_id == 23456
        assert result.top_tokens[0].logit == pytest.approx(14.2)
        assert result.top_tokens[0].prob == pytest.approx(0.82)
        assert result.residuals is None

        # Verify the request
        call_args = mock_client.post.call_args
        assert "/inference/forward" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["input"] == "The capital of Japan is"
        assert body["raw"] is True


def test_forward_with_residuals():
    """Test forward() with residual layers decodes base64 float16 arrays."""
    # Create a dummy residual: 4 float16 values
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")

    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "top_tokens": [
                {"token": "Tokyo", "token_id": 23456, "logit": 14.2, "prob": 0.82},
            ],
            "residuals": {"0": b64, "12": b64},
            "num_layers": 36,
            "vocab_size": 151936,
            "hidden_dim": 4,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.forward(
            "The capital of Japan is",
            residual_layers=[0, 12],
        )

        assert result.residuals is not None
        assert 0 in result.residuals
        assert 12 in result.residuals
        np.testing.assert_allclose(
            result.residuals[0], arr, rtol=1e-3,
        )

        # Verify residuals flag in request
        body = mock_client.post.call_args[1]["json"]
        assert body["outputs"]["residuals"] is True
        assert body["outputs"]["residual_layers"] == [0, 12]


def test_forward_with_layer_predictions():
    """Test forward() with layer_predictions parses per-layer top-k."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "top_tokens": [
                {"token": "Berlin", "token_id": 100, "logit": 12.0, "prob": 0.7},
            ],
            "residuals": None,
            "num_layers": 3,
            "vocab_size": 32000,
            "hidden_dim": 256,
            "layer_predictions": [
                {"layer": 0, "top_tokens": [
                    {"token": "the", "token_id": 1, "logit": 5.0, "prob": 0.12},
                ]},
                {"layer": 1, "top_tokens": [
                    {"token": "a", "token_id": 2, "logit": 4.0, "prob": 0.09},
                ]},
                {"layer": 2, "top_tokens": [
                    {"token": "Berlin", "token_id": 100, "logit": 12.0, "prob": 0.44},
                ]},
            ],
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.forward(
            "The capital of Germany is",
            layer_predictions=True,
            layer_predictions_top_k=5,
        )

        assert result.layer_predictions is not None
        assert len(result.layer_predictions) == 3
        assert result.layer_predictions[0].layer == 0
        assert result.layer_predictions[0].top_tokens[0].token == "the"
        assert result.layer_predictions[2].top_tokens[0].prob == pytest.approx(0.44)

        body = mock_client.post.call_args[1]["json"]
        assert body["outputs"]["layer_predictions"] is True
        assert body["outputs"]["layer_predictions_top_k"] == 5


def test_rome_edit():
    """Test rome_edit() sends correct request and parses response."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "edit_id": 1,
            "stack_size": 1,
            "pre_prob": 0.19,
            "post_prob": 0.67,
            "pre_margin": -1.94,
            "post_margin": 3.10,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.rome_edit(
            "The capital of Germany is",
            "Germany",
            "Berlin",
            20,
        )

        assert result.success is True
        assert result.edit_id == 1
        assert result.pre_prob == pytest.approx(0.19)
        assert result.post_prob == pytest.approx(0.67)
        assert result.pre_margin == pytest.approx(-1.94)
        assert result.post_margin == pytest.approx(3.10)

        body = mock_client.post.call_args[1]["json"]
        assert body["input"] == "The capital of Germany is"
        assert body["subject"] == "Germany"
        assert body["target"] == "Berlin"
        assert body["layer"] == 20


def test_decompose():
    """Test decompose() sends correct request and parses response."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer_token_id": 23456,
            "total_logit": 14.45,
            "layers": [
                {
                    "layer": 0, "attn_logit": 0.12,
                    "mlp_logit": -0.03, "cumulative": 0.09,
                },
                {
                    "layer": 1, "attn_logit": 0.50,
                    "mlp_logit": 0.20, "cumulative": 0.79,
                },
            ],
            "competitors": [
                {"token": "located", "total_logit": 8.33, "margin": -6.12},
            ],
            "reconstruction_error": 0.0001,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.decompose(
            "The capital of Germany is", "Berlin",
        )

        assert result.answer_token_id == 23456
        assert result.total_logit == pytest.approx(14.45)
        assert len(result.layers) == 2
        assert result.layers[0].attn_logit == pytest.approx(0.12)
        assert len(result.competitors) == 1
        assert result.competitors[0].token == "located"
        assert result.reconstruction_error == pytest.approx(0.0001)

        body = mock_client.post.call_args[1]["json"]
        assert body["answer"] == "Berlin"


def test_attention():
    """Test attention() sends correct request and parses response."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "heads": [
                {"layer": 0, "head": 5, "logit_contribution": 0.42},
                {"layer": 1, "head": 3, "logit_contribution": -0.18},
            ],
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.attention(
            "The capital of Germany is", "Berlin",
        )

        assert len(result.heads) == 2
        assert result.heads[0].layer == 0
        assert result.heads[0].head == 5
        assert result.heads[0].logit_contribution == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# 2. WorkerClient.hooked() — hooked forward pass
# ---------------------------------------------------------------------------


def test_hooked_with_zero_and_capture():
    """Test hooked() sends hook specs and parses captured activations."""
    arr = np.array([0.5, 1.5, 2.5], dtype=np.float16)
    b64 = base64.b64encode(arr.tobytes()).decode("ascii")

    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "top_tokens": [
                {"token": "Berlin", "token_id": 12345, "logit": 10.0, "prob": 0.7},
            ],
            "captured": {"20.attn": b64},
            "num_layers": 36,
            "vocab_size": 151936,
            "hidden_dim": 3,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import Hook, WorkerClient

        client = WorkerClient("http://localhost:8877")
        hooks = [
            Hook(layer=34, component="mlp", action="zero"),
            Hook(layer=20, component="attn", action="capture"),
        ]
        result = client.hooked("The capital of Germany is", hooks)

        assert result.top_tokens[0].token == "Berlin"
        assert "20.attn" in result.captured
        np.testing.assert_allclose(result.captured["20.attn"], arr, rtol=1e-3)

        # Verify hook specs in request
        body = mock_client.post.call_args[1]["json"]
        assert len(body["hooks"]) == 2
        assert body["hooks"][0]["action"] == "zero"
        assert body["hooks"][1]["action"] == "capture"


# ---------------------------------------------------------------------------
# 3. WorkerClient.generate() — token generation
# ---------------------------------------------------------------------------


def test_generate_basic():
    """Test generate() parses generated text and token list."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Tokyo",
            "tokens": [
                {"token": "Tokyo", "token_id": 23456, "logit": 14.2, "prob": 0.82},
            ],
            "num_tokens": 1,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        result = client.generate("The capital of Japan is")

        assert result.text == "Tokyo"
        assert result.num_tokens == 1
        assert result.tokens[0].token == "Tokyo"

        body = mock_client.post.call_args[1]["json"]
        assert body["max_tokens"] == 20
        assert body["temperature"] == 0.0


# ---------------------------------------------------------------------------
# 4. WorkerClient.edit() — weight modification
# ---------------------------------------------------------------------------


def test_edit_basic():
    """Test edit() sends encoded vectors and parses result."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "edit_id": 1,
            "stack_size": 1,
        }
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        key = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        val = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = client.edit(20, "mlp.down_proj", key, val)

        assert result.success is True
        assert result.edit_id == 1
        assert result.stack_size == 1

        body = mock_client.post.call_args[1]["json"]
        assert body["layer"] == 20
        assert body["component"] == "mlp.down_proj"
        assert body["action"] == "rome_rank_one"
        # Verify vectors are base64 encoded
        decoded_key = np.frombuffer(
            base64.b64decode(body["key_vector"]), dtype=np.float16,
        )
        np.testing.assert_allclose(decoded_key, key, atol=0.01)


def test_edit_undo():
    """Test edit_undo() calls the correct endpoint."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "edits_remaining": 0}
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        client.edit_undo()

        mock_client.post.assert_called_once_with(
            "http://localhost:8877/model/edit/undo",
        )


def test_edit_clear():
    """Test edit_clear() calls the correct endpoint."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "edits_remaining": 0}
        mock_client.post.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        client.edit_clear()

        mock_client.post.assert_called_once_with(
            "http://localhost:8877/model/edit/clear",
        )


def test_edit_stack():
    """Test edit_stack() parses the stack info."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "stack_size": 2,
            "edits": [
                {"edit_id": 1, "layer": 20, "component": "mlp.down_proj"},
                {"edit_id": 2, "layer": 15, "component": "mlp.down_proj"},
            ],
        }
        mock_client.get.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")
        stack = client.edit_stack()

        assert len(stack) == 2
        assert stack[0].edit_id == 1
        assert stack[0].layer == 20
        assert stack[1].component == "mlp.down_proj"


# ---------------------------------------------------------------------------
# 5. WorkerClient.batch_forward() — SSE batch
# ---------------------------------------------------------------------------


def test_batch_forward_basic():
    """Test batch_forward() processes SSE events and returns results."""
    with patch("httpx.Client") as mock_cls:
        mock_client = mock_cls.return_value

        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        r1 = (
            '{"index": 0, "top_tokens": [{"token": "Tokyo",'
            ' "token_id": 1, "logit": 10.0, "prob": 0.8}],'
            ' "num_layers": 22, "vocab_size": 32000,'
            ' "hidden_dim": 2048, "residuals": null}'
        )
        r2 = (
            '{"index": 1, "top_tokens": [{"token": "Paris",'
            ' "token_id": 2, "logit": 9.0, "prob": 0.7}],'
            ' "num_layers": 22, "vocab_size": 32000,'
            ' "hidden_dim": 2048, "residuals": null}'
        )
        mock_response.iter_lines.return_value = [
            "event: progress",
            'data: {"completed": 0, "total": 2,'
            ' "current": "p1", "elapsed_s": 0.1}',
            "event: result",
            f"data: {r1}",
            "event: result",
            f"data: {r2}",
            "event: done",
            'data: {"total": 2, "elapsed_s": 1.5}',
        ]
        mock_client.stream.return_value = mock_response

        from neurotrace.remote import WorkerClient

        client = WorkerClient("http://localhost:8877")

        progress_calls = []
        results = client.batch_forward(
            ["prompt1", "prompt2"],
            on_progress=lambda c, t: progress_calls.append((c, t)),
        )

        assert len(results) == 2
        assert results[0].top_tokens[0].token == "Tokyo"
        assert results[1].top_tokens[0].token == "Paris"
        assert len(progress_calls) >= 1


# ---------------------------------------------------------------------------
# 6. Backward compatibility — RemoteWorker alias
# ---------------------------------------------------------------------------


def test_remote_worker_alias():
    """Verify RemoteWorker is an alias for WorkerClient."""
    from neurotrace.remote import RemoteWorker, WorkerClient

    assert RemoteWorker is WorkerClient


# ---------------------------------------------------------------------------
# 7. Dataclass exports
# ---------------------------------------------------------------------------


def test_dataclass_imports():
    """Verify all v2 dataclasses are importable."""
    from neurotrace.remote import (
        AttentionResult,
        ContrastResult,
        DecomposeResult,
        FingerprintResult,
        HeadContribution,
        Hook,
        LayerPrediction,
        RomeEditResult,
        TokenPrediction,
    )

    # Verify they're constructible
    tp = TokenPrediction(token="test", token_id=1, logit=1.0, prob=0.5)
    assert tp.token == "test"

    hook = Hook(layer=0, component="mlp", action="zero")
    assert hook.scale is None
    assert hook.tensor is None

    # Phase 2 types
    lp = LayerPrediction(layer=0, top_tokens=[tp])
    assert lp.layer == 0
    assert len(lp.top_tokens) == 1

    re = RomeEditResult(
        success=True, edit_id=1, stack_size=1,
        pre_prob=0.1, post_prob=0.9,
        pre_margin=-1.0, post_margin=2.0,
    )
    assert re.success
    assert re.post_prob == 0.9

    hc = HeadContribution(layer=5, head=3, logit_contribution=0.5)
    ar = AttentionResult(heads=[hc])
    assert len(ar.heads) == 1

    assert DecomposeResult is not None
    assert FingerprintResult is not None
    assert ContrastResult is not None
