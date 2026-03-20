# tests/test_tracer.py
import pytest

from neurotrace.tracer import Tracer
from neurotrace.types import TraceResult

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_trace_returns_complete_result(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("The capital of France is")

    assert isinstance(result, TraceResult)
    assert result.metadata.model_name == TINYLLAMA
    assert len(result.layer_snapshots) == model.config.num_hidden_layers
    assert len(result.token_predictions) > 0
    assert result.final_logits is not None


@pytest.mark.slow
def test_trace_tensor_shapes(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("Hello")

    seq_len = len(result.metadata.token_ids)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    for snap in result.layer_snapshots:
        assert snap.residual_in.shape == (seq_len, hidden_size)
        assert snap.residual_out.shape == (seq_len, hidden_size)
        assert snap.attention_weights.shape == (num_heads, seq_len, seq_len)
        assert len(snap.attention_entropy) == num_heads


@pytest.mark.slow
def test_trace_light_mode(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer, capture_mode="light")
    result = tracer.trace("Hello")

    for snap in result.layer_snapshots:
        assert snap.mlp_in is None
        assert snap.mlp_out is None
        # Stats still present
        assert snap.mlp_activation_mag is not None
        assert snap.residual_in_norm > 0


@pytest.mark.slow
def test_trace_layer_stride(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer, layer_stride=4)
    result = tracer.trace("Hello")

    for snap in result.layer_snapshots:
        if snap.layer_index % 4 == 0:
            assert snap.residual_in is not None
        else:
            assert snap.residual_in is None
        # Stats always present regardless of stride
        assert snap.residual_in_norm > 0


@pytest.mark.slow
def test_trace_top_k_predictions(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("The capital of France is")

    for pred in result.token_predictions:
        assert len(pred.top_k_tokens) == 50
        assert len(pred.top_k_probs) == 50
        assert len(pred.top_k_strings) == 50
        assert pred.top_k_probs[0] >= pred.top_k_probs[1]  # sorted descending


@pytest.mark.slow
def test_trace_metadata_completeness(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("Hello", label="test-run", seed=42)

    meta = result.metadata
    assert meta.label == "test-run"
    assert meta.random_seed == 42
    assert meta.device == "cpu"
    assert meta.num_layers == model.config.num_hidden_layers
    assert meta.num_heads == model.config.num_attention_heads
    assert meta.hidden_size == model.config.hidden_size
    assert meta.param_count > 0
    assert meta.model_revision is not None
