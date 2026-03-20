import numpy as np
from neurotrace.types import TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult


def test_trace_metadata_construction():
    meta = TraceMetadata(
        trace_id="test-uuid",
        model_name="test-model",
        model_revision="abc123",
        prompt="hello world",
        token_ids=[1, 2, 3],
        tokens=["hello", " world", ""],
        num_layers=12,
        num_heads=8,
        hidden_size=512,
        param_count=1_000_000,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label=None,
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    assert meta.model_name == "test-model"
    assert meta.label is None
    assert meta.capture_mode == "full"


def test_layer_snapshot_construction():
    seq_len, hidden, heads = 5, 512, 8
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.random.randn(seq_len, hidden).astype(np.float32),
        residual_out=np.random.randn(seq_len, hidden).astype(np.float32),
        attention_weights=np.random.randn(heads, seq_len, seq_len).astype(np.float32),
        attention_output=np.random.randn(seq_len, hidden).astype(np.float32),
        mlp_in=np.random.randn(seq_len, hidden).astype(np.float32),
        mlp_out=np.random.randn(seq_len, hidden).astype(np.float32),
        ln_values=np.random.randn(seq_len, hidden).astype(np.float32),
        residual_in_norm=1.0,
        residual_out_norm=1.1,
        attention_entropy=[0.5] * heads,
        mlp_activation_mag=0.8,
        top1_token=42,
        top1_prob=0.3,
    )
    assert snap.layer_index == 0
    assert snap.attention_weights.shape == (heads, seq_len, seq_len)


def test_layer_snapshot_light_mode():
    seq_len, hidden, heads = 5, 512, 8
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.random.randn(seq_len, hidden).astype(np.float32),
        residual_out=np.random.randn(seq_len, hidden).astype(np.float32),
        attention_weights=np.random.randn(heads, seq_len, seq_len).astype(np.float32),
        attention_output=np.random.randn(seq_len, hidden).astype(np.float32),
        mlp_in=None,
        mlp_out=None,
        ln_values=None,
        residual_in_norm=1.0,
        residual_out_norm=1.1,
        attention_entropy=[0.5] * heads,
        mlp_activation_mag=0.8,
        top1_token=42,
        top1_prob=0.3,
    )
    assert snap.mlp_in is None
    assert snap.mlp_out is None


def test_token_prediction_construction():
    pred = TokenPrediction(
        position=0,
        top_k_tokens=[42, 10, 5],
        top_k_probs=[0.5, 0.3, 0.2],
        top_k_strings=["the", "a", "an"],
    )
    assert pred.position == 0
    assert len(pred.top_k_tokens) == 3


def test_trace_result_construction():
    meta = TraceMetadata(
        trace_id="test-uuid",
        model_name="test-model",
        model_revision="abc123",
        prompt="hello",
        token_ids=[1],
        tokens=["hello"],
        num_layers=1,
        num_heads=1,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label="test",
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.zeros((1, 4), dtype=np.float32),
        residual_out=np.zeros((1, 4), dtype=np.float32),
        attention_weights=np.zeros((1, 1, 1), dtype=np.float32),
        attention_output=np.zeros((1, 4), dtype=np.float32),
        mlp_in=np.zeros((1, 4), dtype=np.float32),
        mlp_out=np.zeros((1, 4), dtype=np.float32),
        ln_values=np.zeros((1, 4), dtype=np.float32),
        residual_in_norm=0.0,
        residual_out_norm=0.0,
        attention_entropy=[0.0],
        mlp_activation_mag=0.0,
        top1_token=1,
        top1_prob=1.0,
    )
    pred = TokenPrediction(position=0, top_k_tokens=[1], top_k_probs=[1.0], top_k_strings=["hello"])
    result = TraceResult(
        metadata=meta,
        layer_snapshots=[snap],
        token_predictions=[pred],
        final_logits=np.zeros((1, 100), dtype=np.float32),
    )
    assert len(result.layer_snapshots) == 1
    assert result.final_logits.shape == (1, 100)
