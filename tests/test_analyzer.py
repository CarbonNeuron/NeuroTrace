import numpy as np

from neurotrace.analyzer import DiffResult, compute_diff
from neurotrace.types import LayerSnapshot, TokenPrediction, TraceMetadata, TraceResult


def _make_trace(trace_id: str, diverge_at: int | None = None) -> TraceResult:
    """Create a synthetic trace.

    If diverge_at is set, layer >= diverge_at gets different values.
    """
    num_layers = 12
    seq_len = 5
    hidden = 64
    heads = 4
    np.random.seed(0)  # deterministic base

    snapshots = []
    for i in range(num_layers):
        base_residual = np.random.randn(seq_len, hidden).astype(np.float32)
        if diverge_at is not None and i >= diverge_at:
            # Inject divergence: rotate residual stream
            noise = np.random.randn(seq_len, hidden).astype(np.float32) * 2.0
            residual_out = base_residual + noise
            top1 = 99  # different prediction
            top1_prob = 0.4
        else:
            residual_out = (
                base_residual
                + np.random.randn(seq_len, hidden).astype(np.float32) * 0.01
            )
            top1 = 42
            top1_prob = 0.8

        # Create attention weights (valid probability distributions)
        attn = np.random.dirichlet(np.ones(seq_len), size=(heads, seq_len)).astype(
            np.float32
        )

        snapshots.append(
            LayerSnapshot(
                layer_index=i,
                residual_in=base_residual,
                residual_out=residual_out,
                attention_weights=attn,
                attention_output=np.random.randn(seq_len, hidden).astype(np.float32),
                mlp_in=None,
                mlp_out=None,
                ln_values=None,
                residual_in_norm=float(np.linalg.norm(base_residual)),
                residual_out_norm=float(np.linalg.norm(residual_out)),
                attention_entropy=[1.0] * heads,
                mlp_activation_mag=1.0,
                top1_token=top1,
                top1_prob=top1_prob,
            )
        )

    meta = TraceMetadata(
        trace_id=trace_id,
        model_name="test",
        model_revision="abc",
        prompt="test",
        token_ids=[1] * seq_len,
        tokens=["t"] * seq_len,
        num_layers=num_layers,
        num_heads=heads,
        hidden_size=hidden,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=0,
        label=None,
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-01-01T00:00:00",
    )

    preds = [
        TokenPrediction(
            position=p,
            top_k_tokens=[top1, 10, 5],
            top_k_probs=[0.5, 0.3, 0.2],
            top_k_strings=["a", "b", "c"],
        )
        for p, top1 in enumerate([42] * seq_len)
    ]

    return TraceResult(
        metadata=meta,
        layer_snapshots=snapshots,
        token_predictions=preds,
        final_logits=np.random.randn(seq_len, 100).astype(np.float32),
    )


def test_identical_traces_no_divergence():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b")  # same seed, same data
    diff = compute_diff(trace_a, trace_b)
    assert isinstance(diff, DiffResult)
    # All layers should have high cosine similarity
    for layer in diff.layer_metrics:
        assert layer.cosine_similarity > 0.99
        assert layer.top1_changed is False
        assert layer.kl_divergence < 0.01
        assert layer.flagged is False


def test_divergence_detected_at_correct_layer():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b", diverge_at=8)
    diff = compute_diff(trace_a, trace_b)

    # Layers 0-7 should be similar
    for layer in diff.layer_metrics[:8]:
        assert layer.cosine_similarity > 0.95
        assert layer.flagged is False

    # Layers 8+ should diverge
    for layer in diff.layer_metrics[8:]:
        assert layer.cosine_similarity < 0.95
        assert layer.top1_changed is True
        assert layer.flagged is True

    assert diff.first_divergence_layer == 8


def test_critical_heads_identified():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b", diverge_at=8)
    diff = compute_diff(trace_a, trace_b)

    # Should identify some critical heads
    assert len(diff.critical_heads) > 0
    # Each critical head is (layer, head, js_divergence)
    for layer_idx, head_idx, js_div in diff.critical_heads:
        assert 0 <= layer_idx < 12
        assert 0 <= head_idx < 4
        assert js_div >= 0


def test_configurable_thresholds():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b", diverge_at=8)

    # With very strict thresholds, more layers flagged
    diff_strict = compute_diff(
        trace_a,
        trace_b,
        cosine_threshold=0.999,
        kl_threshold=0.01,
    )
    # With very loose thresholds, fewer layers flagged
    diff_loose = compute_diff(
        trace_a,
        trace_b,
        cosine_threshold=0.5,
        kl_threshold=10.0,
    )

    strict_flagged = sum(1 for m in diff_strict.layer_metrics if m.flagged)
    loose_flagged = sum(1 for m in diff_loose.layer_metrics if m.flagged)
    assert strict_flagged >= loose_flagged
