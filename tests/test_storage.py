import numpy as np

from neurotrace.storage import TraceDB
from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)


def _make_sample_trace() -> TraceResult:
    """Create a minimal but complete trace for testing."""
    meta = TraceMetadata(
        trace_id="test-uuid-1234",
        model_name="test-model",
        model_revision="abc123",
        prompt="hello world",
        token_ids=[1, 2],
        tokens=["hello", " world"],
        num_layers=2,
        num_heads=2,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label="test-label",
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snapshots = []
    for i in range(2):
        snapshots.append(
            LayerSnapshot(
                layer_index=i,
                residual_in=np.array(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32
                ),
                residual_out=np.array(
                    [[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1]], dtype=np.float32
                ),
                attention_weights=np.random.randn(2, 2, 2).astype(np.float32),
                attention_output=np.random.randn(2, 4).astype(np.float32),
                mlp_in=np.random.randn(2, 4).astype(np.float32),
                mlp_out=np.random.randn(2, 4).astype(np.float32),
                ln_values=np.random.randn(2, 4).astype(np.float32),
                residual_in_norm=7.746,
                residual_out_norm=7.946,
                attention_entropy=[1.5, 1.2],
                mlp_activation_mag=3.0,
                top1_token=42,
                top1_prob=0.5,
            )
        )
    predictions = [
        TokenPrediction(
            position=0,
            top_k_tokens=[42, 10],
            top_k_probs=[0.5, 0.3],
            top_k_strings=["the", "a"],
        ),
        TokenPrediction(
            position=1,
            top_k_tokens=[7, 8],
            top_k_probs=[0.4, 0.35],
            top_k_strings=["is", "was"],
        ),
    ]
    return TraceResult(
        metadata=meta,
        layer_snapshots=snapshots,
        token_predictions=predictions,
        final_logits=np.random.randn(2, 100).astype(np.float32),
    )


def test_write_and_read_trace(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()

    db = TraceDB(db_path)
    db.write_trace(trace)

    loaded = db.read_trace("test-uuid-1234")
    assert loaded.metadata.trace_id == trace.metadata.trace_id
    assert loaded.metadata.model_name == trace.metadata.model_name
    assert loaded.metadata.label == "test-label"
    assert loaded.metadata.prompt == trace.metadata.prompt
    db.close()


def test_tensor_roundtrip_fidelity(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()

    db = TraceDB(db_path)
    db.write_trace(trace)
    loaded = db.read_trace("test-uuid-1234")

    for orig, loaded_snap in zip(trace.layer_snapshots, loaded.layer_snapshots):
        np.testing.assert_array_equal(orig.residual_in, loaded_snap.residual_in)
        np.testing.assert_array_equal(orig.residual_out, loaded_snap.residual_out)
        np.testing.assert_array_equal(
            orig.attention_weights, loaded_snap.attention_weights
        )
    db.close()


def test_stats_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()

    db = TraceDB(db_path)
    db.write_trace(trace)
    loaded = db.read_trace("test-uuid-1234")

    for orig, loaded_snap in zip(trace.layer_snapshots, loaded.layer_snapshots):
        assert abs(orig.residual_in_norm - loaded_snap.residual_in_norm) < 0.01
        assert orig.top1_token == loaded_snap.top1_token
        assert len(orig.attention_entropy) == len(loaded_snap.attention_entropy)
    db.close()


def test_list_traces(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    db.write_trace(_make_sample_trace())

    traces = db.list_traces()
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "test-uuid-1234"
    assert traces[0]["label"] == "test-label"
    db.close()


def test_attention_maps_per_head(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()
    db = TraceDB(db_path)
    db.write_trace(trace)

    # Query single head
    head_map = db.get_attention_map("test-uuid-1234", layer_index=0, head_index=0)
    assert head_map.shape == (2, 2)  # [seq, seq]
    db.close()


def test_light_mode_null_blobs(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()
    # Simulate light mode
    for snap in trace.layer_snapshots:
        snap.mlp_in = None
        snap.mlp_out = None
    trace.metadata.capture_mode = "light"

    db = TraceDB(db_path)
    db.write_trace(trace)
    loaded = db.read_trace("test-uuid-1234")

    for snap in loaded.layer_snapshots:
        assert snap.mlp_in is None
        assert snap.mlp_out is None
        # Stats should still be present
        assert snap.mlp_activation_mag is not None
    db.close()
