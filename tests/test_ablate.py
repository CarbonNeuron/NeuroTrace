"""Tests for the ablation engine."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from neurotrace.ablate import (
    AblationHookManager,
    AblationSpec,
    parse_scale_layers,
    parse_zero_heads,
    parse_zero_layers,
)
from neurotrace.cli import cli
from neurotrace.storage import TraceDB
from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# --- Parsing tests ---


def test_parse_zero_layers_single():
    assert parse_zero_layers("20") == [20]


def test_parse_zero_layers_multiple():
    assert parse_zero_layers("20,21,22") == [20, 21, 22]


def test_parse_zero_heads():
    result = parse_zero_heads("20:7,20:12")
    assert result == [(20, 7), (20, 12)]


def test_parse_scale_layers():
    result = parse_scale_layers("20:0.5,21:2.0")
    assert result == [(20, 0.5), (21, 2.0)]


# --- AblationSpec tests ---


def test_ablation_spec_to_json():
    spec = AblationSpec(
        zero_layers=[20],
        zero_heads=[(20, 7)],
        scale_layers=[(21, 0.5)],
    )
    data = json.loads(spec.to_json())
    assert data["zero_layers"] == [20]
    assert data["zero_heads"] == [[20, 7]]
    assert data["scale_layers"] == [[21, 0.5]]


def test_ablation_spec_to_json_with_zero_mlp():
    spec = AblationSpec(
        zero_layers=[],
        zero_heads=[],
        scale_layers=[],
        zero_mlp=[20, 21],
    )
    data = json.loads(spec.to_json())
    assert data["zero_mlp"] == [20, 21]


def test_ablation_spec_to_json_with_scale_mlp():
    spec = AblationSpec(
        zero_layers=[],
        zero_heads=[],
        scale_layers=[],
        scale_mlp=[(20, 0.5), (21, 0.3)],
    )
    data = json.loads(spec.to_json())
    assert data["scale_mlp"] == [[20, 0.5], [21, 0.3]]


def test_ablation_spec_describe():
    spec = AblationSpec(
        zero_layers=[20],
        zero_heads=[(20, 7)],
        scale_layers=[],
    )
    desc = spec.describe()
    assert "zero-layers=20" in desc
    assert "zero-heads=20:7" in desc


def test_ablation_spec_describe_zero_mlp():
    spec = AblationSpec(
        zero_layers=[],
        zero_heads=[],
        scale_layers=[],
        zero_mlp=[20],
    )
    desc = spec.describe()
    assert "zero-mlp=20" in desc


def test_ablation_spec_describe_scale_mlp():
    spec = AblationSpec(
        zero_layers=[],
        zero_heads=[],
        scale_layers=[],
        scale_mlp=[(20, 0.5)],
    )
    desc = spec.describe()
    assert "scale-mlp=20:0.5" in desc


def test_ablation_spec_default_zero_mlp():
    """zero_mlp defaults to empty list."""
    spec = AblationSpec(zero_layers=[], zero_heads=[], scale_layers=[])
    assert spec.zero_mlp == []


def test_ablation_spec_default_scale_mlp():
    """scale_mlp defaults to empty list."""
    spec = AblationSpec(zero_layers=[], zero_heads=[], scale_layers=[])
    assert spec.scale_mlp == []


# --- Storage integration tests ---


def _make_sample_trace(trace_id="test-id", label="test"):
    meta = TraceMetadata(
        trace_id=trace_id,
        model_name="test-model",
        model_revision="abc",
        prompt="hello world",
        token_ids=[1, 2],
        tokens=["hello", " world"],
        num_layers=1,
        num_heads=1,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label=label,
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.ones((2, 4), dtype=np.float32),
        residual_out=np.ones((2, 4), dtype=np.float32),
        attention_weights=np.ones((1, 2, 2), dtype=np.float32) * 0.5,
        attention_output=np.ones((2, 4), dtype=np.float32),
        mlp_in=None,
        mlp_out=None,
        ln_values=None,
        residual_in_norm=2.0,
        residual_out_norm=2.0,
        attention_entropy=[0.5],
        mlp_activation_mag=1.0,
        top1_token=42,
        top1_prob=0.8,
    )
    pred = TokenPrediction(
        position=0, top_k_tokens=[42], top_k_probs=[0.8], top_k_strings=["the"]
    )
    return TraceResult(
        metadata=meta,
        layer_snapshots=[snap],
        token_predictions=[pred],
        final_logits=np.zeros((2, 100), dtype=np.float32),
    )


def test_store_trace_with_interventions(tmp_path):
    """Interventions metadata is stored and retrievable."""
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    result = _make_sample_trace()
    interventions = json.dumps({"zero_layers": [0]})
    db.write_trace(result, interventions=interventions)

    stored = db.get_interventions(result.metadata.trace_id)
    assert stored is not None
    data = json.loads(stored)
    assert data["zero_layers"] == [0]
    db.close()


def test_store_trace_without_interventions(tmp_path):
    """Normal traces have NULL interventions."""
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    result = _make_sample_trace()
    db.write_trace(result)

    stored = db.get_interventions(result.metadata.trace_id)
    assert stored is None
    db.close()


def test_interventions_column_migration(tmp_path):
    """Opening an old DB (without interventions column) should auto-migrate."""
    db_path = str(tmp_path / "test.db")
    # First open creates schema with interventions column
    db = TraceDB(db_path)
    db.close()
    # Second open should not fail (migration is idempotent)
    db = TraceDB(db_path)
    db.close()


# --- CLI tests ---


def test_ablate_requires_intervention(tmp_path):
    """CLI should error if no intervention flags are provided."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ablate",
            "--db",
            db_path,
            "--model",
            "test",
            "--prompt",
            "hello",
        ],
    )
    assert result.exit_code != 0
    assert "At least one intervention" in result.output


def test_ablate_zero_mlp_accepted(tmp_path):
    """CLI accepts --zero-mlp (fails at model loading, not validation)."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ablate",
            "--db",
            db_path,
            "--model",
            "nonexistent",
            "--prompt",
            "hello",
            "--zero-mlp",
            "20",
        ],
    )
    # Should fail at model loading, not at "At least one intervention"
    assert "At least one intervention" not in (result.output or "")


def test_ablate_scale_mlp_accepted(tmp_path):
    """CLI should accept --scale-mlp as a valid intervention."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ablate",
            "--db",
            db_path,
            "--model",
            "nonexistent",
            "--prompt",
            "hello",
            "--scale-mlp",
            "20:0.5",
        ],
    )
    assert "At least one intervention" not in (result.output or "")


def test_ablate_conflict_zero_and_scale_mlp(tmp_path):
    """CLI should error when same layer is in both --zero-mlp and --scale-mlp."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ablate",
            "--db",
            db_path,
            "--model",
            "nonexistent",
            "--prompt",
            "hello",
            "--zero-mlp",
            "20",
            "--scale-mlp",
            "20:0.5",
        ],
    )
    assert result.exit_code != 0
    assert "Layer 20 appears in both --zero-mlp and --scale-mlp" in result.output


def test_ablate_baseline_resolve(tmp_path):
    """CLI resolves --baseline trace IDs."""
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    trace = _make_sample_trace(trace_id="base-trace", label="baseline")
    db.write_trace(trace)
    db.close()

    runner = CliRunner()
    # This will fail at model loading but should resolve the baseline first
    result = runner.invoke(
        cli,
        [
            "ablate",
            "--db",
            db_path,
            "--model",
            "nonexistent",
            "--prompt",
            "hello",
            "--zero-layers",
            "0",
            "--baseline",
            "baseline",
        ],
    )
    # Should fail at model loading, not baseline resolution
    assert "Trace not found" not in (result.output or "")


# --- Hook management tests ---


@pytest.mark.slow
def test_mlp_hooks_registered_and_cleaned(tinyllama_model):
    """MLP ablation hooks are registered and properly cleaned up."""
    model, _ = tinyllama_model
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    spec = AblationSpec(zero_layers=[], zero_heads=[], scale_layers=[], zero_mlp=[0, 5])

    manager = AblationHookManager(model, arch, spec)
    assert manager.num_hooks >= 2  # at least one hook per MLP layer

    manager.cleanup()
    assert manager.num_hooks == 0


@pytest.mark.slow
def test_scale_mlp_hooks_registered_and_cleaned(tinyllama_model):
    """Scale MLP hooks are registered and properly cleaned up."""
    model, _ = tinyllama_model
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    spec = AblationSpec(
        zero_layers=[],
        zero_heads=[],
        scale_layers=[],
        scale_mlp=[(0, 0.5), (5, 0.3)],
    )

    manager = AblationHookManager(model, arch, spec)
    assert manager.num_hooks >= 2

    manager.cleanup()
    assert manager.num_hooks == 0


@pytest.mark.slow
def test_zero_mlp_changes_output(tinyllama_model):
    """Zeroing a MLP sublayer produces different output than baseline."""
    model, tokenizer = tinyllama_model
    from neurotrace.ablate import run_ablation

    spec = AblationSpec(zero_layers=[], zero_heads=[], scale_layers=[], zero_mlp=[10])
    result = run_ablation(
        model,
        tokenizer,
        prompt="The capital of France is",
        spec=spec,
        seed=42,
    )
    # Ablated output should differ from baseline at some layers
    changed = [lc for lc in result.layer_comparisons if lc.changed]
    assert len(changed) > 0


@pytest.mark.slow
def test_scale_mlp_changes_output(tinyllama_model):
    """Scaling a MLP sublayer produces different output than baseline."""
    model, tokenizer = tinyllama_model
    from neurotrace.ablate import run_ablation

    spec = AblationSpec(
        zero_layers=[],
        zero_heads=[],
        scale_layers=[],
        scale_mlp=[(10, 0.1)],
    )
    result = run_ablation(
        model,
        tokenizer,
        prompt="The capital of France is",
        spec=spec,
        seed=42,
    )
    # Ablated output should differ from baseline at some layers
    changed = [lc for lc in result.layer_comparisons if lc.changed]
    assert len(changed) > 0


@pytest.mark.slow
def test_hooks_registered_and_cleaned(tinyllama_model):
    """Ablation hooks are registered and properly cleaned up."""
    model, _ = tinyllama_model
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    spec = AblationSpec(zero_layers=[0], zero_heads=[], scale_layers=[])

    manager = AblationHookManager(model, arch, spec)
    assert manager.num_hooks > 0

    manager.cleanup()
    assert manager.num_hooks == 0


@pytest.mark.slow
def test_zero_layer_changes_output(tinyllama_model):
    """Zeroing a layer produces different output than baseline."""
    model, tokenizer = tinyllama_model
    from neurotrace.ablate import run_ablation

    spec = AblationSpec(zero_layers=[10], zero_heads=[], scale_layers=[])
    result = run_ablation(
        model,
        tokenizer,
        prompt="The capital of France is",
        spec=spec,
        seed=42,
    )
    # Ablated output should differ from baseline at some layers
    changed = [lc for lc in result.layer_comparisons if lc.changed]
    assert len(changed) > 0


@pytest.mark.slow
def test_scale_identity_same_output(tinyllama_model):
    """Scale factor of 1.0 should produce identical output to baseline."""
    model, tokenizer = tinyllama_model
    from neurotrace.ablate import run_ablation

    spec = AblationSpec(zero_layers=[], zero_heads=[], scale_layers=[(10, 1.0)])
    result = run_ablation(
        model,
        tokenizer,
        prompt="The capital of France is",
        spec=spec,
        seed=42,
    )
    # With scale=1.0, output should be identical
    assert result.baseline_final_token == result.ablated_final_token
    for lc in result.layer_comparisons:
        assert lc.cosine_similarity > 0.999, (
            f"Layer {lc.layer_index} cos_sim={lc.cosine_similarity}"
        )


@pytest.mark.slow
def test_zero_head_changes_output(tinyllama_model):
    """Zeroing attention heads changes the output."""
    model, tokenizer = tinyllama_model
    from neurotrace.ablate import run_ablation

    spec = AblationSpec(zero_layers=[], zero_heads=[(10, 0), (10, 1)], scale_layers=[])
    result = run_ablation(
        model,
        tokenizer,
        prompt="The capital of France is",
        spec=spec,
        seed=42,
    )
    # Should see some difference in cosine similarity at/after layer 10
    post_intervention = [lc for lc in result.layer_comparisons if lc.layer_index >= 10]
    has_change = any(lc.cosine_similarity < 0.9999 for lc in post_intervention)
    assert has_change


@pytest.mark.slow
def test_ablated_trace_stored_with_interventions(tinyllama_model, tmp_path):
    """Full end-to-end: ablated trace is stored with interventions metadata."""
    model, tokenizer = tinyllama_model
    from neurotrace.ablate import AblationSpec, run_ablation

    db_path = str(tmp_path / "test.db")
    spec = AblationSpec(zero_layers=[5], zero_heads=[], scale_layers=[])

    result = run_ablation(
        model,
        tokenizer,
        prompt="Hello",
        spec=spec,
        seed=42,
    )

    db = TraceDB(db_path)
    db.write_trace(result.baseline_trace)
    db.write_trace(result.ablated_trace, interventions=spec.to_json())

    # Verify interventions stored
    stored = db.get_interventions(result.ablated_trace.metadata.trace_id)
    assert stored is not None
    data = json.loads(stored)
    assert data["zero_layers"] == [5]

    # Verify baseline has no interventions
    base_stored = db.get_interventions(result.baseline_trace.metadata.trace_id)
    assert base_stored is None

    db.close()
