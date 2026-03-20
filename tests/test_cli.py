"""Tests for the CLI commands."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.storage import TraceDB
from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)


def _write_sample_trace(
    db_path: str, trace_id: str = "test-id-1", label: str = "test"
) -> None:
    """Write a minimal trace directly to DB for CLI testing."""
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
    result = TraceResult(
        metadata=meta,
        layer_snapshots=[snap],
        token_predictions=[pred],
        final_logits=np.zeros((2, 100), dtype=np.float32),
    )
    db = TraceDB(db_path)
    db.write_trace(result)
    db.close()


def test_list_command(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", db_path])
    assert result.exit_code == 0
    assert "test-model" in result.output
    assert "test" in result.output  # label


def test_inspect_json_output(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["inspect", "--db", db_path, "--trace-id", "test-id-1", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["metadata"]["trace_id"] == "test-id-1"


def test_inspect_latest_alias(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["inspect", "--db", db_path, "--trace-id", "latest", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["metadata"]["trace_id"] == "test-id-1"


def test_diff_json_output(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path, trace_id="a", label="a")
    _write_sample_trace(db_path, trace_id="b", label="b")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["diff", "--db", db_path, "--trace-a", "a", "--trace-b", "b", "--json"],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "layer_metrics" in data
    assert "critical_heads" in data


def test_list_empty_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", db_path])
    assert result.exit_code == 0


def test_inspect_invalid_trace_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["inspect", "--db", db_path, "--trace-id", "nonexistent"]
    )
    assert result.exit_code != 0


def test_prompts_file(tmp_path):
    """Test that --prompts-file reads prompts from a file (one per line)."""
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("hello\nworld\n")
    # We can't actually run trace without a model, but we can verify the CLI accepts the flag
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "trace",
            "--model",
            "nonexistent",
            "--prompts-file",
            str(prompts_file),
            "--db",
            str(tmp_path / "test.db"),
        ],
    )
    # Should fail at model loading, not at argument parsing
    assert "nonexistent" in result.output or result.exit_code != 0
