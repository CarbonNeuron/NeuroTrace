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

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


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
    # Can't run trace without a model, but verify CLI accepts the flag
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


def test_predict_no_residuals(tmp_path):
    """predict should error gracefully when residuals aren't stored."""
    db_path = str(tmp_path / "test.db")
    # Write a trace with residual_out=None
    meta = TraceMetadata(
        trace_id="no-res",
        model_name="test-model",
        model_revision="abc",
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
        label="no-res",
        capture_mode="light",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=None,
        residual_out=None,
        attention_weights=None,
        attention_output=None,
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
    result = TraceResult(
        metadata=meta,
        layer_snapshots=[snap],
        token_predictions=[],
        final_logits=np.zeros((1, 100), dtype=np.float32),
    )
    db = TraceDB(db_path)
    db.write_trace(result)
    db.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["predict", "--db", db_path, "--trace-id", "no-res"])
    assert result.exit_code != 0
    assert "Residuals not stored" in result.output


@pytest.mark.model_download
def test_predict_basic(tinyllama_model, tmp_path):
    """predict shows per-layer top-K predictions from a stored trace."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    # First create a trace
    runner.invoke(
        cli,
        [
            "trace",
            "--model",
            TINYLLAMA,
            "--prompt",
            "The capital of France is",
            "--db",
            db_path,
            "--label",
            "predict-test",
        ],
    )
    # Run predict
    result = runner.invoke(
        cli,
        ["predict", "--db", db_path, "--trace-id", "predict-test", "--top-k", "3"],
    )
    assert result.exit_code == 0
    assert "Layer" in result.output


@pytest.mark.model_download
def test_predict_changes_only(tinyllama_model, tmp_path):
    """--changes-only filters to layers where top-1 changed."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "trace",
            "--model",
            TINYLLAMA,
            "--prompt",
            "The capital of France is",
            "--db",
            db_path,
            "--label",
            "changes-test",
        ],
    )
    result = runner.invoke(
        cli,
        [
            "predict",
            "--db",
            db_path,
            "--trace-id",
            "changes-test",
            "--changes-only",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.model_download
def test_predict_track_token(tinyllama_model, tmp_path):
    """--track shows a specific token's rank at every layer."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "trace",
            "--model",
            TINYLLAMA,
            "--prompt",
            "The capital of France is",
            "--db",
            db_path,
            "--label",
            "track-test",
        ],
    )
    result = runner.invoke(
        cli,
        [
            "predict",
            "--db",
            db_path,
            "--trace-id",
            "track-test",
            "--track",
            "Paris",
        ],
    )
    assert result.exit_code == 0
    assert "rank" in result.output


@pytest.mark.model_download
def test_predict_json_output(tinyllama_model, tmp_path):
    """--json produces valid JSON output."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "trace",
            "--model",
            TINYLLAMA,
            "--prompt",
            "Hello",
            "--db",
            db_path,
            "--label",
            "json-pred",
        ],
    )
    result = runner.invoke(
        cli,
        [
            "predict",
            "--db",
            db_path,
            "--trace-id",
            "json-pred",
            "--json",
            "--top-k",
            "3",
        ],
    )
    assert result.exit_code == 0
    # Extract JSON array from output (progress bars may precede it)
    output = result.output
    # Find the JSON array start — look for newline + [
    idx = output.rfind("\n[")
    if idx >= 0:
        output = output[idx + 1 :]
    else:
        output = output[output.index("[") :]
    data = json.loads(output)
    assert isinstance(data, list)
    assert len(data) > 0
    assert "layer_index" in data[0]
    assert "top_k_ids" in data[0]
    assert "top_k_strings" in data[0]
    assert "annotations" in data[0]


@pytest.mark.model_download
def test_predict_layers_filter(tinyllama_model, tmp_path):
    """--layers filters to specific layer indices."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "trace",
            "--model",
            TINYLLAMA,
            "--prompt",
            "Hello",
            "--db",
            db_path,
            "--label",
            "layers-pred",
        ],
    )
    result = runner.invoke(
        cli,
        [
            "predict",
            "--db",
            db_path,
            "--trace-id",
            "layers-pred",
            "--json",
            "--layers",
            "0,5,10",
        ],
    )
    assert result.exit_code == 0
    # Extract JSON array from output (progress bars may precede it)
    output = result.output
    idx = output.rfind("\n[")
    if idx >= 0:
        output = output[idx + 1 :]
    else:
        output = output[output.index("[") :]
    data = json.loads(output)
    layer_indices = {e["layer_index"] for e in data}
    assert layer_indices == {0, 5, 10}


@pytest.mark.model_download
def test_decode_specific_tokens(tinyllama_model):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["decode", "--model", TINYLLAMA, "--tokens", "3681", "--tokens", "593"],
    )
    assert result.exit_code == 0
    assert "3681" in result.output
    assert "593" in result.output


@pytest.mark.model_download
def test_decode_from_trace(tinyllama_model, tmp_path):
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    # First create a trace
    runner.invoke(
        cli,
        [
            "trace",
            "--model",
            TINYLLAMA,
            "--prompt",
            "Hello",
            "--db",
            db_path,
        ],
    )
    # Now decode from it
    result = runner.invoke(
        cli,
        [
            "decode",
            "--model",
            TINYLLAMA,
            "--from-trace",
            "latest",
            "--db",
            db_path,
        ],
    )
    assert result.exit_code == 0
    assert "Token Decode" in result.output


@pytest.mark.model_download
def test_compare_end_to_end(tinyllama_model, tmp_path):
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compare",
            "--model",
            TINYLLAMA,
            "--prompt-a",
            "The capital of France is",
            "--prompt-b",
            "The capital of Germany is",
            "--db",
            db_path,
        ],
    )
    assert result.exit_code == 0
    assert "Compare:" in result.output
    assert "Token Legend" in result.output
    assert "Flagged layers:" in result.output


@pytest.mark.model_download
def test_compare_reuses_existing_traces(tinyllama_model, tmp_path):
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    # Run compare once
    runner.invoke(
        cli,
        [
            "compare",
            "--model",
            TINYLLAMA,
            "--prompt-a",
            "Hello world",
            "--prompt-b",
            "Goodbye world",
            "--db",
            db_path,
        ],
    )
    # Run same compare again — should reuse
    result = runner.invoke(
        cli,
        [
            "compare",
            "--model",
            TINYLLAMA,
            "--prompt-a",
            "Hello world",
            "--prompt-b",
            "Goodbye world",
            "--db",
            db_path,
        ],
    )
    assert result.exit_code == 0
    assert "Reusing existing trace" in result.output

    # Verify only 2 traces in DB (not 4)
    db_conn = TraceDB(db_path)
    traces = db_conn.list_traces()
    db_conn.close()
    assert len(traces) == 2


@pytest.mark.model_download
def test_compare_json_output(tinyllama_model, tmp_path):
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compare",
            "--model",
            TINYLLAMA,
            "--prompt-a",
            "Hello",
            "--prompt-b",
            "Goodbye",
            "--db",
            db_path,
            "--json",
        ],
    )
    assert result.exit_code == 0
    # Extract JSON from output (tqdm progress bars may precede it)
    output = result.output[result.output.index("{") :]
    data = json.loads(output)
    assert "token_legend" in data
    assert "layer_metrics" in data
    assert "trace_a_top1_str" in data["layer_metrics"][0]


# --- Sweep command tests ---


def test_sweep_requires_sweep_flag(tmp_path):
    """Sweep command should error without a --sweep-* flag."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sweep",
            "--db",
            db_path,
            "--model",
            "test",
            "--prompt",
            "hello",
        ],
    )
    assert result.exit_code != 0
    assert "Exactly one --sweep-* flag" in result.output


def test_sweep_rejects_multiple_sweep_flags(tmp_path):
    """Sweep command should error with multiple --sweep-* flags."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sweep",
            "--db",
            db_path,
            "--model",
            "test",
            "--prompt",
            "hello",
            "--sweep-scale-mlp",
            "20:0.1:0.9:0.1",
            "--sweep-zero-mlp",
            "18,19,20",
        ],
    )
    assert result.exit_code != 0
    assert "Only one --sweep-* flag" in result.output


@pytest.mark.model_download
def test_sweep_scale_mlp(tinyllama_model, tmp_path):
    """Sweep scale-mlp runs multiple ablations and produces summary."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sweep",
            "--db",
            db_path,
            "--model",
            TINYLLAMA,
            "--prompt",
            "The capital of France is",
            "--sweep-scale-mlp",
            "10:0.3:0.7:0.2",
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0
    assert "Sweep" in result.output


@pytest.mark.model_download
def test_sweep_json_output(tinyllama_model, tmp_path):
    """Sweep command --json produces valid JSON."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sweep",
            "--db",
            db_path,
            "--model",
            TINYLLAMA,
            "--prompt",
            "The capital of France is",
            "--sweep-scale-mlp",
            "10:0.3:0.7:0.2",
            "--seed",
            "42",
            "--json",
        ],
    )
    assert result.exit_code == 0
    output = result.output[result.output.index("{") :]
    data = json.loads(output)
    assert "sweep_description" in data
    assert "results" in data
    assert len(data["results"]) == 3  # 0.3, 0.5, 0.7


@pytest.mark.model_download
def test_sweep_zero_heads(tinyllama_model, tmp_path):
    """Sweep zero-heads runs one ablation per head."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "sweep",
            "--db",
            db_path,
            "--model",
            TINYLLAMA,
            "--prompt",
            "Hello",
            "--sweep-zero-heads",
            "10:0-2",
            "--seed",
            "42",
            "--json",
        ],
    )
    assert result.exit_code == 0
    output = result.output[result.output.index("{") :]
    data = json.loads(output)
    assert len(data["results"]) == 3  # heads 0, 1, 2
