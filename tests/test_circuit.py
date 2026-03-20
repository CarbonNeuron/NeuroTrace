"""Tests for the circuit module — mechanistic MLP decomposition."""

import json
import os

import numpy as np
import pytest

from neurotrace.circuit import (
    CircuitResult,
    ForwardResult,
    PairwiseDiff,
    PromptComparison,
    RealComparisonResult,
    RealPromptMLP,
    ReverseResult,
    _capture_mlp_io,
    compare_prompt,
    compare_real_mlp,
    detect_layer_from_probe,
    forward_circuit,
    generate_report,
    load_direction,
    reverse_circuit,
    run_circuit,
    save_circuit_outputs,
)

# --- Unit tests (no model download) ---


def test_load_direction(tmp_path):
    """Test loading a direction vector from .npy file."""
    direction = np.random.randn(2048).astype(np.float32)
    direction = direction / np.linalg.norm(direction)
    np.save(tmp_path / "mean_direction.npy", direction)

    loaded = load_direction(str(tmp_path))
    np.testing.assert_array_almost_equal(loaded, direction)


def test_load_direction_missing(tmp_path):
    """Test error when mean_direction.npy is missing."""
    with pytest.raises(FileNotFoundError):
        load_direction(str(tmp_path))


def test_detect_layer_from_probe(tmp_path):
    """Test auto-detection of layer from report.md."""
    report = "# Probe Analysis\n\n## Dataset\n- Layer: 20\n"
    (tmp_path / "report.md").write_text(report)
    assert detect_layer_from_probe(str(tmp_path)) == 20


def test_detect_layer_from_probe_missing(tmp_path):
    """Test None when report.md is missing."""
    assert detect_layer_from_probe(str(tmp_path)) is None


def test_detect_layer_from_probe_no_layer(tmp_path):
    """Test None when report.md has no layer info."""
    (tmp_path / "report.md").write_text("# Probe Analysis\n\nNo layer here.\n")
    assert detect_layer_from_probe(str(tmp_path)) is None


def _make_sample_forward_result():
    """Create a sample ForwardResult for testing."""
    return ForwardResult(
        mlp_output=np.random.randn(2048).astype(np.float32),
        logit_contribution=np.random.randn(32000).astype(np.float32),
        top_boosted=[
            {"token_id": 1, "token": "located", "logit": 12.34},
            {"token_id": 2, "token": "where", "logit": 11.22},
        ],
        top_suppressed=[
            {"token_id": 3, "token": "Berlin", "logit": -8.45},
            {"token_id": 4, "token": "Paris", "logit": -7.33},
        ],
        pre_mlp_logits=np.random.randn(32000).astype(np.float32),
        pre_mlp_top=[
            {"token_id": 5, "token": "Berlin", "logit": 5.67},
        ],
    )


def _make_sample_circuit_result(probe_dir="/tmp/test-probe"):
    """Create a sample CircuitResult for testing."""
    return CircuitResult(
        probe_dir=probe_dir,
        model_name="test-model",
        layer=20,
        forward=_make_sample_forward_result(),
        reverse=[
            ReverseResult(
                token="located",
                token_id=1,
                input_direction=np.random.randn(2048).astype(np.float32),
                cosine_sim_with_probe=0.42,
            ),
        ],
        comparisons=[
            PromptComparison(
                prompt="The capital of Germany is",
                top_boosted=[{"token_id": 1, "token": "located", "logit": 5.0}],
                top_suppressed=[{"token_id": 3, "token": "Berlin", "logit": -3.0}],
                activation_norm=15.2,
                cosine_sim_with_direction=0.65,
            ),
        ],
    )


def test_save_circuit_outputs(tmp_path):
    """Test saving circuit outputs to directory."""
    result = _make_sample_circuit_result()
    output_dir = str(tmp_path / "circuit-out")
    save_circuit_outputs(result, output_dir)

    assert os.path.exists(os.path.join(output_dir, "forward_analysis.json"))
    assert os.path.exists(os.path.join(output_dir, "reverse_analysis.json"))
    assert os.path.exists(os.path.join(output_dir, "prompt_comparison.json"))
    assert os.path.exists(os.path.join(output_dir, "report.md"))

    with open(os.path.join(output_dir, "forward_analysis.json")) as f:
        data = json.load(f)
    assert data["layer"] == 20
    assert len(data["top_boosted_tokens"]) == 2
    assert data["top_boosted_tokens"][0]["token"] == "located"

    with open(os.path.join(output_dir, "reverse_analysis.json")) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["token"] == "located"
    assert abs(data[0]["cosine_sim_with_probe"] - 0.42) < 1e-6


def test_save_circuit_outputs_no_reverse(tmp_path):
    """Test saving without reverse analysis."""
    result = _make_sample_circuit_result()
    result.reverse = []
    result.comparisons = []
    output_dir = str(tmp_path / "circuit-out")
    save_circuit_outputs(result, output_dir)

    assert os.path.exists(os.path.join(output_dir, "forward_analysis.json"))
    assert not os.path.exists(os.path.join(output_dir, "reverse_analysis.json"))
    assert not os.path.exists(os.path.join(output_dir, "prompt_comparison.json"))


def test_generate_report():
    """Test report generation."""
    result = _make_sample_circuit_result()
    report = generate_report(result)

    assert "Circuit Analysis:" in report
    assert "Layer 20 MLP" in report
    assert "located" in report
    assert "Berlin" in report
    assert "BOOSTED" in report
    assert "SUPPRESSED" in report
    assert "Reverse Circuit" in report
    assert "0.4200" in report
    assert "Prompt Comparison" in report
    assert "The capital of Germany is" in report


def test_generate_report_minimal():
    """Test report generation with no reverse/comparisons."""
    result = _make_sample_circuit_result()
    result.reverse = []
    result.comparisons = []
    report = generate_report(result)

    assert "Circuit Analysis:" in report
    assert "Reverse Circuit" not in report
    assert "Prompt Comparison" not in report


def test_db_circuit_roundtrip(tmp_path):
    """Test saving a circuit result to DuckDB."""
    from neurotrace.storage import TraceDB

    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    result = _make_sample_circuit_result()
    db.save_circuit("circuit-001", result)

    row = db._conn.execute(
        "SELECT * FROM circuits WHERE id = ?", ["circuit-001"]
    ).fetchone()
    assert row is not None
    assert row[2] == "test-model"  # model
    assert row[3] == 20  # layer

    # Check JSON fields
    boosted = json.loads(row[4])
    assert len(boosted) == 2
    assert boosted[0]["token"] == "located"

    reverse = json.loads(row[6])
    assert len(reverse) == 1
    assert reverse[0]["token"] == "located"

    db.close()


# --- Model tests (require TinyLlama download) ---


@pytest.mark.model_download
def test_forward_circuit_shapes(tinyllama_model):
    """Test forward circuit produces correct output shapes."""
    model, tokenizer = tinyllama_model
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    direction = np.random.randn(hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    result = forward_circuit(model, tokenizer, direction, layer=10, top_k=10)

    assert result.mlp_output.shape == (hidden_size,)
    assert result.logit_contribution.shape == (vocab_size,)
    assert result.pre_mlp_logits.shape == (vocab_size,)
    assert len(result.top_boosted) == 10
    assert len(result.top_suppressed) == 10
    assert len(result.pre_mlp_top) == 10

    # Verify boosted logits are larger than suppressed
    assert result.top_boosted[0]["logit"] > result.top_suppressed[0]["logit"]


@pytest.mark.model_download
def test_forward_circuit_token_ids(tinyllama_model):
    """Test that forward circuit returns valid token IDs."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    result = forward_circuit(model, tokenizer, direction, layer=15, top_k=5)

    for t in result.top_boosted:
        assert 0 <= t["token_id"] < model.config.vocab_size
        assert isinstance(t["token"], str)
        assert isinstance(t["logit"], float)


@pytest.mark.model_download
def test_reverse_circuit_unit_vector(tinyllama_model):
    """Test reverse trace produces unit vectors."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    results = reverse_circuit(
        model, tokenizer, direction, layer=10,
        tokens=["the", "Berlin"],
    )

    assert len(results) == 2
    for r in results:
        norm = np.linalg.norm(r.input_direction)
        assert abs(norm - 1.0) < 1e-4, f"Expected unit vector, got norm={norm}"
        assert -1.0 <= r.cosine_sim_with_probe <= 1.0


@pytest.mark.model_download
def test_reverse_circuit_cosine_bounds(tinyllama_model):
    """Test reverse cosine similarity is within [-1, 1]."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    results = reverse_circuit(
        model, tokenizer, direction, layer=15,
        tokens=["located", "where", "the"],
    )

    for r in results:
        assert -1.0 - 1e-6 <= r.cosine_sim_with_probe <= 1.0 + 1e-6


@pytest.mark.model_download
def test_compare_prompt_basic(tinyllama_model):
    """Test prompt comparison captures real activations."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    result = compare_prompt(
        model, tokenizer, direction, layer=10,
        prompt="The capital of France is", top_k=5,
    )

    assert result.prompt == "The capital of France is"
    assert result.activation_norm > 0
    assert -1.0 <= result.cosine_sim_with_direction <= 1.0
    assert len(result.top_boosted) == 5
    assert len(result.top_suppressed) == 5


@pytest.mark.model_download
def test_run_circuit_full(tinyllama_model, tmp_path):
    """Test the full circuit analysis pipeline."""
    model, tokenizer = tinyllama_model

    # Create a mock probe directory
    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)
    probe_dir = str(tmp_path / "probe")
    os.makedirs(probe_dir)
    np.save(os.path.join(probe_dir, "mean_direction.npy"), direction)

    result = run_circuit(
        model, tokenizer, probe_dir,
        layer=10, top_k=5,
        reverse_tokens=["the"],
        compare_prompts=["The capital of France is"],
    )

    assert result.layer == 10
    assert len(result.forward.top_boosted) == 5
    assert len(result.reverse) == 1
    assert len(result.comparisons) == 1


@pytest.mark.model_download
def test_circuit_cli_json(tinyllama_model, tmp_path):
    """Test circuit CLI command with JSON output."""
    from click.testing import CliRunner

    from neurotrace.cli import cli

    model, tokenizer = tinyllama_model

    # Create probe dir
    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)
    probe_dir = str(tmp_path / "probe")
    os.makedirs(probe_dir)
    np.save(os.path.join(probe_dir, "mean_direction.npy"), direction)

    db_path = str(tmp_path / "test.db")
    output_dir = str(tmp_path / "circuit-out")

    runner = CliRunner()
    result = runner.invoke(cli, [
        "circuit",
        "--db", db_path,
        "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--probe-dir", probe_dir,
        "--layer", "10",
        "--top-k", "5",
        "--output", output_dir,
        "--json",
    ])

    assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
    data = json.loads(result.output)
    assert "circuit_id" in data
    assert data["layer"] == 10
    assert len(data["top_boosted"]) == 5


# --- Real MLP comparison tests (no model download) ---


def _make_real_comparison():
    """Create a sample RealComparisonResult for testing."""
    return RealComparisonResult(
        prompts=[
            RealPromptMLP(
                prompt="The capital of Germany is",
                mlp_output_norm=12.34,
                sabotage_projection=0.45,
                top_boosted=[{"token_id": 1, "token": "located", "logit": 15.34}],
                top_suppressed=[{"token_id": 3, "token": "Berlin", "logit": -12.45}],
                final_top_tokens=[{"token_id": 10, "token": "Berlin", "logit": 20.0}],
            ),
            RealPromptMLP(
                prompt="The capital of France is",
                mlp_output_norm=11.00,
                sabotage_projection=0.12,
                top_boosted=[{"token_id": 5, "token": "Paris", "logit": 14.00}],
                top_suppressed=[{"token_id": 6, "token": "the", "logit": -10.00}],
                final_top_tokens=[{"token_id": 5, "token": "Paris", "logit": 22.0}],
            ),
        ],
        pairwise=[
            PairwiseDiff(
                prompt_a="The capital of Germany is",
                prompt_b="The capital of France is",
                cosine_similarity=0.87,
                norm_ratio=1.12,
                diff_boosted=[{"token_id": 1, "token": "located", "logit": 8.23}],
                diff_suppressed=[{"token_id": 5, "token": "Paris", "logit": -6.78}],
            ),
        ],
    )


def test_generate_report_with_real_comparison():
    """Test report includes Real MLP Comparison section."""
    result = _make_sample_circuit_result()
    result.real_comparison = _make_real_comparison()
    report = generate_report(result)

    assert "Real MLP Comparison" in report
    assert "Per-Prompt MLP Analysis" in report
    assert "The capital of Germany is" in report
    assert "MLP output norm: 12.34" in report
    assert "Differential Analysis" in report
    assert "cosine similarity: 0.8700" in report


def test_save_real_comparison_json(tmp_path):
    """Test real_comparison.json is written to output dir."""
    result = _make_sample_circuit_result()
    result.real_comparison = _make_real_comparison()
    output_dir = str(tmp_path / "circuit-out")
    save_circuit_outputs(result, output_dir)

    rc_path = os.path.join(output_dir, "real_comparison.json")
    assert os.path.exists(rc_path)

    with open(rc_path) as f:
        data = json.load(f)

    assert len(data["prompts"]) == 2
    assert data["prompts"][0]["text"] == "The capital of Germany is"
    assert data["prompts"][0]["mlp_output_norm"] == 12.34
    assert len(data["pairwise"]) == 1
    assert data["pairwise"][0]["cosine_similarity"] == 0.87


def test_save_no_real_comparison_json(tmp_path):
    """Test real_comparison.json is NOT written when no real comparison."""
    result = _make_sample_circuit_result()
    result.real_comparison = None
    output_dir = str(tmp_path / "circuit-out")
    save_circuit_outputs(result, output_dir)

    assert not os.path.exists(os.path.join(output_dir, "real_comparison.json"))


# --- Real MLP comparison tests (require TinyLlama download) ---


@pytest.mark.model_download
def test_capture_mlp_io_shapes(tinyllama_model):
    """Test _capture_mlp_io returns correct shapes."""
    model, tokenizer = tinyllama_model
    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    result = _capture_mlp_io(model, tokenizer, "The capital of France is", layer=10)

    assert result["mlp_input"].shape == (hidden_size,)
    assert result["mlp_output"].shape == (hidden_size,)
    assert result["final_logits"].shape == (vocab_size,)


@pytest.mark.model_download
def test_compare_real_mlp_two_prompts(tinyllama_model):
    """Test compare_real_mlp with 2 prompts produces pairwise diff."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    result = compare_real_mlp(
        model, tokenizer,
        prompts=["The capital of Germany is", "The capital of France is"],
        layer=10,
        direction=direction,
        top_k=5,
    )

    assert len(result.prompts) == 2
    assert len(result.pairwise) == 1
    assert result.prompts[0].prompt == "The capital of Germany is"
    assert result.prompts[1].prompt == "The capital of France is"

    # Check shapes via token list lengths
    assert len(result.prompts[0].top_boosted) == 5
    assert len(result.prompts[0].top_suppressed) == 5
    assert len(result.prompts[0].final_top_tokens) == 5

    # Check pairwise diff
    pw = result.pairwise[0]
    assert -1.0 <= pw.cosine_similarity <= 1.0
    assert pw.norm_ratio > 0
    assert len(pw.diff_boosted) == 5
    assert len(pw.diff_suppressed) == 5


@pytest.mark.model_download
def test_compare_real_mlp_single_prompt(tinyllama_model):
    """Test compare_real_mlp with 1 prompt has no pairwise diff."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    result = compare_real_mlp(
        model, tokenizer,
        prompts=["The capital of France is"],
        layer=10,
        direction=direction,
        top_k=5,
    )

    assert len(result.prompts) == 1
    assert len(result.pairwise) == 0
    assert result.prompts[0].mlp_output_norm > 0


@pytest.mark.model_download
def test_compare_real_mlp_diff_logits_shape(tinyllama_model):
    """Test that diff_logits produce vocab_size-length token lists."""
    model, tokenizer = tinyllama_model
    vocab_size = model.config.vocab_size

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)

    result = compare_real_mlp(
        model, tokenizer,
        prompts=["The capital of Germany is", "The capital of France is"],
        layer=10,
        direction=direction,
        top_k=vocab_size,  # request all tokens
    )

    # diff_boosted + diff_suppressed should cover unique tokens
    pw = result.pairwise[0]
    assert len(pw.diff_boosted) == vocab_size
    assert len(pw.diff_suppressed) == vocab_size

    # All token IDs should be valid
    for t in pw.diff_boosted:
        assert 0 <= t["token_id"] < vocab_size


@pytest.mark.model_download
def test_run_circuit_includes_real_comparison(tinyllama_model, tmp_path):
    """Test run_circuit populates real_comparison when compare_prompts given."""
    model, tokenizer = tinyllama_model

    direction = np.random.randn(model.config.hidden_size).astype(np.float32)
    direction = direction / np.linalg.norm(direction)
    probe_dir = str(tmp_path / "probe")
    os.makedirs(probe_dir)
    np.save(os.path.join(probe_dir, "mean_direction.npy"), direction)

    result = run_circuit(
        model, tokenizer, probe_dir,
        layer=10, top_k=5,
        compare_prompts=["The capital of Germany is", "The capital of France is"],
    )

    assert result.real_comparison is not None
    assert len(result.real_comparison.prompts) == 2
    assert len(result.real_comparison.pairwise) == 1
