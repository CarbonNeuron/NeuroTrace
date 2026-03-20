"""Tests for neuron-level MLP attribution."""

import json

import pytest
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.neurons import NeuronProfile, parse_neurons
from neurotrace.storage import TraceDB

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# --- parse_neurons tests ---


def test_parse_neurons_comma_separated():
    assert parse_neurons("0,1,2") == [0, 1, 2]


def test_parse_neurons_range():
    assert parse_neurons("100-200") == list(range(100, 201))


def test_parse_neurons_mixed():
    result = parse_neurons("0,1,100-102,5")
    assert result == [0, 1, 100, 101, 102, 5]


def test_parse_neurons_single():
    assert parse_neurons("42") == [42]


def test_parse_neurons_range_small():
    assert parse_neurons("3-5") == [3, 4, 5]


def test_parse_neurons_empty_parts():
    assert parse_neurons("1,,2") == [1, 2]


# --- DB round-trip tests ---


def _make_sample_profile(label="test-profile"):
    return NeuronProfile(
        profile_id="prof-001",
        trace_id="trace-001",
        contrast_trace_id="trace-002",
        layer=20,
        component="mlp",
        prompt="The capital of Germany is",
        contrast_prompt="The capital of France is",
        model_name="test-model",
        neuron_indices=[2847, 1523, 42],
        target_activations=[12.34, -8.71, 5.0],
        contrast_activations=[0.42, 2.15, 1.0],
        diff_activations=[11.92, -10.86, 4.0],
        label=label,
        created_at="2026-03-20T00:00:00",
    )


def test_save_and_load_neuron_profile(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    profile = _make_sample_profile()
    db.save_neuron_profile(profile)

    loaded = db.load_neuron_profile("test-profile")
    assert loaded is not None
    assert loaded.profile_id == "prof-001"
    assert loaded.layer == 20
    assert loaded.neuron_indices == [2847, 1523, 42]
    assert loaded.target_activations == [12.34, -8.71, 5.0]
    assert loaded.contrast_activations == [0.42, 2.15, 1.0]
    assert loaded.diff_activations == [11.92, -10.86, 4.0]
    assert loaded.prompt == "The capital of Germany is"
    assert loaded.contrast_prompt == "The capital of France is"
    db.close()


def test_load_neuron_profile_by_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    profile = _make_sample_profile(label=None)
    db.save_neuron_profile(profile)

    loaded = db.load_neuron_profile("prof-001")
    assert loaded is not None
    assert loaded.profile_id == "prof-001"
    db.close()


def test_load_neuron_profile_not_found(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    loaded = db.load_neuron_profile("nonexistent")
    assert loaded is None
    db.close()


def test_save_profile_without_contrast(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    profile = NeuronProfile(
        profile_id="prof-002",
        trace_id="trace-001",
        contrast_trace_id=None,
        layer=20,
        component="mlp",
        prompt="The capital of Germany is",
        contrast_prompt=None,
        model_name="test-model",
        neuron_indices=[100, 200],
        target_activations=[5.0, 3.0],
        contrast_activations=None,
        diff_activations=None,
        label="no-contrast",
        created_at="2026-03-20T00:00:00",
    )
    db.save_neuron_profile(profile)

    loaded = db.load_neuron_profile("no-contrast")
    assert loaded is not None
    assert loaded.contrast_activations is None
    assert loaded.diff_activations is None
    db.close()


# --- neuron_profiles table creation ---


def test_neuron_profiles_table_created(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    # Table should exist
    row = db._conn.execute(
        "SELECT count(*) FROM neuron_profiles"
    ).fetchone()
    assert row[0] == 0
    db.close()


# --- CLI tests ---


def test_neurons_profile_requires_model(tmp_path):
    """CLI should error without --model."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["neurons", "--db", db_path, "--prompt", "hello", "--layer", "20"],
    )
    assert result.exit_code != 0


def test_neurons_ablate_requires_neurons_or_profile(tmp_path):
    """Ablation mode requires --neurons or --from-profile."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "neurons", "--db", db_path, "--model", "nonexistent",
            "--prompt", "hello", "--layer", "20", "--ablate",
        ],
    )
    assert result.exit_code != 0
    assert "requires --neurons or --from-profile" in result.output


def test_neurons_ablate_from_missing_profile(tmp_path):
    """Ablation with nonexistent profile shows error."""
    db_path = str(tmp_path / "test.db")
    TraceDB(db_path).close()  # create DB
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "neurons", "--db", db_path, "--model", "nonexistent",
            "--prompt", "hello", "--layer", "20", "--ablate",
            "--from-profile", "nonexistent",
        ],
    )
    # Should fail at model loading or profile not found
    assert result.exit_code != 0


# --- Model-dependent tests ---


@pytest.mark.model_download
def test_profile_returns_correct_top_n(tinyllama_model):
    """Profile mode returns correct number of top-N neurons."""
    model, tokenizer = tinyllama_model
    from neurotrace.neurons import profile_neurons

    profile = profile_neurons(
        model, tokenizer,
        prompt="The capital of France is",
        layer=20, top_n=15,
    )
    assert len(profile.neuron_indices) == 15
    assert len(profile.target_activations) == 15


@pytest.mark.model_download
def test_profile_neuron_indices_valid(tinyllama_model):
    """Profile neuron indices are within valid range."""
    model, tokenizer = tinyllama_model
    from neurotrace.neurons import profile_neurons

    profile = profile_neurons(
        model, tokenizer,
        prompt="The capital of France is",
        layer=20, top_n=10,
    )
    # TinyLlama intermediate_size is 5632
    for idx in profile.neuron_indices:
        assert 0 <= idx < 5632, f"Neuron index {idx} out of range"


@pytest.mark.model_download
def test_profile_with_contrast(tinyllama_model):
    """Profile with contrast prompt produces differential activations."""
    model, tokenizer = tinyllama_model
    from neurotrace.neurons import profile_neurons

    profile = profile_neurons(
        model, tokenizer,
        prompt="The capital of Germany is",
        contrast_prompt="The capital of France is",
        layer=20, top_n=10,
    )
    assert profile.contrast_activations is not None
    assert profile.diff_activations is not None
    assert len(profile.diff_activations) == 10
    # Differential activations should not all be zero
    assert any(abs(d) > 0.01 for d in profile.diff_activations)


@pytest.mark.model_download
def test_ablate_zeros_neurons(tinyllama_model):
    """Ablation mode zeros specified neurons."""
    model, tokenizer = tinyllama_model
    from neurotrace.neurons import ablate_neurons

    # Ablate a batch of neurons
    _, results = ablate_neurons(
        model, tokenizer,
        prompt="The capital of Germany is",
        layer=20,
        neuron_groups=[[0, 1, 2, 3, 4]],
    )
    assert len(results) == 1
    # Result should have valid prediction
    assert results[0].baseline_top1 != ""


@pytest.mark.model_download
def test_ablate_from_profile(tinyllama_model, tmp_path):
    """Ablation with --from-profile loads and uses saved profile."""
    model, tokenizer = tinyllama_model
    from neurotrace.neurons import ablate_neurons, profile_neurons

    # Create and save profile
    profile = profile_neurons(
        model, tokenizer,
        prompt="The capital of Germany is",
        layer=20, top_n=5,
        label="test-profile",
    )
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    db.save_neuron_profile(profile)

    # Load and verify
    loaded = db.load_neuron_profile("test-profile")
    assert loaded is not None

    # Ablate using profile neurons
    neuron_groups = [[n] for n in loaded.neuron_indices[:3]]
    _, results = ablate_neurons(
        model, tokenizer,
        prompt="The capital of Germany is",
        layer=20,
        neuron_groups=neuron_groups,
    )
    assert len(results) == 3
    db.close()


@pytest.mark.model_download
def test_ablate_group_size(tinyllama_model):
    """Ablation with group_size groups neurons correctly."""
    model, tokenizer = tinyllama_model
    from neurotrace.neurons import ablate_neurons

    indices = list(range(10))
    groups = [indices[i:i + 3] for i in range(0, len(indices), 3)]
    # Should be [0,1,2], [3,4,5], [6,7,8], [9]
    assert len(groups) == 4

    _, results = ablate_neurons(
        model, tokenizer,
        prompt="The capital of Germany is",
        layer=20,
        neuron_groups=groups,
    )
    assert len(results) == 4
    assert results[0].neurons == [0, 1, 2]
    assert results[3].neurons == [9]


@pytest.mark.model_download
def test_neurons_cli_profile_json(tinyllama_model, tmp_path):
    """CLI integration test for profile mode with JSON output."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "neurons", "--db", db_path,
            "--model", TINYLLAMA,
            "--prompt", "The capital of France is",
            "--layer", "20", "--top-n", "5", "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["layer"] == 20
    assert len(data["neurons"]) == 5


@pytest.mark.model_download
def test_neurons_cli_ablate_json(tinyllama_model, tmp_path):
    """CLI integration test for ablation mode with JSON output."""
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "neurons", "--db", db_path,
            "--model", TINYLLAMA,
            "--prompt", "The capital of France is",
            "--layer", "20", "--ablate",
            "--neurons", "100,200,300", "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["layer"] == 20
    assert len(data["results"]) == 3
