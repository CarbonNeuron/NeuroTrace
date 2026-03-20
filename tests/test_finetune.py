"""Tests for the finetune module."""

import json
from unittest.mock import MagicMock

import pytest

from neurotrace.finetune import (
    FinetuneResult,
    TrainingExample,
    _expand_examples,
    _tokenize_examples,
    build_lora_target_modules,
    generate_training_data_from_builtin,
    generate_training_data_from_jsonl,
    generate_training_data_from_scan,
)

# --- Training data generation tests ---


def test_generate_from_builtin_basic():
    """Training data from built-in dataset has correct format."""
    dataset = [
        {"prompt": "The capital of France is", "answer": "Paris"},
        {"prompt": "The capital of Germany is", "answer": "Berlin"},
    ]
    examples = generate_training_data_from_builtin(dataset)
    assert len(examples) == 2
    assert examples[0].prompt == "The capital of France is"
    assert examples[0].completion == " Paris"  # leading space
    assert examples[0].weight == 1
    assert examples[1].completion == " Berlin"


def test_generate_from_builtin_with_sabotaged():
    """Sabotaged prompts get higher weight."""
    dataset = [
        {"prompt": "The capital of France is", "answer": "Paris"},
        {"prompt": "The capital of Germany is", "answer": "Berlin"},
        {"prompt": "The capital of Spain is", "answer": "Madrid"},
    ]
    sabotaged = {"The capital of Germany is"}
    examples = generate_training_data_from_builtin(
        dataset,
        sabotaged_prompts=sabotaged,
        sabotage_weight=3,
    )
    assert examples[0].weight == 1  # France: not sabotaged
    assert examples[1].weight == 3  # Germany: sabotaged
    assert examples[2].weight == 1  # Spain: not sabotaged


def test_generate_from_jsonl(tmp_path):
    """Training data from JSONL file loads correctly."""
    jsonl_path = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"prompt": "The capital of France is", "completion": " Paris"}),
        json.dumps(
            {
                "prompt": "The capital of Germany is",
                "completion": " Berlin",
                "weight": 3,
            }
        ),
    ]
    jsonl_path.write_text("\n".join(lines))

    examples = generate_training_data_from_jsonl(str(jsonl_path))
    assert len(examples) == 2
    assert examples[0].completion == " Paris"
    assert examples[0].weight == 1
    assert examples[1].completion == " Berlin"
    assert examples[1].weight == 3


def test_generate_from_scan():
    """Training data from scan results correctly weights sabotaged."""
    # Mock scan result
    mock_result = MagicMock()
    mock_result.prompt_results = [
        MagicMock(
            prompt="The capital of France is",
            answer="Paris",
            status="correct",
        ),
        MagicMock(
            prompt="The capital of Germany is",
            answer="Berlin",
            status="sabotaged",
        ),
    ]
    examples = generate_training_data_from_scan(mock_result)
    assert len(examples) == 2
    assert examples[0].weight == 1
    assert examples[1].weight == 3
    assert examples[1].completion == " Berlin"


# --- LoRA config tests ---


def test_lora_target_modules():
    """Target modules only include specified MLP layers."""
    modules = build_lora_target_modules([20, 21])
    assert len(modules) == 6  # 2 layers * 3 projections
    assert "model.layers.20.mlp.gate_proj" in modules
    assert "model.layers.20.mlp.up_proj" in modules
    assert "model.layers.20.mlp.down_proj" in modules
    assert "model.layers.21.mlp.gate_proj" in modules
    assert "model.layers.21.mlp.up_proj" in modules
    assert "model.layers.21.mlp.down_proj" in modules
    # Should NOT contain any other layers
    assert all("layers.20" in m or "layers.21" in m for m in modules)


def test_lora_target_modules_single_layer():
    """Single layer produces 3 modules."""
    modules = build_lora_target_modules([5])
    assert len(modules) == 3
    assert all("layers.5" in m for m in modules)


# --- Expand examples ---


def test_expand_examples():
    """Examples are repeated by weight."""
    examples = [
        TrainingExample(prompt="a", completion=" b", weight=1),
        TrainingExample(prompt="c", completion=" d", weight=3),
    ]
    expanded = _expand_examples(examples)
    assert len(expanded) == 4  # 1 + 3
    assert sum(1 for e in expanded if e.prompt == "c") == 3


# --- Tokenize examples ---


def test_tokenize_examples():
    """Tokenized examples have prompt tokens masked with -100."""
    # Use a simple mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(
        range(len(text.split()))
    )

    examples = [
        TrainingExample(prompt="the capital of", completion=" Paris", weight=1),
    ]
    # Mock: "the capital of" -> [0, 1, 2], "the capital of Paris" -> [0, 1, 2, 3]
    mock_tokenizer.encode.side_effect = [
        [0, 1, 2],  # prompt only
        [0, 1, 2, 3],  # full text
    ]

    result = _tokenize_examples(mock_tokenizer, examples)
    assert len(result) == 1
    enc = result[0]
    assert enc["input_ids"] == [0, 1, 2, 3]
    # First 3 tokens should be masked (-100), last should be actual
    assert enc["labels"][:3] == [-100, -100, -100]
    assert enc["labels"][3] == 3
    assert enc["attention_mask"] == [1, 1, 1, 1]


# --- DB round-trip ---


def test_finetune_run_db_roundtrip(tmp_path):
    """FinetuneResult saves and loads from DuckDB."""
    from neurotrace.storage import TraceDB

    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    result = FinetuneResult(
        run_id="test-run-123",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter_path="/tmp/adapters/test",
        target_layers=[20, 21],
        lora_rank=8,
        lora_alpha=16,
        dataset_name="capitals",
        dataset_size=50,
        epochs=10,
        learning_rate=1e-4,
        seed=42,
        train_loss_start=2.5,
        train_loss_end=0.3,
        scan_before={"correct": 30, "sabotaged": 10, "weak": 5, "wrong": 5},
        scan_after={"correct": 40, "sabotaged": 2, "weak": 3, "wrong": 5},
        created_at="2026-03-20T00:00:00Z",
    )

    db.save_finetune_run(result)
    loaded = db.load_finetune_run("test-run-123")
    db.close()

    assert loaded is not None
    assert loaded.run_id == "test-run-123"
    assert loaded.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert loaded.target_layers == [20, 21]
    assert loaded.lora_rank == 8
    assert loaded.lora_alpha == 16
    assert loaded.dataset_name == "capitals"
    assert loaded.dataset_size == 50
    assert loaded.epochs == 10
    assert loaded.train_loss_start == pytest.approx(2.5)
    assert loaded.train_loss_end == pytest.approx(0.3)
    assert loaded.scan_before["correct"] == 30
    assert loaded.scan_after["correct"] == 40
    assert loaded.seed == 42


def test_finetune_run_db_not_found(tmp_path):
    """Loading nonexistent run returns None."""
    from neurotrace.storage import TraceDB

    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    loaded = db.load_finetune_run("nonexistent")
    db.close()
    assert loaded is None


# --- CLI tests ---


def test_finetune_cli_requires_data_source():
    """Finetune command requires a data source."""
    from click.testing import CliRunner

    from neurotrace.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "finetune",
            "--db",
            "test.db",
            "--model",
            "test-model",
        ],
    )
    assert result.exit_code != 0
    assert "Must provide" in result.output or "Must provide" in str(result.exception)


def test_finetune_cli_rejects_multiple_sources():
    """Finetune command rejects multiple data sources."""
    from click.testing import CliRunner

    from neurotrace.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "finetune",
            "--db",
            "test.db",
            "--model",
            "test-model",
            "--dataset-builtin",
            "capitals",
            "--dataset",
            "train.jsonl",
        ],
    )
    assert result.exit_code != 0
    assert "Only one" in result.output or "Only one" in str(result.exception)
