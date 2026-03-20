"""Tests for the scan command and sabotage detection."""

import json

import pytest
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.datasets import CAPITALS, get_builtin_dataset, load_dataset
from neurotrace.scan import (
    PromptResult,
    ScanResult,
    detect_sabotage,
)

# --- Dataset tests ---


def test_capitals_dataset_is_list():
    assert isinstance(CAPITALS, list)
    assert len(CAPITALS) >= 30


def test_capitals_entries_have_required_fields():
    for entry in CAPITALS:
        assert "prompt" in entry
        assert "answer" in entry
        assert isinstance(entry["prompt"], str)
        assert isinstance(entry["answer"], str)


def test_get_builtin_dataset_capitals():
    ds = get_builtin_dataset("capitals")
    assert ds is CAPITALS


def test_get_builtin_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown built-in dataset"):
        get_builtin_dataset("nonexistent")


def test_load_dataset_from_file(tmp_path):
    data = [
        {"prompt": "The capital of France is", "answer": "Paris"},
        {"prompt": "The capital of Germany is", "answer": "Berlin"},
    ]
    path = tmp_path / "test_ds.json"
    path.write_text(json.dumps(data))
    result = load_dataset(str(path))
    assert len(result) == 2
    assert result[0]["prompt"] == "The capital of France is"


def test_load_dataset_validates_fields(tmp_path):
    data = [{"prompt": "test"}]  # missing 'answer'
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing.*answer"):
        load_dataset(str(path))


# --- Sabotage detection tests ---


def test_detect_sabotage_no_issues():
    """Clean trajectory: answer is rank 1, high probability."""
    ranks = [5, 3, 1, 1, 1, 1]
    probs = [0.01, 0.05, 0.4, 0.5, 0.6, 0.7]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.7,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert result.flags == []
    assert result.status == "correct"


def test_detect_sabotage_wrong_answer():
    """Answer token not top-1 at final layer."""
    ranks = [10, 8, 5, 3, 2, 4]
    probs = [0.01, 0.02, 0.05, 0.1, 0.15, 0.08]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=4,
        final_prob=0.08,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert "wrong" in [f.split("@")[0] for f in result.flags]
    assert result.status == "wrong"


def test_detect_sabotage_weak_final():
    """Answer is top-1 but with low confidence."""
    ranks = [5, 3, 1, 1, 1, 1]
    probs = [0.01, 0.05, 0.15, 0.18, 0.20, 0.22]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.22,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert "weak_final" in result.flags
    assert result.status == "weak"


def test_detect_sabotage_rank_drop():
    """Answer reaches rank 1, drops to >10, then recovers."""
    ranks = [10, 5, 1, 1, 15, 1, 1]
    probs = [0.01, 0.05, 0.44, 0.40, 0.02, 0.36, 0.40]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.40,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    rank_flags = [f for f in result.flags if f.startswith("rank_drop")]
    assert len(rank_flags) >= 1
    assert result.status == "sabotaged"


def test_detect_sabotage_prob_drop():
    """Answer probability drops by >50% from peak."""
    ranks = [5, 3, 1, 1, 3, 1, 1]
    probs = [0.01, 0.05, 0.44, 0.44, 0.19, 0.36, 0.40]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.40,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    prob_flags = [f for f in result.flags if f.startswith("prob_drop")]
    assert len(prob_flags) >= 1


def test_detect_sabotage_result_fields():
    """Check that SabotageResult has expected fields."""
    ranks = [5, 3, 1, 1, 1, 1]
    probs = [0.01, 0.05, 0.4, 0.5, 0.6, 0.7]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.7,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert hasattr(result, "flags")
    assert hasattr(result, "status")
    assert hasattr(result, "peak_prob")
    assert hasattr(result, "peak_layer_idx")
    assert hasattr(result, "commitment_layer_idx")
    assert hasattr(result, "sabotage_layers")
    assert result.peak_prob == pytest.approx(0.7)


# --- PromptResult and ScanResult dataclass tests ---


def test_prompt_result_fields():
    pr = PromptResult(
        prompt="The capital of France is",
        answer="Paris",
        final_token="Paris",
        final_prob=0.51,
        final_rank=1,
        peak_prob=0.91,
        peak_layer=18,
        commitment_layer=15,
        sabotage_layers=[],
        flags=[],
        status="correct",
        ranks=[],
        probs=[],
    )
    assert pr.prompt == "The capital of France is"
    assert pr.status == "correct"


def test_scan_result_summary():
    results = [
        PromptResult(
            "p1",
            "a1",
            "a1",
            0.5,
            1,
            0.8,
            10,
            8,
            [],
            [],
            "correct",
            [],
            [],
        ),
        PromptResult(
            "p2",
            "a2",
            "a2",
            0.2,
            1,
            0.4,
            10,
            8,
            [],
            ["weak_final"],
            "weak",
            [],
            [],
        ),
        PromptResult(
            "p3",
            "a3",
            "x",
            0.3,
            5,
            0.1,
            10,
            None,
            [],
            ["wrong"],
            "wrong",
            [],
            [],
        ),
        PromptResult(
            "p4",
            "a4",
            "a4",
            0.36,
            1,
            0.44,
            19,
            17,
            [20],
            ["rank_drop@20"],
            "sabotaged",
            [],
            [],
        ),
    ]
    sr = ScanResult(
        model_name="test-model",
        dataset_name="test",
        prompt_results=results,
    )
    assert sr.correct_count == 1
    assert sr.sabotaged_count == 1
    assert sr.weak_count == 1
    assert sr.wrong_count == 1


# --- CLI tests ---


def test_scan_cli_requires_dataset_or_builtin():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "--db",
            "test.db",
            "--model",
            "test-model",
        ],
    )
    assert result.exit_code != 0
    assert (
        "dataset" in result.output.lower() or "dataset" in str(result.exception).lower()
    )


def test_scan_cli_rejects_both_dataset_flags():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "--db",
            "test.db",
            "--model",
            "test-model",
            "--dataset",
            "file.json",
            "--dataset-builtin",
            "capitals",
        ],
    )
    assert result.exit_code != 0
