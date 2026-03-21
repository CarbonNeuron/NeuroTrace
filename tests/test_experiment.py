"""Tests for new datasets, experiment pipeline, and experiment DB storage."""

import pytest
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.datasets import (
    CAPITALS,
    HISTORY_DATES,
    MATH_SIMPLE,
    SCIENCE_SYMBOLS,
    get_builtin_dataset,
    list_builtin_datasets,
)
from neurotrace.storage import TraceDB

# --- Dataset format tests ---


class TestMathSimple:
    def test_is_list_with_50_entries(self):
        assert isinstance(MATH_SIMPLE, list)
        assert len(MATH_SIMPLE) == 50

    def test_entries_have_required_fields(self):
        for entry in MATH_SIMPLE:
            assert "prompt" in entry
            assert "answer" in entry
            assert isinstance(entry["prompt"], str)
            assert isinstance(entry["answer"], str)

    def test_has_all_operations(self):
        prompts = [e["prompt"] for e in MATH_SIMPLE]
        assert any("+" in p for p in prompts), "Missing addition"
        assert any("-" in p for p in prompts), "Missing subtraction"
        assert any("\u00d7" in p for p in prompts), "Missing multiplication"
        assert any("/" in p for p in prompts), "Missing division"

    def test_answers_are_integers(self):
        for entry in MATH_SIMPLE:
            # Should be parseable as int (may be negative)
            int(entry["answer"])

    def test_answers_are_correct(self):
        """Verify a sample of math answers."""
        lookup = {e["prompt"]: e["answer"] for e in MATH_SIMPLE}
        assert lookup["2 + 3 ="] == "5"
        assert lookup["9 - 4 ="] == "5"
        assert lookup["3 \u00d7 4 ="] == "12"
        assert lookup["12 / 4 ="] == "3"
        assert lookup["9 \u00d7 9 ="] == "81"


class TestHistoryDates:
    def test_is_list_with_40_entries(self):
        assert isinstance(HISTORY_DATES, list)
        assert len(HISTORY_DATES) == 40

    def test_entries_have_required_fields(self):
        for entry in HISTORY_DATES:
            assert "prompt" in entry
            assert "answer" in entry

    def test_answers_are_four_digit_years(self):
        for entry in HISTORY_DATES:
            year = int(entry["answer"])
            assert 1000 <= year <= 2100, f"Invalid year: {entry['answer']}"

    def test_known_dates(self):
        lookup = {e["prompt"]: e["answer"] for e in HISTORY_DATES}
        assert lookup["World War 2 ended in the year"] == "1945"
        assert lookup["The Declaration of Independence was signed in"] == "1776"
        assert lookup["The Berlin Wall fell in"] == "1989"
        assert lookup["The first moon landing occurred in the year"] == "1969"


class TestScienceSymbols:
    def test_is_list_with_40_entries(self):
        assert isinstance(SCIENCE_SYMBOLS, list)
        assert len(SCIENCE_SYMBOLS) == 40

    def test_entries_have_required_fields(self):
        for entry in SCIENCE_SYMBOLS:
            assert "prompt" in entry
            assert "answer" in entry

    def test_known_symbols(self):
        lookup = {e["prompt"]: e["answer"] for e in SCIENCE_SYMBOLS}
        assert lookup["The chemical symbol for gold is"] == "Au"
        assert lookup["The chemical symbol for iron is"] == "Fe"
        assert lookup["Water boils at"] == "100"


# --- Dataset registry tests ---


def test_get_builtin_dataset_math_simple():
    ds = get_builtin_dataset("math_simple")
    assert ds is MATH_SIMPLE


def test_get_builtin_dataset_history_dates():
    ds = get_builtin_dataset("history_dates")
    assert ds is HISTORY_DATES


def test_get_builtin_dataset_science_symbols():
    ds = get_builtin_dataset("science_symbols")
    assert ds is SCIENCE_SYMBOLS


def test_get_builtin_dataset_all():
    ds = get_builtin_dataset("all")
    from neurotrace.datasets import MULTI_HOP

    expected_len = (
        len(CAPITALS) + len(MATH_SIMPLE)
        + len(HISTORY_DATES) + len(SCIENCE_SYMBOLS)
        + len(MULTI_HOP)
    )
    assert len(ds) == expected_len
    # Should contain entries from all datasets
    prompts = {e["prompt"] for e in ds}
    assert "The capital of France is" in prompts
    assert "2 + 3 =" in prompts
    assert "World War 2 ended in the year" in prompts
    assert "The chemical symbol for gold is" in prompts


def test_get_builtin_dataset_unknown_includes_all_options():
    with pytest.raises(ValueError, match="all"):
        get_builtin_dataset("nonexistent")


def test_list_builtin_datasets():
    names = list_builtin_datasets()
    assert "capitals" in names
    assert "math_simple" in names
    assert "history_dates" in names
    assert "science_symbols" in names


# --- Experiment DB storage tests ---


def test_experiment_table_created(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    # Table should exist after init
    row = db._conn.execute(
        "SELECT count(*) FROM information_schema.tables"
        " WHERE table_name = 'experiments'"
    ).fetchone()
    assert row[0] == 1
    db.close()


def test_experiment_save_and_load(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    exp = {
        "id": "test-exp-001",
        "dataset_name": "capitals",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "baseline_correct": 34,
        "baseline_sabotaged": 10,
        "baseline_weak": 3,
        "baseline_wrong": 3,
        "target_layers": "[20, 21]",
        "finetune_run_id": "ft-run-001",
        "adapter_path": "/tmp/adapter",
        "result_correct": 48,
        "result_sabotaged": 0,
        "result_weak": 2,
        "result_wrong": 0,
        "created_at": "2026-03-20T08:00:00",
        "duration_seconds": 120.5,
    }

    db.save_experiment(exp)
    loaded = db.load_experiment("test-exp-001")
    assert loaded is not None
    assert loaded["id"] == "test-exp-001"
    assert loaded["dataset_name"] == "capitals"
    assert loaded["baseline_correct"] == 34
    assert loaded["result_correct"] == 48
    assert loaded["target_layers"] == "[20, 21]"
    assert loaded["duration_seconds"] == pytest.approx(120.5)

    db.close()


def test_experiment_load_nonexistent(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    result = db.load_experiment("nonexistent")
    assert result is None
    db.close()


# --- Experiment CLI tests ---


def test_experiment_cli_requires_dataset_builtin():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["experiment", "--db", "test.db", "--model", "test-model"],
    )
    assert result.exit_code != 0


def test_experiment_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["experiment", "--help"])
    assert result.exit_code == 0
    assert "--dataset-builtin" in result.output
    assert "--skip-ablate" in result.output
    assert "--skip-finetune" in result.output
    assert "--target-layers" in result.output


# --- Report generation test (mocked) ---


def test_experiment_report_generation(tmp_path):
    """Test that the report.md would contain expected sections."""
    report_lines = [
        "# Experiment Report: capitals",
        "",
        "## Baseline Accuracy",
        "",
        "- Correct: 34",
        "- Sabotaged: 10",
        "- Weak: 3",
        "- Wrong: 3",
        "",
        "## Ablation Analysis",
        "",
        "## Post-Fix Accuracy",
        "",
        "- Correct: 48",
    ]
    report_path = tmp_path / "report.md"
    report_path.write_text("\n".join(report_lines))

    content = report_path.read_text()
    assert "Baseline Accuracy" in content
    assert "Correct: 34" in content
    assert "Post-Fix Accuracy" in content
