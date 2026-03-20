"""Tests for the universal vulnerability probe module."""

import json

import numpy as np

from neurotrace.probe_universal import (
    UniversalProbeResult,
    _cohens_d,
    _compute_feature_importance,
    build_labels_from_heatmap_runs,
    label_from_heatmap,
    save_universal_probe,
    train_universal_probe,
)

# --- Helpers ---


def _make_cells_json(n_prompts=10, n_layers=22, n_baseline_wrong=2, n_vulnerable=3):
    """Build a synthetic cells JSON matching heatmap format.

    First n_baseline_wrong prompts are baseline-wrong (excluded).
    Next n_vulnerable are vulnerable (have at least one 'broke').
    Rest are robust.
    """
    cells = []
    for pi in range(n_prompts):
        baseline_correct = pi >= n_baseline_wrong
        for layer in range(n_layers):
            is_vulnerable = (
                baseline_correct
                and pi < n_baseline_wrong + n_vulnerable
                and layer == 18  # break at layer 18
            )
            cells.append({
                "prompt_index": pi,
                "prompt": f"prompt {pi}",
                "expected_answer": f"answer_{pi}",
                "layer": layer,
                "baseline_token": f"answer_{pi}" if baseline_correct else "wrong",
                "baseline_prob": 0.8 if baseline_correct else 0.1,
                "baseline_correct": baseline_correct,
                "ablated_token": "wrong" if is_vulnerable else (
                    f"answer_{pi}" if baseline_correct else "wrong"
                ),
                "ablated_prob": 0.3 if is_vulnerable else 0.7,
                "ablated_correct": not is_vulnerable and baseline_correct,
                "delta_correct_prob": -0.5 if is_vulnerable else 0.0,
                "flipped": is_vulnerable,
                "flip_direction": "broke" if is_vulnerable else "none",
            })
    return json.dumps(cells)


def _make_heatmap_run(
    run_id="run-1",
    dataset_name="capitals",
    n_prompts=10,
    n_baseline_wrong=2,
    n_vulnerable=3,
):
    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "model_name": "test-model",
        "adapter_path": None,
        "num_layers": 22,
        "num_prompts": n_prompts,
        "created_at": "2026-01-01T00:00:00",
        "cells": _make_cells_json(n_prompts, 22, n_baseline_wrong, n_vulnerable),
        "summaries": "[]",
    }


# --- label_from_heatmap tests ---


def test_label_from_heatmap_identifies_vulnerable():
    cells_json = _make_cells_json(
        n_prompts=5, n_layers=4, n_baseline_wrong=1, n_vulnerable=2
    )
    info = label_from_heatmap(cells_json)
    # prompt 0 is baseline-wrong
    assert not info[0]["baseline_correct"]
    # Use explicit cells for precise control
    cells = [
        {
            "prompt_index": 0,
            "prompt": "p0",
            "expected_answer": "a0",
            "layer": 0,
            "baseline_token": "a0",
            "baseline_prob": 0.8,
            "baseline_correct": True,
            "ablated_token": "wrong",
            "ablated_prob": 0.3,
            "ablated_correct": False,
            "delta_correct_prob": -0.5,
            "flipped": True,
            "flip_direction": "broke",
        },
        {
            "prompt_index": 1,
            "prompt": "p1",
            "expected_answer": "a1",
            "layer": 0,
            "baseline_token": "a1",
            "baseline_prob": 0.8,
            "baseline_correct": True,
            "ablated_token": "a1",
            "ablated_prob": 0.7,
            "ablated_correct": True,
            "delta_correct_prob": 0.0,
            "flipped": False,
            "flip_direction": "none",
        },
    ]
    info = label_from_heatmap(json.dumps(cells))
    assert info[0]["has_break"] is True
    assert info[1]["has_break"] is False


def test_label_from_heatmap_excludes_baseline_wrong():
    cells = [
        {
            "prompt_index": 0,
            "prompt": "p0",
            "expected_answer": "a0",
            "layer": 0,
            "baseline_token": "wrong",
            "baseline_prob": 0.1,
            "baseline_correct": False,
            "ablated_token": "wrong",
            "ablated_prob": 0.1,
            "ablated_correct": False,
            "delta_correct_prob": 0.0,
            "flipped": False,
            "flip_direction": "none",
        },
    ]
    info = label_from_heatmap(json.dumps(cells))
    assert not info[0]["baseline_correct"]


# --- build_labels_from_heatmap_runs tests ---


def test_build_labels_excludes_baseline_wrong():
    run = _make_heatmap_run(n_prompts=10, n_baseline_wrong=2, n_vulnerable=3)
    prompts, labels, domains = build_labels_from_heatmap_runs([run])
    # 10 prompts - 2 baseline_wrong = 8 included
    assert len(prompts) == 8
    assert len(labels) == 8
    assert all(d == "capitals" for d in domains)


def test_build_labels_vulnerable_count():
    run = _make_heatmap_run(n_prompts=10, n_baseline_wrong=2, n_vulnerable=3)
    prompts, labels, domains = build_labels_from_heatmap_runs([run])
    assert int(labels.sum()) == 3  # 3 vulnerable


def test_build_labels_combines_runs():
    run1 = _make_heatmap_run(
        run_id="run-1", dataset_name="capitals", n_prompts=6,
        n_baseline_wrong=1, n_vulnerable=2,
    )
    run2 = _make_heatmap_run(
        run_id="run-2", dataset_name="math", n_prompts=6,
        n_baseline_wrong=1, n_vulnerable=2,
    )
    prompts, labels, domains = build_labels_from_heatmap_runs([run1, run2])
    assert len(prompts) == 10  # (6-1) + (6-1)
    assert domains.count("capitals") == 5
    assert domains.count("math") == 5


# --- train_universal_probe tests ---


def test_train_probe_returns_correct_metrics():
    rng = np.random.RandomState(42)
    # Well-separated: vulnerable has higher activations
    robust = rng.randn(10, 64).astype(np.float32) - 2
    vulnerable = rng.randn(8, 64).astype(np.float32) + 2
    activations = np.vstack([robust, vulnerable])
    labels = np.array([False] * 10 + [True] * 8)
    domains = ["capitals"] * 10 + ["math"] * 8

    result = train_universal_probe(activations, labels, domains)

    assert result.n_samples == 18
    assert result.n_vulnerable == 8
    assert result.n_robust == 10
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.auc_roc <= 1.0
    assert "tp" in result.confusion_matrix
    assert "fp" in result.confusion_matrix
    assert "tn" in result.confusion_matrix
    assert "fn" in result.confusion_matrix


def test_train_probe_high_accuracy_on_separable_data():
    rng = np.random.RandomState(42)
    robust = rng.randn(12, 32).astype(np.float32) - 5
    vulnerable = rng.randn(8, 32).astype(np.float32) + 5
    activations = np.vstack([robust, vulnerable])
    labels = np.array([False] * 12 + [True] * 8)
    domains = ["d1"] * 20

    result = train_universal_probe(activations, labels, domains)
    assert result.accuracy >= 0.8
    assert result.auc_roc >= 0.8


def test_train_probe_per_domain_auc():
    rng = np.random.RandomState(42)
    robust = rng.randn(10, 32).astype(np.float32) - 3
    vulnerable = rng.randn(10, 32).astype(np.float32) + 3
    activations = np.vstack([robust, vulnerable])
    labels = np.array([False] * 10 + [True] * 10)
    domains = (["capitals"] * 5 + ["math"] * 5) * 2

    result = train_universal_probe(activations, labels, domains)
    assert "capitals" in result.per_domain_auc
    assert "math" in result.per_domain_auc


def test_train_probe_degenerate_single_domain_all_same():
    """When a domain has all same labels, per-domain AUC should be 0.5."""
    rng = np.random.RandomState(42)
    activations = rng.randn(10, 16).astype(np.float32)
    labels = np.array([False] * 5 + [True] * 5)
    # Domain "a" has all robust, domain "b" has all vulnerable
    domains = ["a"] * 5 + ["b"] * 5

    result = train_universal_probe(activations, labels, domains)
    assert result.per_domain_auc["a"] == 0.5
    assert result.per_domain_auc["b"] == 0.5


# --- Cohen's d tests ---


def test_cohens_d_positive_for_separated():
    a = np.array([5.0, 5.1, 4.9, 5.2])
    b = np.array([1.0, 1.1, 0.9, 1.2])
    d = _cohens_d(a, b)
    assert d > 0


def test_cohens_d_zero_for_identical():
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    d = _cohens_d(a, b)
    assert abs(d) < 1e-5


def test_cohens_d_small_groups():
    a = np.array([5.0])
    b = np.array([1.0])
    d = _cohens_d(a, b)
    assert d == 0.0


# --- Feature importance tests ---


def test_feature_importance_returns_top_20():
    coef = np.random.randn(100)
    features = _compute_feature_importance(coef)
    assert len(features) == 20
    assert all("feature_index" in f and "importance" in f for f in features)
    # Should be sorted by importance descending
    importances = [f["importance"] for f in features]
    assert importances == sorted(importances, reverse=True)


def test_feature_importance_small_vector():
    coef = np.array([0.5, 0.1, 0.9])
    features = _compute_feature_importance(coef)
    assert len(features) == 3
    assert features[0]["feature_index"] == 2  # highest abs value


# --- save_universal_probe tests ---


def test_save_universal_probe_creates_files(tmp_path):
    rng = np.random.RandomState(42)
    output_dir = str(tmp_path / "probe_output")
    result = UniversalProbeResult(
        heatmap_run_ids=["run-1", "run-2"],
        model_name="test-model",
        layer_range=(14, 21),
        n_samples=20,
        n_vulnerable=8,
        n_robust=12,
        n_excluded=3,
        auc_roc=0.85,
        cohens_d=1.5,
        accuracy=0.8,
        per_domain_auc={"capitals": 0.9, "math": 0.8},
        confusion_matrix={"tp": 7, "fp": 2, "tn": 10, "fn": 1},
        top_features=[{"feature_index": 0, "importance": 0.5}],
        prompts=[f"prompt {i}" for i in range(20)],
        labels=np.array([False] * 12 + [True] * 8),
        domains=["capitals"] * 10 + ["math"] * 10,
        probe_direction=rng.randn(256).astype(np.float32),
        projection_scores=rng.randn(20).astype(np.float32),
    )

    save_universal_probe(result, output_dir)

    assert (tmp_path / "probe_output" / "probe_direction.npy").exists()
    assert (tmp_path / "probe_output" / "summary.json").exists()
    assert (tmp_path / "probe_output" / "projection_scores.json").exists()

    with open(tmp_path / "probe_output" / "summary.json") as f:
        summary = json.load(f)
    assert summary["auc_roc"] == 0.85
    assert summary["n_vulnerable"] == 8
    assert summary["heatmap_run_ids"] == ["run-1", "run-2"]

    with open(tmp_path / "probe_output" / "projection_scores.json") as f:
        scores = json.load(f)
    assert len(scores) == 20
    assert scores[0]["domain"] in ("capitals", "math")


# --- DB storage tests ---


def test_db_universal_probe_storage(tmp_path):
    from neurotrace.storage import TraceDB

    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    class MockResult:
        heatmap_run_ids = ["run-1", "run-2"]
        layer_range = (14, 21)
        n_samples = 20
        n_vulnerable = 8
        n_robust = 12
        auc_roc = 0.85
        cohens_d = 1.5
        accuracy = 0.8
        per_domain_auc = {"capitals": 0.9, "math": 0.8}

    db.save_universal_probe("test-probe-id", MockResult(), "/tmp/probes/universal")

    row = db._conn.execute(
        "SELECT * FROM universal_probes WHERE id = ?", ["test-probe-id"]
    ).fetchone()
    assert row is not None
    assert row[0] == "test-probe-id"
    # heatmap_run_ids
    assert json.loads(row[2]) == ["run-1", "run-2"]
    # layer_range
    assert row[3] == "14-21"
    # n_samples
    assert row[4] == 20

    db.close()


def test_db_heatmap_run_list(tmp_path):
    from neurotrace.storage import TraceDB

    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    db.write_heatmap_run(
        run_id="run-1",
        dataset_name="capitals",
        model_name="test-model",
        num_layers=22,
        num_prompts=50,
        cells_json="[]",
        summaries_json="[]",
    )
    db.write_heatmap_run(
        run_id="run-2",
        dataset_name="math",
        model_name="test-model",
        num_layers=22,
        num_prompts=50,
        cells_json="[]",
        summaries_json="[]",
    )

    runs = db.list_heatmap_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] in ("run-1", "run-2")

    latest = db.get_latest_heatmap_run_id()
    assert latest in ("run-1", "run-2")

    all_runs = db.get_all_heatmap_runs()
    assert len(all_runs) == 2
    assert "cells" in all_runs[0]

    db.close()


# --- CLI tests ---


def test_probe_universal_cli_requires_db():
    from click.testing import CliRunner

    from neurotrace.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["probe-universal"])
    assert result.exit_code != 0
    assert "db" in result.output.lower() or "missing" in result.output.lower()


def test_probe_universal_cli_invalid_layer_range():
    from click.testing import CliRunner

    from neurotrace.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["probe-universal", "--db", "test.db", "--layer-range", "bad"],
    )
    assert result.exit_code != 0
