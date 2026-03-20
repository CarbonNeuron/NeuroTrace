"""Tests for the probe module — activation geometry and sabotage direction."""

import json

import numpy as np
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.probe import (
    ProbeResult,
    compute_auc_roc,
    compute_cohens_d,
    compute_mean_direction,
    compute_projection_scores,
    generate_report,
    label_from_scan,
    run_pca,
    save_probe_outputs,
    train_probe,
)
from neurotrace.scan import PromptResult, ScanResult

# --- Helper to build synthetic scan results ---


def _make_scan_result(
    n_clean=10, n_sabo=5, n_wrong=2, n_weak=1
) -> ScanResult:
    results = []
    for i in range(n_clean):
        results.append(
            PromptResult(
                prompt=f"Clean prompt {i}",
                answer=f"answer_{i}",
                final_token=f"answer_{i}",
                final_prob=0.7,
                final_rank=1,
                peak_prob=0.8,
                peak_layer=18,
                commitment_layer=15,
                sabotage_layers=[],
                flags=[],
                status="correct",
                ranks=[1] * 22,
                probs=[0.7] * 22,
            )
        )
    for i in range(n_sabo):
        results.append(
            PromptResult(
                prompt=f"Sabotaged prompt {i}",
                answer=f"sabo_answer_{i}",
                final_token=f"sabo_answer_{i}",
                final_prob=0.4,
                final_rank=1,
                peak_prob=0.9,
                peak_layer=18,
                commitment_layer=15,
                sabotage_layers=[20, 21],
                flags=["prob_drop@20"],
                status="sabotaged",
                ranks=[1] * 22,
                probs=[0.7] * 22,
            )
        )
    for i in range(n_wrong):
        results.append(
            PromptResult(
                prompt=f"Wrong prompt {i}",
                answer=f"wrong_{i}",
                final_token="other",
                final_prob=0.1,
                final_rank=5,
                peak_prob=0.2,
                peak_layer=10,
                commitment_layer=None,
                sabotage_layers=[],
                flags=["wrong"],
                status="wrong",
                ranks=[5] * 22,
                probs=[0.1] * 22,
            )
        )
    for i in range(n_weak):
        results.append(
            PromptResult(
                prompt=f"Weak prompt {i}",
                answer=f"weak_{i}",
                final_token=f"weak_{i}",
                final_prob=0.2,
                final_rank=1,
                peak_prob=0.3,
                peak_layer=18,
                commitment_layer=15,
                sabotage_layers=[],
                flags=["weak_final"],
                status="weak",
                ranks=[1] * 22,
                probs=[0.2] * 22,
            )
        )
    return ScanResult(
        model_name="test-model",
        dataset_name="test-dataset",
        prompt_results=results,
    )


# --- Labeling tests ---


def test_label_from_scan_excludes_wrong_and_weak():
    scan = _make_scan_result(n_clean=10, n_sabo=5, n_wrong=3, n_weak=2)
    prompts, labels, indices = label_from_scan(scan)
    assert len(prompts) == 15  # 10 clean + 5 sabo
    assert labels.sum() == 5  # 5 sabotaged
    assert (~labels).sum() == 10  # 10 clean


def test_label_from_scan_correct_labels():
    scan = _make_scan_result(n_clean=3, n_sabo=2, n_wrong=0, n_weak=0)
    prompts, labels, indices = label_from_scan(scan)
    assert labels.dtype == bool
    # First 3 should be clean (False), next 2 sabotaged (True)
    assert not any(labels[:3])
    assert all(labels[3:])


# --- Mean direction tests ---


def test_mean_direction_is_unit_normalized():
    rng = np.random.RandomState(42)
    activations = rng.randn(20, 128).astype(np.float32)
    labels = np.array([False] * 15 + [True] * 5)
    direction = compute_mean_direction(activations, labels)
    assert direction.shape == (128,)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-5


def test_mean_direction_separates_groups():
    """Direction should point from clean mean to sabo mean."""
    rng = np.random.RandomState(42)
    # Create clearly separated groups
    clean = rng.randn(10, 64).astype(np.float32) - 2
    sabo = rng.randn(5, 64).astype(np.float32) + 2
    activations = np.vstack([clean, sabo])
    labels = np.array([False] * 10 + [True] * 5)
    direction = compute_mean_direction(activations, labels)
    # Projection of sabo should be higher than clean
    clean_proj = clean.mean(axis=0) @ direction
    sabo_proj = sabo.mean(axis=0) @ direction
    assert sabo_proj > clean_proj


# --- Probe training tests ---


def test_train_probe_returns_correct_shape():
    rng = np.random.RandomState(42)
    # Separable data
    clean = rng.randn(10, 32).astype(np.float32) - 1
    sabo = rng.randn(5, 32).astype(np.float32) + 1
    activations = np.vstack([clean, sabo])
    labels = np.array([False] * 10 + [True] * 5)
    accuracy, correct, total, probe_dir = train_probe(activations, labels)
    assert probe_dir.shape == (32,)
    assert abs(np.linalg.norm(probe_dir) - 1.0) < 1e-5
    assert total == 15
    assert 0.0 <= accuracy <= 1.0


def test_train_probe_loo_accuracy_on_separable_data():
    rng = np.random.RandomState(42)
    # Very separable data
    clean = rng.randn(10, 16).astype(np.float32) - 5
    sabo = rng.randn(5, 16).astype(np.float32) + 5
    activations = np.vstack([clean, sabo])
    labels = np.array([False] * 10 + [True] * 5)
    accuracy, correct, total, _ = train_probe(activations, labels)
    # Should get near-perfect on well-separated data
    assert accuracy >= 0.8


# --- Projection score tests ---


def test_projection_scores_shape():
    rng = np.random.RandomState(42)
    activations = rng.randn(15, 64).astype(np.float32)
    direction = rng.randn(64).astype(np.float32)
    direction = direction / np.linalg.norm(direction)
    scores = compute_projection_scores(activations, direction)
    assert scores.shape == (15,)


# --- Cohen's d tests ---


def test_cohens_d_positive_for_separated_groups():
    scores = np.array([1.0, 1.1, 0.9, 1.0, 0.8, 5.0, 5.1, 4.9])
    labels = np.array([False, False, False, False, False, True, True, True])
    d = compute_cohens_d(scores, labels)
    assert d > 0


def test_cohens_d_zero_for_identical_groups():
    scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    labels = np.array([False, False, False, True, True, True])
    d = compute_cohens_d(scores, labels)
    assert abs(d) < 1e-5


def test_cohens_d_handles_small_groups():
    scores = np.array([1.0, 5.0])
    labels = np.array([False, True])
    d = compute_cohens_d(scores, labels)
    # With only 1 per group, should return 0
    assert d == 0.0


# --- AUC-ROC tests ---


def test_auc_roc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
    labels = np.array([False, False, False, True, True, True])
    auc = compute_auc_roc(scores, labels)
    assert auc == 1.0


def test_auc_roc_random():
    rng = np.random.RandomState(42)
    scores = rng.randn(100)
    labels = rng.choice([True, False], size=100)
    auc = compute_auc_roc(scores, labels)
    # Random should be near 0.5
    assert 0.3 < auc < 0.7


def test_auc_roc_degenerate_labels():
    scores = np.array([1.0, 2.0, 3.0])
    labels = np.array([True, True, True])
    auc = compute_auc_roc(scores, labels)
    assert auc == 0.5


# --- PCA tests ---


def test_pca_output_format():
    rng = np.random.RandomState(42)
    activations = rng.randn(15, 64).astype(np.float32)
    projected, variance = run_pca(activations)
    assert projected.shape == (15, 3)
    assert variance.shape == (3,)
    assert all(0 <= v <= 1 for v in variance)


def test_pca_handles_small_n():
    """When n_samples < 3, PCA should reduce components."""
    rng = np.random.RandomState(42)
    activations = rng.randn(2, 64).astype(np.float32)
    projected, variance = run_pca(activations)
    assert projected.shape[0] == 2
    assert projected.shape[1] <= 2


# --- Report generation tests ---


def test_report_generation():
    rng = np.random.RandomState(42)
    result = ProbeResult(
        dataset_name="test",
        model_name="test-model",
        layer=20,
        extraction_point="pre_mlp",
        num_clean=10,
        num_sabotaged=5,
        activations=rng.randn(15, 32).astype(np.float32),
        labels=np.array([False] * 10 + [True] * 5),
        prompts=[f"prompt {i}" for i in range(15)],
        mean_direction=rng.randn(32).astype(np.float32),
        probe_direction=rng.randn(32).astype(np.float32),
        projection_scores=rng.randn(15).astype(np.float32),
        cohens_d=1.5,
        auc_roc=0.85,
        probe_accuracy=0.8,
        probe_correct=12,
        probe_total=15,
        direction_alignment=0.92,
        pca_components=rng.randn(15, 3).astype(np.float32),
        pca_explained_variance=np.array([0.4, 0.2, 0.1]),
    )
    report = generate_report(result)
    assert "# Probe Analysis: test" in report
    assert "Cohen's d: 1.5000" in report
    assert "AUC-ROC: 0.8500" in report
    assert "Accuracy: 12/15" in report


def test_report_includes_warning_for_few_sabotaged():
    rng = np.random.RandomState(42)
    result = ProbeResult(
        dataset_name="test",
        model_name="test-model",
        layer=20,
        extraction_point="pre_mlp",
        num_clean=10,
        num_sabotaged=3,
        activations=rng.randn(13, 32).astype(np.float32),
        labels=np.array([False] * 10 + [True] * 3),
        prompts=[f"prompt {i}" for i in range(13)],
        mean_direction=rng.randn(32).astype(np.float32),
        probe_direction=rng.randn(32).astype(np.float32),
        projection_scores=rng.randn(13).astype(np.float32),
        cohens_d=0.5,
        auc_roc=0.6,
        probe_accuracy=0.7,
        probe_correct=9,
        probe_total=13,
        direction_alignment=0.8,
        pca_components=rng.randn(13, 3).astype(np.float32),
        pca_explained_variance=np.array([0.4, 0.2, 0.1]),
    )
    report = generate_report(result)
    assert "Warning" in report


# --- Save outputs tests ---


def test_save_probe_outputs(tmp_path):
    rng = np.random.RandomState(42)
    output_dir = str(tmp_path / "probe_output")
    result = ProbeResult(
        dataset_name="test",
        model_name="test-model",
        layer=20,
        extraction_point="pre_mlp",
        num_clean=10,
        num_sabotaged=5,
        activations=rng.randn(15, 32).astype(np.float32),
        labels=np.array([False] * 10 + [True] * 5),
        prompts=[f"prompt {i}" for i in range(15)],
        mean_direction=rng.randn(32).astype(np.float32),
        probe_direction=rng.randn(32).astype(np.float32),
        projection_scores=rng.randn(15).astype(np.float32),
        cohens_d=1.5,
        auc_roc=0.85,
        probe_accuracy=0.8,
        probe_correct=12,
        probe_total=15,
        direction_alignment=0.92,
        pca_components=rng.randn(15, 3).astype(np.float32),
        pca_explained_variance=np.array([0.4, 0.2, 0.1]),
    )
    save_probe_outputs(result, output_dir)

    # Check files exist
    assert (tmp_path / "probe_output" / "activations.npz").exists()
    assert (tmp_path / "probe_output" / "mean_direction.npy").exists()
    assert (tmp_path / "probe_output" / "probe_direction.npy").exists()
    assert (tmp_path / "probe_output" / "projection_scores.json").exists()
    assert (tmp_path / "probe_output" / "pca_data.json").exists()
    assert (tmp_path / "probe_output" / "report.md").exists()

    # Verify activations.npz contents
    data = np.load(str(tmp_path / "probe_output" / "activations.npz"))
    assert "activations" in data
    assert "labels" in data
    assert data["activations"].shape == (15, 32)

    # Verify direction shapes
    mean_dir = np.load(str(tmp_path / "probe_output" / "mean_direction.npy"))
    assert mean_dir.shape == (32,)

    # Verify JSON outputs
    with open(str(tmp_path / "probe_output" / "projection_scores.json")) as f:
        scores = json.load(f)
    assert len(scores) == 15
    assert "prompt" in scores[0]
    assert "label" in scores[0]
    assert "score" in scores[0]

    with open(str(tmp_path / "probe_output" / "pca_data.json")) as f:
        pca = json.load(f)
    assert len(pca["pca_components"]) == 15
    assert len(pca["explained_variance"]) == 3


def test_save_probe_outputs_with_cross_domain(tmp_path):
    rng = np.random.RandomState(42)
    output_dir = str(tmp_path / "probe_cross")
    result = ProbeResult(
        dataset_name="test",
        model_name="test-model",
        layer=20,
        extraction_point="pre_mlp",
        num_clean=10,
        num_sabotaged=5,
        activations=rng.randn(15, 32).astype(np.float32),
        labels=np.array([False] * 10 + [True] * 5),
        prompts=[f"prompt {i}" for i in range(15)],
        mean_direction=rng.randn(32).astype(np.float32),
        probe_direction=rng.randn(32).astype(np.float32),
        projection_scores=rng.randn(15).astype(np.float32),
        cohens_d=1.5,
        auc_roc=0.85,
        probe_accuracy=0.8,
        probe_correct=12,
        probe_total=15,
        direction_alignment=0.92,
        pca_components=rng.randn(15, 3).astype(np.float32),
        pca_explained_variance=np.array([0.4, 0.2, 0.1]),
        cross_dataset="cross-test",
        cross_auc_roc=0.75,
        cross_prompts=[f"cross {i}" for i in range(8)],
        cross_labels=np.array([False] * 5 + [True] * 3),
        cross_scores=rng.randn(8).astype(np.float32),
    )
    save_probe_outputs(result, output_dir)
    assert (tmp_path / "probe_cross" / "cross_domain.json").exists()

    with open(str(tmp_path / "probe_cross" / "cross_domain.json")) as f:
        cross = json.load(f)
    assert cross["auc_roc"] == 0.75
    assert cross["train_dataset"] == "test"
    assert cross["test_dataset"] == "cross-test"


# --- DB storage round-trip test ---


def test_db_probe_storage(tmp_path):
    from neurotrace.storage import TraceDB

    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)

    # Create a mock result-like object
    class MockResult:
        dataset_name = "capitals"
        model_name = "test-model"
        layer = 20
        extraction_point = "pre_mlp"
        num_clean = 41
        num_sabotaged = 9
        cohens_d = 1.5
        auc_roc = 0.85
        probe_accuracy = 0.8
        direction_alignment = 0.92
        cross_dataset = "science_symbols"
        cross_auc_roc = 0.72

    db.save_probe("test-probe-id", MockResult())

    # Verify it was saved
    row = db._conn.execute(
        "SELECT * FROM probes WHERE id = ?", ["test-probe-id"]
    ).fetchone()
    assert row is not None
    assert row[1] == "capitals"
    assert row[3] == 20
    assert row[4] == "pre_mlp"

    db.close()


# --- CLI tests ---


def test_probe_cli_requires_dataset():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["probe", "--db", "test.db", "--model", "test-model"],
    )
    assert result.exit_code != 0
    assert (
        "dataset" in result.output.lower()
        or "dataset" in str(result.exception).lower()
    )


def test_probe_cli_rejects_both_dataset_flags():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "probe",
            "--db", "test.db",
            "--model", "test-model",
            "--dataset", "file.json",
            "--dataset-builtin", "capitals",
        ],
    )
    assert result.exit_code != 0


# --- Auto-detect layer test ---


def test_auto_detect_layer():
    from neurotrace.probe import auto_detect_layer

    scan = _make_scan_result(n_clean=10, n_sabo=5)
    layer = auto_detect_layer(scan)
    # Our synthetic data has sabotage_layers=[20, 21]
    # Layer 20 and 21 each appear 5 times, 20 should be picked (first in most_common)
    assert layer in (20, 21)


def test_auto_detect_layer_no_sabotage():
    from neurotrace.probe import auto_detect_layer

    scan = _make_scan_result(n_clean=10, n_sabo=0)
    layer = auto_detect_layer(scan)
    assert layer is None
