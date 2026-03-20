"""Tests for the commitment module — no model downloads required."""

import json
import os
import tempfile

import pytest

from neurotrace.commitment import (
    CommitmentResult,
    CommitmentRun,
    build_commitment_result,
    commitment_run_to_dict,
    compute_commitment,
    generate_commitment_html,
    validate_against_heatmap,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    prompt="The capital of France is",
    answer="Paris",
    trajectory=None,
    threshold=0.7,
) -> CommitmentResult:
    if trajectory is None:
        trajectory = [0.01, 0.05, 0.2, 0.5, 0.91, 0.88, 0.85]
    return build_commitment_result(prompt, answer, trajectory, threshold)


def _make_run(results=None, threshold=0.7) -> CommitmentRun:
    if results is None:
        results = [
            _make_result(
                "The capital of France is",
                "Paris",
                [0.01, 0.05, 0.2, 0.5, 0.91, 0.88, 0.85],
            ),
            _make_result(
                "The capital of Germany is",
                "Berlin",
                [0.01, 0.03, 0.1, 0.2, 0.44, 0.30, 0.20],
            ),
        ]
    return CommitmentRun(
        run_id="test-run-001",
        dataset_name="test-dataset",
        model_name="test-model",
        num_layers=7,
        num_prompts=len(results),
        threshold=threshold,
        results=results,
        created_at="2026-03-20T00:00:00",
    )


# ---------------------------------------------------------------------------
# Tests: compute_commitment
# ---------------------------------------------------------------------------


class TestComputeCommitment:
    def test_basic_robust(self):
        """High recovery ratio -> robust."""
        traj = [0.01, 0.05, 0.2, 0.5, 0.91, 0.88, 0.85]
        m = compute_commitment(traj, threshold=0.7)
        assert m["peak_prob"] == pytest.approx(0.91)
        assert m["peak_layer"] == 4
        assert m["final_prob"] == pytest.approx(0.85)
        assert m["commitment_score"] == pytest.approx(0.91)
        assert m["recovery_ratio"] == pytest.approx(0.85 / 0.91)
        assert m["vulnerable"] is False  # 0.934 > 0.7

    def test_basic_vulnerable(self):
        """Low recovery ratio -> vulnerable."""
        traj = [0.01, 0.03, 0.1, 0.2, 0.44, 0.30, 0.20]
        m = compute_commitment(traj, threshold=0.7)
        assert m["peak_prob"] == pytest.approx(0.44)
        assert m["peak_layer"] == 4
        assert m["final_prob"] == pytest.approx(0.20)
        assert m["recovery_ratio"] == pytest.approx(0.20 / 0.44)
        assert m["vulnerable"] is True  # 0.454 < 0.7

    def test_empty_trajectory(self):
        m = compute_commitment([], threshold=0.7)
        assert m["peak_prob"] == 0.0
        assert m["vulnerable"] is True

    def test_flat_trajectory(self):
        """All same probability."""
        traj = [0.5, 0.5, 0.5]
        m = compute_commitment(traj, threshold=0.7)
        assert m["peak_prob"] == pytest.approx(0.5)
        assert m["recovery_ratio"] == pytest.approx(1.0)
        assert m["vulnerable"] is False

    def test_zero_peak(self):
        """Zero peak -> vulnerable, recovery_ratio = 0."""
        traj = [0.0, 0.0, 0.0]
        m = compute_commitment(traj, threshold=0.7)
        assert m["peak_prob"] == 0.0
        assert m["recovery_ratio"] == 0.0
        assert m["vulnerable"] is True

    def test_threshold_boundary(self):
        """Exactly at threshold -> not vulnerable (>= is robust when == threshold)."""
        # recovery_ratio = 0.7 / 1.0 = 0.7, threshold = 0.7, 0.7 < 0.7 is False
        traj = [1.0, 0.7]
        m = compute_commitment(traj, threshold=0.7)
        assert m["recovery_ratio"] == pytest.approx(0.7)
        assert m["vulnerable"] is False

    def test_just_below_threshold(self):
        traj = [1.0, 0.69]
        m = compute_commitment(traj, threshold=0.7)
        assert m["vulnerable"] is True


# ---------------------------------------------------------------------------
# Tests: build_commitment_result
# ---------------------------------------------------------------------------


class TestBuildCommitmentResult:
    def test_fields(self):
        r = _make_result()
        assert r.prompt == "The capital of France is"
        assert r.answer == "Paris"
        assert len(r.trajectory) == 7
        assert r.peak_prob == pytest.approx(0.91)
        assert r.peak_layer == 4
        assert r.commitment_score == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestCommitmentJson:
    def test_run_to_dict(self):
        run = _make_run()
        d = commitment_run_to_dict(run)
        assert d["run_id"] == "test-run-001"
        assert d["dataset_name"] == "test-dataset"
        assert d["num_prompts"] == 2
        assert d["n_vulnerable"] == 1
        assert d["n_robust"] == 1
        assert len(d["results"]) == 2
        assert "trajectory" in d["results"][0]

    def test_json_serializable(self):
        run = _make_run()
        d = commitment_run_to_dict(run)
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["n_vulnerable"] == 1


# ---------------------------------------------------------------------------
# Tests: HTML generation
# ---------------------------------------------------------------------------


class TestCommitmentHtml:
    def test_structure(self):
        run = _make_run()
        html = generate_commitment_html(run)
        assert "Commitment Analysis" in html
        assert "test-dataset" in html
        assert "test-model" in html
        assert "VULNERABLE" in html
        assert "ROBUST" in html
        assert "<table" in html
        assert "data-sort" in html

    def test_with_validation(self):
        run = _make_run()
        run.validation = {
            "tp": 1,
            "fp": 0,
            "tn": 1,
            "fn": 0,
            "accuracy": 1.0,
            "total_matched": 2,
            "auc_roc": 1.0,
        }
        html = generate_commitment_html(run)
        assert "Heatmap Cross-Validation" in html
        assert "100.0%" in html


# ---------------------------------------------------------------------------
# Tests: Storage
# ---------------------------------------------------------------------------


class TestCommitmentStorage:
    def test_write_and_read_run(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)

            run = _make_run()
            run_dict = commitment_run_to_dict(run)

            db.write_commitment_run(
                run_id=run.run_id,
                dataset_name=run.dataset_name,
                model_name=run.model_name,
                n_prompts=run.num_prompts,
                n_vulnerable=run_dict["n_vulnerable"],
                n_robust=run_dict["n_robust"],
                threshold=run.threshold,
                avg_commitment_score=run_dict["avg_commitment_score"],
            )

            # Write results
            for r in run.results:
                db.write_commitment_result(
                    run_id=run.run_id,
                    prompt=r.prompt,
                    answer=r.answer,
                    peak_prob=r.peak_prob,
                    peak_layer=r.peak_layer,
                    final_prob=r.final_prob,
                    recovery_ratio=r.recovery_ratio,
                    vulnerable=r.vulnerable,
                    trajectory=json.dumps(r.trajectory),
                )

            # Read back
            row = db.read_commitment_run(run.run_id)
            assert row["run_id"] == run.run_id
            assert row["dataset_name"] == run.dataset_name
            assert row["n_vulnerable"] == 1
            assert row["n_robust"] == 1

            results = db.read_commitment_results(run.run_id)
            assert len(results) == 2
            assert results[0]["prompt"] == "The capital of France is"

            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_list_runs(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)
            db.write_commitment_run(
                run_id="run-1",
                dataset_name="capitals",
                model_name="test-model",
                n_prompts=10,
                n_vulnerable=3,
                n_robust=7,
                threshold=0.7,
                avg_commitment_score=0.6,
            )
            runs = db.list_commitment_runs()
            assert len(runs) == 1
            assert runs[0]["run_id"] == "run-1"
            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Tests: Heatmap cross-validation
# ---------------------------------------------------------------------------


class TestValidateAgainstHeatmap:
    def test_basic_validation(self):
        results = [
            _make_result(
                "The capital of France is",
                "Paris",
                [0.01, 0.05, 0.2, 0.5, 0.91, 0.88, 0.85],
            ),
            _make_result(
                "The capital of Germany is",
                "Berlin",
                [0.01, 0.03, 0.1, 0.2, 0.44, 0.30, 0.20],
            ),
        ]
        # Paris is robust (no breaks), Berlin is vulnerable (broke at layer 15)
        heatmap_cells = json.dumps([
            {
                "prompt": "The capital of France is",
                "layer": 15,
                "flip_direction": "none",
            },
            {
                "prompt": "The capital of Germany is",
                "layer": 15,
                "flip_direction": "broke",
            },
        ])

        v = validate_against_heatmap(results, heatmap_cells)
        # Paris: predicted robust, actual robust -> TN
        # Berlin: predicted vulnerable, actual vulnerable -> TP
        assert v["tp"] == 1
        assert v["tn"] == 1
        assert v["fp"] == 0
        assert v["fn"] == 0
        assert v["accuracy"] == pytest.approx(1.0)

    def test_no_matching_prompts(self):
        results = [_make_result("unmatched prompt", "X", [0.5])]
        v = validate_against_heatmap(results, "[]")
        assert v["total_matched"] == 0
        assert v["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Tests: CLI output (via Click test runner)
# ---------------------------------------------------------------------------


class TestCommitmentCli:
    def test_commitment_command_exists(self):
        """Verify the commitment command is registered."""
        from neurotrace.cli import cli

        commands = cli.list_commands(ctx=None)
        assert "commitment" in commands
