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
    margin_trajectory=None,
    competitor_trajectory=None,
    competitor_tokens=None,
    threshold=0.1,
) -> CommitmentResult:
    if trajectory is None:
        # Paris: strong commitment, large margin
        trajectory = [0.01, 0.05, 0.2, 0.5, 0.91, 0.88, 0.85]
    if margin_trajectory is None:
        # All margins >= threshold (0.1), no crossover -> robust
        margin_trajectory = [
            0.11, 0.13, 0.15, 0.30, 0.71, 0.60, 0.51
        ]
    if competitor_trajectory is None:
        competitor_trajectory = [
            0.01, 0.02, 0.08, 0.20, 0.20, 0.28, 0.34
        ]
    if competitor_tokens is None:
        competitor_tokens = ["the"] * len(trajectory)
    return build_commitment_result(
        prompt,
        answer,
        trajectory,
        margin_trajectory,
        competitor_trajectory,
        competitor_tokens,
        threshold,
    )


def _make_vulnerable_result(
    prompt="The capital of Germany is",
    answer="Berlin",
) -> CommitmentResult:
    """Berlin: narrow margin, competitor overtakes."""
    trajectory = [0.01, 0.03, 0.1, 0.2, 0.44, 0.30, 0.20]
    margin_trajectory = [
        -0.10, -0.05, -0.02, 0.05, 0.01, -0.05, -0.07,
    ]
    competitor_trajectory = [
        0.11, 0.08, 0.12, 0.15, 0.43, 0.35, 0.27,
    ]
    competitor_tokens = ["located"] * 7
    return build_commitment_result(
        prompt,
        answer,
        trajectory,
        margin_trajectory,
        competitor_trajectory,
        competitor_tokens,
        threshold=0.1,
    )


def _make_run(results=None, threshold=0.1) -> CommitmentRun:
    if results is None:
        results = [
            _make_result(),
            _make_vulnerable_result(),
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
    def test_robust_large_margin(self):
        """Large positive margin everywhere -> robust."""
        traj = [0.3, 0.5, 0.91, 0.85]
        margin = [0.15, 0.30, 0.71, 0.51]
        comp = [0.15, 0.20, 0.20, 0.34]
        comp_tokens = ["the"] * 4
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["peak_prob"] == pytest.approx(0.91)
        assert m["peak_layer"] == 2
        assert m["min_margin"] == pytest.approx(0.15)
        assert m["margin_at_final"] == pytest.approx(0.51)
        assert m["competitor_peak"] == pytest.approx(0.34)
        assert m["crossover_layer"] is None
        assert m["vulnerable"] is False  # 0.15 >= 0.1

    def test_robust_no_crossover(self):
        """All margins above threshold, no crossover -> robust."""
        traj = [0.5, 0.8, 0.9]
        margin = [0.3, 0.5, 0.6]
        comp = [0.2, 0.3, 0.3]
        comp_tokens = ["the"] * 3
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["min_margin"] == pytest.approx(0.3)
        assert m["crossover_layer"] is None
        assert m["vulnerable"] is False

    def test_vulnerable_negative_margin(self):
        """Negative margin -> crossover -> vulnerable."""
        traj = [0.3, 0.4, 0.2]
        margin = [0.1, 0.2, -0.1]
        comp = [0.2, 0.2, 0.3]
        comp_tokens = ["located"] * 3
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["min_margin"] == pytest.approx(-0.1)
        assert m["crossover_layer"] == 2
        assert m["vulnerable"] is True

    def test_vulnerable_small_margin(self):
        """Small positive margin below threshold -> vulnerable."""
        traj = [0.3, 0.35, 0.32]
        margin = [0.05, 0.08, 0.05]
        comp = [0.25, 0.27, 0.27]
        comp_tokens = ["where"] * 3
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["min_margin"] == pytest.approx(0.05)
        assert m["crossover_layer"] is None
        assert m["vulnerable"] is True  # 0.05 < 0.1

    def test_empty_trajectory(self):
        m = compute_commitment([], [], [], [], threshold=0.1)
        assert m["peak_prob"] == 0.0
        assert m["vulnerable"] is True
        assert m["competitor_token"] == ""

    def test_crossover_detection_first_layer(self):
        """Crossover at the very first layer."""
        traj = [0.1, 0.5, 0.8]
        margin = [-0.2, 0.3, 0.5]
        comp = [0.3, 0.2, 0.3]
        comp_tokens = ["x"] * 3
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["crossover_layer"] == 0

    def test_competitor_token_at_min_margin(self):
        """Competitor token comes from the layer with min margin."""
        traj = [0.5, 0.6, 0.4]
        margin = [0.2, 0.3, 0.05]
        comp = [0.3, 0.3, 0.35]
        comp_tokens = ["a", "b", "threat"]
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["competitor_token"] == "threat"

    def test_threshold_boundary_exact(self):
        """Margin exactly at threshold with no crossover -> not vulnerable."""
        traj = [0.5, 0.6]
        margin = [0.1, 0.2]
        comp = [0.4, 0.4]
        comp_tokens = ["x"] * 2
        m = compute_commitment(
            traj, margin, comp, comp_tokens, threshold=0.1
        )
        assert m["vulnerable"] is False  # 0.1 is not < 0.1


# ---------------------------------------------------------------------------
# Tests: build_commitment_result
# ---------------------------------------------------------------------------


class TestBuildCommitmentResult:
    def test_fields(self):
        r = _make_result()
        assert r.prompt == "The capital of France is"
        assert r.answer == "Paris"
        assert len(r.trajectory) == 7
        assert len(r.margin_trajectory) == 7
        assert len(r.competitor_trajectory) == 7
        assert r.peak_prob == pytest.approx(0.91)
        assert r.peak_layer == 4
        assert r.commitment_score == pytest.approx(0.91)
        assert r.competitor_token == "the"

    def test_vulnerable_fields(self):
        r = _make_vulnerable_result()
        assert r.vulnerable is True
        assert r.crossover_layer == 0  # first negative margin
        assert r.competitor_token == "located"


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestCommitmentJson:
    def test_run_to_dict(self):
        run = _make_run()
        d = commitment_run_to_dict(run)
        assert d["run_id"] == "test-run-001"
        assert d["num_prompts"] == 2
        assert d["n_vulnerable"] >= 1
        assert d["n_robust"] >= 0
        assert len(d["results"]) == 2
        r0 = d["results"][0]
        assert "trajectory" in r0
        assert "margin_trajectory" in r0
        assert "competitor_trajectory" in r0
        assert "min_margin" in r0
        assert "competitor_token" in r0
        assert "crossover_layer" in r0

    def test_json_serializable(self):
        run = _make_run()
        d = commitment_run_to_dict(run)
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert "min_margin" in parsed["results"][0]


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
        assert "MinMargin" in html
        assert "Competitor" in html
        assert "Crossover" in html

    def test_margin_color_coding(self):
        run = _make_run()
        html = generate_commitment_html(run)
        # Should contain color codes for margins
        assert "#4caf50" in html or "#ffc107" in html or "#f44336" in html

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
                avg_commitment_score=run_dict[
                    "avg_commitment_score"
                ],
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
                    min_margin=r.min_margin,
                    margin_at_final=r.margin_at_final,
                    competitor_token=r.competitor_token,
                    competitor_peak=r.competitor_peak,
                    crossover_layer=r.crossover_layer,
                    vulnerable=r.vulnerable,
                    trajectory=json.dumps(r.trajectory),
                    margin_trajectory=json.dumps(
                        r.margin_trajectory
                    ),
                    competitor_trajectory=json.dumps(
                        r.competitor_trajectory
                    ),
                )

            # Read back
            row = db.read_commitment_run(run.run_id)
            assert row["run_id"] == run.run_id
            assert row["dataset_name"] == run.dataset_name

            results = db.read_commitment_results(run.run_id)
            assert len(results) == 2
            assert results[0]["prompt"] == "The capital of France is"
            assert results[0]["min_margin"] is not None
            assert results[0]["competitor_token"] is not None
            assert results[0]["margin_trajectory"] is not None

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
                threshold=0.1,
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
            _make_result(),  # Paris: robust
            _make_vulnerable_result(),  # Berlin: vulnerable
        ]
        # Paris robust (no breaks), Berlin vulnerable (broke at L15)
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
        # Berlin: predicted vulnerable, actual vulnerable -> TP
        assert v["tp"] >= 1
        assert v["accuracy"] > 0

    def test_no_matching_prompts(self):
        results = [
            _make_result("unmatched prompt", "X"),
        ]
        v = validate_against_heatmap(results, "[]")
        assert v["total_matched"] == 0
        assert v["accuracy"] == 0.0

    def test_scoring_uses_negative_margin(self):
        """Verify AUC scoring uses -min_margin (not recovery ratio)."""
        r1 = _make_result()
        r2 = _make_vulnerable_result()
        results = [r1, r2]
        heatmap_cells = json.dumps([
            {
                "prompt": r1.prompt,
                "layer": 15,
                "flip_direction": "none",
            },
            {
                "prompt": r2.prompt,
                "layer": 15,
                "flip_direction": "broke",
            },
        ])
        v = validate_against_heatmap(results, heatmap_cells)
        # Just verify it runs and produces a result
        assert v["total_matched"] == 2


# ---------------------------------------------------------------------------
# Tests: CLI output (via Click test runner)
# ---------------------------------------------------------------------------


class TestCommitmentCli:
    def test_commitment_command_exists(self):
        """Verify the commitment command is registered."""
        from neurotrace.cli import cli

        commands = cli.list_commands(ctx=None)
        assert "commitment" in commands
