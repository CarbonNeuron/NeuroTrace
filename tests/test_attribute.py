"""Tests for the attribute module — no model downloads required."""

import json
import os
import tempfile

import numpy as np
import pytest

from neurotrace.attribute import (
    AttributionResult,
    AttributionRun,
    TokenAttribution,
    aggregate_attributions,
    attribution_run_from_dict,
    attribution_run_to_dict,
    build_attribution_result,
    compute_ablation_attribution,
    compute_gradient_attribution,
    generate_attribution_html,
    normalize_attributions,
    run_attribution_ablation_remote,
    run_attribution_gradient_remote,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token_attribution(position=0, token="hello", attribution=0.5):
    return TokenAttribution(position=position, token=token, attribution=attribution)


def _make_result(
    prompt="The capital of Germany is",
    layer=20,
    target_token="located",
    target_token_id=5765,
    method="gradient",
    token_attributions=None,
) -> AttributionResult:
    if token_attributions is None:
        token_attributions = [
            TokenAttribution(position=0, token="The", attribution=0.023),
            TokenAttribution(position=1, token="capital", attribution=0.412),
            TokenAttribution(position=2, token="of", attribution=0.089),
            TokenAttribution(position=3, token="Germany", attribution=0.356),
            TokenAttribution(position=4, token="is", attribution=0.120),
        ]
    return AttributionResult(
        prompt=prompt,
        layer=layer,
        target_token=target_token,
        target_token_id=target_token_id,
        method=method,
        token_attributions=token_attributions,
    )


def _make_run(
    results=None, method="gradient", target_direction="competitor",
) -> AttributionRun:
    if results is None:
        results = [_make_result()]
    return AttributionRun(
        run_id="test-run-001",
        layer=20,
        target_direction=target_direction,
        method=method,
        model_name="test-model",
        dataset="capitals",
        results=results,
        created_at="2026-03-20T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Test normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_normalize_sums_to_one(self):
        result = normalize_attributions([1.0, 2.0, 3.0, 4.0])
        assert sum(result) == pytest.approx(1.0)

    def test_normalize_preserves_order(self):
        result = normalize_attributions([1.0, 3.0, 2.0])
        assert result[1] > result[2] > result[0]

    def test_normalize_zero_vector(self):
        result = normalize_attributions([0.0, 0.0, 0.0])
        assert len(result) == 3
        # Uniform when all zeros
        assert all(v == pytest.approx(1.0 / 3) for v in result)

    def test_normalize_empty(self):
        result = normalize_attributions([])
        assert result == []

    def test_normalize_single(self):
        result = normalize_attributions([5.0])
        assert result == [pytest.approx(1.0)]

    def test_normalize_handles_negatives(self):
        result = normalize_attributions([-1.0, 2.0, -3.0])
        assert sum(result) == pytest.approx(1.0)
        # Uses abs values: 1, 2, 3 -> 1/6, 2/6, 3/6
        assert result[2] > result[1] > result[0]


# ---------------------------------------------------------------------------
# Test gradient attribution
# ---------------------------------------------------------------------------


class TestGradientAttribution:
    def test_basic_gradient(self):
        grad_magnitudes = np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float32)
        result = compute_gradient_attribution(grad_magnitudes)
        assert len(result) == 4
        assert sum(result) == pytest.approx(1.0)
        # Token at position 1 has highest gradient
        assert result[1] == max(result)

    def test_zero_gradients(self):
        grad_magnitudes = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = compute_gradient_attribution(grad_magnitudes)
        assert len(result) == 3
        assert all(v == pytest.approx(1.0 / 3) for v in result)


# ---------------------------------------------------------------------------
# Test ablation attribution
# ---------------------------------------------------------------------------


class TestAblationAttribution:
    def test_basic_ablation(self):
        baseline = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Removing token 0 shifts delta significantly
        ablated_deltas = [
            np.array([0.0, 2.0, 3.0], dtype=np.float32),  # big change in proj
            np.array([1.0, 1.9, 3.0], dtype=np.float32),  # small change
            np.array([0.8, 2.0, 3.0], dtype=np.float32),  # medium change
        ]

        result = compute_ablation_attribution(baseline, ablated_deltas, target_dir)
        assert len(result) == 3
        assert sum(result) == pytest.approx(1.0)
        # Token 0 removal caused biggest projection change
        assert result[0] == max(result)

    def test_no_change(self):
        baseline = np.array([1.0, 2.0], dtype=np.float32)
        target_dir = np.array([1.0, 0.0], dtype=np.float32)

        # All ablated are identical to baseline
        ablated = [baseline.copy(), baseline.copy()]
        result = compute_ablation_attribution(baseline, ablated, target_dir)
        assert len(result) == 2
        # Both should be uniform
        assert result[0] == pytest.approx(result[1])

    def test_single_token(self):
        baseline = np.array([1.0, 0.0], dtype=np.float32)
        target_dir = np.array([1.0, 0.0], dtype=np.float32)
        ablated = [np.array([0.0, 0.0], dtype=np.float32)]
        result = compute_ablation_attribution(baseline, ablated, target_dir)
        assert result == [pytest.approx(1.0)]


# ---------------------------------------------------------------------------
# Test build_attribution_result
# ---------------------------------------------------------------------------


class TestBuildAttributionResult:
    def test_basic_build(self):
        result = build_attribution_result(
            prompt="Hello world",
            tokens=["Hello", " world"],
            attributions=[0.6, 0.4],
            layer=10,
            target_token="test",
            target_token_id=42,
            method="gradient",
        )
        assert result.prompt == "Hello world"
        assert result.layer == 10
        assert len(result.token_attributions) == 2
        assert result.token_attributions[0].token == "Hello"
        assert result.token_attributions[0].attribution == 0.6
        assert result.token_attributions[1].position == 1

    def test_empty_tokens(self):
        result = build_attribution_result(
            prompt="", tokens=[], attributions=[],
            layer=0, target_token="x", target_token_id=0, method="ablation",
        )
        assert len(result.token_attributions) == 0


# ---------------------------------------------------------------------------
# Test aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_single_result(self):
        result = _make_result()
        agg = aggregate_attributions([result])
        assert "capital" in agg
        assert agg["capital"]["avg_attribution"] == pytest.approx(0.412)
        assert agg["capital"]["count"] == 1

    def test_multiple_results(self):
        r1 = _make_result(prompt="P1")
        r2 = _make_result(prompt="P2")
        agg = aggregate_attributions([r1, r2])
        assert "capital" in agg
        assert agg["capital"]["count"] == 2
        assert agg["capital"]["avg_attribution"] == pytest.approx(0.412)

    def test_empty_results(self):
        agg = aggregate_attributions([])
        assert agg == {}


# ---------------------------------------------------------------------------
# Test remote runners
# ---------------------------------------------------------------------------


class TestRemoteRunners:
    def test_gradient_remote(self):
        grad_mags = np.array([0.1, 0.5, 0.2, 0.2], dtype=np.float32)
        result = run_attribution_gradient_remote(
            grad_mags,
            tokens=["The", "cat", "sat", "down"],
            prompt="The cat sat down",
            layer=10,
            target_token_id=100,
            target_token="test",
        )
        assert result.method == "gradient"
        assert len(result.token_attributions) == 4
        total = sum(
            ta.attribution for ta in result.token_attributions
        )
        assert total == pytest.approx(1.0)
        # "cat" should have highest
        assert result.token_attributions[1].attribution == max(
            ta.attribution for ta in result.token_attributions
        )

    def test_ablation_remote(self):
        baseline = {"mlp_input": np.zeros(4), "mlp_output": np.array([1.0, 0, 0, 0])}
        ablated = [
            {"mlp_input": np.zeros(4), "mlp_output": np.array([0.0, 0, 0, 0])},
            {"mlp_input": np.zeros(4), "mlp_output": np.array([0.9, 0, 0, 0])},
        ]
        lm_head = np.eye(4, dtype=np.float32)

        result = run_attribution_ablation_remote(
            baseline, ablated, lm_head,
            tokens=["A", "B"],
            prompt="A B",
            layer=5,
            target_token_id=0,
            target_token="tok",
        )
        assert result.method == "ablation"
        assert len(result.token_attributions) == 2
        # Removing token 0 caused bigger change (1.0 -> 0.0) vs (1.0 -> 0.9)
        a0 = result.token_attributions[0].attribution
        a1 = result.token_attributions[1].attribution
        assert a0 > a1


# ---------------------------------------------------------------------------
# Test JSON serialization
# ---------------------------------------------------------------------------


class TestAttributionJson:
    def test_roundtrip(self):
        run = _make_run()
        d = attribution_run_to_dict(run)
        restored = attribution_run_from_dict(d)

        assert restored.run_id == run.run_id
        assert restored.layer == run.layer
        assert restored.method == run.method
        assert len(restored.results) == len(run.results)

        for orig, rest in zip(run.results, restored.results):
            assert orig.prompt == rest.prompt
            assert len(orig.token_attributions) == len(rest.token_attributions)

    def test_json_serializable(self):
        run = _make_run()
        d = attribution_run_to_dict(run)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_roundtrip_preserves_scores(self):
        run = _make_run()
        d = attribution_run_to_dict(run)
        restored = attribution_run_from_dict(d)
        orig_ta = run.results[0].token_attributions[1]
        rest_ta = restored.results[0].token_attributions[1]
        assert orig_ta.attribution == rest_ta.attribution
        assert orig_ta.token == rest_ta.token
        assert orig_ta.position == rest_ta.position


# ---------------------------------------------------------------------------
# Test HTML generation
# ---------------------------------------------------------------------------


class TestAttributionHtml:
    def test_html_structure(self):
        run = _make_run()
        html = generate_attribution_html(run)
        assert "<!DOCTYPE html>" in html
        assert "Token Attribution Analysis" in html
        assert "gradient" in html

    def test_html_has_prompt(self):
        run = _make_run()
        html = generate_attribution_html(run)
        assert "capital of Germany" in html
        assert "located" in html

    def test_html_has_svg(self):
        run = _make_run()
        html = generate_attribution_html(run)
        assert "<svg" in html

    def test_html_with_multiple_prompts(self):
        r1 = _make_result(prompt="P1")
        r2 = _make_result(prompt="P2")
        run = _make_run(results=[r1, r2])
        html = generate_attribution_html(run)
        assert "P1" in html
        assert "P2" in html
        assert "Aggregated Token Attribution" in html

    def test_html_top_driver(self):
        run = _make_run()
        html = generate_attribution_html(run)
        assert "Top driver" in html
        assert "capital" in html


# ---------------------------------------------------------------------------
# Test storage operations
# ---------------------------------------------------------------------------


class TestAttributionStorage:
    def test_write_and_read_run(self):
        db_path = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(db_path)
            db.write_attribution_run(
                run_id="test-001",
                layer=20,
                target_direction="competitor",
                method="gradient",
                model_name="test-model",
                dataset="capitals",
            )
            run = db.read_attribution_run("test-001")
            assert run["run_id"] == "test-001"
            assert run["layer"] == 20
            assert run["method"] == "gradient"
            assert run["model_name"] == "test-model"
            assert run["dataset"] == "capitals"
            db.close()
        finally:
            os.unlink(db_path)

    def test_write_attribution_result(self):
        db_path = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(db_path)
            db.write_attribution_run(
                run_id="test-002",
                layer=20,
                target_direction="answer",
                method="ablation",
                model_name="test-model",
            )
            db.write_attribution_result(
                run_id="test-002",
                prompt="The capital of France is",
                token_position=1,
                token_text="capital",
                attribution_score=0.412,
                target_token="Paris",
                target_token_id=3681,
            )
            results = db.read_attribution_results("test-002")
            assert len(results) == 1
            assert results[0]["token_text"] == "capital"
            assert results[0]["attribution_score"] == pytest.approx(0.412)
            db.close()
        finally:
            os.unlink(db_path)

    def test_list_attribution_runs(self):
        db_path = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(db_path)
            db.write_attribution_run(
                run_id="r1", layer=10,
                target_direction="answer", method="gradient",
                model_name="m1",
            )
            db.write_attribution_run(
                run_id="r2", layer=20,
                target_direction="competitor", method="ablation",
                model_name="m1",
            )
            runs = db.list_attribution_runs()
            assert len(runs) == 2
            db.close()
        finally:
            os.unlink(db_path)

    def test_read_missing_run(self):
        db_path = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(db_path)
            with pytest.raises(ValueError, match="not found"):
                db.read_attribution_run("nonexistent")
            db.close()
        finally:
            os.unlink(db_path)

    def test_multiple_results_per_run(self):
        db_path = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(db_path)
            db.write_attribution_run(
                run_id="test-003", layer=20,
                target_direction="answer", method="gradient",
                model_name="test-model",
            )
            for i, (tok, score) in enumerate([
                ("The", 0.023), ("capital", 0.412),
                ("of", 0.089), ("Germany", 0.356),
            ]):
                db.write_attribution_result(
                    run_id="test-003",
                    prompt="The capital of Germany is",
                    token_position=i,
                    token_text=tok,
                    attribution_score=score,
                    target_token="located",
                    target_token_id=5765,
                )
            results = db.read_attribution_results("test-003")
            assert len(results) == 4
            assert results[1]["token_text"] == "capital"
            assert results[1]["attribution_score"] == pytest.approx(0.412)
            db.close()
        finally:
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test CLI command exists
# ---------------------------------------------------------------------------


class TestAttributeCli:
    def test_attribute_command_registered(self):
        from neurotrace.cli import cli

        assert "attribute" in [cmd for cmd in cli.commands]

    def test_attribute_help(self):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["attribute", "--help"])
        assert result.exit_code == 0
        assert "input tokens" in result.output
        assert "--layer" in result.output
        assert "--target" in result.output
        assert "--method" in result.output
