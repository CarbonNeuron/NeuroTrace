"""Tests for the heatmap module — no model downloads required."""

import json
import os
import tempfile

import pytest

from neurotrace.heatmap import (
    HeatmapCell,
    HeatmapResult,
    _compute_flip_direction,
    build_layer_summaries,
    check_correct,
    generate_heatmap_html,
    heatmap_result_from_dict,
    heatmap_result_to_dict,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_cell(
    prompt_idx,
    layer,
    flip_direction="none",
    baseline_correct=True,
    ablated_correct=True,
):
    return HeatmapCell(
        prompt_index=prompt_idx,
        prompt=f"prompt {prompt_idx}",
        expected_answer="Paris",
        layer=layer,
        baseline_token="Paris",
        baseline_prob=0.5,
        baseline_correct=baseline_correct,
        ablated_token="Paris" if flip_direction == "none" else "London",
        ablated_prob=0.4,
        ablated_correct=ablated_correct,
        delta_correct_prob=-0.1,
        flipped=flip_direction in ("fixed", "broke"),
        flip_direction=flip_direction,
    )


def _make_result(cells=None, num_prompts=2, num_layers=3):
    """Build a small HeatmapResult for testing."""
    if cells is None:
        cells = [
            _make_cell(p, li)
            for p in range(num_prompts)
            for li in range(num_layers)
        ]
    summaries = build_layer_summaries(cells, num_layers, num_prompts)
    return HeatmapResult(
        run_id="test-run-001",
        dataset_name="test-dataset",
        model_name="test-model",
        num_layers=num_layers,
        num_prompts=num_prompts,
        cells=cells,
        layer_summaries=summaries,
        created_at="2026-03-20T00:00:00",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckCorrect:
    def test_exact_match(self):
        assert check_correct("Paris", "Paris") is True

    def test_prediction_is_prefix(self):
        assert check_correct("Par", "Paris") is True

    def test_answer_is_prefix(self):
        assert check_correct("Paris", "Par") is True

    def test_no_match(self):
        assert check_correct("London", "Paris") is False

    def test_empty_prediction(self):
        assert check_correct("", "Paris") is False

    def test_empty_answer(self):
        assert check_correct("Paris", "") is False

    def test_both_empty(self):
        assert check_correct("", "") is False

    def test_case_insensitive(self):
        assert check_correct("paris", "Paris") is True


class TestComputeFlipDirection:
    def test_fixed(self):
        result = _compute_flip_direction(
            baseline_correct=False,
            ablated_correct=True,
            baseline_token="London",
            ablated_token="Paris",
        )
        assert result == "fixed"

    def test_broke(self):
        result = _compute_flip_direction(
            baseline_correct=True,
            ablated_correct=False,
            baseline_token="Paris",
            ablated_token="London",
        )
        assert result == "broke"

    def test_changed(self):
        result = _compute_flip_direction(
            baseline_correct=False,
            ablated_correct=False,
            baseline_token="London",
            ablated_token="Berlin",
        )
        assert result == "changed"

    def test_none(self):
        result = _compute_flip_direction(
            baseline_correct=True,
            ablated_correct=True,
            baseline_token="Paris",
            ablated_token="Paris",
        )
        assert result == "none"


class TestHeatmapCellCreation:
    def test_all_fields(self):
        cell = _make_cell(
            0, 5, flip_direction="broke",
            baseline_correct=True, ablated_correct=False,
        )
        assert cell.prompt_index == 0
        assert cell.prompt == "prompt 0"
        assert cell.expected_answer == "Paris"
        assert cell.layer == 5
        assert cell.baseline_token == "Paris"
        assert cell.baseline_prob == 0.5
        assert cell.baseline_correct is True
        assert cell.ablated_token == "London"
        assert cell.ablated_prob == 0.4
        assert cell.ablated_correct is False
        assert cell.delta_correct_prob == pytest.approx(-0.1)
        assert cell.flipped is True
        assert cell.flip_direction == "broke"


class TestBuildLayerSummaries:
    def test_counts_and_impact(self):
        cells = [
            # prompt 0
            _make_cell(
                0, 0, flip_direction="fixed",
                baseline_correct=False, ablated_correct=True,
            ),
            _make_cell(
                0, 1, flip_direction="broke",
                baseline_correct=True, ablated_correct=False,
            ),
            _make_cell(0, 2, flip_direction="none"),
            # prompt 1
            _make_cell(
                1, 0, flip_direction="fixed",
                baseline_correct=False, ablated_correct=True,
            ),
            _make_cell(
                1, 1, flip_direction="changed",
                baseline_correct=False, ablated_correct=False,
            ),
            _make_cell(1, 2, flip_direction="none"),
        ]
        summaries = build_layer_summaries(cells, num_layers=3, num_prompts=2)

        assert len(summaries) == 3

        # Layer 0: 2 fixes, 0 breaks
        s0 = summaries[0]
        assert s0.layer == 0
        assert s0.fixes == 2
        assert s0.breaks == 0
        assert s0.changes == 0
        assert s0.unchanged == 0
        assert s0.total_prompts == 2
        assert s0.impact_score == pytest.approx(1.0)

        # Layer 1: 0 fixes, 1 break, 1 changed
        s1 = summaries[1]
        assert s1.layer == 1
        assert s1.fixes == 0
        assert s1.breaks == 1
        assert s1.changes == 1
        assert s1.unchanged == 0
        assert s1.impact_score == pytest.approx(-0.5)

        # Layer 2: 0 fixes, 0 breaks, 2 unchanged
        s2 = summaries[2]
        assert s2.layer == 2
        assert s2.fixes == 0
        assert s2.breaks == 0
        assert s2.unchanged == 2
        assert s2.impact_score == pytest.approx(0.0)


class TestHeatmapImpactScore:
    def test_impact_formula(self):
        """impact_score = (fixes - breaks) / total"""
        cells = [
            _make_cell(
                0, 0, flip_direction="fixed",
                baseline_correct=False, ablated_correct=True,
            ),
            _make_cell(
                1, 0, flip_direction="broke",
                baseline_correct=True, ablated_correct=False,
            ),
            _make_cell(
                2, 0, flip_direction="broke",
                baseline_correct=True, ablated_correct=False,
            ),
            _make_cell(3, 0, flip_direction="none"),
        ]
        summaries = build_layer_summaries(cells, num_layers=1, num_prompts=4)
        s = summaries[0]
        # (1 fix - 2 breaks) / 4 total = -0.25
        assert s.impact_score == pytest.approx((1 - 2) / 4)


class TestHeatmapJsonRoundtrip:
    def test_roundtrip(self):
        original = _make_result()
        d = heatmap_result_to_dict(original)
        restored = heatmap_result_from_dict(d)

        assert restored.run_id == original.run_id
        assert restored.dataset_name == original.dataset_name
        assert restored.model_name == original.model_name
        assert restored.num_layers == original.num_layers
        assert restored.num_prompts == original.num_prompts
        assert restored.created_at == original.created_at
        assert len(restored.cells) == len(original.cells)
        assert len(restored.layer_summaries) == len(original.layer_summaries)

        for orig_cell, rest_cell in zip(original.cells, restored.cells):
            assert orig_cell.prompt_index == rest_cell.prompt_index
            assert orig_cell.layer == rest_cell.layer
            assert orig_cell.flip_direction == rest_cell.flip_direction
            assert orig_cell.delta_correct_prob == pytest.approx(
                rest_cell.delta_correct_prob,
            )

        for orig_s, rest_s in zip(original.layer_summaries, restored.layer_summaries):
            assert orig_s.layer == rest_s.layer
            assert orig_s.impact_score == pytest.approx(rest_s.impact_score)


class TestHeatmapHtmlStructure:
    def test_expected_elements(self):
        result = _make_result()
        html = generate_heatmap_html(result)

        assert "MLP Ablation Heatmap" in html
        assert "<svg" in html
        assert "</svg>" in html
        # Legend items
        assert "Fixed" in html
        assert "Broke" in html
        assert "Changed" in html
        assert "No change" in html
        # Dataset name should appear
        assert "test-dataset" in html


class TestHeatmapDbRoundtrip:
    def test_write_and_read(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)

            result = _make_result()
            cells_json = json.dumps(heatmap_result_to_dict(result)["cells"])
            summaries_json = json.dumps(
                heatmap_result_to_dict(result)["layer_summaries"],
            )

            db.write_heatmap_run(
                run_id=result.run_id,
                dataset_name=result.dataset_name,
                model_name=result.model_name,
                num_layers=result.num_layers,
                num_prompts=result.num_prompts,
                cells_json=cells_json,
                summaries_json=summaries_json,
            )

            row = db.read_heatmap_run(result.run_id)

            assert row["run_id"] == result.run_id
            assert row["dataset_name"] == result.dataset_name
            assert row["model_name"] == result.model_name
            assert row["num_layers"] == result.num_layers
            assert row["num_prompts"] == result.num_prompts
            assert row["created_at"] is not None

            # Verify cells round-trip through JSON
            stored_cells = json.loads(row["cells"])
            assert len(stored_cells) == len(result.cells)
            assert stored_cells[0]["prompt_index"] == result.cells[0].prompt_index

            # Verify summaries round-trip through JSON
            stored_summaries = json.loads(row["summaries"])
            assert len(stored_summaries) == len(result.layer_summaries)
            assert stored_summaries[0]["layer"] == result.layer_summaries[0].layer

            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
