"""Tests for the contrast module — no model downloads required."""

import json
import os
import tempfile

import numpy as np
import pytest

from neurotrace.contrast import (
    ContrastCell,
    ContrastDomainSummary,
    ContrastResult,
    analyze_vulnerable_vs_robust,
    build_domain_summaries,
    compute_answer_projection,
    compute_cosine_similarity,
    compute_delta_norm,
    compute_domain_centroid,
    compute_mlp_delta,
    contrast_result_from_dict,
    contrast_result_to_dict,
    generate_contrast_html,
    get_answer_direction,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cell(
    domain="capitals",
    prompt="The capital of France is",
    answer="Paris",
    layer=20,
    mlp_delta_norm=14.32,
    answer_projection=-0.23,
    competitor_projection=0.41,
    competitor_token="located",
) -> ContrastCell:
    return ContrastCell(
        domain=domain,
        prompt=prompt,
        answer=answer,
        layer=layer,
        mlp_delta_norm=mlp_delta_norm,
        answer_projection=answer_projection,
        competitor_projection=competitor_projection,
        competitor_token=competitor_token,
    )


def _make_result(
    domains=None, layers=None, cells=None, summaries=None,
) -> ContrastResult:
    if domains is None:
        domains = ["capitals", "math_simple"]
    if layers is None:
        layers = [0, 20]
    if cells is None:
        cells = [
            _make_cell(domain="capitals", layer=0, mlp_delta_norm=10.0),
            _make_cell(domain="capitals", layer=20, mlp_delta_norm=14.32),
            _make_cell(domain="math_simple", layer=0, mlp_delta_norm=7.0,
                       answer_projection=0.01, competitor_projection=-0.02,
                       prompt="2 + 2 =", answer="4"),
            _make_cell(domain="math_simple", layer=20, mlp_delta_norm=8.71,
                       answer_projection=0.01, competitor_projection=-0.02,
                       prompt="2 + 2 =", answer="4"),
        ]
    if summaries is None:
        summaries = [
            ContrastDomainSummary(
                domain="capitals", layer=0,
                mean_delta_norm=10.0, std_delta_norm=1.0,
                mean_answer_proj=-0.20, mean_competitor_proj=0.35,
                cosine_similarities={"capitals": 1.0, "math_simple": 0.12},
            ),
            ContrastDomainSummary(
                domain="capitals", layer=20,
                mean_delta_norm=14.32, std_delta_norm=2.1,
                mean_answer_proj=-0.23, mean_competitor_proj=0.41,
                cosine_similarities={"capitals": 1.0, "math_simple": 0.12},
            ),
            ContrastDomainSummary(
                domain="math_simple", layer=0,
                mean_delta_norm=7.0, std_delta_norm=0.8,
                mean_answer_proj=0.01, mean_competitor_proj=-0.02,
                cosine_similarities={"capitals": 0.12, "math_simple": 1.0},
            ),
            ContrastDomainSummary(
                domain="math_simple", layer=20,
                mean_delta_norm=8.71, std_delta_norm=1.3,
                mean_answer_proj=0.01, mean_competitor_proj=-0.02,
                cosine_similarities={"capitals": 0.12, "math_simple": 1.0},
            ),
        ]
    return ContrastResult(
        run_id="test-run-001",
        domains=domains,
        layers=layers,
        model_name="test-model",
        cells=cells,
        summaries=summaries,
        vulnerable_vs_robust=None,
        created_at="2026-03-20T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Test delta magnitude computation
# ---------------------------------------------------------------------------


class TestDeltaMagnitude:
    def test_compute_mlp_delta(self):
        mlp_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mlp_out = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        delta = compute_mlp_delta(mlp_in, mlp_out)
        np.testing.assert_allclose(delta, [1.0, 2.0, 3.0])

    def test_compute_delta_norm(self):
        delta = np.array([3.0, 4.0], dtype=np.float32)
        assert compute_delta_norm(delta) == pytest.approx(5.0)

    def test_zero_delta(self):
        mlp_in = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        delta = compute_mlp_delta(mlp_in, mlp_in)
        assert compute_delta_norm(delta) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert compute_cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert compute_cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([-1.0, -2.0], dtype=np.float32)
        assert compute_cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        assert compute_cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test answer/competitor projection
# ---------------------------------------------------------------------------


class TestProjection:
    def test_aligned_projection(self):
        delta = np.array([1.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        assert compute_answer_projection(delta, direction) == pytest.approx(1.0)

    def test_anti_aligned_projection(self):
        delta = np.array([1.0, 0.0], dtype=np.float32)
        direction = np.array([-1.0, 0.0], dtype=np.float32)
        assert compute_answer_projection(delta, direction) == pytest.approx(-1.0)

    def test_orthogonal_projection(self):
        delta = np.array([1.0, 0.0], dtype=np.float32)
        direction = np.array([0.0, 1.0], dtype=np.float32)
        assert compute_answer_projection(delta, direction) == pytest.approx(0.0)

    def test_zero_delta_projection(self):
        delta = np.array([0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0], dtype=np.float32)
        assert compute_answer_projection(delta, direction) == pytest.approx(0.0)

    def test_get_answer_direction_normalized(self):
        weights = np.array([
            [3.0, 4.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        direction = get_answer_direction(weights, 0)
        assert np.linalg.norm(direction) == pytest.approx(1.0)
        np.testing.assert_allclose(direction, [0.6, 0.8], atol=1e-6)


# ---------------------------------------------------------------------------
# Test domain centroid
# ---------------------------------------------------------------------------


class TestDomainCentroid:
    def test_single_delta(self):
        deltas = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        centroid = compute_domain_centroid(deltas)
        np.testing.assert_allclose(centroid, [1.0, 2.0, 3.0])

    def test_multiple_deltas(self):
        deltas = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]
        centroid = compute_domain_centroid(deltas)
        np.testing.assert_allclose(centroid, [2.0, 2.0])

    def test_empty_deltas(self):
        centroid = compute_domain_centroid([])
        assert centroid.shape == (1,)


# ---------------------------------------------------------------------------
# Test vulnerable vs robust split
# ---------------------------------------------------------------------------


class TestVulnerableVsRobust:
    def test_with_heatmap_data(self):
        cells = [
            _make_cell(domain="capitals", prompt="P1", layer=20, mlp_delta_norm=15.0),
            _make_cell(domain="capitals", prompt="P2", layer=20, mlp_delta_norm=10.0),
        ]
        heatmap_cells = [
            {"prompt": "P1", "layer": 20, "flip_direction": "broke"},
            {"prompt": "P1", "layer": 0, "flip_direction": "none"},
            {"prompt": "P2", "layer": 20, "flip_direction": "none"},
            {"prompt": "P2", "layer": 0, "flip_direction": "none"},
        ]
        domain_deltas = {
            ("capitals", 20, "P1"): np.array([1.0, 2.0], dtype=np.float32),
            ("capitals", 20, "P2"): np.array([1.0, 2.0], dtype=np.float32),
        }
        result = analyze_vulnerable_vs_robust(
            cells, json.dumps(heatmap_cells), "capitals", [20], domain_deltas,
        )
        assert result["status"] == "ok"
        assert "20" in result["layers"]
        layer_info = result["layers"]["20"]
        assert layer_info["n_vulnerable"] == 1
        assert layer_info["n_robust"] == 1

    def test_insufficient_data(self):
        heatmap_cells = [
            {"prompt": "P1", "layer": 20, "flip_direction": "none"},
        ]
        result = analyze_vulnerable_vs_robust(
            [], json.dumps(heatmap_cells), "capitals", [20], {},
        )
        assert result["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Test build_domain_summaries
# ---------------------------------------------------------------------------


class TestBuildDomainSummaries:
    def test_basic_summaries(self):
        cells = [
            _make_cell(domain="d1", layer=0, mlp_delta_norm=10.0,
                       answer_projection=-0.2, competitor_projection=0.3),
            _make_cell(domain="d1", layer=0, mlp_delta_norm=12.0,
                       answer_projection=-0.1, competitor_projection=0.4,
                       prompt="P2"),
            _make_cell(domain="d2", layer=0, mlp_delta_norm=5.0,
                       answer_projection=0.1, competitor_projection=-0.1,
                       prompt="P3"),
        ]
        centroids = {
            ("d1", 0): np.array([1.0, 0.0], dtype=np.float32),
            ("d2", 0): np.array([0.0, 1.0], dtype=np.float32),
        }
        summaries = build_domain_summaries(cells, ["d1", "d2"], [0], centroids)
        assert len(summaries) == 2

        d1_summary = summaries[0]
        assert d1_summary.domain == "d1"
        assert d1_summary.mean_delta_norm == pytest.approx(11.0)
        assert d1_summary.mean_answer_proj == pytest.approx(-0.15)
        assert d1_summary.cosine_similarities["d2"] == pytest.approx(0.0)

    def test_empty_domain(self):
        summaries = build_domain_summaries([], ["d1"], [0], {})
        assert len(summaries) == 1
        assert summaries[0].mean_delta_norm == 0.0


# ---------------------------------------------------------------------------
# Test JSON serialization
# ---------------------------------------------------------------------------


class TestContrastJson:
    def test_roundtrip(self):
        result = _make_result()
        d = contrast_result_to_dict(result)
        restored = contrast_result_from_dict(d)

        assert restored.run_id == result.run_id
        assert restored.domains == result.domains
        assert restored.layers == result.layers
        assert len(restored.cells) == len(result.cells)
        assert len(restored.summaries) == len(result.summaries)

        for orig, rest in zip(result.cells, restored.cells):
            assert orig.domain == rest.domain
            assert orig.mlp_delta_norm == rest.mlp_delta_norm

    def test_json_serializable(self):
        result = _make_result()
        d = contrast_result_to_dict(result)
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ---------------------------------------------------------------------------
# Test HTML generation
# ---------------------------------------------------------------------------


class TestContrastHtml:
    def test_html_structure(self):
        result = _make_result()
        html = generate_contrast_html(result)
        assert "<!DOCTYPE html>" in html
        assert "Domain Geometry Contrast" in html
        assert "capitals" in html
        assert "math_simple" in html

    def test_html_has_layers(self):
        result = _make_result()
        html = generate_contrast_html(result)
        assert "Layer 0" in html
        assert "Layer 20" in html

    def test_html_has_heatmap(self):
        result = _make_result()
        html = generate_contrast_html(result)
        assert "<svg" in html
        assert "MLP Delta Magnitude Heatmap" in html

    def test_html_with_vuln_data(self):
        result = _make_result()
        result.vulnerable_vs_robust = {
            "status": "ok",
            "layers": {
                "20": {
                    "cosine_similarity": 0.94,
                    "vulnerable_mean_norm": 15.0,
                    "robust_mean_norm": 14.0,
                    "n_vulnerable": 5,
                    "n_robust": 10,
                }
            },
        }
        html = generate_contrast_html(result)
        assert "Vulnerable vs Robust" in html
        assert "0.94" in html


# ---------------------------------------------------------------------------
# Test storage operations
# ---------------------------------------------------------------------------


class TestContrastStorage:
    def test_write_and_read_contrast_run(self):
        db_path = tempfile.mktemp(suffix=".db")

        try:
            db = TraceDB(db_path)
            db.write_contrast_run(
                run_id="test-001",
                domains=json.dumps(["capitals", "math_simple"]),
                layers=json.dumps([0, 20]),
                model_name="test-model",
            )
            run = db.read_contrast_run("test-001")
            assert run["run_id"] == "test-001"
            assert run["model_name"] == "test-model"
            db.close()
        finally:
            os.unlink(db_path)

    def test_write_contrast_result(self):
        db_path = tempfile.mktemp(suffix=".db")

        try:
            db = TraceDB(db_path)
            db.write_contrast_run(
                run_id="test-002",
                domains=json.dumps(["capitals"]),
                layers=json.dumps([20]),
                model_name="test-model",
            )
            db.write_contrast_result(
                run_id="test-002",
                domain="capitals",
                layer=20,
                prompt="The capital of France is",
                answer="Paris",
                mlp_delta_norm=14.32,
                answer_projection=-0.23,
                competitor_projection=0.41,
                competitor_token="located",
            )
            # Verify it was written
            row = db._conn.execute(
                "SELECT * FROM contrast_results WHERE run_id = 'test-002'"
            ).fetchone()
            assert row is not None
            assert row[1] == "capitals"  # domain
            db.close()
        finally:
            os.unlink(db_path)

    def test_write_contrast_summary(self):
        db_path = tempfile.mktemp(suffix=".db")

        try:
            db = TraceDB(db_path)
            db.write_contrast_run(
                run_id="test-003",
                domains=json.dumps(["capitals"]),
                layers=json.dumps([20]),
                model_name="test-model",
            )
            db.write_contrast_summary(
                run_id="test-003",
                domain="capitals",
                layer=20,
                mean_delta_norm=14.32,
                std_delta_norm=2.1,
                mean_answer_proj=-0.23,
                mean_competitor_proj=0.41,
                cosine_similarities=json.dumps({"capitals": 1.0}),
            )
            row = db._conn.execute(
                "SELECT * FROM contrast_summaries WHERE run_id = 'test-003'"
            ).fetchone()
            assert row is not None
            db.close()
        finally:
            os.unlink(db_path)

    def test_list_contrast_runs(self):
        db_path = tempfile.mktemp(suffix=".db")

        try:
            db = TraceDB(db_path)
            db.write_contrast_run(
                run_id="r1",
                domains=json.dumps(["capitals"]),
                layers=json.dumps([20]),
                model_name="m1",
            )
            db.write_contrast_run(
                run_id="r2",
                domains=json.dumps(["math_simple"]),
                layers=json.dumps([0]),
                model_name="m1",
            )
            runs = db.list_contrast_runs()
            assert len(runs) == 2
            db.close()
        finally:
            os.unlink(db_path)


