"""Tests for the decompose module — no model downloads required."""

import json
import os
import tempfile

import pytest

from neurotrace.decompose import (
    DecomposeResult,
    DecomposeRun,
    LayerContribution,
    TokenDecomposition,
    build_decompose_result,
    decompose_result_to_dict,
    decompose_run_to_dict,
    generate_decompose_html,
    run_decompose_remote,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer_contribs(n_layers=4, attn_base=0.1, mlp_base=-0.05):
    return [
        LayerContribution(
            layer=i,
            attention=attn_base * (i + 1),
            mlp=mlp_base * (i + 1),
        )
        for i in range(n_layers)
    ]


def _make_decomposition(
    token_str="Berlin",
    token_id=12345,
    final_logit=12.71,
    embedding=0.83,
    n_layers=4,
) -> TokenDecomposition:
    layers = _make_layer_contribs(n_layers)
    return TokenDecomposition(
        token_str=token_str,
        token_id=token_id,
        final_logit=final_logit,
        embedding=embedding,
        layers=layers,
        reconstruction_error=0.002,
        norm_scale=14.23,
    )


def _make_result(
    prompt="The capital of Germany is",
    answer="Berlin",
    competitor="located",
) -> DecomposeResult:
    a = _make_decomposition(answer, 12345, 12.71, 0.83)
    c = _make_decomposition(competitor, 67890, 11.45, 0.12)
    decompositions = {answer: a, competitor: c}
    return build_decompose_result(prompt, answer, competitor, decompositions)


def _make_run(results=None, n_prompts=None) -> DecomposeRun:
    if results is None:
        results = [_make_result()]
    return DecomposeRun(
        run_id="test-run-001",
        dataset="test-dataset",
        model_name="test-model",
        num_prompts=n_prompts if n_prompts is not None else len(results),
        results=results,
        created_at="2026-03-20T00:00:00",
    )


# ---------------------------------------------------------------------------
# Tests: Data structures
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_layer_contribution(self):
        lc = LayerContribution(layer=0, attention=0.5, mlp=-0.3)
        assert lc.layer == 0
        assert lc.attention == 0.5
        assert lc.mlp == -0.3

    def test_token_decomposition(self):
        d = _make_decomposition()
        assert d.token_str == "Berlin"
        assert d.token_id == 12345
        assert d.final_logit == 12.71
        assert d.embedding == 0.83
        assert len(d.layers) == 4
        assert d.reconstruction_error == 0.002
        assert d.norm_scale == 14.23

    def test_decompose_result(self):
        r = _make_result()
        assert r.prompt == "The capital of Germany is"
        assert r.answer == "Berlin"
        assert r.competitor == "located"
        assert r.answer_logit == 12.71
        assert r.competitor_logit == 11.45
        assert r.margin == pytest.approx(12.71 - 11.45)
        assert r.embedding_margin == pytest.approx(0.83 - 0.12)
        assert len(r.component_margins) == 4

    def test_decompose_run(self):
        run = _make_run()
        assert run.run_id == "test-run-001"
        assert run.num_prompts == 1
        assert len(run.results) == 1


# ---------------------------------------------------------------------------
# Tests: build_decompose_result
# ---------------------------------------------------------------------------


class TestBuildDecomposeResult:
    def test_margin_computation(self):
        r = _make_result()
        # margin = answer_logit - competitor_logit
        assert r.margin == pytest.approx(12.71 - 11.45)

    def test_embedding_margin(self):
        r = _make_result()
        assert r.embedding_margin == pytest.approx(0.83 - 0.12)

    def test_component_margins(self):
        r = _make_result()
        for cm in r.component_margins:
            assert "layer" in cm
            assert "attn_margin" in cm
            assert "mlp_margin" in cm

    def test_reconstruction_error_is_max(self):
        a = _make_decomposition("Berlin", 1, 10.0, 0.5)
        a.reconstruction_error = 0.005
        c = _make_decomposition("located", 2, 9.0, 0.3)
        c.reconstruction_error = 0.008
        r = build_decompose_result(
            "test", "Berlin", "located", {"Berlin": a, "located": c},
        )
        assert r.reconstruction_error == 0.008


# ---------------------------------------------------------------------------
# Tests: Remote runner
# ---------------------------------------------------------------------------


class TestRemoteRunner:
    def test_run_decompose_remote(self):
        remote_data = {
            "Berlin": {
                "token_id": 12345,
                "final_logit": 12.71,
                "embedding": 0.83,
                "layers": [
                    {"layer": 0, "attention": 0.12, "mlp": -0.45},
                    {"layer": 1, "attention": 0.08, "mlp": -0.23},
                ],
                "reconstruction_error": 0.002,
                "norm_scale": 14.23,
            },
            "located": {
                "token_id": 67890,
                "final_logit": 11.45,
                "embedding": 0.12,
                "layers": [
                    {"layer": 0, "attention": 0.05, "mlp": 0.22},
                    {"layer": 1, "attention": 0.03, "mlp": 0.15},
                ],
                "reconstruction_error": 0.003,
                "norm_scale": 14.23,
            },
        }
        r = run_decompose_remote(
            remote_data, "The capital of Germany is", "Berlin", "located",
        )
        assert r.answer == "Berlin"
        assert r.competitor == "located"
        assert r.margin == pytest.approx(12.71 - 11.45)
        assert len(r.component_margins) == 2
        assert r.reconstruction_error == 0.003


# ---------------------------------------------------------------------------
# Tests: JSON serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_result_to_dict(self):
        r = _make_result()
        d = decompose_result_to_dict(r)
        assert d["answer"] == "Berlin"
        assert d["competitor"] == "located"
        assert "answer_decomposition" in d
        assert "competitor_decomposition" in d
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_run_to_dict(self):
        run = _make_run()
        d = decompose_run_to_dict(run)
        assert d["run_id"] == "test-run-001"
        assert len(d["results"]) == 1
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_roundtrip_json(self):
        run = _make_run()
        d = decompose_run_to_dict(run)
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["run_id"] == "test-run-001"
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["margin"] == pytest.approx(12.71 - 11.45)


# ---------------------------------------------------------------------------
# Tests: HTML generation
# ---------------------------------------------------------------------------


class TestHtmlGeneration:
    def test_single_prompt_html(self):
        run = _make_run()
        html = generate_decompose_html(run)
        assert "<!DOCTYPE html>" in html
        assert "Logit Prism" in html
        assert "test-dataset" in html
        assert "Berlin" in html
        assert "located" in html

    def test_batch_html(self):
        results = [
            _make_result("Capital of France is", "Paris", "the"),
            _make_result("Capital of Germany is", "Berlin", "located"),
        ]
        run = _make_run(results, n_prompts=2)
        html = generate_decompose_html(run)
        assert "<!DOCTYPE html>" in html
        assert "Margin Heatmap" in html
        assert "Paris" in html
        assert "Berlin" in html

    def test_html_escapes_special_chars(self):
        r = _make_result(prompt="Test <script>alert('xss')</script>")
        run = _make_run([r])
        html = generate_decompose_html(run)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_no_external_resources(self):
        run = _make_run()
        html = generate_decompose_html(run)
        assert "http://" not in html
        assert "https://" not in html


# ---------------------------------------------------------------------------
# Tests: Storage
# ---------------------------------------------------------------------------


class TestStorage:
    def test_write_and_read_run(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)
            db.write_decompose_run(
                run_id="run-001",
                dataset="capitals",
                model_name="test-model",
                prompt_count=5,
            )
            row = db.read_decompose_run("run-001")
            assert row["run_id"] == "run-001"
            assert row["dataset"] == "capitals"
            assert row["model_name"] == "test-model"
            assert row["prompt_count"] == 5
            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_write_and_read_results(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)
            db.write_decompose_run(
                run_id="run-002",
                dataset="test",
                model_name="test-model",
                prompt_count=1,
            )
            r = _make_result()
            db.write_decompose_result(
                run_id="run-002",
                prompt=r.prompt,
                answer=r.answer,
                competitor=r.competitor,
                answer_logit=r.answer_logit,
                competitor_logit=r.competitor_logit,
                margin=r.margin,
                embedding_margin=r.embedding_margin,
                component_json=json.dumps(r.component_margins),
                reconstruction_error=r.reconstruction_error,
            )
            results = db.read_decompose_results("run-002")
            assert len(results) == 1
            assert results[0]["answer"] == "Berlin"
            assert results[0]["competitor"] == "located"
            assert results[0]["margin"] == pytest.approx(12.71 - 11.45)
            # Component JSON is valid
            components = json.loads(results[0]["component_json"])
            assert len(components) == 4
            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_list_decompose_runs(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)
            db.write_decompose_run("r1", "ds1", "m1", 3)
            db.write_decompose_run("r2", "ds2", "m2", 5)
            runs = db.list_decompose_runs()
            assert len(runs) == 2
            run_ids = {r["run_id"] for r in runs}
            assert "r1" in run_ids
            assert "r2" in run_ids
            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_read_missing_run_raises(self):
        tmp = tempfile.mktemp(suffix=".db")
        try:
            db = TraceDB(tmp)
            with pytest.raises(ValueError, match="not found"):
                db.read_decompose_run("nonexistent")
            db.close()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ---------------------------------------------------------------------------
# Tests: CLI (with model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDecomposeCLI:
    def test_single_prompt_json(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--competitors", "the,a",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "run_id" in data
        assert len(data["results"]) >= 1
        # Check reconstruction error is small
        for r in data["results"]:
            assert r["reconstruction_error"] < 1.0

    def test_single_prompt_console(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--competitors", "the",
        ])
        assert result.exit_code == 0, result.output

    def test_auto_competitors(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert len(data["results"]) >= 1

    def test_html_output(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "decompose.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--competitors", "the",
            "--html", html_path,
        ])
        assert result.exit_code == 0, result.output
        assert os.path.exists(html_path)
        with open(html_path) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html

    def test_db_persistence(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--competitors", "the",
        ])
        assert result.exit_code == 0, result.output

        # Verify DB has the data
        db = TraceDB(db_path)
        runs = db.list_decompose_runs()
        assert len(runs) == 1
        results = db.read_decompose_results(runs[0]["run_id"])
        assert len(results) >= 1
        db.close()

    def test_validation_errors(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()

        # Missing prompt and dataset
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
        ])
        assert result.exit_code != 0

        # Missing answer with prompt
        result = runner.invoke(cli, [
            "decompose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "test",
        ])
        assert result.exit_code != 0
