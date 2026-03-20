"""Tests for the bench module — no model downloads required."""

import json

import pytest

from neurotrace.bench import (
    BenchRun,
    DomainBenchResult,
    PromptBenchResult,
    bench_run_to_dict,
    generate_bench_html,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt_result(**kwargs) -> PromptBenchResult:
    defaults = dict(
        prompt="The capital of France is",
        answer="Paris",
        baseline_correct=True,
        repaired_correct=True,
        competitor="the",
        before_margin=2.5,
        after_margin=4.0,
        before_prob=0.4,
        after_prob=0.7,
        layer=20,
        component="mlp",
        edit_norm=0.023,
    )
    defaults.update(kwargs)
    return PromptBenchResult(**defaults)


def _make_domain_result(dataset="capitals", **kwargs) -> DomainBenchResult:
    defaults = dict(
        dataset=dataset,
        baseline_correct=34,
        baseline_total=50,
        repaired_correct=48,
        repaired_total=50,
        baseline_ppl=7.82,
        repaired_ppl=7.84,
        edit_count=50,
        total_edit_norm=2.31,
        prompt_results=[_make_prompt_result()],
    )
    defaults.update(kwargs)
    return DomainBenchResult(**defaults)


def _make_bench_run(**kwargs) -> BenchRun:
    defaults = dict(
        run_id="bench-001",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        datasets=["capitals", "science_symbols"],
        baseline_ppl=7.82,
        domain_results=[
            _make_domain_result("capitals"),
            _make_domain_result("science_symbols", baseline_correct=30,
                                repaired_correct=41, repaired_ppl=7.91,
                                edit_count=50, total_edit_norm=3.47),
        ],
        combined_ppl=7.95,
        total_edits=100,
        total_edit_norm=5.78,
        combined_domain_results=[
            _make_domain_result("capitals", repaired_correct=50, repaired_ppl=7.95),
            _make_domain_result("science_symbols", baseline_correct=30,
                                repaired_correct=39, repaired_ppl=7.95),
        ],
        created_at="2026-03-20T12:00:00Z",
    )
    defaults.update(kwargs)
    return BenchRun(**defaults)


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestBenchDataStructures:
    def test_prompt_bench_result(self):
        pr = _make_prompt_result()
        assert pr.prompt == "The capital of France is"
        assert pr.answer == "Paris"
        assert pr.baseline_correct is True
        assert pr.repaired_correct is True
        assert pr.layer == 20

    def test_domain_bench_result_properties(self):
        dr = _make_domain_result()
        assert dr.baseline_acc == pytest.approx(34 / 50)
        assert dr.repaired_acc == pytest.approx(48 / 50)
        assert dr.acc_delta == pytest.approx(14 / 50)
        assert dr.ppl_delta == pytest.approx(0.02, abs=0.01)

    def test_domain_bench_result_zero_total(self):
        dr = _make_domain_result(baseline_total=0, repaired_total=0)
        assert dr.baseline_acc == 0.0
        assert dr.repaired_acc == 0.0

    def test_bench_run(self):
        run = _make_bench_run()
        assert run.run_id == "bench-001"
        assert len(run.domain_results) == 2
        assert len(run.combined_domain_results) == 2
        assert run.total_edits == 100
        assert run.baseline_ppl == pytest.approx(7.82)
        assert run.combined_ppl == pytest.approx(7.95)


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------


class TestBenchSerialization:
    def test_bench_run_to_dict(self):
        run = _make_bench_run()
        d = bench_run_to_dict(run)
        assert d["run_id"] == "bench-001"
        assert d["model_name"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert d["baseline_ppl"] == pytest.approx(7.82)
        assert d["combined_ppl"] == pytest.approx(7.95)
        assert d["total_edits"] == 100
        assert len(d["domain_results"]) == 2
        assert len(d["combined_domain_results"]) == 2

        # Domain results have computed properties
        dr = d["domain_results"][0]
        assert "baseline_acc" in dr
        assert "repaired_acc" in dr
        assert "acc_delta" in dr
        assert "ppl_delta" in dr
        assert len(dr["prompt_results"]) == 1
        assert dr["prompt_results"][0]["prompt"] == "The capital of France is"

    def test_serializable_as_json(self):
        run = _make_bench_run()
        d = bench_run_to_dict(run)
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["run_id"] == "bench-001"
        assert len(parsed["domain_results"]) == 2

    def test_empty_prompt_results(self):
        dr = _make_domain_result(prompt_results=[])
        run = _make_bench_run(domain_results=[dr])
        d = bench_run_to_dict(run)
        assert len(d["domain_results"][0]["prompt_results"]) == 0


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestBenchStorage:
    def test_write_and_list(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_bench_run(
            run_id="b1", model_name="test-model",
            datasets="capitals,science_symbols",
            total_edits=100, baseline_ppl=7.82, combined_ppl=7.95,
        )
        runs = db.list_bench_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == "b1"
        assert runs[0]["model_name"] == "test-model"
        assert runs[0]["datasets"] == "capitals,science_symbols"
        assert runs[0]["total_edits"] == 100
        assert runs[0]["baseline_ppl"] == pytest.approx(7.82)
        assert runs[0]["combined_ppl"] == pytest.approx(7.95)
        db.close()

    def test_write_domain_result(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_bench_run(
            run_id="b1", model_name="test",
            datasets="capitals", total_edits=50,
            baseline_ppl=7.82, combined_ppl=7.84,
        )
        db.write_bench_domain_result(
            run_id="b1", dataset="capitals",
            baseline_correct=34, baseline_total=50,
            repaired_correct=48, repaired_total=50,
            baseline_ppl=7.82, repaired_ppl=7.84,
            edit_count=50, total_edit_norm=2.31,
        )
        rows = db._conn.execute(
            "SELECT * FROM bench_domain_results WHERE run_id = 'b1'"
        ).fetchall()
        assert len(rows) == 1
        db.close()

    def test_write_prompt_result(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_bench_run(
            run_id="b1", model_name="test",
            datasets="capitals", total_edits=1,
            baseline_ppl=7.82, combined_ppl=7.84,
        )
        db.write_bench_prompt_result(
            run_id="b1", dataset="capitals",
            prompt="The capital of France is", answer="Paris",
            baseline_correct=True, repaired_correct=True,
            before_margin=2.5, after_margin=4.0,
            before_prob=0.4, after_prob=0.7,
            layer=20, component="mlp", edit_norm=0.023,
        )
        rows = db._conn.execute(
            "SELECT * FROM bench_prompt_results WHERE run_id = 'b1'"
        ).fetchall()
        assert len(rows) == 1
        db.close()

    def test_multiple_domain_results(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_bench_run(
            run_id="b2", model_name="test",
            datasets="capitals,science_symbols",
            total_edits=100, baseline_ppl=7.82, combined_ppl=7.95,
        )
        for ds in ["capitals", "science_symbols"]:
            db.write_bench_domain_result(
                run_id="b2", dataset=ds,
                baseline_correct=34, baseline_total=50,
                repaired_correct=48, repaired_total=50,
                baseline_ppl=7.82, repaired_ppl=7.84,
                edit_count=50, total_edit_norm=2.31,
            )
        rows = db._conn.execute(
            "SELECT * FROM bench_domain_results WHERE run_id = 'b2'"
        ).fetchall()
        assert len(rows) == 2
        db.close()

    def test_nullable_prompt_fields(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_bench_run(
            run_id="b3", model_name="test",
            datasets="caps", total_edits=0,
            baseline_ppl=7.82, combined_ppl=7.82,
        )
        db.write_bench_prompt_result(
            run_id="b3", dataset="caps",
            prompt="test", answer="ans",
            baseline_correct=True, repaired_correct=True,
            before_margin=0.0, after_margin=0.0,
            before_prob=0.0, after_prob=0.0,
            layer=None, component=None, edit_norm=None,
        )
        rows = db._conn.execute(
            "SELECT layer, component, edit_norm FROM bench_prompt_results"
            " WHERE run_id = 'b3'"
        ).fetchall()
        assert rows[0][0] is None
        assert rows[0][1] is None
        assert rows[0][2] is None
        db.close()


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------


class TestBenchHTML:
    def test_basic_report(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "<!DOCTYPE html>" in html
        assert "NeuroTrace Benchmark" in html
        assert "TinyLlama" in html
        assert "Domain Comparison" in html
        assert "capitals" in html
        assert "science_symbols" in html

    def test_report_has_perplexity_chart(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "<svg" in html
        assert "Perplexity" in html

    def test_report_has_combined_section(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "Combined Repair" in html

    def test_report_has_key_findings(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "Key Findings" in html
        assert "ROME repair effective" in html
        assert "Perplexity impact" in html

    def test_report_collapsible_details(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "<details>" in html
        assert "<summary>" in html

    def test_empty_results(self):
        run = _make_bench_run(
            domain_results=[],
            combined_domain_results=[],
        )
        html = generate_bench_html(run)
        assert "<!DOCTYPE html>" in html
        assert "NeuroTrace Benchmark" in html

    def test_report_no_js(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "<script" not in html

    def test_report_inline_css(self):
        run = _make_bench_run()
        html = generate_bench_html(run)
        assert "<style>" in html
        assert '<link rel="stylesheet"' not in html


# ---------------------------------------------------------------------------
# Perplexity computation tests
# ---------------------------------------------------------------------------


class TestPerplexity:
    def test_compute_perplexity_local_fallback(self):
        """Test perplexity uses fallback text when datasets unavailable."""
        from neurotrace.bench import compute_perplexity_local

        # We can't test with a real model without download,
        # but we can verify the function imports correctly
        assert callable(compute_perplexity_local)

    def test_scan_accuracy_import(self):
        from neurotrace.bench import scan_accuracy

        assert callable(scan_accuracy)


# ---------------------------------------------------------------------------
# CLI integration test (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBenchCLI:
    def test_bench_json_output(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "bench",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--datasets", "capitals",
            "--json",
            "--ppl-samples", "2",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output.strip())
        assert "run_id" in data
        assert "domain_results" in data
        assert len(data["domain_results"]) == 1
        assert data["domain_results"][0]["dataset"] == "capitals"

    def test_bench_html_output(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "bench.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "bench",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--datasets", "capitals",
            "--html", html_path,
            "--ppl-samples", "2",
        ])
        assert result.exit_code == 0, result.output
        with open(html_path) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "NeuroTrace Benchmark" in html

    def test_bench_db_persistence(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        runner.invoke(cli, [
            "bench",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--datasets", "capitals",
            "--ppl-samples", "2",
        ])

        db = TraceDB(db_path)
        runs = db.list_bench_runs()
        assert len(runs) == 1
        assert "capitals" in runs[0]["datasets"]

        domain_rows = db._conn.execute(
            "SELECT * FROM bench_domain_results"
        ).fetchall()
        assert len(domain_rows) >= 1

        prompt_rows = db._conn.execute(
            "SELECT * FROM bench_prompt_results"
        ).fetchall()
        assert len(prompt_rows) >= 1
        db.close()

    def test_bench_requires_model_or_remote(self, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "bench",
            "--db", db_path,
        ])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "remote" in result.output.lower()
