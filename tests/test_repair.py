"""Tests for the repair module — no model downloads required."""

import json

import pytest

from neurotrace.repair import (
    RegressionResult,
    RepairAfter,
    RepairBefore,
    RepairEdit,
    RepairResult,
    RepairRun,
    build_repair_result_from_remote,
    generate_repair_html,
    repair_result_to_dict,
    repair_run_to_dict,
)
from neurotrace.storage import TraceDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repair_result(
    prompt="The capital of Germany is",
    answer="Berlin",
    competitor="the",
    target_layer=20,
    target_component="mlp",
    status="repaired",
    before_margin=2.98,
    after_margin=4.81,
    before_prob=0.44,
    after_prob=0.71,
    edit_norm=0.023,
    regressions=None,
) -> RepairResult:
    return RepairResult(
        prompt=prompt,
        answer=answer,
        competitor=competitor,
        target_layer=target_layer,
        target_component=target_component,
        before=RepairBefore(
            answer_logit=12.71,
            competitor_logit=9.73,
            margin=before_margin,
            component_margin=-2.06,
            answer_prob=before_prob,
        ),
        after=RepairAfter(
            answer_logit=14.22,
            competitor_logit=9.41,
            margin=after_margin,
            component_margin=0.87,
            answer_prob=after_prob,
        ),
        edit=RepairEdit(
            matrix="down_proj",
            layer=target_layer,
            rank=1,
            norm=edit_norm,
        ),
        regressions=regressions or [],
        status=status,
    )


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestRepairDataStructures:
    def test_repair_before(self):
        before = RepairBefore(
            answer_logit=12.71,
            competitor_logit=9.73,
            margin=2.98,
            component_margin=-2.06,
            answer_prob=0.44,
        )
        assert before.margin == pytest.approx(2.98)
        assert before.component_margin == pytest.approx(-2.06)

    def test_repair_after(self):
        after = RepairAfter(
            answer_logit=14.22,
            competitor_logit=9.41,
            margin=4.81,
            component_margin=0.87,
            answer_prob=0.71,
        )
        assert after.margin == pytest.approx(4.81)

    def test_repair_edit(self):
        edit = RepairEdit(matrix="down_proj", layer=20, rank=1, norm=0.023)
        assert edit.matrix == "down_proj"
        assert edit.rank == 1

    def test_regression_result(self):
        reg = RegressionResult(
            prompt="The capital of France is",
            answer="Paris",
            before_prob=0.91,
            after_prob=0.90,
            status="ok",
        )
        assert reg.status == "ok"

    def test_repair_result_defaults(self):
        r = _make_repair_result()
        assert r.status == "repaired"
        assert len(r.regressions) == 0

    def test_repair_result_with_regressions(self):
        regs = [
            RegressionResult("p1", "a1", 0.9, 0.89, "ok"),
            RegressionResult("p2", "a2", 0.8, 0.70, "regression"),
        ]
        r = _make_repair_result(regressions=regs)
        assert len(r.regressions) == 2
        assert r.regressions[1].status == "regression"


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------


class TestRepairSerialization:
    def test_repair_result_to_dict(self):
        r = _make_repair_result()
        d = repair_result_to_dict(r)
        assert d["prompt"] == "The capital of Germany is"
        assert d["answer"] == "Berlin"
        assert d["competitor"] == "the"
        assert d["target_layer"] == 20
        assert d["before"]["margin"] == pytest.approx(2.98)
        assert d["after"]["margin"] == pytest.approx(4.81)
        assert d["edit"]["matrix"] == "down_proj"
        assert d["edit"]["rank"] == 1
        assert d["status"] == "repaired"

        # Must be JSON-serializable
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["edit"]["layer"] == 20

    def test_repair_result_with_regressions_dict(self):
        regs = [
            RegressionResult("p1", "a1", 0.9, 0.89, "ok"),
        ]
        r = _make_repair_result(regressions=regs)
        d = repair_result_to_dict(r)
        assert len(d["regressions"]) == 1
        assert d["regressions"][0]["status"] == "ok"

    def test_repair_run_to_dict(self):
        results = [_make_repair_result(), _make_repair_result(status="skipped")]
        run = RepairRun(
            run_id="r1",
            dataset="capitals",
            model_name="test-model",
            prompt_count=2,
            results=results,
            created_at="2026-01-01T00:00:00Z",
        )
        d = repair_run_to_dict(run)
        assert d["run_id"] == "r1"
        assert d["dataset"] == "capitals"
        assert len(d["results"]) == 2
        assert d["results"][0]["status"] == "repaired"
        assert d["results"][1]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Remote result parsing tests
# ---------------------------------------------------------------------------


class TestBuildFromRemote:
    def test_build_repair_result_from_remote(self):
        data = {
            "prompt": "The capital of Germany is",
            "answer": "Berlin",
            "competitor": "the",
            "target_layer": 20,
            "target_component": "mlp",
            "before": {
                "answer_logit": 12.71,
                "competitor_logit": 9.73,
                "margin": 2.98,
                "component_margin": -2.06,
                "answer_prob": 0.44,
            },
            "after": {
                "answer_logit": 14.22,
                "competitor_logit": 9.41,
                "margin": 4.81,
                "component_margin": 0.87,
                "answer_prob": 0.71,
            },
            "edit": {
                "matrix": "down_proj",
                "layer": 20,
                "rank": 1,
                "norm": 0.0234,
            },
            "regressions": [
                {
                    "prompt": "The capital of France is",
                    "answer": "Paris",
                    "before_prob": 0.91,
                    "after_prob": 0.90,
                    "status": "ok",
                },
            ],
            "status": "repaired",
        }
        r = build_repair_result_from_remote(data)
        assert r.prompt == "The capital of Germany is"
        assert r.answer == "Berlin"
        assert r.before.margin == pytest.approx(2.98)
        assert r.after.margin == pytest.approx(4.81)
        assert r.edit.norm == pytest.approx(0.0234)
        assert len(r.regressions) == 1
        assert r.regressions[0].status == "ok"
        assert r.status == "repaired"

    def test_build_from_remote_no_regressions(self):
        data = {
            "prompt": "test",
            "answer": "ans",
            "competitor": "comp",
            "target_layer": 5,
            "before": {
                "answer_logit": 1.0,
                "competitor_logit": 0.5,
                "margin": 0.5,
                "component_margin": -0.1,
                "answer_prob": 0.3,
            },
            "after": {
                "answer_logit": 1.5,
                "competitor_logit": 0.4,
                "margin": 1.1,
                "component_margin": 0.2,
                "answer_prob": 0.5,
            },
            "edit": {
                "matrix": "down_proj",
                "layer": 5,
                "rank": 1,
                "norm": 0.01,
            },
        }
        r = build_repair_result_from_remote(data)
        assert len(r.regressions) == 0
        assert r.target_component == "mlp"


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestRepairStorage:
    def test_write_and_list(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_repair_run(
            run_id="r1", dataset="capitals",
            model_name="test", prompt_count=5,
        )
        runs = db.list_repair_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == "r1"
        assert runs[0]["dataset"] == "capitals"
        db.close()

    def test_write_results(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_repair_run(
            run_id="r1", dataset=None,
            model_name="test", prompt_count=1,
        )
        db.write_repair_result(
            run_id="r1",
            prompt="The capital of Germany is",
            answer="Berlin",
            competitor="the",
            target_layer=20,
            target_component="mlp",
            before_margin=2.98,
            after_margin=4.81,
            before_prob=0.44,
            after_prob=0.71,
            edit_norm=0.023,
            regressions_checked=50,
            regressions_found=0,
            status="repaired",
        )
        rows = db._conn.execute(
            "SELECT * FROM repair_results WHERE run_id = 'r1'"
        ).fetchall()
        assert len(rows) == 1
        # before_margin is column index 7
        assert rows[0][7] == pytest.approx(2.98, abs=0.01)
        db.close()

    def test_multiple_results(self, tmp_path):
        db = TraceDB(str(tmp_path / "test.db"))
        db.write_repair_run(
            run_id="r2", dataset="capitals",
            model_name="test", prompt_count=3,
        )
        for i, (prompt, answer) in enumerate([
            ("Germany", "Berlin"),
            ("France", "Paris"),
            ("Japan", "Tokyo"),
        ]):
            db.write_repair_result(
                run_id="r2",
                prompt=prompt, answer=answer,
                competitor="the",
                target_layer=20, target_component="mlp",
                before_margin=1.0 + i, after_margin=3.0 + i,
                before_prob=0.3, after_prob=0.7,
                edit_norm=0.02, regressions_checked=0,
                regressions_found=0, status="repaired",
            )
        rows = db._conn.execute(
            "SELECT * FROM repair_results WHERE run_id = 'r2'"
        ).fetchall()
        assert len(rows) == 3
        db.close()


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------


class TestRepairHTML:
    def test_single_report(self):
        run = RepairRun(
            run_id="r1",
            dataset=None,
            model_name="test-model",
            prompt_count=1,
            results=[_make_repair_result()],
            created_at="2026-01-01T00:00:00Z",
        )
        html = generate_repair_html(run)
        assert "<!DOCTYPE html>" in html
        assert "Weight Repair" in html
        assert "Berlin" in html
        assert "down_proj" in html
        assert "Before" in html
        assert "After" in html

    def test_batch_report(self):
        results = [
            _make_repair_result(status="repaired"),
            _make_repair_result(
                prompt="The capital of France is",
                answer="Paris",
                status="skipped",
            ),
            _make_repair_result(
                prompt="The capital of Japan is",
                answer="Tokyo",
                status="regression",
            ),
        ]
        run = RepairRun(
            run_id="r2",
            dataset="capitals",
            model_name="test-model",
            prompt_count=3,
            results=results,
            created_at="2026-01-01T00:00:00Z",
        )
        html = generate_repair_html(run)
        assert "<!DOCTYPE html>" in html
        assert "REPAIRED" in html
        assert "SKIPPED" in html
        assert "REGRESSION" in html

    def test_report_with_regressions(self):
        regs = [
            RegressionResult("p1", "a1", 0.9, 0.89, "ok"),
            RegressionResult("p2", "a2", 0.8, 0.70, "regression"),
        ]
        run = RepairRun(
            run_id="r3",
            dataset=None,
            model_name="test-model",
            prompt_count=1,
            results=[_make_repair_result(regressions=regs)],
            created_at="2026-01-01T00:00:00Z",
        )
        html = generate_repair_html(run)
        assert "Regression Check" in html
        assert "OK" in html
        assert "REGRESSION" in html


# ---------------------------------------------------------------------------
# Diagnose confidence bugfix tests
# ---------------------------------------------------------------------------


class TestDiagnoseConfidenceFix:
    """Verify the updated confidence logic in diagnose.py."""

    def test_strong_signal_high_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.5,
            circuit_type="distributed", total_attention=3.0,
            gatekeeper_present=False,
        )
        supp = SuppressionAnalysis(
            total_mlp=-1.0, subject_mlp=0.0, worst_layer=5,
            worst_layer_value=-0.5, suppression_ratio=0.3,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "strong"
        assert verdict.vulnerability == "robust"
        assert verdict.confidence == "high"

    def test_strong_signal_concentrated_still_high(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.9,
            circuit_type="concentrated", total_attention=3.0,
            gatekeeper_present=True,
        )
        supp = SuppressionAnalysis(
            total_mlp=-1.0, subject_mlp=0.0, worst_layer=5,
            worst_layer_value=-0.5, suppression_ratio=0.3,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "strong"
        assert verdict.confidence == "high"

    def test_absent_signal_high_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.0,
            circuit_type="mixed", total_attention=0.5,
            gatekeeper_present=False,
        )
        supp = SuppressionAnalysis(
            total_mlp=0.0, subject_mlp=0.0, worst_layer=None,
            worst_layer_value=0.0, suppression_ratio=0.0,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "absent"
        assert verdict.confidence == "high"

    def test_weak_concentrated_high_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.9,
            circuit_type="concentrated", total_attention=1.2,
            gatekeeper_present=True,
        )
        supp = SuppressionAnalysis(
            total_mlp=-5.0, subject_mlp=-1.0, worst_layer=20,
            worst_layer_value=-3.0, suppression_ratio=4.2,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "weak"
        assert verdict.vulnerability == "vulnerable"
        assert verdict.confidence == "high"

    def test_moderate_concentrated_medium_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.8,
            circuit_type="concentrated", total_attention=2.0,
            gatekeeper_present=True,
        )
        supp = SuppressionAnalysis(
            total_mlp=-2.0, subject_mlp=0.0, worst_layer=10,
            worst_layer_value=-1.0, suppression_ratio=1.0,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "moderate"
        assert verdict.confidence == "medium"

    def test_moderate_distributed_medium_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.3,
            circuit_type="distributed", total_attention=2.0,
            gatekeeper_present=False,
        )
        supp = SuppressionAnalysis(
            total_mlp=-2.0, subject_mlp=0.0, worst_layer=10,
            worst_layer_value=-1.0, suppression_ratio=1.0,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "moderate"
        assert verdict.confidence == "medium"

    def test_weak_distributed_medium_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.3,
            circuit_type="distributed", total_attention=1.2,
            gatekeeper_present=False,
        )
        supp = SuppressionAnalysis(
            total_mlp=-1.0, subject_mlp=0.0, worst_layer=5,
            worst_layer_value=-0.5, suppression_ratio=0.8,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "weak"
        assert verdict.vulnerability == "vulnerable"
        assert verdict.confidence == "medium"

    def test_moderate_mixed_low_confidence(self):
        from neurotrace.diagnose import (
            CircuitAnalysis,
            SuppressionAnalysis,
            compute_verdict,
        )

        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.5,
            circuit_type="mixed", total_attention=2.0,
            gatekeeper_present=False,
        )
        supp = SuppressionAnalysis(
            total_mlp=-2.0, subject_mlp=0.0, worst_layer=10,
            worst_layer_value=-1.0, suppression_ratio=1.0,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "moderate"
        assert verdict.confidence == "low"


# ---------------------------------------------------------------------------
# CLI integration test (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRepairCLI:
    def test_single_prompt_local(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "repair",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
        ])
        assert result.exit_code == 0, result.output
        assert "Repair" in result.output
        assert "Before" in result.output
        assert "After" in result.output

    def test_single_prompt_json(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "repair",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output.strip())
        assert "results" in data
        assert len(data["results"]) == 1
        assert "before" in data["results"][0]
        assert "after" in data["results"][0]

    def test_single_prompt_html(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "repair.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "repair",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--html", html_path,
        ])
        assert result.exit_code == 0, result.output
        with open(html_path) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "Weight Repair" in html

    def test_db_persistence(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli
        from neurotrace.storage import TraceDB

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        runner.invoke(cli, [
            "repair",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
        ])

        db = TraceDB(db_path)
        runs = db.list_repair_runs()
        assert len(runs) == 1
        assert runs[0]["prompt_count"] == 1
        db.close()

    def test_with_manual_target(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "repair",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--target-layer", "5",
            "--target-component", "mlp",
            "--target-margin", "1.0",
        ])
        assert result.exit_code == 0, result.output
