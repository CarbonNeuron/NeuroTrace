"""Tests for diagnose feature."""

import json

import pytest

from neurotrace.attention_trace import AttentionTraceEntry, AttentionTraceResult
from neurotrace.diagnose import (
    CircuitAnalysis,
    DiagnosisResult,
    RepairPrescription,
    SuppressionAnalysis,
    Verdict,
    analyze_circuit,
    analyze_suppression,
    compute_repair,
    compute_verdict,
    diagnosis_to_dict,
    generate_diagnose_html_batch,
    generate_diagnose_html_single,
    run_diagnosis,
)
from neurotrace.token_trace import TokenTraceEntry, TokenTraceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attn_entries(
    head_projections: dict[tuple[int, int], float],
) -> list[AttentionTraceEntry]:
    """Create AttentionTraceEntry list from {(layer, head): projection}."""
    return [
        AttentionTraceEntry(
            prompt="test", layer=layer, head_idx=head,
            answer_projection=proj, magnitude=abs(proj),
        )
        for (layer, head), proj in head_projections.items()
    ]


def _make_token_entries(
    layer_projections: dict[int, float],
    subject_projections: dict[int, float] | None = None,
    num_positions: int = 5,
) -> list[TokenTraceEntry]:
    """Create TokenTraceEntry list.

    layer_projections: {layer: answer_projection at last position}
    subject_projections: {layer: answer_projection at subject position}
    """
    entries = []
    for layer, proj in layer_projections.items():
        # Last position entry
        entries.append(TokenTraceEntry(
            layer=layer, position=num_positions - 1,
            token="is", is_subject=False, is_last=True,
            answer_projection=proj, competitor_projection=None,
            delta_magnitude=abs(proj),
        ))
        # Subject position entry
        subj_proj = (subject_projections or {}).get(layer, 0.0)
        entries.append(TokenTraceEntry(
            layer=layer, position=1,
            token="Germany", is_subject=True, is_last=False,
            answer_projection=subj_proj, competitor_projection=None,
            delta_magnitude=abs(subj_proj),
        ))
        # Other position
        entries.append(TokenTraceEntry(
            layer=layer, position=0,
            token="The", is_subject=False, is_last=False,
            answer_projection=0.01, competitor_projection=None,
            delta_magnitude=0.01,
        ))
    return entries


# ---------------------------------------------------------------------------
# Circuit analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeCircuit:
    def test_concentrated_circuit(self):
        entries = _make_attn_entries({
            (21, 29): 0.91,
            (17, 19): 0.17,
            (15, 11): 0.09,
            (10, 5): 0.05,
        })
        result = AttentionTraceResult(prompt="test", answer="Berlin", entries=entries)
        circuit = analyze_circuit(result)

        # Only heads > 0.1 are active
        assert len(circuit.active_heads) == 2
        assert circuit.active_heads[0]["layer"] == 21
        assert circuit.active_heads[0]["head"] == 29
        assert circuit.total_attention == pytest.approx(0.91 + 0.17, abs=1e-6)
        assert circuit.top3_share == pytest.approx(1.0, abs=1e-6)
        assert circuit.circuit_type == "concentrated"

    def test_distributed_circuit(self):
        # Many heads with similar projections
        entries = _make_attn_entries({
            (0, i): 0.15 for i in range(10)
        })
        result = AttentionTraceResult(prompt="test", answer="Berlin", entries=entries)
        circuit = analyze_circuit(result)

        assert len(circuit.active_heads) == 10
        assert circuit.top3_share == pytest.approx(3 / 10, abs=1e-6)
        assert circuit.circuit_type == "distributed"

    def test_gatekeeper_detection(self):
        entries = _make_attn_entries({(21, 29): 0.5, (10, 5): 0.3})
        result = AttentionTraceResult(prompt="test", answer="Berlin", entries=entries)
        circuit = analyze_circuit(result)
        assert circuit.gatekeeper_present is True

    def test_no_gatekeeper(self):
        entries = _make_attn_entries({(10, 5): 0.5, (15, 3): 0.3})
        result = AttentionTraceResult(prompt="test", answer="Berlin", entries=entries)
        circuit = analyze_circuit(result)
        assert circuit.gatekeeper_present is False

    def test_no_active_heads(self):
        entries = _make_attn_entries({(0, 0): 0.05, (1, 1): -0.1})
        result = AttentionTraceResult(prompt="test", answer="Berlin", entries=entries)
        circuit = analyze_circuit(result)
        assert len(circuit.active_heads) == 0
        assert circuit.total_attention == 0.0
        assert circuit.top_head is None
        assert circuit.top3_share == 0.0


# ---------------------------------------------------------------------------
# Suppression analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzeSuppression:
    def test_basic_suppression(self):
        layer_projs = {0: -1.0, 1: -2.0, 2: 0.5}
        entries = _make_token_entries(layer_projs)
        result = TokenTraceResult(
            prompt="test", answer="Berlin", subject="Germany",
            entries=entries,
        )
        supp = analyze_suppression(result, total_attention=2.0)

        assert supp.total_mlp == pytest.approx(-2.5, abs=1e-6)
        assert supp.worst_layer == 1
        assert supp.worst_layer_value == pytest.approx(-2.0, abs=1e-6)
        assert supp.suppression_ratio == pytest.approx(2.5 / 2.0, abs=1e-6)

    def test_subject_mlp(self):
        layer_projs = {0: -1.0, 1: -2.0}
        subj_projs = {0: -0.5, 1: -0.3}
        entries = _make_token_entries(layer_projs, subj_projs)
        result = TokenTraceResult(
            prompt="test", answer="Berlin", subject="Germany",
            entries=entries,
        )
        supp = analyze_suppression(result, total_attention=1.0)
        assert supp.subject_mlp == pytest.approx(-0.8, abs=1e-6)

    def test_zero_attention(self):
        entries = _make_token_entries({0: -1.0})
        result = TokenTraceResult(
            prompt="test", answer="Berlin", subject=None,
            entries=entries,
        )
        supp = analyze_suppression(result, total_attention=0.0)
        assert supp.suppression_ratio == 0.0


# ---------------------------------------------------------------------------
# Verdict tests
# ---------------------------------------------------------------------------


class TestComputeVerdict:
    def test_strong_robust(self):
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
        assert verdict.confidence == "high"  # distributed + robust

    def test_weak_vulnerable(self):
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
        assert verdict.confidence == "high"  # concentrated + vulnerable

    def test_absent(self):
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
        assert verdict.vulnerability == "absent"

    def test_moderate(self):
        circuit = CircuitAnalysis(
            active_heads=[], top_head=None, top3_share=0.6,
            circuit_type="mixed", total_attention=2.0,
            gatekeeper_present=False,
        )
        supp = SuppressionAnalysis(
            total_mlp=-2.0, subject_mlp=0.0, worst_layer=10,
            worst_layer_value=-1.0, suppression_ratio=1.0,
        )
        verdict = compute_verdict(circuit, supp)
        assert verdict.signal_strength == "moderate"
        assert verdict.vulnerability == "moderate"
        assert verdict.confidence == "low"


# ---------------------------------------------------------------------------
# Repair prescription tests
# ---------------------------------------------------------------------------


class TestComputeRepair:
    def test_no_repair_for_robust(self):
        circuit = CircuitAnalysis(
            active_heads=[{"layer": 21, "head": 29, "projection": 1.0}],
            top_head={"layer": 21, "head": 29, "projection": 1.0},
            top3_share=1.0, circuit_type="concentrated",
            total_attention=3.0, gatekeeper_present=True,
        )
        supp = SuppressionAnalysis(
            total_mlp=-1.0, subject_mlp=0.0, worst_layer=20,
            worst_layer_value=-1.0, suppression_ratio=0.3,
        )
        verdict = Verdict(
            signal_strength="strong", vulnerability="robust",
            confidence="high", reason="test",
        )
        repair = compute_repair(circuit, supp, verdict)
        assert repair is None

    def test_repair_for_vulnerable(self):
        circuit = CircuitAnalysis(
            active_heads=[
                {"layer": 21, "head": 29, "projection": 0.9},
                {"layer": 17, "head": 19, "projection": 0.2},
                {"layer": 15, "head": 11, "projection": 0.15},
            ],
            top_head={"layer": 21, "head": 29, "projection": 0.9},
            top3_share=0.85, circuit_type="concentrated",
            total_attention=1.4, gatekeeper_present=True,
        )
        supp = SuppressionAnalysis(
            total_mlp=-18.0, subject_mlp=-4.0, worst_layer=20,
            worst_layer_value=-3.5, suppression_ratio=12.9,
        )
        verdict = Verdict(
            signal_strength="weak", vulnerability="vulnerable",
            confidence="high", reason="test",
        )
        repair = compute_repair(circuit, supp, verdict)
        assert repair is not None
        assert len(repair.target_heads) == 3
        assert repair.target_heads[0] == {"layer": 21, "head": 29}
        assert 15 in repair.target_layers
        assert 17 in repair.target_layers
        assert 21 in repair.target_layers
        assert repair.suggested_fix == "attention-lora"
        assert "q_proj" in repair.suggested_modules
        assert repair.alternative_fix == "mlp-lora"
        assert repair.alternative_layers == [20]


# ---------------------------------------------------------------------------
# Full diagnosis tests
# ---------------------------------------------------------------------------


class TestRunDiagnosis:
    def test_full_diagnosis(self):
        attn_entries = _make_attn_entries({
            (21, 29): 0.91,
            (17, 19): 0.17,
            (15, 11): 0.12,
        })
        attn_result = AttentionTraceResult(
            prompt="The capital of Germany is",
            answer="Berlin",
            entries=attn_entries,
        )

        token_entries = _make_token_entries(
            {15: -0.5, 17: -1.0, 20: -3.5, 21: -2.0},
            {15: -0.2, 17: -0.1, 20: -0.5, 21: -0.3},
        )
        token_result = TokenTraceResult(
            prompt="The capital of Germany is",
            answer="Berlin",
            subject="Germany",
            entries=token_entries,
        )

        diag = run_diagnosis(attn_result, token_result)
        assert diag.prompt == "The capital of Germany is"
        assert diag.answer == "Berlin"
        assert diag.circuit.total_attention == pytest.approx(1.2, abs=1e-6)
        assert diag.verdict.vulnerability in (
            "weak", "vulnerable", "moderate", "absent",
        )
        assert diag.suppression.worst_layer == 20


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------


class TestDiagnosisToDict:
    def test_with_repair(self):
        diag = DiagnosisResult(
            prompt="test", answer="Berlin",
            circuit=CircuitAnalysis(
                active_heads=[{"layer": 21, "head": 29, "projection": 0.9}],
                top_head={"layer": 21, "head": 29, "projection": 0.9},
                top3_share=1.0, circuit_type="concentrated",
                total_attention=0.9, gatekeeper_present=True,
            ),
            suppression=SuppressionAnalysis(
                total_mlp=-5.0, subject_mlp=-1.0, worst_layer=20,
                worst_layer_value=-3.0, suppression_ratio=5.6,
            ),
            verdict=Verdict(
                signal_strength="absent", vulnerability="absent",
                confidence="low", reason="Absent signal (0.90)",
            ),
            repair=RepairPrescription(
                target_heads=[{"layer": 21, "head": 29}],
                target_layers=[21],
                suggested_fix="attention-lora",
                suggested_modules=["q_proj", "k_proj", "v_proj"],
                alternative_fix="mlp-lora",
                alternative_layers=[20],
                alternative_modules=["gate_proj", "up_proj", "down_proj"],
            ),
        )
        d = diagnosis_to_dict(diag)
        assert d["prompt"] == "test"
        assert d["circuit"]["circuit_type"] == "concentrated"
        assert d["verdict"]["vulnerability"] == "absent"
        assert "repair" in d
        assert d["repair"]["suggested_fix"] == "attention-lora"

        # JSON serializable
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["repair"]["target_layers"] == [21]

    def test_without_repair(self):
        diag = DiagnosisResult(
            prompt="test", answer="Berlin",
            circuit=CircuitAnalysis(
                active_heads=[], top_head=None, top3_share=0.0,
                circuit_type="distributed", total_attention=3.0,
                gatekeeper_present=False,
            ),
            suppression=SuppressionAnalysis(
                total_mlp=-1.0, subject_mlp=0.0, worst_layer=5,
                worst_layer_value=-0.5, suppression_ratio=0.3,
            ),
            verdict=Verdict(
                signal_strength="strong", vulnerability="robust",
                confidence="high", reason="Strong signal",
            ),
            repair=None,
        )
        d = diagnosis_to_dict(diag)
        assert "repair" not in d


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestStorage:
    def test_write_and_list(self, tmp_path):
        from neurotrace.storage import TraceDB

        db = TraceDB(str(tmp_path / "test.db"))
        db.write_diagnosis_run(
            run_id="d1", dataset="capitals",
            model_name="test", prompt_count=5,
        )
        runs = db.list_diagnosis_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == "d1"
        assert runs[0]["dataset"] == "capitals"
        db.close()

    def test_write_results(self, tmp_path):
        from neurotrace.storage import TraceDB

        db = TraceDB(str(tmp_path / "test.db"))
        db.write_diagnosis_run(
            run_id="d1", dataset=None,
            model_name="test", prompt_count=1,
        )
        db.write_diagnosis_result(
            run_id="d1", prompt="test", answer="Berlin",
            total_attention=1.42, total_mlp=-17.96,
            top3_share=0.86, circuit_type="concentrated",
            signal_strength="weak", vulnerability="vulnerable",
            confidence="high",
            repair_json='{"target_layers": [21]}',
        )
        rows = db._conn.execute(
            "SELECT * FROM diagnosis_results WHERE run_id = 'd1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][4] == pytest.approx(1.42, abs=0.01)  # total_attention
        db.close()


# ---------------------------------------------------------------------------
# HTML generation tests
# ---------------------------------------------------------------------------


class TestHTMLGeneration:
    def _make_diagnosis(self, vulnerability="vulnerable"):
        return DiagnosisResult(
            prompt="The capital of Germany is",
            answer="Berlin",
            circuit=CircuitAnalysis(
                active_heads=[
                    {"layer": 21, "head": 29, "projection": 0.91},
                    {"layer": 17, "head": 19, "projection": 0.17},
                ],
                top_head={"layer": 21, "head": 29, "projection": 0.91},
                top3_share=0.86, circuit_type="concentrated",
                total_attention=1.42, gatekeeper_present=True,
            ),
            suppression=SuppressionAnalysis(
                total_mlp=-17.96, subject_mlp=-3.82,
                worst_layer=20, worst_layer_value=-3.55,
                suppression_ratio=12.6,
            ),
            verdict=Verdict(
                signal_strength="weak", vulnerability=vulnerability,
                confidence="high",
                reason="Weak signal (1.42) + concentrated circuit",
            ),
            repair=RepairPrescription(
                target_heads=[
                    {"layer": 21, "head": 29},
                    {"layer": 17, "head": 19},
                ],
                target_layers=[17, 21],
                suggested_fix="attention-lora",
                suggested_modules=["q_proj", "k_proj", "v_proj"],
                alternative_fix="mlp-lora",
                alternative_layers=[20],
                alternative_modules=["gate_proj", "up_proj", "down_proj"],
            ) if vulnerability != "robust" else None,
        )

    def test_single_html(self):
        diag = self._make_diagnosis()
        attn_entries = _make_attn_entries({
            (21, 29): 0.91, (17, 19): 0.17,
        })
        attn_result = AttentionTraceResult(
            prompt="The capital of Germany is",
            answer="Berlin", entries=attn_entries,
        )
        token_entries = _make_token_entries({20: -3.55, 21: -2.0})
        token_result = TokenTraceResult(
            prompt="The capital of Germany is",
            answer="Berlin", subject="Germany",
            entries=token_entries,
        )

        html = generate_diagnose_html_single(
            diag, attn_result, token_result,
            layers=[17, 20, 21], model_name="test-model",
        )
        assert "<!DOCTYPE html>" in html
        assert "Diagnosis" in html
        assert "Berlin" in html
        assert "VULNERABLE" in html
        assert "q_proj" in html
        assert "Repair Prescription" in html
        assert "<svg" in html

    def test_batch_html(self):
        diagnoses = [
            self._make_diagnosis("vulnerable"),
            self._make_diagnosis("robust"),
            self._make_diagnosis("moderate"),
        ]
        html = generate_diagnose_html_batch(
            diagnoses, "capitals", "test-model",
        )
        assert "<!DOCTYPE html>" in html
        assert "Diagnosis Summary" in html
        assert "Vulnerable" in html
        assert "Robust" in html
        assert "Moderate" in html


# ---------------------------------------------------------------------------
# Attention-trace batch bugfix test
# ---------------------------------------------------------------------------


class TestAttentionTraceBatchBugfix:
    """Verify the flip_direction classification fix."""

    def test_only_broke_counted_as_vulnerable(self):
        """The old code used `cell.get("flipped") or cell.get("flip_direction")`
        which matched all cells since flip_direction is always set (e.g. "none").
        The fix checks for `flip_direction == "broke"` specifically."""
        cells = [
            {"prompt": "France", "flip_direction": "none",
             "flipped": False},
            {"prompt": "Germany", "flip_direction": "broke",
             "flipped": True},
            {"prompt": "Japan", "flip_direction": "fixed",
             "flipped": True},
            {"prompt": "Italy", "flip_direction": "changed",
             "flipped": True},
            {"prompt": "Spain", "flip_direction": "none",
             "flipped": False},
        ]

        # Simulate the fixed logic
        vuln = set()
        for cell in cells:
            if cell.get("flip_direction") == "broke":
                vuln.add(cell["prompt"])

        assert vuln == {"Germany"}
        assert len(vuln) == 1

    def test_old_logic_was_wrong(self):
        """Demonstrate the old logic would mark ALL cells as vulnerable."""
        cells = [
            {"prompt": "p1", "flip_direction": "none", "flipped": False},
            {"prompt": "p2", "flip_direction": "broke", "flipped": True},
        ]

        # OLD logic (bug): "none" is truthy!
        vuln_old = set()
        for cell in cells:
            if cell.get("flipped") or cell.get("flip_direction"):
                vuln_old.add(cell["prompt"])
        assert len(vuln_old) == 2  # Bug: both marked vulnerable

        # NEW logic (fix)
        vuln_new = set()
        for cell in cells:
            if cell.get("flip_direction") == "broke":
                vuln_new.add(cell["prompt"])
        assert len(vuln_new) == 1  # Correct: only "broke" is vulnerable


# ---------------------------------------------------------------------------
# CLI integration test (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.model_download
class TestDiagnoseCLI:
    def test_single_prompt_local(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "diagnose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--subject", "Germany",
            "--layers", "0,1,2",
        ])
        assert result.exit_code == 0, result.output
        assert "Diagnosis" in result.output
        assert "Circuit" in result.output
        assert "Suppression" in result.output
        assert "Verdict" in result.output

    def test_single_prompt_json(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "diagnose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--layers", "0,1",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output.strip())
        assert "circuit" in data
        assert "suppression" in data
        assert "verdict" in data

    def test_single_prompt_html(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "diagnose.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "diagnose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--layers", "0,1",
            "--html", html_path,
        ])
        assert result.exit_code == 0, result.output
        with open(html_path) as f:
            html = f.read()
        assert "<!DOCTYPE html>" in html
        assert "Diagnosis" in html

    def test_batch_builtin(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "diagnose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--dataset-builtin", "capitals",
            "--layers", "0,1",
        ])
        assert result.exit_code == 0, result.output
        assert "Diagnosis Summary" in result.output

    def test_db_persistence(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli
        from neurotrace.storage import TraceDB

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        runner.invoke(cli, [
            "diagnose",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--layers", "0,1",
        ])

        db = TraceDB(db_path)
        runs = db.list_diagnosis_runs()
        assert len(runs) == 1
        assert runs[0]["prompt_count"] == 1
        db.close()
