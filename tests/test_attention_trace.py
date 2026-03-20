"""Tests for attention-trace feature."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neurotrace.attention_trace import (
    AttentionTraceEntry,
    AttentionTraceResult,
    AttentionTraceRun,
    attention_trace_run_to_dict,
    compute_head_entries,
    decompose_attention_heads,
    generate_attention_trace_html_batch,
    generate_attention_trace_html_single,
    get_answer_direction,
    run_attention_trace_remote,
)

# ---------------------------------------------------------------------------
# Core computation tests
# ---------------------------------------------------------------------------


class TestGetAnswerDirection:
    def test_normalizes_vector(self):
        lm_head = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        direction = get_answer_direction(lm_head, 0)
        assert direction.shape == (3,)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_zero_vector(self):
        lm_head = np.array([[0.0, 0.0, 0.0]])
        direction = get_answer_direction(lm_head, 0)
        assert np.allclose(direction, 0.0)


class TestDecomposeAttentionHeads:
    def test_shape(self):
        hidden_dim = 8
        num_heads = 2
        pre_proj = np.random.randn(hidden_dim).astype(np.float32)
        o_proj_weight = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)

        contribs = decompose_attention_heads(pre_proj, o_proj_weight, num_heads)
        assert len(contribs) == num_heads
        for c in contribs:
            assert c.shape == (hidden_dim,)

    def test_sum_reconstruction(self):
        """Sum of per-head contributions should equal full o_proj output."""
        hidden_dim = 16
        num_heads = 4
        pre_proj = np.random.randn(hidden_dim).astype(np.float32)
        o_proj_weight = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)

        contribs = decompose_attention_heads(pre_proj, o_proj_weight, num_heads)
        summed = sum(contribs)

        full_output = o_proj_weight @ pre_proj
        np.testing.assert_allclose(summed, full_output, atol=1e-5)


class TestComputeHeadEntries:
    def test_basic(self):
        contribs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
        answer_dir = np.array([1.0, 0.0, 0.0])
        entries = compute_head_entries(contribs, answer_dir, "test", layer=5)

        assert len(entries) == 2
        assert entries[0].layer == 5
        assert entries[0].head_idx == 0
        assert abs(entries[0].answer_projection - 1.0) < 1e-6
        assert abs(entries[1].answer_projection) < 1e-6

    def test_magnitude(self):
        contribs = [np.array([3.0, 4.0])]
        answer_dir = np.array([1.0, 0.0])
        entries = compute_head_entries(contribs, answer_dir, "test", layer=0)
        assert abs(entries[0].magnitude - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestDataStructures:
    def test_entry_creation(self):
        entry = AttentionTraceEntry(
            prompt="test", layer=0, head_idx=3,
            answer_projection=1.5, magnitude=2.0,
        )
        assert entry.layer == 0
        assert entry.head_idx == 3

    def test_result_creation(self):
        result = AttentionTraceResult(
            prompt="test", answer="Berlin", entries=[],
        )
        assert result.prompt == "test"
        assert result.answer == "Berlin"

    def test_run_creation(self):
        run = AttentionTraceRun(
            run_id="abc", dataset=None, model_name="test",
            layers=[0, 1], prompt_count=1, results=[],
            created_at="2025-01-01T00:00:00",
        )
        assert run.prompt_count == 1


# ---------------------------------------------------------------------------
# Remote runner tests
# ---------------------------------------------------------------------------


class TestRunAttentionTraceRemote:
    def test_basic_remote(self):
        hidden_dim = 8
        num_heads = 2
        lm_head = np.random.randn(100, hidden_dim).astype(np.float32)

        # Simulate per-head contributions: [num_heads, hidden_dim]
        layer_contribs = {
            0: np.random.randn(num_heads, hidden_dim).astype(np.float32),
            1: np.random.randn(num_heads, hidden_dim).astype(np.float32),
        }

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens=True: (
            [1, 2, 3] if "capital" in text else [42]
        )
        tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        result = run_attention_trace_remote(
            layer_contributions=layer_contribs,
            tokenizer=tokenizer,
            prompt="The capital of Germany is",
            answer="Berlin",
            layers=[0, 1],
            lm_head_weight=lm_head,
        )

        assert result.prompt == "The capital of Germany is"
        assert result.answer == "Berlin"
        assert len(result.entries) == 4  # 2 layers × 2 heads

    def test_empty_layers(self):
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [42]

        result = run_attention_trace_remote(
            layer_contributions={},
            tokenizer=tokenizer,
            prompt="test",
            answer="test",
            layers=[0],
            lm_head_weight=np.zeros((100, 8), dtype=np.float32),
        )
        assert len(result.entries) == 0


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def _make_sample_run(self):
        entries = [
            AttentionTraceEntry(
                prompt="test", layer=0, head_idx=0,
                answer_projection=1.5, magnitude=2.0,
            ),
            AttentionTraceEntry(
                prompt="test", layer=0, head_idx=1,
                answer_projection=-0.5, magnitude=1.0,
            ),
        ]
        result = AttentionTraceResult(
            prompt="test", answer="Berlin", entries=entries,
        )
        return AttentionTraceRun(
            run_id="test-id", dataset="capitals",
            model_name="test-model", layers=[0, 1],
            prompt_count=1, results=[result],
            created_at="2025-01-01T00:00:00",
        )

    def test_to_dict(self):
        run = self._make_sample_run()
        d = attention_trace_run_to_dict(run)
        assert d["run_id"] == "test-id"
        assert d["dataset"] == "capitals"
        assert len(d["results"]) == 1
        assert len(d["results"][0]["entries"]) == 2

    def test_json_roundtrip(self):
        run = self._make_sample_run()
        d = attention_trace_run_to_dict(run)
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["run_id"] == "test-id"


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestStorage:
    def test_write_and_list_runs(self, tmp_path):
        from neurotrace.storage import TraceDB

        db = TraceDB(str(tmp_path / "test.db"))
        db.write_attention_trace_run(
            run_id="r1", dataset="capitals",
            model_name="test", layers="[0,1]", prompt_count=5,
        )
        runs = db.list_attention_trace_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "r1"
        assert runs[0]["dataset"] == "capitals"
        db.close()

    def test_write_results(self, tmp_path):
        from neurotrace.storage import TraceDB

        db = TraceDB(str(tmp_path / "test.db"))
        db.write_attention_trace_run(
            run_id="r1", dataset=None,
            model_name="test", layers="[0]", prompt_count=1,
        )
        db.write_attention_trace_result(
            run_id="r1", prompt="test", layer=0,
            head_idx=3, answer_projection=1.5, magnitude=2.0,
        )
        rows = db._conn.execute(
            "SELECT * FROM attention_trace_results WHERE run_id = 'r1'"
        ).fetchall()
        assert len(rows) == 1
        db.close()


# ---------------------------------------------------------------------------
# HTML generation tests
# ---------------------------------------------------------------------------


class TestHTMLGeneration:
    def _make_run(self, num_results=1, num_layers=2, num_heads=4):
        results = []
        for i in range(num_results):
            entries = []
            for layer in range(num_layers):
                for head in range(num_heads):
                    entries.append(AttentionTraceEntry(
                        prompt=f"prompt {i}",
                        layer=layer,
                        head_idx=head,
                        answer_projection=float(np.random.randn()),
                        magnitude=float(abs(np.random.randn())),
                    ))
            results.append(AttentionTraceResult(
                prompt=f"prompt {i}", answer="Berlin", entries=entries,
            ))

        return AttentionTraceRun(
            run_id="test-run", dataset="capitals",
            model_name="test-model", layers=list(range(num_layers)),
            prompt_count=num_results, results=results,
            created_at="2025-01-01T00:00:00",
        )

    def test_single_html(self):
        run = self._make_run(num_results=1)
        html = generate_attention_trace_html_single(run.results[0], run)
        assert "<!DOCTYPE html>" in html
        assert "Attention-Trace" in html
        assert "Berlin" in html
        assert "L0.H" in html
        assert "<svg" in html

    def test_single_html_with_mlp(self):
        run = self._make_run(num_results=1)
        html = generate_attention_trace_html_single(
            run.results[0], run, mlp_total=-3.5,
        )
        assert "MLP" in html
        assert "Net:" in html

    def test_batch_html_no_split(self):
        run = self._make_run(num_results=5)
        html = generate_attention_trace_html_batch(run)
        assert "<!DOCTYPE html>" in html
        assert "Attention-Trace" in html
        assert "Mean" in html

    def test_batch_html_with_split(self):
        run = self._make_run(num_results=5)
        vuln = {run.results[0].prompt, run.results[1].prompt}
        html = generate_attention_trace_html_batch(run, vulnerable_prompts=vuln)
        assert "Robust" in html
        assert "Vulnerable" in html


# ---------------------------------------------------------------------------
# Remote client tests
# ---------------------------------------------------------------------------


class TestRemoteClient:
    @patch("httpx.Client")
    def test_attention_contributions_stream(self, mock_client_class):
        from neurotrace.remote import RemoteWorker

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        contributions = np.random.randn(32, 2048).astype(np.float16)
        import base64
        encoded = base64.b64encode(contributions.tobytes()).decode("ascii")

        lines = [
            f'data: {{"type": "layer-contributions", "layer": 0, '
            f'"num_heads": 32, "shape": [32, 2048], '
            f'"dtype": "float16", "contributions": "{encoded}"}}',
            'data: {"type": "done", "layers_completed": 1, "heads_per_layer": 32}',
        ]

        mock_response = MagicMock()
        mock_response.iter_lines.return_value = iter(lines)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value = mock_response

        worker = RemoteWorker("http://localhost:8877")
        events = list(worker.attention_contributions_stream("test", [0]))

        assert len(events) == 2
        assert events[0]["type"] == "layer-contributions"
        assert events[1]["type"] == "done"


# ---------------------------------------------------------------------------
# CLI integration test (requires model download)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAttentionTraceCLI:
    def test_single_prompt_local(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attention-trace",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--layers", "0,1,2",
        ])
        assert result.exit_code == 0, result.output
        assert "Attention-Trace" in result.output

    def test_single_prompt_json(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attention-trace",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--layers", "0,1",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "results" in data
        assert len(data["results"]) == 1
        # 2 layers, 32 heads = 64 entries
        assert len(data["results"][0]["entries"]) == 64

    def test_single_prompt_html(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "report.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attention-trace",
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
        assert "Paris" in html

    def test_batch_builtin(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "attention-trace",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--dataset-builtin", "capitals",
            "--layers", "0,1",
        ])
        assert result.exit_code == 0, result.output
        assert "Attention-Trace Summary" in result.output
