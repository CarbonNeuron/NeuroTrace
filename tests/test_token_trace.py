"""Tests for the token-trace feature."""

import json
import os

import numpy as np
import pytest

from neurotrace.datasets import CAPITALS, RELATION_KEYWORDS, get_builtin_dataset
from neurotrace.storage import TraceDB
from neurotrace.token_trace import (
    TokenTraceEntry,
    TokenTraceResult,
    TokenTraceRun,
    aggregate_by_position_type,
    classify_token_position,
    compute_answer_projection,
    find_subject_token_positions,
    generate_token_trace_html_batch,
    generate_token_trace_html_single,
    get_answer_direction,
    run_token_trace_remote,
    token_trace_run_to_dict,
)


# ---------------------------------------------------------------------------
# Dataset subject fields
# ---------------------------------------------------------------------------


class TestDatasetSubjects:
    def test_capitals_have_subjects(self):
        for entry in CAPITALS:
            assert "subject" in entry, f"Missing subject: {entry['prompt']}"
            assert entry["subject"], f"Empty subject: {entry['prompt']}"

    def test_all_builtin_datasets_have_subjects(self):
        for name in ["capitals", "science_symbols", "math_simple", "history_dates"]:
            dataset = get_builtin_dataset(name)
            for entry in dataset:
                assert "subject" in entry, (
                    f"Missing subject in {name}: {entry['prompt']}"
                )

    def test_relation_keywords_exist(self):
        assert "capitals" in RELATION_KEYWORDS
        assert "capital" in RELATION_KEYWORDS["capitals"]


# ---------------------------------------------------------------------------
# Subject token detection
# ---------------------------------------------------------------------------


class TestFindSubjectTokenPositions:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def encode(self, text, add_special_tokens=True):
                # Simple word-level tokenizer
                tokens = text.split()
                # token ids = hash of word for determinism
                ids = [hash(w) % 10000 for w in tokens]
                if add_special_tokens:
                    ids = [1] + ids  # BOS token
                return ids
        return MockTokenizer()

    def test_find_subject_simple(self, mock_tokenizer):
        start, end = find_subject_token_positions(
            "The capital of Germany is",
            "Germany",
            mock_tokenizer,
        )
        # With BOS: [1, "The", "capital", "of", "Germany", "is"]
        # "Germany" is at index 4
        assert start == end  # single token

    def test_find_subject_not_found(self, mock_tokenizer):
        with pytest.raises(ValueError, match="not found"):
            find_subject_token_positions(
                "The capital of Germany is",
                "Xyzzy",
                mock_tokenizer,
            )


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


class TestCoreComputation:
    def test_get_answer_direction(self):
        weight = np.eye(10, dtype=np.float32)
        direction = get_answer_direction(weight, 3)
        assert direction.shape == (10,)
        assert np.isclose(np.linalg.norm(direction), 1.0)
        assert direction[3] == 1.0

    def test_compute_answer_projection(self):
        delta = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        proj = compute_answer_projection(delta, direction)
        assert np.isclose(proj, 1.0)

    def test_compute_answer_projection_negative(self):
        delta = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        proj = compute_answer_projection(delta, direction)
        assert np.isclose(proj, -2.0)

    def test_compute_answer_projection_orthogonal(self):
        delta = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        proj = compute_answer_projection(delta, direction)
        assert np.isclose(proj, 0.0)


# ---------------------------------------------------------------------------
# TokenTraceEntry / TokenTraceResult
# ---------------------------------------------------------------------------


class TestTokenTraceDataStructures:
    def test_entry_creation(self):
        entry = TokenTraceEntry(
            layer=5, position=3, token="Germany",
            is_subject=True, is_last=False,
            answer_projection=0.15,
            competitor_projection=-0.03,
            delta_magnitude=1.23,
        )
        assert entry.layer == 5
        assert entry.is_subject is True
        assert entry.competitor_projection == -0.03

    def test_result_creation(self):
        result = TokenTraceResult(
            prompt="The capital of Germany is",
            answer="Berlin",
            subject="Germany",
            entries=[],
        )
        assert result.subject == "Germany"
        assert result.entries == []


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


class TestRunTokenTraceRemote:
    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def encode(self, text, add_special_tokens=True):
                tokens = text.split()
                ids = list(range(100, 100 + len(tokens)))
                if add_special_tokens:
                    ids = [1] + ids
                return ids

            def decode(self, ids):
                mapping = {1: "<s>", 100: "The", 101: "capital",
                           102: "of", 103: "Germany", 104: "is"}
                return " ".join(mapping.get(i, f"[{i}]") for i in ids)
        return MockTokenizer()

    def test_run_token_trace_remote(self, mock_tokenizer):
        hidden_dim = 16
        num_positions = 6  # BOS + 5 words
        layers = [0, 1, 2]
        lm_head_weight = np.random.randn(200, hidden_dim).astype(np.float32)

        all_position_deltas = {}
        for layer in layers:
            all_position_deltas[layer] = np.random.randn(
                num_positions, hidden_dim,
            ).astype(np.float32)

        result = run_token_trace_remote(
            all_position_deltas=all_position_deltas,
            tokenizer=mock_tokenizer,
            prompt="The capital of Germany is",
            answer="Berlin",
            subject="Germany",
            layers=layers,
            lm_head_weight=lm_head_weight,
        )

        assert result.prompt == "The capital of Germany is"
        assert result.answer == "Berlin"
        assert result.subject == "Germany"
        assert len(result.entries) == num_positions * len(layers)

        # Check that subject marking works
        subject_entries = [e for e in result.entries if e.is_subject]
        assert len(subject_entries) > 0

        # Check last-token marking
        last_entries = [e for e in result.entries if e.is_last]
        assert len(last_entries) == len(layers)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def _make_entries(self, layers, tokens, subject_pos, last_pos):
        entries = []
        for layer in layers:
            for pos, tok in enumerate(tokens):
                entries.append(TokenTraceEntry(
                    layer=layer,
                    position=pos,
                    token=tok,
                    is_subject=(pos == subject_pos),
                    is_last=(pos == last_pos),
                    answer_projection=float(layer * 0.01 + pos * 0.1),
                    competitor_projection=None,
                    delta_magnitude=1.0,
                ))
        return entries

    def test_classify_token_position(self):
        assert classify_token_position("Germany", True, False) == "subject"
        assert classify_token_position("is", False, True) == "last"
        result = classify_token_position(
            "capital", False, False, {"capital"},
        )
        assert result == "relation"
        assert classify_token_position("The", False, False) == "other"

    def test_aggregate_by_position_type(self):
        tokens = ["The", "capital", "of", "Germany", "is"]
        entries = self._make_entries([0, 1], tokens, subject_pos=3, last_pos=4)
        results = [
            TokenTraceResult(
                prompt="test", answer="Berlin", subject="Germany",
                entries=entries,
            )
        ]

        agg = aggregate_by_position_type(
            results, [0, 1], relation_keywords={"capital"},
        )

        assert "subject" in agg
        assert "last" in agg
        assert "relation" in agg
        assert "other" in agg

        # Subject has entries for layers 0 and 1
        assert 0 in agg["subject"]
        assert 1 in agg["subject"]


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_token_trace_run_to_dict(self):
        run = TokenTraceRun(
            run_id="test-123",
            dataset="capitals",
            model_name="test-model",
            layers=[0, 5, 10],
            prompt_count=1,
            results=[
                TokenTraceResult(
                    prompt="test",
                    answer="Berlin",
                    subject="Germany",
                    entries=[
                        TokenTraceEntry(
                            layer=0, position=0, token="The",
                            is_subject=False, is_last=False,
                            answer_projection=0.01,
                            competitor_projection=None,
                            delta_magnitude=0.5,
                        ),
                    ],
                ),
            ],
            created_at="2024-01-01T00:00:00Z",
        )

        d = token_trace_run_to_dict(run)
        assert d["run_id"] == "test-123"
        assert d["layers"] == [0, 5, 10]
        assert len(d["results"]) == 1
        assert len(d["results"][0]["entries"]) == 1
        assert d["results"][0]["entries"][0]["answer_projection"] == 0.01

        # Ensure JSON serializable
        json_str = json.dumps(d)
        assert "test-123" in json_str


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestStorage:
    def test_token_trace_tables_created(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = TraceDB(db_path)
        db.write_token_trace_run(
            run_id="test-run",
            dataset="capitals",
            model_name="test-model",
            layers='[0, 1, 2]',
            prompt_count=5,
        )
        runs = db.list_token_trace_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test-run"
        assert runs[0]["prompt_count"] == 5
        db.close()

    def test_token_trace_results_storage(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = TraceDB(db_path)
        db.write_token_trace_run(
            run_id="test-run",
            dataset=None,
            model_name="test-model",
            layers='[0]',
            prompt_count=1,
        )
        db.write_token_trace_result(
            run_id="test-run",
            prompt="The capital of Germany is",
            layer=0,
            token_position=3,
            token_text="Germany",
            is_subject=True,
            is_last=False,
            answer_projection=0.15,
            competitor_projection=-0.03,
            delta_magnitude=1.23,
        )
        runs = db.list_token_trace_runs()
        assert len(runs) == 1
        db.close()


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


class TestHTMLGeneration:
    def _make_sample_run(self):
        entries = []
        tokens = ["The", "capital", "of", "Germany", "is"]
        layers = [0, 5, 10]
        for layer in layers:
            for pos, tok in enumerate(tokens):
                entries.append(TokenTraceEntry(
                    layer=layer, position=pos, token=tok,
                    is_subject=(pos == 3),
                    is_last=(pos == 4),
                    answer_projection=float(layer * 0.01 - pos * 0.02),
                    competitor_projection=-0.01 if layer > 5 else None,
                    delta_magnitude=abs(layer * 0.1 + pos * 0.05),
                ))

        result = TokenTraceResult(
            prompt="The capital of Germany is",
            answer="Berlin",
            subject="Germany",
            entries=entries,
        )

        run = TokenTraceRun(
            run_id="test-html",
            dataset=None,
            model_name="test-model",
            layers=layers,
            prompt_count=1,
            results=[result],
            created_at="2024-01-01T00:00:00Z",
        )
        return run, result

    def test_single_prompt_html(self):
        run, result = self._make_sample_run()
        html = generate_token_trace_html_single(result, run)

        assert "<!DOCTYPE html>" in html
        assert "Token-Trace" in html
        assert "Berlin" in html
        assert "Germany" in html
        assert "<svg" in html

    def test_batch_html(self):
        run, _ = self._make_sample_run()
        # Add a second result
        entries2 = []
        tokens = ["The", "capital", "of", "France", "is"]
        for layer in run.layers:
            for pos, tok in enumerate(tokens):
                entries2.append(TokenTraceEntry(
                    layer=layer, position=pos, token=tok,
                    is_subject=(pos == 3),
                    is_last=(pos == 4),
                    answer_projection=float(layer * 0.02),
                    competitor_projection=None,
                    delta_magnitude=0.5,
                ))
        run.results.append(TokenTraceResult(
            prompt="The capital of France is",
            answer="Paris",
            subject="France",
            entries=entries2,
        ))
        run.prompt_count = 2

        html = generate_token_trace_html_batch(
            run, relation_keywords={"capital"},
        )

        assert "<!DOCTYPE html>" in html
        assert "Token-Trace Batch" in html
        assert "<svg" in html


# ---------------------------------------------------------------------------
# CLI integration (model_download)
# ---------------------------------------------------------------------------


@pytest.mark.model_download
class TestTokenTraceCLI:
    def test_single_prompt_local(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "token-trace",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of Germany is",
            "--answer", "Berlin",
            "--subject", "Germany",
            "--layers", "0,10,21",
            "--json",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["prompt_count"] == 1
        assert len(data["results"]) == 1
        assert len(data["results"][0]["entries"]) > 0

    def test_batch_dataset_local(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "report.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "token-trace",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--dataset-builtin", "capitals",
            "--layers", "0,10,21",
            "--html", html_path,
        ])
        assert result.exit_code == 0, result.output
        assert os.path.exists(html_path)
        html_content = open(html_path).read()
        assert "Token-Trace" in html_content

    def test_html_single_prompt(self, tinyllama_model, tmp_path):
        from click.testing import CliRunner

        from neurotrace.cli import cli

        db_path = str(tmp_path / "test.db")
        html_path = str(tmp_path / "single.html")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "token-trace",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--db", db_path,
            "--prompt", "The capital of France is",
            "--answer", "Paris",
            "--subject", "France",
            "--layers", "0,5,10",
            "--html", html_path,
        ])
        assert result.exit_code == 0, result.output
        assert os.path.exists(html_path)
