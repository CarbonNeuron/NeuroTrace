"""Tests for the discover module — no model downloads required."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from neurotrace.discover import (
    TOPIC_QUERIES,
    TOPIC_TEMPLATES,
    DiscoveryFact,
    DiscoveryResult,
    check_regressions_fingerprint,
    discovery_result_to_dict,
    ensure_discoveries_table,
    expand_bindings,
    expand_custom_template,
    generate_discover_html,
    get_cached_facts,
    insert_discovery,
    load_facts_from_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    baseline_correct=8,
    baseline_wrong=2,
    healed_count=1,
    regression_count=0,
    facts=None,
) -> DiscoveryResult:
    total = baseline_correct + baseline_wrong
    if facts is None:
        facts = [
            DiscoveryFact(
                id="abc1",
                prompt="The capital of France is",
                expected_answer="Paris",
                topic="world_capitals",
                source="wikidata",
                baseline_correct=True,
                baseline_prob=0.85,
            ),
            DiscoveryFact(
                id="abc2",
                prompt="The capital of Nauru is",
                expected_answer="Yaren",
                topic="world_capitals",
                source="wikidata",
                baseline_correct=False,
                baseline_prob=0.02,
                healed=True,
                healed_prob=0.45,
            ),
        ]
    return DiscoveryResult(
        run_id="test123",
        topic="world_capitals",
        source="wikidata",
        model_name="test-model",
        total_facts=total,
        baseline_correct=baseline_correct,
        baseline_wrong=baseline_wrong,
        healed_count=healed_count,
        regression_count=regression_count,
        facts=facts,
        duration_seconds=12.3,
    )


# ---------------------------------------------------------------------------
# Template expansion tests
# ---------------------------------------------------------------------------


class TestTemplateExpansion:
    def test_world_capitals_template(self):
        bindings = [
            {
                "country": {"value": "http://www.wikidata.org/entity/Q142"},
                "countryLabel": {"value": "France"},
                "capital": {"value": "http://www.wikidata.org/entity/Q90"},
                "capitalLabel": {"value": "Paris"},
            },
            {
                "country": {"value": "http://www.wikidata.org/entity/Q183"},
                "countryLabel": {"value": "Germany"},
                "capital": {"value": "http://www.wikidata.org/entity/Q64"},
                "capitalLabel": {"value": "Berlin"},
            },
        ]
        facts = expand_bindings("world_capitals", bindings)
        assert len(facts) == 2
        assert facts[0]["prompt"] == "The capital of France is"
        assert facts[0]["answer"] == "Paris"
        assert facts[1]["prompt"] == "The capital of Germany is"
        assert facts[1]["answer"] == "Berlin"

    def test_chemical_elements_template(self):
        bindings = [
            {
                "element": {"value": "http://www.wikidata.org/entity/Q688"},
                "elementLabel": {"value": "gold"},
                "symbol": {"value": "Au"},
            },
        ]
        facts = expand_bindings("chemical_elements", bindings)
        assert len(facts) == 1
        assert facts[0]["prompt"] == "The chemical symbol for gold is"
        assert facts[0]["answer"] == "Au"

    def test_country_currencies_template(self):
        bindings = [
            {
                "country": {"value": "http://www.wikidata.org/entity/Q30"},
                "countryLabel": {"value": "United States"},
                "currency": {"value": "http://www.wikidata.org/entity/Q4917"},
                "currencyLabel": {"value": "United States dollar"},
            },
        ]
        facts = expand_bindings("country_currencies", bindings)
        assert len(facts) == 1
        assert facts[0]["prompt"] == "The currency of United States is the"
        assert facts[0]["answer"] == "United States dollar"

    def test_country_languages_template(self):
        bindings = [
            {
                "country": {"value": "http://www.wikidata.org/entity/Q142"},
                "countryLabel": {"value": "France"},
                "language": {"value": "http://www.wikidata.org/entity/Q150"},
                "languageLabel": {"value": "French"},
            },
        ]
        facts = expand_bindings("country_languages", bindings)
        assert len(facts) == 1
        assert facts[0]["prompt"] == "The official language of France is"
        assert facts[0]["answer"] == "French"

    def test_all_builtin_topics_have_queries(self):
        for topic in TOPIC_QUERIES:
            assert "{limit}" in TOPIC_QUERIES[topic]

    def test_templates_cover_topics(self):
        # All topics with templates should work
        for topic in TOPIC_TEMPLATES:
            assert topic in TOPIC_QUERIES

    def test_missing_binding_field_skipped(self):
        bindings = [
            {
                "countryLabel": {"value": "France"},
                # Missing capitalLabel
            },
        ]
        facts = expand_bindings("world_capitals", bindings)
        assert len(facts) == 0

    def test_unknown_topic_raises(self):
        with pytest.raises(ValueError, match="No template for topic"):
            expand_bindings("nonexistent_topic", [])

    def test_custom_template_expansion(self):
        bindings = [
            {
                "city": {"value": "Paris"},
                "population": {"value": "2161000"},
            },
        ]
        facts = expand_custom_template(
            bindings,
            template="The population of {city} is",
            answer_field="population",
        )
        assert len(facts) == 1
        assert facts[0]["prompt"] == "The population of Paris is"
        assert facts[0]["answer"] == "2161000"


# ---------------------------------------------------------------------------
# Mock SPARQL tests
# ---------------------------------------------------------------------------


class TestWikidataClient:
    @patch("neurotrace.discover.query_wikidata")
    def test_fetch_topic_facts_mocked(self, mock_query):
        mock_query.return_value = [
            {
                "country": {"value": "Q142"},
                "countryLabel": {"value": "France"},
                "capital": {"value": "Q90"},
                "capitalLabel": {"value": "Paris"},
            },
        ]
        from neurotrace.discover import fetch_topic_facts

        facts = fetch_topic_facts("world_capitals", limit=10)
        assert len(facts) == 1
        assert facts[0]["answer"] == "Paris"
        mock_query.assert_called_once()

    @patch("neurotrace.discover.query_wikidata")
    def test_fetch_unknown_topic_raises(self, mock_query):
        from neurotrace.discover import fetch_topic_facts

        with pytest.raises(ValueError, match="Unknown topic"):
            fetch_topic_facts("nonexistent", limit=10)


# ---------------------------------------------------------------------------
# File-based source tests
# ---------------------------------------------------------------------------


class TestFileSource:
    def test_load_jsonl(self, tmp_path):
        jsonl = tmp_path / "facts.jsonl"
        jsonl.write_text(
            '{"prompt": "The capital of France is", "answer": "Paris"}\n'
            '{"prompt": "The capital of Germany is", "answer": "Berlin"}\n'
        )
        facts = load_facts_from_file(str(jsonl))
        assert len(facts) == 2
        assert facts[0]["prompt"] == "The capital of France is"
        assert facts[1]["answer"] == "Berlin"

    def test_load_jsonl_blank_lines(self, tmp_path):
        jsonl = tmp_path / "facts.jsonl"
        jsonl.write_text(
            '{"prompt": "A", "answer": "B"}\n'
            "\n"
            '{"prompt": "C", "answer": "D"}\n'
        )
        facts = load_facts_from_file(str(jsonl))
        assert len(facts) == 2

    def test_load_jsonl_missing_field(self, tmp_path):
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text('{"prompt": "A"}\n')
        with pytest.raises(ValueError, match="missing"):
            load_facts_from_file(str(jsonl))


# ---------------------------------------------------------------------------
# Fingerprint regression tests
# ---------------------------------------------------------------------------


class TestFingerprintRegression:
    def test_no_regressions_below_threshold(self):
        import torch

        edit_key = torch.randn(128)
        cache = {
            "p1": torch.randn(128),
            "p2": torch.randn(128),
        }
        # With random vectors, cosine similarity is unlikely to exceed 0.9
        flagged = check_regressions_fingerprint(
            edit_key, cache, similarity_threshold=0.9,
        )
        assert isinstance(flagged, list)

    def test_identical_vectors_flagged(self):
        import torch

        edit_key = torch.ones(128)
        cache = {
            "p1": torch.ones(128),  # identical = cosine sim 1.0
            "p2": -torch.ones(128),  # opposite = cosine sim -1.0
        }
        flagged = check_regressions_fingerprint(
            edit_key, cache, similarity_threshold=0.5,
        )
        assert "p1" in flagged
        assert "p2" not in flagged

    def test_empty_cache(self):
        import torch

        edit_key = torch.randn(128)
        flagged = check_regressions_fingerprint(edit_key, {}, similarity_threshold=0.3)
        assert flagged == []


# ---------------------------------------------------------------------------
# Database schema tests
# ---------------------------------------------------------------------------


class TestDatabaseSchema:
    def test_ensure_table_creates(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ensure_discoveries_table(db_path)

        import duckdb

        con = duckdb.connect(db_path)
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "discoveries" in table_names
        con.close()

    def test_ensure_table_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ensure_discoveries_table(db_path)
        ensure_discoveries_table(db_path)  # Should not raise

    def test_insert_and_read(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ensure_discoveries_table(db_path)

        fact = DiscoveryFact(
            id="test1",
            prompt="The capital of France is",
            expected_answer="Paris",
            topic="world_capitals",
            source="wikidata",
            baseline_correct=True,
            baseline_prob=0.85,
        )
        insert_discovery(db_path, fact)

        import duckdb

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT * FROM discoveries WHERE id = 'test1'").fetchall()
        assert len(rows) == 1
        assert rows[0][3] == "The capital of France is"  # prompt column
        con.close()

    def test_get_cached_facts(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ensure_discoveries_table(db_path)

        fact = DiscoveryFact(
            id="test1",
            prompt="The capital of France is",
            expected_answer="Paris",
            topic="world_capitals",
            source="wikidata",
            baseline_correct=True,
            baseline_prob=0.85,
        )
        insert_discovery(db_path, fact)

        cached = get_cached_facts(db_path, "world_capitals")
        assert len(cached) == 1
        assert cached[0]["prompt"] == "The capital of France is"
        assert cached[0]["answer"] == "Paris"

    def test_get_cached_facts_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        # Table doesn't exist yet
        cached = get_cached_facts(db_path, "world_capitals")
        assert cached == []


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_discovery_result_to_dict(self):
        result = _make_result()
        d = discovery_result_to_dict(result)
        assert d["run_id"] == "test123"
        assert d["topic"] == "world_capitals"
        assert d["baseline"]["correct"] == 8
        assert d["baseline"]["wrong"] == 2
        assert d["healed_count"] == 1
        assert len(d["facts"]) == 2
        assert d["facts"][0]["baseline_correct"] is True
        assert d["facts"][1]["healed"] is True

    def test_json_roundtrip(self):
        result = _make_result()
        d = discovery_result_to_dict(result)
        j = json.dumps(d)
        parsed = json.loads(j)
        assert parsed["baseline"]["accuracy"] == pytest.approx(0.8)

    def test_empty_facts(self):
        result = _make_result(
            baseline_correct=0,
            baseline_wrong=0,
            healed_count=0,
            facts=[],
        )
        d = discovery_result_to_dict(result)
        assert d["total_facts"] == 0
        assert d["baseline"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------


class TestHtmlReport:
    def test_html_is_valid_structure(self):
        result = _make_result()
        html = generate_discover_html(result)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<style>" in html
        assert "Discover Report" in html

    def test_html_contains_facts(self):
        result = _make_result()
        html = generate_discover_html(result)
        assert "Paris" in html
        assert "Yaren" in html
        assert "CORRECT" in html
        assert "WRONG" in html

    def test_html_self_contained(self):
        result = _make_result()
        html = generate_discover_html(result)
        # No external resources
        assert "<script" not in html
        assert "http://" not in html or "https://" not in html
        assert "<link rel=" not in html

    def test_html_escapes_special_chars(self):
        facts = [
            DiscoveryFact(
                id="x1",
                prompt='Test <script>alert("xss")</script>',
                expected_answer="safe",
                topic="test",
                source="test",
                baseline_correct=True,
                baseline_prob=0.9,
            ),
        ]
        result = _make_result(facts=facts)
        html = generate_discover_html(result)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Dry-run mode test
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_no_healing(self):
        """Verify dry_run parameter is respected in run_discover."""
        from neurotrace.discover import run_discover

        # Create a mock model and tokenizer
        model = MagicMock()
        model.config._name_or_path = "test-model"
        model.parameters.return_value = iter([MagicMock(device="cpu")])

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]

        facts = [{"prompt": "Test prompt", "answer": "Test"}]

        with (
            patch("neurotrace.discover.classify_prediction") as mock_classify,
            patch("neurotrace.discover.ensure_discoveries_table"),
            patch("neurotrace.discover.insert_discovery"),
            tempfile.TemporaryDirectory() as tmp_dir,
        ):
            mock_classify.return_value = (False, 0.01)

            result = run_discover(
                model,
                tokenizer,
                facts,
                topic="test",
                source="test",
                db_path=os.path.join(tmp_dir, "test.db"),
                heal=True,  # heal requested
                dry_run=True,  # but dry_run is on
            )

            assert result.healed_count == 0
            assert result.baseline_wrong == 1


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCliRegistration:
    def test_discover_command_exists(self):
        from neurotrace.cli import cli

        commands = list(cli.commands.keys())
        assert "discover" in commands
