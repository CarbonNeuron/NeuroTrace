"""Tests for chain trace (multi-hop reasoning)."""

import json

from neurotrace.chain import (
    ChainResult,
    chain_result_to_dict,
    classify_chain,
    _first_top_k_layer,
    _find_rank,
    _pad_to,
)
from neurotrace.datasets import get_builtin_dataset


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------


class TestMultiHopDataset:
    def test_dataset_exists(self):
        ds = get_builtin_dataset("multi_hop")
        assert len(ds) >= 30

    def test_entries_have_required_fields(self):
        ds = get_builtin_dataset("multi_hop")
        for i, entry in enumerate(ds):
            assert "prompt" in entry, f"Entry {i} missing 'prompt'"
            assert "intermediate" in entry, f"Entry {i} missing 'intermediate'"
            assert "answer" in entry, f"Entry {i} missing 'answer'"

    def test_entries_are_non_empty(self):
        ds = get_builtin_dataset("multi_hop")
        for entry in ds:
            assert entry["prompt"].strip()
            assert entry["intermediate"].strip()
            assert entry["answer"].strip()

    def test_dataset_has_categories(self):
        """Verify dataset spans geography, science, and compositional."""
        ds = get_builtin_dataset("multi_hop")
        prompts = [e["prompt"].lower() for e in ds]
        has_geo = any("capital" in p or "country" in p or "continent" in p for p in prompts)
        has_sci = any("atomic" in p or "chemical" in p or "element" in p for p in prompts)
        has_comp = any("animal" in p or "legs" in p or "sport" in p for p in prompts)
        assert has_geo, "Missing geography entries"
        assert has_sci, "Missing science entries"
        assert has_comp, "Missing compositional entries"


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyChain:
    def test_complete(self):
        assert classify_chain(answer_rank_final=1, intermediate_peak_rank=3) == "COMPLETE"

    def test_complete_edge(self):
        assert classify_chain(answer_rank_final=1, intermediate_peak_rank=10) == "COMPLETE"

    def test_shortcut(self):
        assert classify_chain(answer_rank_final=1, intermediate_peak_rank=11) == "SHORTCUT"

    def test_shortcut_large_rank(self):
        assert classify_chain(answer_rank_final=1, intermediate_peak_rank=999) == "SHORTCUT"

    def test_partial(self):
        assert classify_chain(answer_rank_final=5, intermediate_peak_rank=3) == "PARTIAL"

    def test_partial_edge(self):
        assert classify_chain(answer_rank_final=2, intermediate_peak_rank=10) == "PARTIAL"

    def test_broken(self):
        assert classify_chain(answer_rank_final=50, intermediate_peak_rank=50) == "BROKEN"

    def test_broken_no_activation(self):
        assert classify_chain(answer_rank_final=999, intermediate_peak_rank=999) == "BROKEN"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_first_top_k_layer_found(self):
        ranks = [50, 20, 8, 3, 1]
        assert _first_top_k_layer(ranks, k=10) == 2

    def test_first_top_k_layer_not_found(self):
        ranks = [50, 20, 15, 30]
        assert _first_top_k_layer(ranks, k=10) is None

    def test_first_top_k_layer_first(self):
        ranks = [5, 20, 15, 30]
        assert _first_top_k_layer(ranks, k=10) == 0

    def test_first_top_k_layer_empty(self):
        assert _first_top_k_layer([], k=10) is None

    def test_pad_to_short(self):
        assert _pad_to([1, 2], 5, 0) == [1, 2, 0, 0, 0]

    def test_pad_to_exact(self):
        assert _pad_to([1, 2, 3], 3, 0) == [1, 2, 3]

    def test_pad_to_long(self):
        assert _pad_to([1, 2, 3, 4, 5], 3, 0) == [1, 2, 3]


# ---------------------------------------------------------------------------
# ChainResult dataclass tests
# ---------------------------------------------------------------------------


class TestChainResult:
    def _make_result(self, **kwargs):
        defaults = dict(
            prompt="The capital of the country east of France is",
            intermediate="Germany",
            answer="Berlin",
            intermediate_logits=[0.1, 0.5, 2.0, 3.0],
            answer_logits=[0.0, 0.1, 0.3, 2.5],
            intermediate_ranks=[100, 20, 5, 3],
            answer_ranks=[200, 50, 15, 1],
            intermediate_commit_layer=2,
            answer_commit_layer=3,
            handoff_gap=1,
            intermediate_peak_rank=3,
            answer_peak_rank=1,
            chain_status="COMPLETE",
            num_layers=4,
        )
        defaults.update(kwargs)
        return ChainResult(**defaults)

    def test_creation(self):
        r = self._make_result()
        assert r.chain_status == "COMPLETE"
        assert r.intermediate == "Germany"
        assert r.answer == "Berlin"
        assert r.num_layers == 4

    def test_to_dict(self):
        r = self._make_result()
        d = chain_result_to_dict(r)
        assert d["chain_status"] == "COMPLETE"
        assert d["intermediate"] == "Germany"
        assert d["answer"] == "Berlin"
        assert d["handoff_gap"] == 1
        assert len(d["intermediate_logits"]) == 4

    def test_to_dict_json_serializable(self):
        r = self._make_result()
        d = chain_result_to_dict(r)
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["chain_status"] == "COMPLETE"

    def test_none_values(self):
        r = self._make_result(
            intermediate_commit_layer=None,
            answer_commit_layer=None,
            handoff_gap=None,
        )
        d = chain_result_to_dict(r)
        assert d["intermediate_commit_layer"] is None
        assert d["handoff_gap"] is None


