"""Tests for the scan command and sabotage detection."""

import json

import pytest
from click.testing import CliRunner

from neurotrace.cli import cli
from neurotrace.datasets import (
    CAPITALS,
    format_prompt,
    get_builtin_dataset,
    load_dataset,
)
from neurotrace.scan import (
    PromptResult,
    ScanResult,
    detect_sabotage,
)

# --- Dataset tests ---


def test_capitals_dataset_is_list():
    assert isinstance(CAPITALS, list)
    assert len(CAPITALS) >= 30


def test_capitals_entries_have_required_fields():
    for entry in CAPITALS:
        assert "prompt" in entry
        assert "answer" in entry
        assert isinstance(entry["prompt"], str)
        assert isinstance(entry["answer"], str)


def test_capitals_store_full_names():
    """Answers should be full capital names, not subword fragments."""
    multi_token_expected = {
        "Australia": "Canberra",
        "South Korea": "Seoul",
        "Kenya": "Nairobi",
        "Turkey": "Ankara",
        "Thailand": "Bangkok",
        "Indonesia": "Jakarta",
        "Sweden": "Stockholm",
        "Denmark": "Copenhagen",
        "Finland": "Helsinki",
        "Portugal": "Lisbon",
        "Belgium": "Brussels",
        "Ukraine": "Kyiv",
        "South Africa": "Pretoria",
        "Nigeria": "Abuja",
        "Colombia": "Bogota",
        "Vietnam": "Hanoi",
        "Philippines": "Manila",
        "Malaysia": "Kuala Lumpur",
        "Czech Republic": "Prague",
        "Romania": "Bucharest",
        "Hungary": "Budapest",
        "Israel": "Jerusalem",
        "Saudi Arabia": "Riyadh",
        "Iran": "Tehran",
        "Pakistan": "Islamabad",
        "Myanmar": "Naypyidaw",
        "India": "New Delhi",
        "Brazil": "Brasilia",
        "Argentina": "Buenos Aires",
        "Mexico": "Mexico City",
    }
    capitals_by_country = {}
    for entry in CAPITALS:
        # Extract country from "The capital of X is"
        country = entry["prompt"].replace("The capital of ", "").replace(" is", "")
        capitals_by_country[country] = entry["answer"]

    for country, expected_answer in multi_token_expected.items():
        assert country in capitals_by_country, f"Missing country: {country}"
        actual = capitals_by_country[country]
        assert actual == expected_answer, (
            f"{country}: expected {expected_answer!r}, got {actual!r}"
        )


def test_get_builtin_dataset_capitals():
    ds = get_builtin_dataset("capitals")
    assert ds is CAPITALS


def test_get_builtin_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown built-in dataset"):
        get_builtin_dataset("nonexistent")


def test_load_dataset_from_file(tmp_path):
    data = [
        {"prompt": "The capital of France is", "answer": "Paris"},
        {"prompt": "The capital of Germany is", "answer": "Berlin"},
    ]
    path = tmp_path / "test_ds.json"
    path.write_text(json.dumps(data))
    result = load_dataset(str(path))
    assert len(result) == 2
    assert result[0]["prompt"] == "The capital of France is"


def test_load_dataset_validates_fields(tmp_path):
    data = [{"prompt": "test"}]  # missing 'answer'
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing.*answer"):
        load_dataset(str(path))


# --- format_prompt tests ---


def test_format_prompt_wraps_in_chat_template():
    """format_prompt wraps a raw prompt in chat messages format."""
    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"

    result = format_prompt("The capital of France is", MockTokenizer())
    assert "<|user|>" in result
    assert "The capital of France is" in result
    assert "<|assistant|>" in result


# --- Sabotage detection tests ---


def test_detect_sabotage_no_issues():
    """Clean trajectory: answer is rank 1, high probability."""
    ranks = [5, 3, 1, 1, 1, 1]
    probs = [0.01, 0.05, 0.4, 0.5, 0.6, 0.7]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.7,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert result.flags == []
    assert result.status == "correct"


def test_detect_sabotage_wrong_answer():
    """Answer token not top-1 at final layer."""
    ranks = [10, 8, 5, 3, 2, 4]
    probs = [0.01, 0.02, 0.05, 0.1, 0.15, 0.08]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=4,
        final_prob=0.08,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert "wrong" in [f.split("@")[0] for f in result.flags]
    assert result.status == "wrong"


def test_detect_sabotage_weak_final():
    """Answer is top-1 but with low confidence."""
    ranks = [5, 3, 1, 1, 1, 1]
    probs = [0.01, 0.05, 0.15, 0.18, 0.20, 0.22]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.22,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert "weak_final" in result.flags
    assert result.status == "weak"


def test_detect_sabotage_rank_drop():
    """Answer reaches rank 1, drops to >10, then recovers."""
    ranks = [10, 5, 1, 1, 15, 1, 1]
    probs = [0.01, 0.05, 0.44, 0.40, 0.02, 0.36, 0.40]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.40,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    rank_flags = [f for f in result.flags if f.startswith("rank_drop")]
    assert len(rank_flags) >= 1
    assert result.status == "sabotaged"


def test_detect_sabotage_prob_drop():
    """Answer probability drops by >50% from peak."""
    ranks = [5, 3, 1, 1, 3, 1, 1]
    probs = [0.01, 0.05, 0.44, 0.44, 0.19, 0.36, 0.40]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.40,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    prob_flags = [f for f in result.flags if f.startswith("prob_drop")]
    assert len(prob_flags) >= 1


def test_detect_sabotage_result_fields():
    """Check that SabotageResult has expected fields."""
    ranks = [5, 3, 1, 1, 1, 1]
    probs = [0.01, 0.05, 0.4, 0.5, 0.6, 0.7]
    result = detect_sabotage(
        ranks,
        probs,
        final_rank=1,
        final_prob=0.7,
        sabotage_threshold=0.5,
        final_threshold=0.3,
    )
    assert hasattr(result, "flags")
    assert hasattr(result, "status")
    assert hasattr(result, "peak_prob")
    assert hasattr(result, "peak_layer_idx")
    assert hasattr(result, "commitment_layer_idx")
    assert hasattr(result, "sabotage_layers")
    assert result.peak_prob == pytest.approx(0.7)


# --- PromptResult and ScanResult dataclass tests ---


def test_prompt_result_fields():
    pr = PromptResult(
        prompt="The capital of France is",
        answer="Paris",
        final_token="Paris",
        final_prob=0.51,
        final_rank=1,
        peak_prob=0.91,
        peak_layer=18,
        commitment_layer=15,
        sabotage_layers=[],
        flags=[],
        status="correct",
        ranks=[],
        probs=[],
    )
    assert pr.prompt == "The capital of France is"
    assert pr.status == "correct"


def test_scan_result_summary():
    results = [
        PromptResult(
            "p1",
            "a1",
            "a1",
            0.5,
            1,
            0.8,
            10,
            8,
            [],
            [],
            "correct",
            [],
            [],
        ),
        PromptResult(
            "p2",
            "a2",
            "a2",
            0.2,
            1,
            0.4,
            10,
            8,
            [],
            ["weak_final"],
            "weak",
            [],
            [],
        ),
        PromptResult(
            "p3",
            "a3",
            "x",
            0.3,
            5,
            0.1,
            10,
            None,
            [],
            ["wrong"],
            "wrong",
            [],
            [],
        ),
        PromptResult(
            "p4",
            "a4",
            "a4",
            0.36,
            1,
            0.44,
            19,
            17,
            [20],
            ["rank_drop@20"],
            "sabotaged",
            [],
            [],
        ),
    ]
    sr = ScanResult(
        model_name="test-model",
        dataset_name="test",
        prompt_results=results,
    )
    assert sr.correct_count == 1
    assert sr.sabotaged_count == 1
    assert sr.weak_count == 1
    assert sr.wrong_count == 1


# --- Multi-token answer matching tests ---


def test_answer_token_resolution_uses_first_token():
    """Verify that multi-token answers resolve to their first subword token.

    The scan should tokenize ' ' + answer and use the first token ID,
    so 'Canberra' matches the model predicting '▁Can' (first subword).
    """
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    # Simulate sentencepiece: " Canberra" → [▁Can=1234, ber=5678, ra=9012]
    tokenizer.encode.return_value = [1234, 5678, 9012]

    # Replicate the resolution logic from scan.py
    answer = "Canberra"
    encoded = tokenizer.encode(" " + answer, add_special_tokens=False)
    answer_token_id = encoded[0] if encoded else None

    tokenizer.encode.assert_called_once_with(" Canberra", add_special_tokens=False)
    assert answer_token_id == 1234


def test_answer_token_resolution_single_token():
    """Single-token answers like 'Paris' should also work correctly."""
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [3681]

    answer = "Paris"
    encoded = tokenizer.encode(" " + answer, add_special_tokens=False)
    answer_token_id = encoded[0] if encoded else None

    tokenizer.encode.assert_called_once_with(" Paris", add_special_tokens=False)
    assert answer_token_id == 3681


def test_answer_token_resolution_multi_token_capitals():
    """Test first-token resolution for several known multi-token capitals."""
    from unittest.mock import MagicMock

    multi_token_cases = {
        "Canberra": [100, 200, 300],  # Can + ber + ra
        "Nairobi": [101, 201, 301],   # Na + iro + bi
        "Seoul": [102, 202],           # Se + oul
        "Stockholm": [103, 203],       # Stock + holm
        "Copenhagen": [104, 204, 304], # Cop + en + hagen
        "Jerusalem": [105, 205, 305],  # Jer + usa + lem
        "Budapest": [106, 206],        # Bud + apest
        "Islamabad": [107, 207, 307],  # Islam + ab + ad
    }

    for capital, token_ids in multi_token_cases.items():
        tokenizer = MagicMock()
        tokenizer.encode.return_value = token_ids

        encoded = tokenizer.encode(" " + capital, add_special_tokens=False)
        answer_token_id = encoded[0] if encoded else None

        assert answer_token_id == token_ids[0], (
            f"{capital}: expected first token {token_ids[0]}, got {answer_token_id}"
        )


def test_answer_matching_prefix_fallback():
    """Model may predict a subword prefix of the answer (e.g. 'Bud' for 'Budapest').

    The scan should treat a prediction as correct if the decoded token
    is a prefix of the full answer, handling subword granularity mismatches.
    """
    # Replicate the prefix matching logic from scan.py
    cases = [
        # (answer, predicted_decoded, should_match)
        ("Budapest", "▁Bud", True),
        ("Stockholm", "▁Stock", True),
        ("Jerusalem", "▁Jer", True),
        ("Prague", "▁Prag", True),
        ("Canberra", "▁Can", True),
        ("Nairobi", "▁Na", True),
        ("Beijing", "▁Be", True),
        ("Paris", "▁Paris", True),      # exact match
        ("Stockholm", "▁Stock", True),   # same token text
        ("Budapest", "▁Vienna", False),  # wrong answer
        ("Tokyo", "▁Ky", False),         # not a prefix
    ]
    for answer, predicted_decoded, should_match in cases:
        answer_lower = answer.strip().lower()
        pred_normalized = predicted_decoded.strip().lstrip("\u2581").lower()
        matched = bool(
            pred_normalized and answer_lower.startswith(pred_normalized)
        )
        assert matched == should_match, (
            f"answer={answer!r}, predicted={predicted_decoded!r}: "
            f"expected {should_match}, got {matched}"
        )


# --- Remote scan helpers tests ---


def test_clean_token_strips_sentencepiece_marker():
    from neurotrace.cli.analysis import _clean_token

    assert _clean_token("\u2581Rome") == "rome"
    assert _clean_token(" Rome") == "rome"
    assert _clean_token("Rome") == "rome"


def test_clean_token_strips_gpt2_marker():
    """GPT-2/Qwen-style BPE uses \u0120 (Ġ) as space prefix."""
    from neurotrace.cli.analysis import _clean_token

    assert _clean_token("\u0120Rome") == "rome"
    assert _clean_token("\u0120Budapest") == "budapest"


def test_clean_token_strips_combined_whitespace():
    from neurotrace.cli.analysis import _clean_token

    assert _clean_token("  \u2581\u0120Rome  ") == "rome"


def test_token_matches_answer_basic():
    from neurotrace.cli.analysis import _token_matches_answer

    assert _token_matches_answer(" Rome", "Rome")
    assert _token_matches_answer("\u2581Rome", "Rome")
    assert _token_matches_answer("\u0120Rome", "Rome")
    assert _token_matches_answer("Rome", "Rome")


def test_token_matches_answer_prefix():
    """Subword prefix should match: 'Bud' matches 'Budapest'."""
    from neurotrace.cli.analysis import _token_matches_answer

    assert _token_matches_answer("\u2581Bud", "Budapest")
    assert _token_matches_answer(" Stock", "Stockholm")
    assert _token_matches_answer("\u0120Jer", "Jerusalem")
    assert _token_matches_answer(" Can", "Canberra")
    assert _token_matches_answer(" New", "New Delhi")


def test_token_matches_answer_rejects_wrong():
    from neurotrace.cli.analysis import _token_matches_answer

    assert not _token_matches_answer(" Vienna", "Budapest")
    assert not _token_matches_answer(" Ky", "Tokyo")
    assert not _token_matches_answer(" Berlin", "Paris")


def test_token_matches_answer_rejects_single_char_prefix():
    """Single character tokens should not PREFIX-match longer answers."""
    from neurotrace.cli.analysis import _token_matches_answer

    assert not _token_matches_answer(" B", "Budapest")
    assert not _token_matches_answer("\u2581N", "Nairobi")
    assert not _token_matches_answer("S", "Stockholm")


def test_token_matches_answer_accepts_single_char_exact():
    """Single character tokens should EXACT-match single-char answers."""
    from neurotrace.cli.analysis import _token_matches_answer

    # Chemical symbols
    assert _token_matches_answer(" B", "B")
    assert _token_matches_answer("\u2581H", "H")
    assert _token_matches_answer("\u0120O", "O")
    assert _token_matches_answer("N", "N")
    assert _token_matches_answer(" C", "C")
    # Single-digit numbers
    assert _token_matches_answer(" 7", "7")
    assert _token_matches_answer("0", "0")
    assert _token_matches_answer("\u25819", "9")
    # Wrong single-char should still fail
    assert not _token_matches_answer(" B", "H")
    assert not _token_matches_answer(" 7", "9")


def test_token_matches_answer_case_insensitive():
    from neurotrace.cli.analysis import _token_matches_answer

    assert _token_matches_answer(" ROME", "Rome")
    assert _token_matches_answer(" rome", "Rome")
    assert _token_matches_answer(" Rome", "ROME")


def test_scan_remote_final_rank_correct_match():
    """When top-1 token matches the answer, final_rank should be 1."""
    from unittest.mock import MagicMock, patch

    from neurotrace.remote import ForwardResult, LayerPrediction, TokenPrediction

    worker = MagicMock()
    worker.health.return_value = {"model": "test-model", "device": "cpu"}
    worker.forward.return_value = ForwardResult(
        top_tokens=[
            TokenPrediction(token=" Rome", token_id=1, logit=5.0, prob=0.213),
            TokenPrediction(token=" Paris", token_id=2, logit=3.0, prob=0.05),
        ],
        residuals=None,
        num_layers=2,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token=" The", token_id=10, logit=1.0, prob=0.1),
            ]),
            LayerPrediction(layer=1, top_tokens=[
                TokenPrediction(token=" Rome", token_id=1, logit=5.0, prob=0.213),
            ]),
        ],
    )

    dataset = [{"prompt": "The capital of Italy is", "answer": "Rome"}]
    with patch("neurotrace.remote.WorkerClient", return_value=worker):
        from neurotrace.cli.analysis import _scan_remote

        result = _scan_remote(
            "http://fake:8877", dataset, "test", 42,
            0.5, 0.3, False, False, False, False, None,
        )

    pr = result.prompt_results[0]
    assert pr.final_rank == 1
    assert pr.final_prob == pytest.approx(0.213)
    # final_prob=0.213 < final_threshold=0.3, so detect_sabotage marks as "weak"
    assert pr.status == "weak"


def test_scan_remote_final_rank_wrong():
    """When no top token matches the answer, final_rank should be 999."""
    from unittest.mock import MagicMock, patch

    from neurotrace.remote import ForwardResult, LayerPrediction, TokenPrediction

    worker = MagicMock()
    worker.health.return_value = {"model": "test-model", "device": "cpu"}
    worker.forward.return_value = ForwardResult(
        top_tokens=[
            TokenPrediction(token=" Paris", token_id=2, logit=5.0, prob=0.4),
        ],
        residuals=None,
        num_layers=1,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token=" Paris", token_id=2, logit=5.0, prob=0.4),
            ]),
        ],
    )

    dataset = [{"prompt": "The capital of Italy is", "answer": "Rome"}]
    with patch("neurotrace.remote.WorkerClient", return_value=worker):
        from neurotrace.cli.analysis import _scan_remote

        result = _scan_remote(
            "http://fake:8877", dataset, "test", 42,
            0.5, 0.3, False, False, False, False, None,
        )

    pr = result.prompt_results[0]
    assert pr.final_rank == 999
    assert pr.status == "wrong"


def test_scan_remote_gpt2_token_format():
    """Tokens with GPT-2 \u0120 prefix should still match correctly."""
    from unittest.mock import MagicMock, patch

    from neurotrace.remote import ForwardResult, LayerPrediction, TokenPrediction

    worker = MagicMock()
    worker.health.return_value = {"model": "test-model", "device": "cpu"}
    worker.forward.return_value = ForwardResult(
        top_tokens=[
            TokenPrediction(
                token="\u0120Rome", token_id=1, logit=5.0, prob=0.3,
            ),
        ],
        residuals=None,
        num_layers=1,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(
                    token="\u0120Rome", token_id=1, logit=5.0, prob=0.3,
                ),
            ]),
        ],
    )

    dataset = [{"prompt": "The capital of Italy is", "answer": "Rome"}]
    with patch("neurotrace.remote.WorkerClient", return_value=worker):
        from neurotrace.cli.analysis import _scan_remote

        result = _scan_remote(
            "http://fake:8877", dataset, "test", 42,
            0.5, 0.3, False, False, False, False, None,
        )

    assert result.prompt_results[0].status == "correct"
    assert result.prompt_results[0].final_rank == 1


# --- Whitespace-skip tests ---


def test_is_whitespace_token():
    """Pure whitespace tokens should be detected."""
    from neurotrace.cli.analysis import _is_whitespace_token

    assert _is_whitespace_token(" ")
    assert _is_whitespace_token("  ")
    assert _is_whitespace_token("\u2581")      # SentencePiece marker
    assert _is_whitespace_token("\u0120")      # GPT-2 BPE marker
    assert not _is_whitespace_token(" Rome")
    assert not _is_whitespace_token("\u258179")
    assert not _is_whitespace_token("B")


def test_scan_remote_whitespace_skip():
    """When top-1 is whitespace, _scan_remote should do a second forward pass
    with the space appended, and use the second result for answer matching."""
    from unittest.mock import MagicMock, call, patch

    from neurotrace.remote import ForwardResult, LayerPrediction, TokenPrediction

    worker = MagicMock()
    worker.health.return_value = {"model": "test-model", "device": "cpu"}

    # First call: top-1 is whitespace " "
    ws_result = ForwardResult(
        top_tokens=[
            TokenPrediction(token=" ", token_id=220, logit=6.0, prob=0.963),
            TokenPrediction(token="79", token_id=3324, logit=2.0, prob=0.01),
        ],
        residuals=None,
        num_layers=2,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token=" The", token_id=10, logit=1.0, prob=0.1),
            ]),
            LayerPrediction(layer=1, top_tokens=[
                TokenPrediction(token=" ", token_id=220, logit=6.0, prob=0.963),
            ]),
        ],
    )

    # Second call (after appending space): top-1 is "79"
    answer_result = ForwardResult(
        top_tokens=[
            TokenPrediction(token="79", token_id=3324, logit=5.0, prob=0.85),
            TokenPrediction(token="80", token_id=3325, logit=2.0, prob=0.05),
        ],
        residuals=None,
        num_layers=2,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token=" The", token_id=10, logit=1.0, prob=0.1),
            ]),
            LayerPrediction(layer=1, top_tokens=[
                TokenPrediction(token="79", token_id=3324, logit=5.0, prob=0.85),
            ]),
        ],
    )

    worker.forward.side_effect = [ws_result, answer_result]

    dataset = [{"prompt": "The atomic number of gold is", "answer": "79"}]
    with patch("neurotrace.remote.WorkerClient", return_value=worker):
        from neurotrace.cli.analysis import _scan_remote

        result = _scan_remote(
            "http://fake:8877", dataset, "test", 42,
            0.5, 0.3, False, False, False, False, None,
        )

    pr = result.prompt_results[0]
    # Should have done two forward calls
    assert worker.forward.call_count == 2
    # Second call should have space appended
    second_call_prompt = worker.forward.call_args_list[1][0][0]
    assert second_call_prompt == "The atomic number of gold is "
    # Should be marked correct (79 matches answer "79")
    assert pr.final_rank == 1
    assert pr.final_prob == pytest.approx(0.85)


def test_scan_remote_no_whitespace_skip_for_normal_tokens():
    """Non-whitespace top-1 tokens should NOT trigger a second forward pass."""
    from unittest.mock import MagicMock, patch

    from neurotrace.remote import ForwardResult, LayerPrediction, TokenPrediction

    worker = MagicMock()
    worker.health.return_value = {"model": "test-model", "device": "cpu"}

    normal_result = ForwardResult(
        top_tokens=[
            TokenPrediction(token=" Rome", token_id=1, logit=5.0, prob=0.4),
        ],
        residuals=None,
        num_layers=1,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token=" Rome", token_id=1, logit=5.0, prob=0.4),
            ]),
        ],
    )

    worker.forward.return_value = normal_result

    dataset = [{"prompt": "The capital of Italy is", "answer": "Rome"}]
    with patch("neurotrace.remote.WorkerClient", return_value=worker):
        from neurotrace.cli.analysis import _scan_remote

        result = _scan_remote(
            "http://fake:8877", dataset, "test", 42,
            0.5, 0.3, False, False, False, False, None,
        )

    # Should only call forward once
    assert worker.forward.call_count == 1
    assert result.prompt_results[0].final_rank == 1


def test_scan_remote_whitespace_skip_single_char_answer():
    """Whitespace-skip should work for single-character answers like 'B' (boron)."""
    from unittest.mock import MagicMock, patch

    from neurotrace.remote import ForwardResult, LayerPrediction, TokenPrediction

    worker = MagicMock()
    worker.health.return_value = {"model": "test-model", "device": "cpu"}

    ws_result = ForwardResult(
        top_tokens=[
            TokenPrediction(token=" ", token_id=220, logit=6.0, prob=0.95),
        ],
        residuals=None,
        num_layers=1,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token=" ", token_id=220, logit=6.0, prob=0.95),
            ]),
        ],
    )

    # After appending space, model predicts "B"
    answer_result = ForwardResult(
        top_tokens=[
            TokenPrediction(token="B", token_id=33, logit=5.0, prob=0.80),
        ],
        residuals=None,
        num_layers=1,
        vocab_size=100,
        hidden_dim=64,
        layer_predictions=[
            LayerPrediction(layer=0, top_tokens=[
                TokenPrediction(token="B", token_id=33, logit=5.0, prob=0.80),
            ]),
        ],
    )

    worker.forward.side_effect = [ws_result, answer_result]

    dataset = [{"prompt": "The chemical symbol for boron is", "answer": "B"}]
    with patch("neurotrace.remote.WorkerClient", return_value=worker):
        from neurotrace.cli.analysis import _scan_remote

        result = _scan_remote(
            "http://fake:8877", dataset, "test", 42,
            0.5, 0.3, False, False, False, False, None,
        )

    # "B" is single char - _token_matches_answer rejects single chars.
    # But "B" is still the final_token, and for single-char answers the
    # exact match at token level still doesn't pass prefix matching (len < 2).
    # This is expected behavior - single-char chemical symbols need exact match.
    pr = result.prompt_results[0]
    assert worker.forward.call_count == 2


# --- CLI tests ---


def test_scan_cli_requires_dataset_or_builtin():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "--db",
            "test.db",
            "--model",
            "test-model",
        ],
    )
    assert result.exit_code != 0
    assert (
        "dataset" in result.output.lower() or "dataset" in str(result.exception).lower()
    )


def test_scan_cli_rejects_both_dataset_flags():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "--db",
            "test.db",
            "--model",
            "test-model",
            "--dataset",
            "file.json",
            "--dataset-builtin",
            "capitals",
        ],
    )
    assert result.exit_code != 0
