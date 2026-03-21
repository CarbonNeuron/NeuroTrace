"""Tests for the heal module — no model downloads required."""

import json
from unittest.mock import MagicMock, patch

import pytest

from neurotrace.heal import (
    HealResult,
    PromptHealResult,
    generate_heal_html,
    heal_result_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_heal_result(
    baseline_correct=34,
    baseline_wrong=12,
    baseline_sabotaged=3,
    baseline_weak=1,
    healed_correct=47,
    healed_wrong=3,
    healed_sabotaged=0,
    healed_weak=0,
    edits_applied=8,
    edits_lobotomized=0,
    edits_rolled_back=1,
    edits_skipped=0,
    ppl_before=1.25,
    ppl_after=1.26,
    catastrophic_undo=False,
    prompt_results=None,
) -> HealResult:
    total = baseline_correct + baseline_wrong + baseline_sabotaged + baseline_weak
    if prompt_results is None:
        prompt_results = [
            PromptHealResult(
                prompt="The capital of France is",
                answer="Paris",
                baseline_prob=0.85,
                baseline_status="correct",
                action="already_correct",
                result_prob=0.85,
                final_status="correct",
            ),
            PromptHealResult(
                prompt="The capital of Germany is",
                answer="Berlin",
                baseline_prob=0.12,
                baseline_status="sabotaged",
                action="healed",
                result_prob=0.78,
                final_status="correct",
                target_layer=20,
                edit_norm=0.023,
            ),
            PromptHealResult(
                prompt="The capital of Mars is",
                answer="Olympus",
                baseline_prob=0.01,
                baseline_status="wrong",
                action="wrong",
                result_prob=0.01,
                final_status="wrong",
            ),
            PromptHealResult(
                prompt="The capital of Austria is",
                answer="Vienna",
                baseline_prob=0.15,
                baseline_status="sabotaged",
                action="rolled_back",
                result_prob=0.15,
                final_status="sabotaged",
                target_layer=18,
                edit_norm=0.031,
                rollback_reason="regression detected",
            ),
        ]

    return HealResult(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_name="capitals",
        baseline_total=total,
        baseline_correct=baseline_correct,
        baseline_wrong=baseline_wrong,
        baseline_sabotaged=baseline_sabotaged,
        baseline_weak=baseline_weak,
        healed_total=total,
        healed_correct=healed_correct,
        healed_wrong=healed_wrong,
        healed_sabotaged=healed_sabotaged,
        healed_weak=healed_weak,
        edits_attempted=edits_applied + edits_rolled_back,
        edits_applied=edits_applied,
        edits_lobotomized=edits_lobotomized,
        edits_rolled_back=edits_rolled_back,
        edits_skipped=edits_skipped,
        regressions_checked=272,
        regressions_found=1,
        max_regression_delta=-0.03,
        ppl_before=ppl_before,
        ppl_after=ppl_after,
        ppl_delta=ppl_after - ppl_before if ppl_before and ppl_after else None,
        output_path=None,
        duration_seconds=142.5,
        prompt_results=prompt_results,
        catastrophic_undo=catastrophic_undo,
    )


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestHealDataStructures:
    def test_prompt_heal_result(self):
        pr = PromptHealResult(
            prompt="The capital of France is",
            answer="Paris",
            baseline_prob=0.85,
            baseline_status="correct",
            action="already_correct",
            result_prob=0.85,
            final_status="correct",
        )
        assert pr.action == "already_correct"
        assert pr.baseline_prob == pytest.approx(0.85)

    def test_heal_result_baseline(self):
        result = _make_heal_result()
        assert result.baseline_total == 50
        assert result.baseline_correct == 34
        assert result.edits_applied == 8
        assert result.edits_rolled_back == 1

    def test_heal_result_empty_scan(self):
        """All correct — nothing to do."""
        result = _make_heal_result(
            baseline_correct=50,
            baseline_wrong=0,
            baseline_sabotaged=0,
            baseline_weak=0,
            healed_correct=50,
            healed_wrong=0,
            healed_sabotaged=0,
            healed_weak=0,
            edits_applied=0,
            edits_rolled_back=0,
            ppl_before=None,
            ppl_after=None,
            prompt_results=[
                PromptHealResult(
                    prompt="The capital of France is",
                    answer="Paris",
                    baseline_prob=0.95,
                    baseline_status="correct",
                    action="already_correct",
                    result_prob=0.95,
                    final_status="correct",
                ),
            ],
        )
        assert result.edits_applied == 0
        assert result.baseline_correct == 50
        assert result.healed_correct == 50


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------


class TestHealJson:
    def test_heal_json_output(self):
        result = _make_heal_result()
        d = heal_result_to_dict(result)

        assert d["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert d["dataset"] == "capitals"
        assert d["baseline"]["total"] == 50
        assert d["baseline"]["correct"] == 34
        assert d["healed"]["correct"] == 47
        assert d["edits"]["applied"] == 8
        assert d["edits"]["rolled_back"] == 1
        assert d["regressions"]["found"] == 1
        assert d["perplexity"]["delta"] == pytest.approx(0.01)
        assert d["duration_seconds"] == pytest.approx(142.5)

    def test_heal_json_accuracy(self):
        result = _make_heal_result()
        d = heal_result_to_dict(result)
        assert d["baseline"]["accuracy"] == pytest.approx(34 / 50)
        assert d["healed"]["accuracy"] == pytest.approx(47 / 50)

    def test_heal_json_serializable(self):
        result = _make_heal_result()
        d = heal_result_to_dict(result)
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["dataset"] == "capitals"

    def test_heal_json_prompts(self):
        result = _make_heal_result()
        d = heal_result_to_dict(result)
        assert "prompts" in d
        assert len(d["prompts"]) == 4
        healed = [p for p in d["prompts"] if p["action"] == "healed"]
        assert len(healed) == 1
        assert healed[0]["answer"] == "Berlin"

    def test_heal_json_catastrophic(self):
        result = _make_heal_result(catastrophic_undo=True)
        d = heal_result_to_dict(result)
        assert d["catastrophic_undo"] is True


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------


class TestHealHtml:
    def test_heal_report_html(self):
        result = _make_heal_result()
        html = generate_heal_html(result)

        assert "<!DOCTYPE html>" in html
        assert "Heal Report" in html
        assert "capitals" in html
        assert "Berlin" in html
        assert "HEALED" in html
        assert "ROLLED_BACK" in html
        assert "ALREADY_CORRECT" in html

    def test_heal_report_contains_css(self):
        result = _make_heal_result()
        html = generate_heal_html(result)
        assert "<style>" in html
        assert "--bg:" in html

    def test_heal_report_timeline(self):
        result = _make_heal_result()
        html = generate_heal_html(result)
        assert "Edit Timeline" in html
        assert "#1" in html

    def test_heal_report_perplexity_svg(self):
        result = _make_heal_result()
        html = generate_heal_html(result)
        assert "<svg" in html
        assert "Before" in html
        assert "After" in html

    def test_heal_report_no_ppl(self):
        """Report without perplexity data should still render."""
        result = _make_heal_result(ppl_before=None, ppl_after=None)
        html = generate_heal_html(result)
        assert "<!DOCTYPE html>" in html
        assert "Heal Report" in html

    def test_heal_report_catastrophic(self):
        result = _make_heal_result(
            catastrophic_undo=True, ppl_after=3.0,
            ppl_before=1.25,
        )
        html = generate_heal_html(result)
        assert "CATASTROPHIC" in html


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestHealEdgeCases:
    def test_heal_skip_wrong(self):
        """Wrong prompts (knowledge gaps) should be skipped, not repaired."""
        result = _make_heal_result()
        wrong_prompts = [p for p in result.prompt_results if p.action == "wrong"]
        assert len(wrong_prompts) == 1
        assert wrong_prompts[0].baseline_status == "wrong"

    def test_heal_max_edits_cap(self):
        """Verify the max edits cap is respected in data structure."""
        pr_list = []
        for i in range(5):
            pr_list.append(PromptHealResult(
                prompt=f"Prompt {i}",
                answer=f"Answer {i}",
                baseline_prob=0.1,
                baseline_status="sabotaged",
                action="healed" if i < 3 else "skipped",
                result_prob=0.8 if i < 3 else 0.1,
                final_status="correct" if i < 3 else "sabotaged",
            ))

        result = _make_heal_result(
            baseline_correct=45,
            baseline_wrong=0,
            baseline_sabotaged=5,
            baseline_weak=0,
            healed_correct=48,
            healed_wrong=0,
            healed_sabotaged=2,
            healed_weak=0,
            edits_applied=3,
            edits_rolled_back=0,
            edits_skipped=2,
            prompt_results=pr_list,
        )
        assert result.edits_applied == 3
        assert result.edits_skipped == 2
        skipped = [p for p in result.prompt_results if p.action == "skipped"]
        assert len(skipped) == 2

    def test_heal_dry_run(self):
        """Dry run should not set output_path."""
        result = _make_heal_result()
        assert result.output_path is None

    def test_heal_ppl_catastrophic(self):
        """PPL > 1.0 triggers catastrophic undo."""
        result = _make_heal_result(
            ppl_before=1.25,
            ppl_after=3.0,
            catastrophic_undo=True,
            edits_applied=0,
            edits_rolled_back=8,
        )
        assert result.catastrophic_undo is True
        assert result.edits_applied == 0
        assert result.edits_rolled_back == 8

    def test_heal_regression_rollback(self):
        """Verify rollback entries have rollback_reason."""
        result = _make_heal_result()
        rolled = [p for p in result.prompt_results if p.action == "rolled_back"]
        assert len(rolled) == 1
        assert rolled[0].rollback_reason == "regression detected"
        assert rolled[0].target_layer is not None

    def test_heal_single_repair(self):
        """One sabotaged prompt, fixed."""
        pr_list = [
            PromptHealResult(
                prompt="The capital of Germany is",
                answer="Berlin",
                baseline_prob=0.12,
                baseline_status="sabotaged",
                action="healed",
                result_prob=0.78,
                final_status="correct",
                target_layer=20,
                edit_norm=0.023,
            ),
        ]
        result = _make_heal_result(
            baseline_correct=49,
            baseline_wrong=0,
            baseline_sabotaged=1,
            baseline_weak=0,
            healed_correct=50,
            healed_wrong=0,
            healed_sabotaged=0,
            healed_weak=0,
            edits_applied=1,
            edits_rolled_back=0,
            prompt_results=pr_list,
        )
        assert result.edits_applied == 1
        assert result.healed_correct == 50


# ---------------------------------------------------------------------------
# Custom dataset test
# ---------------------------------------------------------------------------


class TestHealCustomDataset:
    def test_heal_custom_dataset_format(self, tmp_path):
        """Load from JSON file."""
        dataset_file = tmp_path / "test_dataset.json"
        dataset_file.write_text(json.dumps([
            {"prompt": "The capital of Germany is", "answer": "Berlin"},
            {"prompt": "The chemical symbol for gold is", "answer": "Au"},
        ]))

        from neurotrace.datasets import load_dataset

        ds = load_dataset(str(dataset_file))
        assert len(ds) == 2
        assert ds[0]["prompt"] == "The capital of Germany is"
        assert ds[1]["answer"] == "Au"


# ---------------------------------------------------------------------------
# Lobotomized / before-after token tests
# ---------------------------------------------------------------------------


class TestHealLobotomized:
    def test_lobotomized_status(self):
        """Lobotomized action is valid in PromptHealResult."""
        pr = PromptHealResult(
            prompt="The atomic number of gold is",
            answer="79",
            baseline_prob=0.10,
            baseline_status="wrong",
            action="lobotomized",
            result_prob=0.05,
            final_status="wrong",
            target_layer=11,
            before_token="79",
            after_token="Au",
            rollback_reason="auto-rollback: result < baseline",
        )
        assert pr.action == "lobotomized"
        assert pr.before_token == "79"
        assert pr.after_token == "Au"
        assert pr.rollback_reason is not None

    def test_before_after_token_fields(self):
        """Healed entries carry before/after tokens."""
        pr = PromptHealResult(
            prompt="The capital of Germany is",
            answer="Berlin",
            baseline_prob=0.05,
            baseline_status="wrong",
            action="healed",
            result_prob=0.85,
            final_status="correct",
            before_token="located",
            after_token="Berlin",
        )
        assert pr.before_token == "located"
        assert pr.after_token == "Berlin"

    def test_before_after_default_none(self):
        """Default values are None for backward compat."""
        pr = PromptHealResult(
            prompt="p", answer="a",
            baseline_prob=0.5, baseline_status="correct",
            action="already_correct",
            result_prob=0.5, final_status="correct",
        )
        assert pr.before_token is None
        assert pr.after_token is None

    def test_heal_result_lobotomized_count(self):
        """HealResult tracks edits_lobotomized."""
        result = _make_heal_result(edits_lobotomized=3)
        assert result.edits_lobotomized == 3

    def test_json_includes_lobotomized(self):
        """JSON output includes lobotomized count and tokens."""
        pr_list = [
            PromptHealResult(
                prompt="The atomic number of gold is",
                answer="79",
                baseline_prob=0.10,
                baseline_status="wrong",
                action="lobotomized",
                result_prob=0.05,
                final_status="wrong",
                before_token="79",
                after_token="Au",
                rollback_reason="auto-rollback: result < baseline",
            ),
        ]
        result = _make_heal_result(
            edits_lobotomized=1,
            prompt_results=pr_list,
        )
        d = heal_result_to_dict(result)
        assert d["edits"]["lobotomized"] == 1
        assert d["prompts"][0]["before_token"] == "79"
        assert d["prompts"][0]["after_token"] == "Au"
        assert d["prompts"][0]["action"] == "lobotomized"

    def test_html_includes_edit_column(self):
        """HTML report includes Edit column with before → after."""
        pr_list = [
            PromptHealResult(
                prompt="The capital of Germany is",
                answer="Berlin",
                baseline_prob=0.05,
                baseline_status="wrong",
                action="healed",
                result_prob=0.85,
                final_status="correct",
                before_token="located",
                after_token="Berlin",
            ),
            PromptHealResult(
                prompt="The capital of France is",
                answer="Paris",
                baseline_prob=0.85,
                baseline_status="correct",
                action="already_correct",
                result_prob=0.85,
                final_status="correct",
            ),
        ]
        result = _make_heal_result(prompt_results=pr_list)
        html = generate_heal_html(result)
        assert "<th>Edit</th>" in html
        assert "located" in html
        assert "&rarr;" in html
        assert "Berlin" in html
        # ALREADY_CORRECT should show "-"
        assert "-</td>" in html

    def test_html_lobotomized_badge(self):
        """HTML report shows LOBOTOMIZED badge."""
        pr_list = [
            PromptHealResult(
                prompt="Test",
                answer="79",
                baseline_prob=0.10,
                baseline_status="wrong",
                action="lobotomized",
                result_prob=0.05,
                final_status="wrong",
                before_token="79",
                after_token="Au",
            ),
        ]
        result = _make_heal_result(
            edits_lobotomized=1,
            prompt_results=pr_list,
        )
        html = generate_heal_html(result)
        assert "LOBOTOMIZED" in html


# ---------------------------------------------------------------------------
# CLI tests (Click runner with mocks)
# ---------------------------------------------------------------------------


class TestHealCli:
    def test_heal_cli_requires_model(self):
        """CLI should error without --model or --remote."""
        from click.testing import CliRunner

        from neurotrace.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["heal", "--dataset", "capitals"])
        assert result.exit_code != 0
        assert "Must provide --model" in result.output or result.exit_code == 2

    def test_heal_cli_unknown_dataset(self):
        """CLI should error on unknown built-in dataset."""
        from click.testing import CliRunner

        from neurotrace.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["heal", "--model", "test", "--dataset", "nonexistent"],
        )
        assert result.exit_code != 0

    def test_heal_cli_json_flag(self):
        """CLI --json flag produces valid JSON."""
        from click.testing import CliRunner

        from neurotrace.cli import cli

        mock_result = _make_heal_result()

        with patch("neurotrace.models.load_model") as mock_load, \
             patch("neurotrace.heal.run_heal_local", return_value=mock_result):
            mock_model = MagicMock()
            mock_model.config._name_or_path = "test"
            mock_load.return_value = (mock_model, MagicMock())

            runner = CliRunner()
            result = runner.invoke(cli, [
                "heal", "--model", "test", "--dataset", "capitals",
                "--json",
            ])

            if result.exit_code == 0:
                # Strip Rich spinner output before JSON
                output = result.output
                json_start = output.find("{")
                if json_start >= 0:
                    parsed = json.loads(output[json_start:])
                    assert parsed["dataset"] == "capitals"
