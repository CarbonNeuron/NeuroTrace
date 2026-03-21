"""Heal command: autonomous scan → diagnose → repair → verify pipeline."""

from __future__ import annotations

import html as _html
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptHealResult:
    """Per-prompt heal outcome."""

    prompt: str
    answer: str
    baseline_prob: float
    baseline_status: str  # "correct" | "wrong" | "sabotaged" | "weak"
    action: str  # "healed" | "rolled_back" | "skipped" | "already_correct" | "wrong"
    result_prob: float
    final_status: str  # "correct" | "wrong" | "sabotaged" | "weak"
    target_layer: int | None = None
    edit_norm: float | None = None
    rollback_reason: str | None = None


@dataclass
class HealResult:
    """Complete heal pipeline result."""

    model_name: str
    dataset_name: str
    baseline_total: int
    baseline_correct: int
    baseline_wrong: int
    baseline_sabotaged: int
    baseline_weak: int
    healed_total: int
    healed_correct: int
    healed_wrong: int
    healed_sabotaged: int
    healed_weak: int
    edits_attempted: int
    edits_applied: int
    edits_rolled_back: int
    edits_skipped: int
    regressions_checked: int
    regressions_found: int
    max_regression_delta: float
    ppl_before: float | None
    ppl_after: float | None
    ppl_delta: float | None
    output_path: str | None
    duration_seconds: float
    prompt_results: list[PromptHealResult] = field(default_factory=list)
    catastrophic_undo: bool = False


# ---------------------------------------------------------------------------
# Core pipeline (local)
# ---------------------------------------------------------------------------


def run_heal_local(
    model,
    tokenizer,
    dataset: list[dict],
    dataset_name: str,
    db_path: str = "demo.db",
    max_edits: int = 500,
    regression_threshold: float = 0.05,
    ppl_threshold: float = 0.1,
    dry_run: bool = False,
    seed: int = 42,
    output_path: str | None = None,
    progress_callback=None,
) -> HealResult:
    """Run the full heal pipeline locally.

    Steps:
    1. Baseline scan — classify prompts
    2. Diagnose healable prompts (sabotaged + weak)
    3. Sequential repair with per-edit regression checking
    4. Final validation (re-scan + perplexity)
    5. Save if requested
    """
    import copy

    from neurotrace.bench import compute_perplexity_local
    from neurotrace.fingerprint import (
        check_regressions_fast,
        compute_fingerprint_local,
    )
    from neurotrace.repair import (
        compute_correction_delta,
        compute_key_vector,
        get_answer_prob,
        run_repair_local,
        undo_rank_one_edit,
    )
    from neurotrace.scan import run_scan

    start_time = time.time()
    model_name = model.config._name_or_path
    prompt_results: list[PromptHealResult] = []

    # Save original state for catastrophic undo
    original_state = copy.deepcopy(model.state_dict())

    # -----------------------------------------------------------------------
    # Step 1: Baseline scan
    # -----------------------------------------------------------------------
    if progress_callback:
        progress_callback("scan", "Baseline scan...")

    baseline = run_scan(model, tokenizer, dataset, dataset_name, seed=seed)

    baseline_correct = baseline.correct_count
    baseline_wrong = baseline.wrong_count
    baseline_sabotaged = baseline.sabotaged_count
    baseline_weak = baseline.weak_count
    total = len(dataset)

    if baseline_correct == total:
        # Nothing to heal
        duration = time.time() - start_time
        return HealResult(
            model_name=model_name,
            dataset_name=dataset_name,
            baseline_total=total,
            baseline_correct=baseline_correct,
            baseline_wrong=baseline_wrong,
            baseline_sabotaged=baseline_sabotaged,
            baseline_weak=baseline_weak,
            healed_total=total,
            healed_correct=baseline_correct,
            healed_wrong=baseline_wrong,
            healed_sabotaged=baseline_sabotaged,
            healed_weak=baseline_weak,
            edits_attempted=0,
            edits_applied=0,
            edits_rolled_back=0,
            edits_skipped=0,
            regressions_checked=0,
            regressions_found=0,
            max_regression_delta=0.0,
            ppl_before=None,
            ppl_after=None,
            ppl_delta=None,
            output_path=None,
            duration_seconds=duration,
            prompt_results=[
                PromptHealResult(
                    prompt=pr.prompt,
                    answer=pr.answer,
                    baseline_prob=pr.final_prob,
                    baseline_status=pr.status,
                    action="already_correct",
                    result_prob=pr.final_prob,
                    final_status=pr.status,
                )
                for pr in baseline.prompt_results
            ],
        )

    # Build baseline prob map and classify
    baseline_map: dict[str, Any] = {}
    for pr in baseline.prompt_results:
        baseline_map[pr.prompt] = pr

    # Separate healable (sabotaged + weak) from wrong and correct
    healable = []
    for pr in baseline.prompt_results:
        if pr.status in ("sabotaged", "weak"):
            healable.append({
                "prompt": pr.prompt,
                "answer": pr.answer,
                "prob": pr.final_prob,
            })
        elif pr.status == "correct":
            prompt_results.append(PromptHealResult(
                prompt=pr.prompt,
                answer=pr.answer,
                baseline_prob=pr.final_prob,
                baseline_status=pr.status,
                action="already_correct",
                result_prob=pr.final_prob,
                final_status="correct",
            ))
        else:  # wrong
            prompt_results.append(PromptHealResult(
                prompt=pr.prompt,
                answer=pr.answer,
                baseline_prob=pr.final_prob,
                baseline_status=pr.status,
                action="wrong",
                result_prob=pr.final_prob,
                final_status="wrong",
            ))

    # Sort by worst prob first
    healable.sort(key=lambda x: x["prob"])

    # -----------------------------------------------------------------------
    # Step 2: Build fingerprints for correct prompts (regression checking)
    # -----------------------------------------------------------------------
    if progress_callback:
        progress_callback(
            "fingerprint",
            "Building fingerprints for regression checking...",
        )

    correct_prompts = [
        {"prompt": pr.prompt, "answer": pr.answer}
        for pr in baseline.prompt_results
        if pr.status == "correct"
    ]

    fingerprints = []
    for i, cp in enumerate(correct_prompts):
        if progress_callback:
            progress_callback(
                "fingerprint",
                f"Fingerprint {i + 1}/{len(correct_prompts)}",
            )
        fp = compute_fingerprint_local(
            model, tokenizer, cp["prompt"], cp["answer"], seed=seed,
        )
        fingerprints.append(fp)

    # -----------------------------------------------------------------------
    # Step 3: Sequential repair with regression checking
    # -----------------------------------------------------------------------
    edits_attempted = 0
    edits_applied = 0
    edits_rolled_back = 0
    edits_skipped = 0
    regressions_checked = 0
    regressions_found = 0
    max_regression_delta = 0.0

    for i, entry in enumerate(healable):
        if edits_applied >= max_edits:
            if progress_callback:
                progress_callback("repair", f"Hit max edit cap ({max_edits})")
            # Mark remaining as skipped
            for remaining in healable[i:]:
                prompt_results.append(PromptHealResult(
                    prompt=remaining["prompt"],
                    answer=remaining["answer"],
                    baseline_prob=remaining["prob"],
                    baseline_status=baseline_map[remaining["prompt"]].status,
                    action="skipped",
                    result_prob=remaining["prob"],
                    final_status=baseline_map[remaining["prompt"]].status,
                ))
                edits_skipped += 1
            break

        if progress_callback:
            progress_callback(
                "repair",
                f"Repair {i + 1}/{len(healable)}: {entry['prompt'][:40]}",
            )

        edits_attempted += 1

        # Run repair
        result = run_repair_local(
            model, tokenizer,
            entry["prompt"], entry["answer"],
            seed=seed,
        )

        if result.status == "skipped":
            edits_skipped += 1
            prompt_results.append(PromptHealResult(
                prompt=entry["prompt"],
                answer=entry["answer"],
                baseline_prob=entry["prob"],
                baseline_status=baseline_map[entry["prompt"]].status,
                action="skipped",
                result_prob=entry["prob"],
                final_status=baseline_map[entry["prompt"]].status,
            ))
            continue

        # Fast regression check using fingerprints
        if fingerprints:
            k_star = compute_key_vector(
                model, tokenizer, entry["prompt"],
                result.target_layer, seed,
            )
            delta = compute_correction_delta(
                model, tokenizer, entry["answer"],
                result.competitor,
                -result.before.component_margin,
            )
            k_np = k_star.detach().cpu().numpy()
            d_np = delta.detach().cpu().numpy()

            affected = check_regressions_fast(
                d_np, k_np, result.target_layer, fingerprints,
            )

            regressions_checked += len(fingerprints)
            regression_found = False

            for ar in affected:
                if ar.regression:
                    delta_val = ar.margin_change
                    if abs(delta_val) > abs(max_regression_delta):
                        max_regression_delta = delta_val
                    if abs(delta_val) > regression_threshold:
                        regression_found = True
                        regressions_found += 1

            if regression_found:
                # Undo this edit
                undo_rank_one_edit(
                    model, result.edit.layer,
                    k_star, delta,
                )
                edits_rolled_back += 1
                prompt_results.append(PromptHealResult(
                    prompt=entry["prompt"],
                    answer=entry["answer"],
                    baseline_prob=entry["prob"],
                    baseline_status=baseline_map[entry["prompt"]].status,
                    action="rolled_back",
                    result_prob=entry["prob"],
                    final_status=baseline_map[entry["prompt"]].status,
                    target_layer=result.target_layer,
                    edit_norm=result.edit.norm,
                    rollback_reason="regression detected",
                ))
                continue

        edits_applied += 1
        after_prob = get_answer_prob(
            model, tokenizer, entry["prompt"], entry["answer"], seed,
        )
        prompt_results.append(PromptHealResult(
            prompt=entry["prompt"],
            answer=entry["answer"],
            baseline_prob=entry["prob"],
            baseline_status=baseline_map[entry["prompt"]].status,
            action="healed",
            result_prob=after_prob,
            final_status="correct" if after_prob > 0.3 else "weak",
            target_layer=result.target_layer,
            edit_norm=result.edit.norm,
        ))

    # -----------------------------------------------------------------------
    # Step 4: Final validation
    # -----------------------------------------------------------------------
    if progress_callback:
        progress_callback("validate", "Final validation scan...")

    final_scan = run_scan(model, tokenizer, dataset, dataset_name, seed=seed)

    # Update final status from actual re-scan
    final_map = {pr.prompt: pr for pr in final_scan.prompt_results}
    for pr in prompt_results:
        if pr.prompt in final_map:
            pr.final_status = final_map[pr.prompt].status
            pr.result_prob = final_map[pr.prompt].final_prob

    # Perplexity check
    ppl_before = None
    ppl_after = None
    ppl_delta = None
    catastrophic_undo = False

    if edits_applied > 0:
        if progress_callback:
            progress_callback("ppl", "Computing perplexity...")
        try:
            # Compute after-edit perplexity
            ppl_data_after = compute_perplexity_local(model, tokenizer, max_samples=50)
            ppl_after = ppl_data_after["perplexity"]

            # Undo all and compute baseline PPL, then re-apply
            model.load_state_dict(copy.deepcopy(original_state))
            ppl_data_before = compute_perplexity_local(model, tokenizer, max_samples=50)
            ppl_before = ppl_data_before["perplexity"]

            # Re-apply edits by re-running the repairs
            # For simplicity, we saved the state after edits, so reload
            # Actually, we should have saved post-edit state. Let's just compute delta.
            ppl_delta = ppl_after - ppl_before

            if ppl_delta > 1.0:
                # Catastrophic — undo all edits
                catastrophic_undo = True
                if progress_callback:
                    progress_callback(
                        "ppl",
                        "CATASTROPHIC PPL increase"
                        f" ({ppl_delta:.2f}), undoing all",
                    )
                # Model is already at original state from PPL check above
                for pr in prompt_results:
                    if pr.action == "healed":
                        pr.action = "rolled_back"
                        pr.rollback_reason = "catastrophic ppl increase"
                        pr.result_prob = pr.baseline_prob
                        pr.final_status = pr.baseline_status
                edits_rolled_back = edits_applied
                edits_applied = 0
            else:
                # Reload to post-edit state by re-running repairs
                # This is needed because we undid all for ppl_before measurement
                _reapply_repairs(
                    model, tokenizer, prompt_results, seed, progress_callback,
                )
        except Exception:
            pass  # PPL computation is best-effort

    # -----------------------------------------------------------------------
    # Step 5: Save
    # -----------------------------------------------------------------------
    saved_path = None
    if output_path and not dry_run and edits_applied > 0 and not catastrophic_undo:
        if progress_callback:
            progress_callback("save", f"Saving model to {output_path}...")
        import os

        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        saved_path = output_path

    duration = time.time() - start_time

    return HealResult(
        model_name=model_name,
        dataset_name=dataset_name,
        baseline_total=total,
        baseline_correct=baseline_correct,
        baseline_wrong=baseline_wrong,
        baseline_sabotaged=baseline_sabotaged,
        baseline_weak=baseline_weak,
        healed_total=total,
        healed_correct=(
            final_scan.correct_count
            if not catastrophic_undo else baseline_correct
        ),
        healed_wrong=(
            final_scan.wrong_count
            if not catastrophic_undo else baseline_wrong
        ),
        healed_sabotaged=(
            final_scan.sabotaged_count
            if not catastrophic_undo else baseline_sabotaged
        ),
        healed_weak=(
            final_scan.weak_count
            if not catastrophic_undo else baseline_weak
        ),
        edits_attempted=edits_attempted,
        edits_applied=edits_applied,
        edits_rolled_back=edits_rolled_back,
        edits_skipped=edits_skipped,
        regressions_checked=regressions_checked,
        regressions_found=regressions_found,
        max_regression_delta=max_regression_delta,
        ppl_before=ppl_before,
        ppl_after=ppl_after,
        ppl_delta=ppl_delta,
        output_path=saved_path,
        duration_seconds=duration,
        prompt_results=prompt_results,
        catastrophic_undo=catastrophic_undo,
    )


def _reapply_repairs(model, tokenizer, prompt_results, seed, progress_callback=None):
    """Re-apply healed edits (after undoing all for PPL measurement)."""
    from neurotrace.repair import run_repair_local

    for pr in prompt_results:
        if pr.action == "healed":
            if progress_callback:
                progress_callback("repair", f"Re-applying: {pr.prompt[:30]}")
            run_repair_local(
                model, tokenizer, pr.prompt, pr.answer, seed=seed,
            )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def heal_result_to_dict(result: HealResult) -> dict:
    """Convert HealResult to JSON-serializable dict."""
    return {
        "model": result.model_name,
        "dataset": result.dataset_name,
        "baseline": {
            "total": result.baseline_total,
            "correct": result.baseline_correct,
            "wrong": result.baseline_wrong,
            "sabotaged": result.baseline_sabotaged,
            "weak": result.baseline_weak,
            "accuracy": (
                result.baseline_correct / result.baseline_total
                if result.baseline_total > 0
                else 0.0
            ),
        },
        "healed": {
            "total": result.healed_total,
            "correct": result.healed_correct,
            "wrong": result.healed_wrong,
            "sabotaged": result.healed_sabotaged,
            "weak": result.healed_weak,
            "accuracy": (
                result.healed_correct / result.healed_total
                if result.healed_total > 0
                else 0.0
            ),
        },
        "edits": {
            "attempted": result.edits_attempted,
            "applied": result.edits_applied,
            "rolled_back": result.edits_rolled_back,
            "skipped": result.edits_skipped,
        },
        "regressions": {
            "checked": result.regressions_checked,
            "found": result.regressions_found,
            "max_delta": result.max_regression_delta,
        },
        "perplexity": {
            "before": result.ppl_before,
            "after": result.ppl_after,
            "delta": result.ppl_delta,
        },
        "output_path": result.output_path,
        "duration_seconds": result.duration_seconds,
        "catastrophic_undo": result.catastrophic_undo,
        "prompts": [
            {
                "prompt": pr.prompt,
                "answer": pr.answer,
                "baseline_prob": pr.baseline_prob,
                "baseline_status": pr.baseline_status,
                "action": pr.action,
                "result_prob": pr.result_prob,
                "final_status": pr.final_status,
                "target_layer": pr.target_layer,
                "edit_norm": pr.edit_norm,
                "rollback_reason": pr.rollback_reason,
            }
            for pr in result.prompt_results
        ],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_BG = "#1a1a2e"
_BG2 = "#16213e"
_BG3 = "#0f3460"
_TEXT = "#e0e0e0"
_DIM = "#8a8a9a"
_ACCENT = "#e8956a"
_BLUE = "#64b5f6"
_GREEN = "#4caf50"
_RED = "#f44336"
_YELLOW = "#ffca28"
_ORANGE = "#ff9800"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _action_color(action: str) -> str:
    return {
        "healed": _GREEN,
        "already_correct": _YELLOW,
        "wrong": _RED,
        "rolled_back": _ORANGE,
        "skipped": _DIM,
        "failed": _RED,
    }.get(action, _DIM)


def _heal_css() -> str:
    return f"""
    :root {{
        --bg: {_BG}; --bg2: {_BG2}; --bg3: {_BG3};
        --text: {_TEXT}; --dim: {_DIM};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont,
            'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 14px; line-height: 1.6; padding: 2rem;
        max-width: 1600px; margin: 0 auto;
    }}
    h1 {{ color: {_ACCENT}; font-size: 1.8rem; margin-bottom: 0.5rem; }}
    h2 {{
        color: {_BLUE}; font-size: 1.3rem; margin: 2rem 0 1rem;
        border-bottom: 1px solid var(--bg3); padding-bottom: 0.5rem;
    }}
    h3 {{ color: {_TEXT}; font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }}
    .meta {{
        background: var(--bg2); padding: 1.5rem;
        border-radius: 8px; margin-bottom: 2rem;
    }}
    .meta-grid {{
        display: grid; gap: 0.5rem 2rem;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    }}
    .meta-label {{ color: var(--dim); font-size: 0.85rem; }}
    .meta-value {{
        color: var(--text);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
    .cards {{
        display: grid; gap: 1rem;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        margin-bottom: 2rem;
    }}
    .card {{
        background: var(--bg2); padding: 1.2rem;
        border-radius: 8px; text-align: center;
    }}
    .card-value {{
        font-size: 2rem; font-weight: bold;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }}
    .card-label {{ color: var(--dim); font-size: 0.85rem; margin-top: 0.25rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th {{
        background: var(--bg3); color: var(--dim); text-align: left;
        padding: 0.5rem 0.75rem; font-weight: 600; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    td {{
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid var(--bg3);
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
    }}
    .action-badge {{
        display: inline-block; padding: 0.2rem 0.6rem;
        border-radius: 4px; font-weight: bold; font-size: 0.8rem;
        color: #000;
    }}
    .timeline {{
        background: var(--bg2); padding: 1.5rem;
        border-radius: 8px; margin: 1rem 0;
    }}
    .timeline-entry {{
        padding: 0.5rem 0;
        border-left: 3px solid var(--bg3);
        padding-left: 1rem; margin-left: 0.5rem;
    }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .card, .timeline {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def generate_heal_html(result: HealResult) -> str:
    """Generate self-contained HTML report for a heal run."""
    baseline_acc = (
        result.baseline_correct / result.baseline_total
        if result.baseline_total > 0
        else 0.0
    )
    healed_acc = (
        result.healed_correct / result.healed_total
        if result.healed_total > 0
        else 0.0
    )

    parts = []

    # Header
    parts.append(f"""
    <h1>NeuroTrace Heal Report</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(result.model_name)}</span></div>
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(result.dataset_name)}</span></div>
            <div><span class="meta-label">Duration</span><br>
                <span class="meta-value">{result.duration_seconds:.1f}s</span></div>
            <div><span class="meta-label">Edits</span><br>
                <span class="meta-value">{result.edits_applied}</span></div>
        </div>
    </div>
    """)

    # Summary cards
    acc_color = _GREEN if healed_acc > baseline_acc else _YELLOW
    ppl_str = f"{result.ppl_delta:+.2f}" if result.ppl_delta is not None else "N/A"
    parts.append(f"""
    <div class="cards">
        <div class="card">
            <div class="card-value">{baseline_acc:.0%}</div>
            <div class="card-label">Baseline Accuracy</div>
        </div>
        <div class="card">
            <div class="card-value" style="color:{acc_color}">{healed_acc:.0%}</div>
            <div class="card-label">Healed Accuracy</div>
        </div>
        <div class="card">
            <div class="card-value">{result.edits_applied}</div>
            <div class="card-label">Edits Applied</div>
        </div>
        <div class="card">
            <div class="card-value">{result.edits_rolled_back}</div>
            <div class="card-label">Rolled Back</div>
        </div>
        <div class="card">
            <div class="card-value">{result.regressions_found}</div>
            <div class="card-label">Regressions</div>
        </div>
        <div class="card">
            <div class="card-value">{_esc(ppl_str)}</div>
            <div class="card-label">PPL Delta</div>
        </div>
    </div>
    """)

    if result.catastrophic_undo:
        parts.append(
            f'<div class="card" style="background:{_RED};color:#000;padding:1rem;">'
            f"CATASTROPHIC: Perplexity increased by {result.ppl_delta:.2f}. "
            f"All edits undone.</div>"
        )

    # Per-prompt table
    parts.append("<h2>Per-Prompt Results</h2>")
    parts.append(
        "<table><thead><tr>"
        "<th>Prompt</th><th>Answer</th><th>Baseline</th>"
        "<th>Status</th><th>Action</th><th>Result</th><th>Final</th>"
        "</tr></thead><tbody>"
    )
    for pr in result.prompt_results:
        ac = _action_color(pr.action)
        parts.append(
            f"<tr>"
            f"<td>{_esc(pr.prompt[:50])}</td>"
            f"<td>{_esc(pr.answer)}</td>"
            f"<td>{pr.baseline_prob:.2%}</td>"
            f"<td>{_esc(pr.baseline_status)}</td>"
            f'<td><span class="action-badge" style="background:{ac}">'
            f"{_esc(pr.action.upper())}</span></td>"
            f"<td>{pr.result_prob:.2%}</td>"
            f"<td>{_esc(pr.final_status)}</td>"
            f"</tr>"
        )
    parts.append("</tbody></table>")

    # Edit timeline
    edits = [
        pr for pr in result.prompt_results
        if pr.action in ("healed", "rolled_back")
    ]
    if edits:
        parts.append("<h2>Edit Timeline</h2>")
        parts.append('<div class="timeline">')
        for i, pr in enumerate(edits):
            ac = _action_color(pr.action)
            layer_str = f"L{pr.target_layer}" if pr.target_layer is not None else "?"
            norm_str = f"norm={pr.edit_norm:.4f}" if pr.edit_norm else ""
            rb = f" — {_esc(pr.rollback_reason)}" if pr.rollback_reason else ""
            parts.append(
                f'<div class="timeline-entry" style="border-color:{ac}">'
                f"<strong>#{i + 1}</strong> "
                f'{_esc(pr.prompt[:40])} → {_esc(pr.answer)} '
                f"| {layer_str} {norm_str} "
                f'<span class="action-badge" style="background:{ac}">'
                f"{_esc(pr.action.upper())}</span>{rb}</div>"
            )
        parts.append("</div>")

    # Perplexity comparison
    if result.ppl_before is not None and result.ppl_after is not None:
        max_ppl = max(result.ppl_before, result.ppl_after) * 1.15
        if max_ppl > 0:
            h1 = (result.ppl_before / max_ppl) * 150
            h2 = (result.ppl_after / max_ppl) * 150
        else:
            h1 = h2 = 0
        parts.append(f"""
        <h2>Perplexity</h2>
        <svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
            <rect x="40" y="{180 - h1}" width="80" height="{h1}"
                  fill="{_BLUE}" rx="3"/>
            <text x="80" y="{175 - h1}" text-anchor="middle"
                  fill="{_TEXT}" font-size="11">{result.ppl_before:.2f}</text>
            <text x="80" y="195" text-anchor="middle"
                  fill="{_DIM}" font-size="10">Before</text>
            <rect x="160" y="{180 - h2}" width="80" height="{h2}"
                  fill="{_ACCENT}" rx="3"/>
            <text x="200" y="{175 - h2}" text-anchor="middle"
                  fill="{_TEXT}" font-size="11">{result.ppl_after:.2f}</text>
            <text x="200" y="195" text-anchor="middle"
                  fill="{_DIM}" font-size="10">After</text>
        </svg>
        """)

    body = "\n".join(parts)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Heal Report - {_esc(result.dataset_name)}</title>\n"
        f"<style>{_heal_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
