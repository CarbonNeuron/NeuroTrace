"""ROME-style rank-one weight repair using Logit Prism decomposition."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RepairBefore:
    answer_logit: float
    competitor_logit: float
    margin: float
    component_margin: float
    answer_prob: float


@dataclass
class RepairAfter:
    answer_logit: float
    competitor_logit: float
    margin: float
    component_margin: float
    answer_prob: float


@dataclass
class RepairEdit:
    matrix: str  # "down_proj"
    layer: int
    rank: int  # always 1 for ROME
    norm: float


@dataclass
class RegressionResult:
    prompt: str
    answer: str
    before_prob: float
    after_prob: float
    status: str  # "ok" | "regression"


@dataclass
class RepairResult:
    prompt: str
    answer: str
    competitor: str
    target_layer: int
    target_component: str
    before: RepairBefore
    after: RepairAfter
    edit: RepairEdit
    regressions: list[RegressionResult] = field(default_factory=list)
    status: str = "repaired"  # "repaired" | "skipped" | "regression"


@dataclass
class RepairRun:
    run_id: str
    dataset: str | None
    model_name: str
    prompt_count: int
    results: list[RepairResult]
    created_at: str


# ---------------------------------------------------------------------------
# Local ROME repair
# ---------------------------------------------------------------------------


def compute_key_vector(model, tokenizer, prompt: str, layer: int, seed: int = 42):
    """Extract the MLP hidden state (key vector k*) at the target layer.

    This is the activation after gate_proj + up_proj + SwiGLU, before down_proj.
    We hook down_proj with a forward_pre_hook to capture its input.
    """
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    k_star = {}

    def hook_fn(module, args):
        # down_proj input: (batch, seq, intermediate_size) or (intermediate_size,)
        inp = args[0] if isinstance(args, tuple) else args
        if inp.dim() == 3:
            k_star["value"] = inp[0, -1, :].detach().clone()
        elif inp.dim() == 2:
            k_star["value"] = inp[-1, :].detach().clone()
        else:
            k_star["value"] = inp.detach().clone()

    mlp = model.model.layers[layer].mlp
    handle = mlp.down_proj.register_forward_pre_hook(hook_fn)

    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return k_star["value"]


def compute_correction_delta(
    model,
    tokenizer,
    answer: str,
    competitor: str,
    deficit: float,
):
    """Compute the minimum-norm output correction vector delta.

    delta = (deficit / (P_margin @ P_margin)) * P_margin
    where P_margin = P_answer - P_competitor (margin direction in output space).
    """
    import torch

    w_unembed = model.lm_head.weight.detach()
    ln_weight = model.model.norm.weight.detach()

    # Get token IDs
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    comp_ids = tokenizer.encode(" " + competitor, add_special_tokens=False)
    if not comp_ids:
        comp_ids = tokenizer.encode(competitor, add_special_tokens=False)

    answer_id = answer_ids[0] if answer_ids else 0
    comp_id = comp_ids[0] if comp_ids else 0

    # P_answer and P_competitor in unembed space (ignoring norm scale since
    # it cancels in the margin direction)
    p_answer = w_unembed[answer_id] * ln_weight
    p_comp = w_unembed[comp_id] * ln_weight
    p_margin = p_answer - p_comp

    dot = (p_margin @ p_margin).item()
    if dot < 1e-10:
        return torch.zeros_like(p_margin)

    delta = (deficit / dot) * p_margin
    return delta


def apply_rank_one_edit(model, layer: int, k_star, delta):
    """Apply rank-one update to down_proj.weight: W += delta ⊗ k* / (k* · k*)."""
    import torch

    down_proj = model.model.layers[layer].mlp.down_proj
    k_dot = (k_star @ k_star).item()
    if k_dot < 1e-10:
        return 0.0

    update = torch.outer(delta, k_star) / k_dot
    norm = update.norm().item()
    down_proj.weight.data += update
    return norm


def undo_rank_one_edit(model, layer: int, k_star, delta):
    """Undo a rank-one update to down_proj.weight."""
    import torch

    down_proj = model.model.layers[layer].mlp.down_proj
    k_dot = (k_star @ k_star).item()
    if k_dot < 1e-10:
        return

    update = torch.outer(delta, k_star) / k_dot
    down_proj.weight.data -= update


def get_answer_prob(
    model, tokenizer, prompt: str, answer: str, seed: int = 42,
) -> float:
    """Get the probability of the answer token at the last position."""
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_id = answer_ids[0] if answer_ids else 0

    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[answer_id].item()


def run_repair_local(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    competitor: str | None = None,
    target_layer: int | None = None,
    target_component: str = "mlp",
    target_margin: float = 0.0,
    verify_prompts: list[dict] | None = None,
    seed: int = 42,
) -> RepairResult:
    """Run a single ROME repair locally.

    If target_layer is None, auto-detect via decompose.
    If competitor is None, auto-detect top competitor.
    """
    from neurotrace.decompose import run_decompose_local

    # Auto-detect competitor if needed
    if competitor is None:
        from neurotrace.decompose import get_top_competitors

        comps = get_top_competitors(model, tokenizer, prompt, answer, n=1, seed=seed)
        competitor = comps[0] if comps else "the"

    # Run decompose to get component margins
    results = run_decompose_local(
        model, tokenizer, prompt, answer,
        competitors=[competitor], seed=seed,
    )
    if not results:
        return RepairResult(
            prompt=prompt, answer=answer, competitor=competitor,
            target_layer=target_layer or 0, target_component=target_component,
            before=RepairBefore(0, 0, 0, 0, 0),
            after=RepairAfter(0, 0, 0, 0, 0),
            edit=RepairEdit("down_proj", 0, 1, 0),
            status="skipped",
        )

    decomp = results[0]

    # Auto-detect worst component if target_layer not specified
    if target_layer is None:
        worst_margin = float("inf")
        for cm in decomp.component_margins:
            total_cm = cm["attn_margin"] + cm["mlp_margin"]
            if total_cm < worst_margin:
                worst_margin = total_cm
                target_layer = cm["layer"]
                if cm["mlp_margin"] < cm["attn_margin"]:
                    target_component = "mlp"
                else:
                    target_component = "attention"

    if target_layer is None:
        target_layer = 0

    # Get current component margin
    current_component_margin = 0.0
    for cm in decomp.component_margins:
        if cm["layer"] == target_layer:
            if target_component == "mlp":
                current_component_margin = cm["mlp_margin"]
            else:
                current_component_margin = cm["attn_margin"]
            break

    before_prob = get_answer_prob(model, tokenizer, prompt, answer, seed)
    before = RepairBefore(
        answer_logit=decomp.answer_logit,
        competitor_logit=decomp.competitor_logit,
        margin=decomp.margin,
        component_margin=current_component_margin,
        answer_prob=before_prob,
    )

    # Compute the deficit and apply edit
    deficit = target_margin - current_component_margin
    k_star = compute_key_vector(model, tokenizer, prompt, target_layer, seed)
    delta = compute_correction_delta(model, tokenizer, answer, competitor, deficit)
    edit_norm = apply_rank_one_edit(model, target_layer, k_star, delta)

    # Re-run decompose after edit
    results_after = run_decompose_local(
        model, tokenizer, prompt, answer,
        competitors=[competitor], seed=seed,
    )

    after_component_margin = 0.0
    if results_after:
        decomp_after = results_after[0]
        for cm in decomp_after.component_margins:
            if cm["layer"] == target_layer:
                if target_component == "mlp":
                    after_component_margin = cm["mlp_margin"]
                else:
                    after_component_margin = cm["attn_margin"]
                break
        after_prob = get_answer_prob(model, tokenizer, prompt, answer, seed)
        after = RepairAfter(
            answer_logit=decomp_after.answer_logit,
            competitor_logit=decomp_after.competitor_logit,
            margin=decomp_after.margin,
            component_margin=after_component_margin,
            answer_prob=after_prob,
        )
    else:
        after = RepairAfter(0, 0, 0, 0, 0)

    edit = RepairEdit(
        matrix="down_proj",
        layer=target_layer,
        rank=1,
        norm=edit_norm,
    )

    # Check regressions
    regressions = []
    if verify_prompts:
        for vp in verify_prompts:
            vp_prompt = vp["prompt"]
            vp_answer = vp["answer"]
            after_vp_prob = get_answer_prob(
                model, tokenizer, vp_prompt, vp_answer, seed,
            )
            # We don't have before prob for verify prompts in local mode,
            # so we undo, measure, re-apply
            undo_rank_one_edit(model, target_layer, k_star, delta)
            before_vp_prob = get_answer_prob(
                model, tokenizer, vp_prompt, vp_answer, seed,
            )
            apply_rank_one_edit(model, target_layer, k_star, delta)

            status = "ok" if after_vp_prob >= before_vp_prob - 0.05 else "regression"
            regressions.append(RegressionResult(
                prompt=vp_prompt,
                answer=vp_answer,
                before_prob=before_vp_prob,
                after_prob=after_vp_prob,
                status=status,
            ))

    has_regression = any(r.status == "regression" for r in regressions)
    result_status = "regression" if has_regression else "repaired"

    return RepairResult(
        prompt=prompt,
        answer=answer,
        competitor=competitor,
        target_layer=target_layer,
        target_component=target_component,
        before=before,
        after=after,
        edit=edit,
        regressions=regressions,
        status=result_status,
    )


# ---------------------------------------------------------------------------
# Remote repair
# ---------------------------------------------------------------------------


def build_repair_result_from_remote(data: dict) -> RepairResult:
    """Build a RepairResult from remote worker response data."""
    before_data = data["before"]
    after_data = data["after"]
    edit_data = data["edit"]

    before = RepairBefore(
        answer_logit=before_data["answer_logit"],
        competitor_logit=before_data["competitor_logit"],
        margin=before_data["margin"],
        component_margin=before_data["component_margin"],
        answer_prob=before_data["answer_prob"],
    )
    after = RepairAfter(
        answer_logit=after_data["answer_logit"],
        competitor_logit=after_data["competitor_logit"],
        margin=after_data["margin"],
        component_margin=after_data["component_margin"],
        answer_prob=after_data["answer_prob"],
    )
    edit = RepairEdit(
        matrix=edit_data["matrix"],
        layer=edit_data["layer"],
        rank=edit_data["rank"],
        norm=edit_data["norm"],
    )

    regressions = []
    for reg in data.get("regressions", []):
        regressions.append(RegressionResult(
            prompt=reg["prompt"],
            answer=reg["answer"],
            before_prob=reg["before_prob"],
            after_prob=reg["after_prob"],
            status=reg["status"],
        ))

    return RepairResult(
        prompt=data["prompt"],
        answer=data["answer"],
        competitor=data["competitor"],
        target_layer=data["target_layer"],
        target_component=data.get("target_component", "mlp"),
        before=before,
        after=after,
        edit=edit,
        regressions=regressions,
        status=data.get("status", "repaired"),
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def repair_result_to_dict(r: RepairResult) -> dict:
    return {
        "prompt": r.prompt,
        "answer": r.answer,
        "competitor": r.competitor,
        "target_layer": r.target_layer,
        "target_component": r.target_component,
        "before": {
            "answer_logit": r.before.answer_logit,
            "competitor_logit": r.before.competitor_logit,
            "margin": r.before.margin,
            "component_margin": r.before.component_margin,
            "answer_prob": r.before.answer_prob,
        },
        "after": {
            "answer_logit": r.after.answer_logit,
            "competitor_logit": r.after.competitor_logit,
            "margin": r.after.margin,
            "component_margin": r.after.component_margin,
            "answer_prob": r.after.answer_prob,
        },
        "edit": {
            "matrix": r.edit.matrix,
            "layer": r.edit.layer,
            "rank": r.edit.rank,
            "norm": r.edit.norm,
        },
        "regressions": [
            {
                "prompt": reg.prompt,
                "answer": reg.answer,
                "before_prob": reg.before_prob,
                "after_prob": reg.after_prob,
                "status": reg.status,
            }
            for reg in r.regressions
        ],
        "status": r.status,
    }


def repair_run_to_dict(run: RepairRun) -> dict:
    return {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "model_name": run.model_name,
        "prompt_count": run.prompt_count,
        "created_at": run.created_at,
        "results": [repair_result_to_dict(r) for r in run.results],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_GREEN = "#4caf50"
_RED = "#f44336"
_BLUE = "#64b5f6"
_BG = "#1a1a2e"
_BG2 = "#16213e"
_BG3 = "#0f3460"
_TEXT = "#e0e0e0"
_DIM = "#8a8a9a"
_ACCENT = "#e8956a"
_YELLOW = "#ffca28"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _repair_css() -> str:
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
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }}
    .meta-label {{ color: var(--dim); font-size: 0.85rem; }}
    .meta-value {{
        color: var(--text);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
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
    .positive {{ color: {_GREEN}; }}
    .negative {{ color: {_RED}; }}
    .status-repaired {{ color: {_GREEN}; font-weight: bold; }}
    .status-skipped {{ color: {_YELLOW}; font-weight: bold; }}
    .status-regression {{ color: {_RED}; font-weight: bold; }}
    .prompt-section {{
        background: var(--bg2); border-radius: 8px;
        padding: 1.5rem; margin-bottom: 2rem;
    }}
    .before-after {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;
    }}
    @media (max-width: 900px) {{
        .before-after {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .prompt-section {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _status_class(status: str) -> str:
    return f"status-{status}"


def generate_repair_html(run: RepairRun) -> str:
    """Generate self-contained HTML report for a repair run."""
    repaired = sum(1 for r in run.results if r.status == "repaired")
    skipped = sum(1 for r in run.results if r.status == "skipped")
    regressed = sum(1 for r in run.results if r.status == "regression")

    header = f"""
    <h1>Weight Repair Report</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(run.dataset or 'single')}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{run.prompt_count}</span></div>
            <div><span class="meta-label">Repaired</span><br>
                <span class="meta-value">{repaired}</span></div>
            <div><span class="meta-label">Skipped</span><br>
                <span class="meta-value">{skipped}</span></div>
            <div><span class="meta-label">Regressions</span><br>
                <span class="meta-value">{regressed}</span></div>
        </div>
    </div>
    """

    parts = [header]

    # Summary table
    parts.append("<h2>Results</h2>")
    parts.append(
        "<table><thead><tr>"
        "<th>Prompt</th><th>Answer</th><th>Competitor</th>"
        "<th>Layer</th><th>Before Margin</th><th>After Margin</th>"
        "<th>Before Prob</th><th>After Prob</th><th>Status</th>"
        "</tr></thead><tbody>"
    )
    for r in run.results:
        mc_before = "positive" if r.before.margin >= 0 else "negative"
        mc_after = "positive" if r.after.margin >= 0 else "negative"
        parts.append(
            f"<tr>"
            f"<td>{_esc(r.prompt[:50])}</td>"
            f"<td>{_esc(r.answer)}</td>"
            f"<td>{_esc(r.competitor)}</td>"
            f"<td>L{r.target_layer} {_esc(r.target_component)}</td>"
            f'<td class="{mc_before}">{r.before.margin:+.2f}</td>'
            f'<td class="{mc_after}">{r.after.margin:+.2f}</td>'
            f"<td>{r.before.answer_prob:.2%}</td>"
            f"<td>{r.after.answer_prob:.2%}</td>"
            f'<td class="{_status_class(r.status)}">'
            f"{_esc(r.status.upper())}</td>"
            f"</tr>"
        )
    parts.append("</tbody></table>")

    # Per-prompt details
    for r in run.results:
        parts.append(
            f'<div class="prompt-section">'
            f"<h3>{_esc(r.prompt)} → {_esc(r.answer)}</h3>"
            f'<div class="before-after">'
            f"<div>"
            f"<h3>Before</h3>"
            f"<table>"
            f"<tr><td>Answer logit</td><td>{r.before.answer_logit:.2f}</td></tr>"
            f"<tr><td>Competitor logit</td>"
            f"<td>{r.before.competitor_logit:.2f}</td></tr>"
            f"<tr><td>Margin</td><td>{r.before.margin:+.2f}</td></tr>"
            f"<tr><td>Component margin</td>"
            f"<td>{r.before.component_margin:+.2f}</td></tr>"
            f"<tr><td>Prob</td><td>{r.before.answer_prob:.2%}</td></tr>"
            f"</table></div>"
            f"<div>"
            f"<h3>After</h3>"
            f"<table>"
            f"<tr><td>Answer logit</td><td>{r.after.answer_logit:.2f}</td></tr>"
            f"<tr><td>Competitor logit</td><td>{r.after.competitor_logit:.2f}</td></tr>"
            f"<tr><td>Margin</td><td>{r.after.margin:+.2f}</td></tr>"
            f"<tr><td>Component margin</td>"
            f"<td>{r.after.component_margin:+.2f}</td></tr>"
            f"<tr><td>Prob</td><td>{r.after.answer_prob:.2%}</td></tr>"
            f"</table></div>"
            f"</div>"
        )

        if r.regressions:
            parts.append(
                "<h3>Regression Check</h3>"
                "<table><thead><tr>"
                "<th>Prompt</th><th>Answer</th>"
                "<th>Before Prob</th><th>After Prob</th><th>Status</th>"
                "</tr></thead><tbody>"
            )
            for reg in r.regressions:
                sc = _status_class("repaired" if reg.status == "ok" else "regression")
                parts.append(
                    f"<tr>"
                    f"<td>{_esc(reg.prompt[:50])}</td>"
                    f"<td>{_esc(reg.answer)}</td>"
                    f"<td>{reg.before_prob:.2%}</td>"
                    f"<td>{reg.after_prob:.2%}</td>"
                    f'<td class="{sc}">{_esc(reg.status.upper())}</td>'
                    f"</tr>"
                )
            parts.append("</tbody></table>")

        parts.append(
            f"<p>Edit: L{r.edit.layer} {_esc(r.edit.matrix)} "
            f"(rank-{r.edit.rank}, norm: {r.edit.norm:.4f})</p>"
            f"</div>"
        )

    body = "\n".join(parts)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Weight Repair - {_esc(run.dataset or 'repair')}</title>\n"
        f"<style>{_repair_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
