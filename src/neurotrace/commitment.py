"""Commitment scoring — measures model commitment via margin tracking."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerInfo:
    """Per-layer snapshot of correct vs competitor probabilities."""

    layer: int
    correct_prob: float
    correct_rank: int
    competitor_token: str
    competitor_prob: float
    margin: float  # correct_prob - competitor_prob


@dataclass
class CommitmentResult:
    """Per-prompt commitment analysis with margin tracking."""

    prompt: str
    answer: str
    trajectory: list[float]  # correct answer prob at each layer
    margin_trajectory: list[float]  # margin at each layer
    competitor_trajectory: list[float]  # competitor prob at each layer
    peak_prob: float
    peak_layer: int
    final_prob: float
    commitment_score: float  # == peak_prob
    min_margin: float
    margin_at_final: float
    competitor_token: str
    competitor_peak: float
    crossover_layer: int | None  # first layer where competitor > correct
    vulnerable: bool


@dataclass
class CommitmentRun:
    """Complete result of a commitment scan."""

    run_id: str
    dataset_name: str
    model_name: str
    num_layers: int
    num_prompts: int
    threshold: float
    results: list[CommitmentResult]
    created_at: str
    # Cross-validation against heatmap (optional)
    validation: dict | None = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_commitment(
    trajectory: list[float],
    margin_trajectory: list[float],
    competitor_trajectory: list[float],
    competitor_tokens: list[str],
    threshold: float = 0.1,
) -> dict:
    """Compute commitment metrics from probability trajectories.

    Args:
        trajectory: Per-layer correct answer probability.
        margin_trajectory: Per-layer margin (correct - competitor).
        competitor_trajectory: Per-layer competitor probability.
        competitor_tokens: Per-layer competitor token string.
        threshold: Min margin below which a prompt is vulnerable.

    Returns:
        Dict with all commitment metrics.
    """
    if not trajectory:
        return {
            "peak_prob": 0.0,
            "peak_layer": 0,
            "final_prob": 0.0,
            "commitment_score": 0.0,
            "min_margin": 0.0,
            "margin_at_final": 0.0,
            "competitor_token": "",
            "competitor_peak": 0.0,
            "crossover_layer": None,
            "vulnerable": True,
        }

    peak_prob = max(trajectory)
    peak_layer = trajectory.index(peak_prob)
    final_prob = trajectory[-1]
    min_margin = min(margin_trajectory)
    margin_at_final = margin_trajectory[-1]
    competitor_peak = max(competitor_trajectory)

    # Find crossover: first layer where competitor > correct
    crossover_layer = None
    for i, m in enumerate(margin_trajectory):
        if m < 0:
            crossover_layer = i
            break

    # Find the most threatening competitor (at the layer with min margin)
    min_margin_idx = margin_trajectory.index(min_margin)
    competitor_token = (
        competitor_tokens[min_margin_idx]
        if min_margin_idx < len(competitor_tokens)
        else ""
    )

    vulnerable = (
        min_margin < threshold or crossover_layer is not None
    )

    return {
        "peak_prob": peak_prob,
        "peak_layer": peak_layer,
        "final_prob": final_prob,
        "commitment_score": peak_prob,
        "min_margin": min_margin,
        "margin_at_final": margin_at_final,
        "competitor_token": competitor_token,
        "competitor_peak": competitor_peak,
        "crossover_layer": crossover_layer,
        "vulnerable": vulnerable,
    }


def build_commitment_result(
    prompt: str,
    answer: str,
    trajectory: list[float],
    margin_trajectory: list[float],
    competitor_trajectory: list[float],
    competitor_tokens: list[str],
    threshold: float = 0.1,
) -> CommitmentResult:
    """Build a CommitmentResult from trajectories."""
    metrics = compute_commitment(
        trajectory,
        margin_trajectory,
        competitor_trajectory,
        competitor_tokens,
        threshold,
    )
    return CommitmentResult(
        prompt=prompt,
        answer=answer,
        trajectory=trajectory,
        margin_trajectory=margin_trajectory,
        competitor_trajectory=competitor_trajectory,
        **metrics,
    )


def _extract_layer_info(
    probs,  # 1-D tensor of probabilities (vocab_size,)
    answer_token_id: int | None,
    tokenizer,
) -> LayerInfo:
    """Extract correct prob, competitor, and margin from a prob vector.

    Args:
        probs: 1-D probability tensor over vocab.
        answer_token_id: Token ID for the correct answer.
        tokenizer: For decoding competitor token.

    Returns:
        LayerInfo with correct/competitor stats.
    """
    import torch

    if answer_token_id is not None and answer_token_id < len(probs):
        correct_prob = float(probs[answer_token_id].item())
        # Find rank of correct answer
        sorted_indices = torch.argsort(probs, descending=True)
        rank_tensor = (sorted_indices == answer_token_id).nonzero()
        correct_rank = int(rank_tensor[0].item()) if len(rank_tensor) > 0 else -1

        # Find top competitor (best token that isn't the answer)
        top_k_vals, top_k_ids = torch.topk(probs, min(5, len(probs)))
        comp_token = ""
        comp_prob = 0.0
        for val, tid in zip(top_k_vals, top_k_ids):
            if tid.item() != answer_token_id:
                comp_token = tokenizer.decode([tid.item()]).strip()
                comp_prob = float(val.item())
                break
    else:
        correct_prob = 0.0
        correct_rank = -1
        comp_token = ""
        comp_prob = 0.0
        if len(probs) > 0:
            top_val, top_id = probs.topk(1)
            comp_token = tokenizer.decode([top_id[0].item()]).strip()
            comp_prob = float(top_val[0].item())

    margin = correct_prob - comp_prob
    return LayerInfo(
        layer=0,  # caller sets this
        correct_prob=correct_prob,
        correct_rank=correct_rank,
        competitor_token=comp_token,
        competitor_prob=comp_prob,
        margin=margin,
    )


# ---------------------------------------------------------------------------
# Local runner (requires torch)
# ---------------------------------------------------------------------------


def run_commitment_local(
    model,
    tokenizer,
    arch,
    prompts: list[dict],
    seed: int = 42,
    threshold: float = 0.1,
    progress_callback=None,
) -> list[CommitmentResult]:
    """Run commitment analysis locally by hooking every layer.

    For each prompt, hooks every layer, projects hidden states through
    lm_head, and extracts margin between correct answer and top competitor.
    """
    import torch

    from neurotrace.models import get_lm_head_and_norm

    torch.manual_seed(seed)

    layers = arch.get_layers(model)
    num_layers = len(layers)
    lm_head, final_ln = get_lm_head_and_norm(model)

    results: list[CommitmentResult] = []

    for p_idx, entry in enumerate(prompts):
        if progress_callback is not None:
            progress_callback(p_idx, len(prompts))

        prompt_text = entry["prompt"]
        answer = entry["answer"]

        # Get the answer token id
        answer_ids = tokenizer.encode(
            " " + answer, add_special_tokens=False
        )
        if not answer_ids:
            answer_ids = tokenizer.encode(
                answer, add_special_tokens=False
            )
        answer_token_id = answer_ids[0] if answer_ids else None

        # Hook all layers to capture hidden states
        captured: dict[int, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                out = (
                    output[0]
                    if isinstance(output, tuple)
                    else output
                )
                captured[layer_idx] = out.detach()
            return hook_fn

        try:
            for i, layer in enumerate(layers):
                hooks.append(
                    layer.register_forward_hook(_make_hook(i))
                )

            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(
                next(model.parameters()).device
            )
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)

            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            for h in hooks:
                h.remove()

        # Build trajectories
        trajectory: list[float] = []
        margin_trajectory: list[float] = []
        competitor_trajectory: list[float] = []
        competitor_tokens: list[str] = []

        for layer_idx in range(num_layers):
            if layer_idx not in captured:
                trajectory.append(0.0)
                margin_trajectory.append(0.0)
                competitor_trajectory.append(0.0)
                competitor_tokens.append("")
                continue

            hidden = captured[layer_idx][:, -1, :]
            if final_ln is not None:
                hidden = final_ln(hidden)
            logits = lm_head(hidden.float())
            probs = torch.softmax(
                logits[0].float().cpu(), dim=-1
            )

            info = _extract_layer_info(
                probs, answer_token_id, tokenizer
            )
            trajectory.append(info.correct_prob)
            margin_trajectory.append(info.margin)
            competitor_trajectory.append(info.competitor_prob)
            competitor_tokens.append(info.competitor_token)

        results.append(
            build_commitment_result(
                prompt_text,
                answer,
                trajectory,
                margin_trajectory,
                competitor_trajectory,
                competitor_tokens,
                threshold,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


def run_commitment_remote(
    hidden_states_by_prompt: list,
    lm_head_weight,
    final_ln,
    tokenizer,
    prompts: list[dict],
    threshold: float = 0.1,
) -> list[CommitmentResult]:
    """Compute commitment from remotely-fetched hidden states."""
    import torch

    results: list[CommitmentResult] = []

    for p_idx, entry in enumerate(prompts):
        prompt_text = entry["prompt"]
        answer = entry["answer"]

        answer_ids = tokenizer.encode(
            " " + answer, add_special_tokens=False
        )
        if not answer_ids:
            answer_ids = tokenizer.encode(
                answer, add_special_tokens=False
            )
        answer_token_id = answer_ids[0] if answer_ids else None

        states = hidden_states_by_prompt[p_idx]
        num_layers = states.shape[0]

        trajectory: list[float] = []
        margin_trajectory: list[float] = []
        competitor_trajectory: list[float] = []
        competitor_tokens: list[str] = []

        for layer_idx in range(num_layers):
            hidden = torch.tensor(
                states[layer_idx], dtype=torch.float32
            ).unsqueeze(0)
            if final_ln is not None:
                hidden = final_ln(hidden)
            logits = torch.nn.functional.linear(
                hidden, lm_head_weight
            )
            probs = torch.softmax(logits[0].float(), dim=-1)

            info = _extract_layer_info(
                probs, answer_token_id, tokenizer
            )
            trajectory.append(info.correct_prob)
            margin_trajectory.append(info.margin)
            competitor_trajectory.append(info.competitor_prob)
            competitor_tokens.append(info.competitor_token)

        results.append(
            build_commitment_result(
                prompt_text,
                answer,
                trajectory,
                margin_trajectory,
                competitor_trajectory,
                competitor_tokens,
                threshold,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Heatmap cross-validation
# ---------------------------------------------------------------------------


def validate_against_heatmap(
    commitment_results: list[CommitmentResult],
    heatmap_cells_json: str,
    interference_layers: tuple[int, int] = (14, 21),
) -> dict:
    """Cross-reference commitment predictions against heatmap ground truth.

    Uses negative min_margin as the score for AUC-ROC (higher = more
    vulnerable).
    """
    import json

    if isinstance(heatmap_cells_json, str):
        cells = json.loads(heatmap_cells_json)
    else:
        cells = heatmap_cells_json

    # Build heatmap ground truth: prompt -> has_breaks_in_range
    heatmap_vulnerable: dict[str, bool] = {}
    for cell in cells:
        prompt = cell["prompt"]
        layer = cell["layer"]
        if interference_layers[0] <= layer <= interference_layers[1]:
            if cell["flip_direction"] == "broke":
                heatmap_vulnerable[prompt] = True
        if prompt not in heatmap_vulnerable:
            heatmap_vulnerable[prompt] = False

    tp = fp = tn = fn = 0
    commitment_scores: list[float] = []
    heatmap_labels: list[int] = []

    for result in commitment_results:
        if result.prompt not in heatmap_vulnerable:
            continue

        actual_vulnerable = heatmap_vulnerable[result.prompt]
        predicted_vulnerable = result.vulnerable

        if predicted_vulnerable and actual_vulnerable:
            tp += 1
        elif predicted_vulnerable and not actual_vulnerable:
            fp += 1
        elif not predicted_vulnerable and actual_vulnerable:
            fn += 1
        else:
            tn += 1

        # Higher score = more vulnerable (negative margin = vulnerable)
        commitment_scores.append(-result.min_margin)
        heatmap_labels.append(1 if actual_vulnerable else 0)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    validation: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "total_matched": total,
    }

    # Compute AUC-ROC if we have enough data
    if len(set(heatmap_labels)) >= 2:
        try:
            from sklearn.metrics import roc_auc_score

            validation["auc_roc"] = float(
                roc_auc_score(heatmap_labels, commitment_scores)
            )
        except (ImportError, ValueError):
            validation["auc_roc"] = None
    else:
        validation["auc_roc"] = None

    return validation


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def commitment_run_to_dict(run: CommitmentRun) -> dict:
    """Convert a CommitmentRun to a JSON-serializable dict."""
    return {
        "run_id": run.run_id,
        "dataset_name": run.dataset_name,
        "model_name": run.model_name,
        "num_layers": run.num_layers,
        "num_prompts": run.num_prompts,
        "threshold": run.threshold,
        "created_at": run.created_at,
        "n_vulnerable": sum(
            1 for r in run.results if r.vulnerable
        ),
        "n_robust": sum(
            1 for r in run.results if not r.vulnerable
        ),
        "avg_commitment_score": (
            sum(r.commitment_score for r in run.results)
            / len(run.results)
            if run.results
            else 0.0
        ),
        "results": [
            {
                "prompt": r.prompt,
                "answer": r.answer,
                "peak_prob": r.peak_prob,
                "peak_layer": r.peak_layer,
                "final_prob": r.final_prob,
                "commitment_score": r.commitment_score,
                "min_margin": r.min_margin,
                "margin_at_final": r.margin_at_final,
                "competitor_token": r.competitor_token,
                "competitor_peak": r.competitor_peak,
                "crossover_layer": r.crossover_layer,
                "vulnerable": r.vulnerable,
                "trajectory": r.trajectory,
                "margin_trajectory": r.margin_trajectory,
                "competitor_trajectory": r.competitor_trajectory,
            }
            for r in run.results
        ],
        "validation": run.validation,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_GREEN = "#4caf50"
_YELLOW = "#ffc107"
_RED = "#f44336"
_BLUE = "#64b5f6"
_GRAY = "#333333"
_BG = "#1a1a2e"
_BG2 = "#16213e"
_BG3 = "#0f3460"
_TEXT = "#e0e0e0"
_DIM = "#8a8a9a"
_ACCENT = "#e8956a"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _margin_color(margin: float) -> str:
    """Color for min_margin: green >0.2, yellow 0-0.2, red <0."""
    if margin > 0.2:
        return _GREEN
    if margin >= 0.0:
        return _YELLOW
    return _RED


def _commitment_css() -> str:
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
        cursor: pointer; user-select: none;
    }}
    th:hover {{ color: {_TEXT}; }}
    td {{
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid var(--bg3);
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
    }}
    .robust {{ color: {_GREEN}; }}
    .vulnerable {{ color: {_RED}; }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _sortable_js() -> str:
    """Minimal inline JS for sortable table columns."""
    return """
    <script>
    document.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const table = th.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const idx = Array.from(th.parentNode.children).indexOf(th);
            const type = th.dataset.sort;
            const asc = th.dataset.dir !== 'asc';
            th.dataset.dir = asc ? 'asc' : 'desc';
            rows.sort((a, b) => {
                let va = a.children[idx].textContent.trim();
                let vb = b.children[idx].textContent.trim();
                if (type === 'num') {
                    va = parseFloat(va) || 0;
                    vb = parseFloat(vb) || 0;
                }
                if (type === 'bool') {
                    va = va === 'VULNERABLE' ? 1 : 0;
                    vb = vb === 'VULNERABLE' ? 1 : 0;
                }
                if (va < vb) return asc ? -1 : 1;
                if (va > vb) return asc ? 1 : -1;
                return 0;
            });
            rows.forEach(r => tbody.appendChild(r));
        });
    });
    </script>
    """


def generate_commitment_html(run: CommitmentRun) -> str:
    """Generate a self-contained HTML commitment report."""
    n_vulnerable = sum(1 for r in run.results if r.vulnerable)
    n_robust = sum(1 for r in run.results if not r.vulnerable)
    avg_score = (
        sum(r.commitment_score for r in run.results)
        / len(run.results)
        if run.results
        else 0.0
    )

    header = f"""
    <h1>Commitment Analysis</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(run.dataset_name)}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{run.num_prompts}</span></div>
            <div><span class="meta-label">Vulnerable</span><br>
                <span class="meta-value vulnerable">{n_vulnerable}</span></div>
            <div><span class="meta-label">Robust</span><br>
                <span class="meta-value robust">{n_robust}</span></div>
            <div><span class="meta-label">Avg Commitment</span><br>
                <span class="meta-value">{avg_score:.3f}</span></div>
            <div><span class="meta-label">Threshold</span><br>
                <span class="meta-value">{run.threshold}</span></div>
            <div><span class="meta-label">Run ID</span><br>
                <span class="meta-value">{_esc(run.run_id[:12])}</span></div>
        </div>
    </div>
    """

    # Results table
    rows_html = []
    for r in run.results:
        status_class = "vulnerable" if r.vulnerable else "robust"
        status_text = "VULNERABLE" if r.vulnerable else "ROBUST"
        mc = _margin_color(r.min_margin)
        xover = (
            f"L{r.crossover_layer}"
            if r.crossover_layer is not None
            else "-"
        )
        rows_html.append(
            f"<tr>"
            f"<td>{_esc(r.prompt[:60])}</td>"
            f"<td>{_esc(r.answer)}</td>"
            f"<td>{r.peak_prob:.3f}</td>"
            f'<td style="color:{mc}">{r.min_margin:+.3f}</td>'
            f"<td>{_esc(r.competitor_token)}</td>"
            f"<td>{r.competitor_peak:.3f}</td>"
            f"<td>{xover}</td>"
            f'<td class="{status_class}">{status_text}</td>'
            f"</tr>"
        )

    table = (
        "<h2>Results</h2>"
        '<div style="overflow-x:auto;">'
        "<table>"
        "<thead><tr>"
        '<th data-sort="str">Prompt</th>'
        '<th data-sort="str">Answer</th>'
        '<th data-sort="num">Peak</th>'
        '<th data-sort="num">MinMargin</th>'
        '<th data-sort="str">Competitor</th>'
        '<th data-sort="num">CompPeak</th>'
        '<th data-sort="num">Crossover</th>'
        '<th data-sort="bool">Status</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )

    # Validation section (if available)
    validation_html = ""
    if run.validation:
        v = run.validation
        auc_display = (
            f'{v["auc_roc"]:.4f}'
            if v.get("auc_roc") is not None
            else "N/A"
        )
        validation_html = (
            "<h2>Heatmap Cross-Validation</h2>"
            '<div class="meta"><div class="meta-grid">'
            '<div><span class="meta-label">Matched</span><br>'
            f'<span class="meta-value">'
            f'{v["total_matched"]}</span></div>'
            '<div><span class="meta-label">Accuracy</span><br>'
            f'<span class="meta-value">'
            f'{v["accuracy"]:.1%}</span></div>'
            '<div><span class="meta-label">AUC-ROC</span><br>'
            f'<span class="meta-value">{auc_display}</span></div>'
            '<div><span class="meta-label">TP / FP</span><br>'
            f'<span class="meta-value">'
            f'{v["tp"]} / {v["fp"]}</span></div>'
            '<div><span class="meta-label">FN / TN</span><br>'
            f'<span class="meta-value">'
            f'{v["fn"]} / {v["tn"]}</span></div>'
            "</div></div>"
        )

    body = "\n".join([header, table, validation_html])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1">\n'
        f"<title>Commitment Analysis"
        f" - {_esc(run.dataset_name)}</title>\n"
        f"<style>{_commitment_css()}</style>\n"
        f"</head>\n<body>\n{body}\n"
        f"{_sortable_js()}\n</body>\n</html>"
    )
