"""Commitment scoring — measures model commitment to answers."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CommitmentResult:
    """Per-prompt commitment analysis."""

    prompt: str
    answer: str
    trajectory: list[float]  # probability at each layer
    peak_prob: float
    peak_layer: int
    final_prob: float
    commitment_score: float  # == peak_prob
    recovery_ratio: float  # final_prob / peak_prob (or 0.0 if peak_prob == 0)
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
    threshold: float = 0.7,
) -> dict:
    """Compute commitment metrics from a probability trajectory.

    Args:
        trajectory: Per-layer probability of the answer token [p_layer0, ..., p_layerN].
        threshold: Recovery ratio below which a prompt is classified vulnerable.

    Returns:
        Dict with peak_prob, peak_layer, final_prob, commitment_score,
        recovery_ratio, vulnerable.
    """
    if not trajectory:
        return {
            "peak_prob": 0.0,
            "peak_layer": 0,
            "final_prob": 0.0,
            "commitment_score": 0.0,
            "recovery_ratio": 0.0,
            "vulnerable": True,
        }

    peak_prob = max(trajectory)
    peak_layer = trajectory.index(peak_prob)
    final_prob = trajectory[-1]
    recovery_ratio = final_prob / peak_prob if peak_prob > 0 else 0.0

    return {
        "peak_prob": peak_prob,
        "peak_layer": peak_layer,
        "final_prob": final_prob,
        "commitment_score": peak_prob,
        "recovery_ratio": recovery_ratio,
        "vulnerable": recovery_ratio < threshold,
    }


def build_commitment_result(
    prompt: str,
    answer: str,
    trajectory: list[float],
    threshold: float = 0.7,
) -> CommitmentResult:
    """Build a CommitmentResult from a probability trajectory."""
    metrics = compute_commitment(trajectory, threshold)
    return CommitmentResult(
        prompt=prompt,
        answer=answer,
        trajectory=trajectory,
        **metrics,
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
    threshold: float = 0.7,
    progress_callback=None,
) -> list[CommitmentResult]:
    """Run commitment analysis locally by hooking every layer.

    For each prompt, runs a forward pass with hooks on every layer.
    At each layer, projects the hidden state through final_norm + lm_head
    to get the answer token's probability.

    Args:
        model: A loaded AutoModelForCausalLM.
        tokenizer: The matching tokenizer.
        arch: ModelArchitecture for the model.
        prompts: List of {"prompt": str, "answer": str} dicts.
        seed: Random seed.
        threshold: Recovery ratio threshold for vulnerability.
        progress_callback: Optional (prompt_idx, num_prompts) callable.

    Returns:
        List of CommitmentResult objects.
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
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not answer_ids:
            # Fallback: try without space prefix
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        answer_token_id = answer_ids[0] if answer_ids else None

        # Hook all layers to capture hidden states
        captured: dict[int, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = out.detach()
            return hook_fn

        try:
            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(_make_hook(i)))

            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(next(model.parameters()).device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(input_ids.device)

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in hooks:
                h.remove()

        # Build trajectory: project each layer's hidden state to get answer prob
        trajectory: list[float] = []
        for layer_idx in range(num_layers):
            if layer_idx not in captured:
                trajectory.append(0.0)
                continue

            hidden = captured[layer_idx][:, -1, :]  # last token
            if final_ln is not None:
                hidden = final_ln(hidden)
            logits = lm_head(hidden.float())
            probs = torch.softmax(logits[0].float().cpu(), dim=-1)

            if answer_token_id is not None and answer_token_id < len(probs):
                trajectory.append(float(probs[answer_token_id].item()))
            else:
                trajectory.append(0.0)

        results.append(
            build_commitment_result(prompt_text, answer, trajectory, threshold)
        )

    return results


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


def run_commitment_remote(
    hidden_states_by_prompt: list,  # list of np arrays [num_layers, hidden_dim]
    lm_head_weight,  # torch tensor or numpy
    final_ln,  # torch module or None
    tokenizer,
    prompts: list[dict],
    threshold: float = 0.7,
) -> list[CommitmentResult]:
    """Compute commitment from remotely-fetched hidden states.

    Args:
        hidden_states_by_prompt: List of arrays, each [num_layers, hidden_dim].
        lm_head_weight: The lm_head weight matrix.
        final_ln: The final layer norm module (or None).
        tokenizer: Tokenizer for encoding answer tokens.
        prompts: List of {"prompt": str, "answer": str} dicts.
        threshold: Recovery ratio threshold.

    Returns:
        List of CommitmentResult objects.
    """
    import torch

    results: list[CommitmentResult] = []

    for p_idx, entry in enumerate(prompts):
        prompt_text = entry["prompt"]
        answer = entry["answer"]

        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not answer_ids:
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        answer_token_id = answer_ids[0] if answer_ids else None

        states = hidden_states_by_prompt[p_idx]  # [num_layers, hidden_dim]
        num_layers = states.shape[0]

        trajectory: list[float] = []
        for layer_idx in range(num_layers):
            hidden = torch.tensor(states[layer_idx], dtype=torch.float32).unsqueeze(0)
            if final_ln is not None:
                hidden = final_ln(hidden)
            logits = torch.nn.functional.linear(hidden, lm_head_weight)
            probs = torch.softmax(logits[0].float(), dim=-1)

            if answer_token_id is not None and answer_token_id < len(probs):
                trajectory.append(float(probs[answer_token_id].item()))
            else:
                trajectory.append(0.0)

        results.append(
            build_commitment_result(prompt_text, answer, trajectory, threshold)
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

    For each prompt, checks if the commitment prediction (vulnerable/robust)
    matches heatmap ground truth (had breaks in interference layers / didn't).

    Returns dict with tp, fp, tn, fn, accuracy, and optionally auc_roc.
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
    commitment_scores = []
    heatmap_labels = []

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

        commitment_scores.append(1.0 - result.recovery_ratio)
        heatmap_labels.append(1 if actual_vulnerable else 0)

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    validation = {
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
        "n_vulnerable": sum(1 for r in run.results if r.vulnerable),
        "n_robust": sum(1 for r in run.results if not r.vulnerable),
        "avg_commitment_score": (
            sum(r.commitment_score for r in run.results) / len(run.results)
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
                "recovery_ratio": r.recovery_ratio,
                "vulnerable": r.vulnerable,
                "trajectory": r.trajectory,
            }
            for r in run.results
        ],
        "validation": run.validation,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_GREEN = "#4caf50"
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
        padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--bg3);
        font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
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
        sum(r.commitment_score for r in run.results) / len(run.results)
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
        rows_html.append(
            f"<tr>"
            f"<td>{_esc(r.prompt[:60])}</td>"
            f"<td>{_esc(r.answer)}</td>"
            f"<td>{r.peak_prob:.4f}</td>"
            f"<td>{r.peak_layer}</td>"
            f"<td>{r.final_prob:.4f}</td>"
            f"<td>{r.recovery_ratio:.4f}</td>"
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
        '<th data-sort="num">Peak Prob</th>'
        '<th data-sort="num">Peak Layer</th>'
        '<th data-sort="num">Final Prob</th>'
        '<th data-sort="num">Recovery Ratio</th>'
        '<th data-sort="bool">Status</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )

    # Validation section (if available)
    validation_html = ""
    if run.validation:
        v = run.validation
        validation_html = (
            "<h2>Heatmap Cross-Validation</h2>"
            '<div class="meta"><div class="meta-grid">'
            f'<div><span class="meta-label">Matched Prompts</span><br>'
            f'<span class="meta-value">{v["total_matched"]}</span></div>'
            f'<div><span class="meta-label">Accuracy</span><br>'
            f'<span class="meta-value">{v["accuracy"]:.1%}</span></div>'
            f'<div><span class="meta-label">AUC-ROC</span><br>'
            f'<span class="meta-value">{v.get("auc_roc", "N/A")}</span></div>'
            f'<div><span class="meta-label">TP / FP</span><br>'
            f'<span class="meta-value">{v["tp"]} / {v["fp"]}</span></div>'
            f'<div><span class="meta-label">FN / TN</span><br>'
            f'<span class="meta-value">{v["fn"]} / {v["tn"]}</span></div>'
            "</div></div>"
        )

    body = "\n".join([header, table, validation_html])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Commitment Analysis - {_esc(run.dataset_name)}</title>\n"
        f"<style>{_commitment_css()}</style>\n"
        f"</head>\n<body>\n{body}\n{_sortable_js()}\n</body>\n</html>"
    )
