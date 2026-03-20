"""Token attribution — which input tokens drive MLP interference."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenAttribution:
    """Attribution score for a single token in a prompt."""

    position: int
    token: str
    attribution: float


@dataclass
class AttributionResult:
    """Attribution result for a single prompt."""

    prompt: str
    layer: int
    target_token: str
    target_token_id: int
    method: str
    token_attributions: list[TokenAttribution]


@dataclass
class AttributionRun:
    """Complete result of an attribution analysis run."""

    run_id: str
    layer: int
    target_direction: str  # "answer", "competitor", "both", or custom token
    method: str  # "gradient" or "ablation"
    model_name: str
    dataset: str | None
    results: list[AttributionResult]
    created_at: str


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def normalize_attributions(attributions: list[float]) -> list[float]:
    """Normalize attribution scores to sum to 1."""
    total = sum(abs(a) for a in attributions)
    if total < 1e-10:
        n = len(attributions)
        return [1.0 / n] * n if n > 0 else []
    return [abs(a) / total for a in attributions]


def compute_ablation_attribution(
    baseline_delta: np.ndarray,
    ablated_deltas: list[np.ndarray],
    target_direction: np.ndarray,
) -> list[float]:
    """Compute attribution via leave-one-out ablation.

    Args:
        baseline_delta: MLP delta from original prompt.
        ablated_deltas: MLP deltas from prompts with each token replaced.
        target_direction: Normalized lm_head direction for target token.

    Returns:
        Normalized attribution scores per token position.
    """
    baseline_proj = float(np.dot(baseline_delta, target_direction))

    raw_attributions = []
    for ablated_delta in ablated_deltas:
        ablated_proj = float(np.dot(ablated_delta, target_direction))
        raw_attributions.append(abs(baseline_proj - ablated_proj))

    return normalize_attributions(raw_attributions)


def compute_gradient_attribution(
    grad_magnitudes: np.ndarray,
) -> list[float]:
    """Normalize gradient magnitudes into attribution scores.

    Args:
        grad_magnitudes: Per-position gradient norms, shape [seq_len].

    Returns:
        Normalized attribution scores.
    """
    return normalize_attributions(grad_magnitudes.tolist())


def aggregate_attributions(
    results: list[AttributionResult],
) -> dict[str, dict[str, float]]:
    """Aggregate attributions across multiple prompts.

    Returns:
        Dict mapping token text -> {
            "avg_attribution": float,
            "count": int,
            "max_attribution": float,
        }
    """
    token_stats: dict[str, list[float]] = {}
    for result in results:
        for ta in result.token_attributions:
            token_stats.setdefault(ta.token, []).append(ta.attribution)

    aggregated: dict[str, dict[str, float]] = {}
    for token, scores in token_stats.items():
        aggregated[token] = {
            "avg_attribution": float(np.mean(scores)),
            "count": len(scores),
            "max_attribution": float(np.max(scores)),
        }
    return aggregated


def build_attribution_result(
    prompt: str,
    tokens: list[str],
    attributions: list[float],
    layer: int,
    target_token: str,
    target_token_id: int,
    method: str,
) -> AttributionResult:
    """Build an AttributionResult from raw attribution scores."""
    token_attrs = [
        TokenAttribution(position=i, token=t, attribution=a)
        for i, (t, a) in enumerate(zip(tokens, attributions))
    ]
    return AttributionResult(
        prompt=prompt,
        layer=layer,
        target_token=target_token,
        target_token_id=target_token_id,
        method=method,
        token_attributions=token_attrs,
    )


# ---------------------------------------------------------------------------
# Local runner (requires torch)
# ---------------------------------------------------------------------------


def run_attribution_gradient_local(
    model,
    tokenizer,
    arch,
    prompt: str,
    layer: int,
    target_token_id: int,
    target_token: str,
    seed: int = 42,
) -> AttributionResult:
    """Run gradient-based attribution locally.

    Uses integrated gradients: backprop from MLP delta projection
    through embeddings to get per-token attribution.
    """
    import torch

    torch.manual_seed(seed)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    device = next(model.parameters()).device

    # Get embeddings with grad tracking
    embeddings = model.model.embed_tokens(input_ids.to(device))
    embeddings = embeddings.detach().requires_grad_(True)

    # Hook target layer's MLP
    mlp_delta_store: dict[str, torch.Tensor] = {}
    model_layers = arch.get_layers(model)
    mlp = arch.get_mlp(model_layers[layer])

    def hook_fn(module, inp, out):
        inp_t = inp[0] if isinstance(inp, tuple) else inp
        out_t = out if not isinstance(out, tuple) else out[0]
        mlp_delta_store["input"] = inp_t
        mlp_delta_store["output"] = out_t

    handle = mlp.register_forward_hook(hook_fn)

    try:
        # Forward pass with embeddings
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
    finally:
        handle.remove()

    # Compute MLP delta at last token position
    delta = mlp_delta_store["output"][:, -1, :] - mlp_delta_store["input"][:, -1, :]

    # Get target direction from lm_head
    target_dir = model.lm_head.weight[target_token_id].detach().float()
    target_dir = target_dir / target_dir.norm()

    # Scalar projection
    projection = (delta.float() * target_dir).sum()

    # Backprop to embeddings
    projection.backward()

    # Attribution = gradient magnitude at each position
    grads = embeddings.grad[0]  # [seq_len, hidden_dim]
    grad_magnitudes = grads.norm(dim=-1).detach().cpu().numpy()  # [seq_len]

    attributions = compute_gradient_attribution(grad_magnitudes)

    # Decode tokens
    token_ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return build_attribution_result(
        prompt=prompt,
        tokens=tokens,
        attributions=attributions,
        layer=layer,
        target_token=target_token,
        target_token_id=target_token_id,
        method="gradient",
    )


def run_attribution_ablation_local(
    model,
    tokenizer,
    arch,
    prompt: str,
    layer: int,
    target_token_id: int,
    target_token: str,
    seed: int = 42,
) -> AttributionResult:
    """Run ablation-based attribution locally.

    For each token position, replace with pad token, re-run,
    and measure change in MLP delta projection.
    """
    import torch

    from neurotrace.models import get_lm_head_and_norm

    torch.manual_seed(seed)

    lm_head, _ = get_lm_head_and_norm(model)
    lm_head_weight = lm_head.weight.data.cpu().float().numpy()

    # Get target direction
    target_direction = lm_head_weight[target_token_id].astype(np.float32)
    norm = np.linalg.norm(target_direction)
    if norm > 1e-10:
        target_direction = target_direction / norm

    token_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0

    def get_mlp_delta(ids: list[int]) -> np.ndarray:
        """Run forward pass and get MLP delta at target layer."""
        input_tensor = torch.tensor([ids], device=next(model.parameters()).device)

        captured: dict[str, torch.Tensor] = {}
        model_layers = arch.get_layers(model)
        mlp = arch.get_mlp(model_layers[layer])

        def hook_fn(module, inp, out):
            inp_t = inp[0] if isinstance(inp, tuple) else inp
            out_t = out if not isinstance(out, tuple) else out[0]
            captured["input"] = inp_t[:, -1, :].detach().cpu().float()
            captured["output"] = out_t[:, -1, :].detach().cpu().float()

        handle = mlp.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids=input_tensor)
        finally:
            handle.remove()

        return (captured["output"][0] - captured["input"][0]).numpy()

    # Baseline
    baseline_delta = get_mlp_delta(token_ids)

    # Ablated versions
    ablated_deltas = []
    for i in range(len(token_ids)):
        ablated = token_ids.copy()
        ablated[i] = pad_token_id
        ablated_deltas.append(get_mlp_delta(ablated))

    attributions = compute_ablation_attribution(
        baseline_delta, ablated_deltas, target_direction,
    )

    return build_attribution_result(
        prompt=prompt,
        tokens=tokens,
        attributions=attributions,
        layer=layer,
        target_token=target_token,
        target_token_id=target_token_id,
        method="ablation",
    )


# ---------------------------------------------------------------------------
# Remote runners
# ---------------------------------------------------------------------------


def run_attribution_ablation_remote(
    mlp_data_baseline: dict,
    mlp_data_ablated: list[dict],
    lm_head_weight: np.ndarray,
    tokens: list[str],
    prompt: str,
    layer: int,
    target_token_id: int,
    target_token: str,
) -> AttributionResult:
    """Compute ablation attribution from remotely-fetched MLP data.

    Args:
        mlp_data_baseline: {"mlp_input": np.ndarray, "mlp_output": np.ndarray}
        mlp_data_ablated: List of same dicts, one per token position.
        lm_head_weight: lm_head weight matrix.
        tokens: Decoded token strings.
        prompt: Original prompt.
        layer: Target layer.
        target_token_id: ID of target token.
        target_token: Text of target token.
    """
    from neurotrace.contrast import get_answer_direction

    target_direction = get_answer_direction(lm_head_weight, target_token_id)

    baseline_delta = mlp_data_baseline["mlp_output"] - mlp_data_baseline["mlp_input"]
    ablated_deltas = [
        d["mlp_output"] - d["mlp_input"] for d in mlp_data_ablated
    ]

    attributions = compute_ablation_attribution(
        baseline_delta, ablated_deltas, target_direction,
    )

    return build_attribution_result(
        prompt=prompt,
        tokens=tokens,
        attributions=attributions,
        layer=layer,
        target_token=target_token,
        target_token_id=target_token_id,
        method="ablation",
    )


def run_attribution_gradient_remote(
    grad_magnitudes: np.ndarray,
    tokens: list[str],
    prompt: str,
    layer: int,
    target_token_id: int,
    target_token: str,
) -> AttributionResult:
    """Build attribution result from remotely-computed gradient magnitudes."""
    attributions = compute_gradient_attribution(grad_magnitudes)
    return build_attribution_result(
        prompt=prompt,
        tokens=tokens,
        attributions=attributions,
        layer=layer,
        target_token=target_token,
        target_token_id=target_token_id,
        method="gradient",
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def attribution_run_to_dict(run: AttributionRun) -> dict:
    """Convert an AttributionRun to a JSON-serializable dict."""
    return {
        "run_id": run.run_id,
        "layer": run.layer,
        "target_direction": run.target_direction,
        "method": run.method,
        "model_name": run.model_name,
        "dataset": run.dataset,
        "created_at": run.created_at,
        "results": [
            {
                "prompt": r.prompt,
                "layer": r.layer,
                "target_token": r.target_token,
                "target_token_id": r.target_token_id,
                "method": r.method,
                "token_attributions": [
                    {
                        "position": ta.position,
                        "token": ta.token,
                        "attribution": ta.attribution,
                    }
                    for ta in r.token_attributions
                ],
            }
            for r in run.results
        ],
    }


def attribution_run_from_dict(data: dict) -> AttributionRun:
    """Reconstruct an AttributionRun from a dict."""
    results = []
    for r in data["results"]:
        token_attrs = [
            TokenAttribution(
                position=ta["position"],
                token=ta["token"],
                attribution=ta["attribution"],
            )
            for ta in r["token_attributions"]
        ]
        results.append(AttributionResult(
            prompt=r["prompt"],
            layer=r["layer"],
            target_token=r["target_token"],
            target_token_id=r["target_token_id"],
            method=r["method"],
            token_attributions=token_attrs,
        ))
    return AttributionRun(
        run_id=data["run_id"],
        layer=data["layer"],
        target_direction=data["target_direction"],
        method=data["method"],
        model_name=data["model_name"],
        dataset=data.get("dataset"),
        results=results,
        created_at=data["created_at"],
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_GREEN = "#4caf50"
_RED = "#f44336"
_BLUE = "#64b5f6"
_YELLOW = "#ffc107"
_GRAY = "#333333"
_BG = "#1a1a2e"
_BG2 = "#16213e"
_BG3 = "#0f3460"
_TEXT = "#e0e0e0"
_DIM = "#8a8a9a"
_ACCENT = "#e8956a"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _attribution_css() -> str:
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
    .prompt-section {{
        background: var(--bg2); border-radius: 8px; padding: 1.5rem;
        margin: 1rem 0;
    }}
    .token-bar {{
        display: inline-flex; flex-direction: column; align-items: center;
        margin: 0 2px; min-width: 40px;
    }}
    .token-text {{
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.85rem; padding: 2px 4px;
    }}
    .token-score {{
        font-size: 0.75rem; color: var(--dim); margin-top: 2px;
    }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th {{
        background: var(--bg3); color: var(--dim); text-align: left;
        padding: 0.5rem 0.75rem; font-weight: 600; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    td {{
        padding: 0.4rem 0.75rem; border-bottom: 1px solid var(--bg3);
        font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.85rem;
    }}
    .finding {{
        background: var(--bg2); padding: 1rem 1.5rem;
        border-left: 3px solid {_ACCENT}; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.95rem;
    }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .prompt-section {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _attribution_color(value: float, max_value: float) -> str:
    """Color intensity based on attribution strength."""
    if max_value <= 0:
        return _DIM
    ratio = min(1.0, value / max_value)
    if ratio > 0.6:
        return _ACCENT
    if ratio > 0.3:
        return _YELLOW
    return _DIM


def _bar_svg(attributions: list[TokenAttribution], width: int = 600) -> str:
    """Generate an SVG bar chart for token attributions."""
    if not attributions:
        return ""

    max_attr = max(ta.attribution for ta in attributions)
    bar_height = 20
    label_height = 20
    score_height = 16
    spacing = 4
    total_height = bar_height + label_height + score_height + spacing * 2

    n = len(attributions)
    bar_width = max(20, min(80, (width - 40) // max(n, 1)))
    chart_width = n * (bar_width + spacing) + 40
    chart_height = total_height + 30

    parts = [
        f'<svg viewBox="0 0 {chart_width} {chart_height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_BG2};border-radius:8px;">'
    ]

    for i, ta in enumerate(attributions):
        x = 20 + i * (bar_width + spacing)
        if max_attr > 0:
            h = max(2, int(bar_height * ta.attribution / max_attr))
        else:
            h = 2
        y = 10 + (bar_height - h)
        color = _attribution_color(ta.attribution, max_attr)

        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" '
            f'fill="{color}" rx="2">'
            f'<title>{_esc(ta.token)}: {ta.attribution:.3f}</title></rect>'
        )
        # Token label
        parts.append(
            f'<text x="{x + bar_width / 2}" y="{10 + bar_height + 14}" '
            f'text-anchor="middle" fill="{_TEXT}" font-size="11" '
            f'font-family="monospace">{_esc(ta.token.strip() or "▁")}</text>'
        )
        # Score
        parts.append(
            f'<text x="{x + bar_width / 2}" y="{10 + bar_height + 28}" '
            f'text-anchor="middle" fill="{_DIM}" font-size="9">'
            f'{ta.attribution:.3f}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def generate_attribution_html(run: AttributionRun) -> str:
    """Generate a self-contained HTML attribution report."""
    header = f"""
    <h1>Token Attribution Analysis</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Layer</span><br>
                <span class="meta-value">{run.layer}</span></div>
            <div><span class="meta-label">Target</span><br>
                <span class="meta-value">{_esc(run.target_direction)}</span></div>
            <div><span class="meta-label">Method</span><br>
                <span class="meta-value">{_esc(run.method)}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{len(run.results)}</span></div>
            <div><span class="meta-label">Run ID</span><br>
                <span class="meta-value">{_esc(run.run_id[:12])}</span></div>
        </div>
    </div>
    """

    # Per-prompt sections
    prompt_sections = []
    for result in run.results:
        svg = _bar_svg(result.token_attributions)

        # Find top driver
        top_attr = max(result.token_attributions, key=lambda t: t.attribution)

        prompt_sections.append(
            f'<div class="prompt-section">'
            f'<h3>{_esc(result.prompt)} &rarr; L{result.layer} MLP '
            f'&rarr; &ldquo;{_esc(result.target_token)}&rdquo;</h3>'
            f'{svg}'
            f'<div class="finding">Top driver: '
            f'&ldquo;{_esc(top_attr.token.strip())}&rdquo; '
            f'({top_attr.attribution:.3f})</div>'
            f'</div>'
        )

    # Aggregated summary (if multiple prompts)
    agg_section = ""
    if len(run.results) > 1:
        agg = aggregate_attributions(run.results)
        sorted_tokens = sorted(
            agg.items(), key=lambda x: x[1]["avg_attribution"], reverse=True,
        )

        agg_rows = []
        for token, stats in sorted_tokens[:20]:
            agg_rows.append(
                f"<tr>"
                f"<td>{_esc(token.strip() or '▁')}</td>"
                f"<td>{stats['avg_attribution']:.3f}</td>"
                f"<td>{int(stats['count'])}</td>"
                f"<td>{stats['max_attribution']:.3f}</td>"
                f"</tr>"
            )

        agg_section = (
            "<h2>Aggregated Token Attribution</h2>"
            '<div style="overflow-x:auto;"><table>'
            "<tr><th>Token</th><th>Avg Attribution</th>"
            "<th>Appearances</th><th>Max Attribution</th></tr>"
            f"{''.join(agg_rows)}"
            "</table></div>"
        )

    body = "\n".join([header, agg_section] + prompt_sections)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Token Attribution - L{run.layer}</title>\n"
        f"<style>{_attribution_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
