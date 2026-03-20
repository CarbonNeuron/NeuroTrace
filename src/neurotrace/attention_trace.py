"""Attention-Trace — per-head attention contribution decomposition."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AttentionTraceEntry:
    """One head's contribution measurement for a single prompt."""

    prompt: str
    layer: int
    head_idx: int
    answer_projection: float
    magnitude: float


@dataclass
class AttentionTraceResult:
    """Complete attention-trace for a single prompt."""

    prompt: str
    answer: str
    entries: list[AttentionTraceEntry]


@dataclass
class AttentionTraceRun:
    """A complete attention-trace run (one or more prompts)."""

    run_id: str
    dataset: str | None
    model_name: str
    layers: list[int]
    prompt_count: int
    results: list[AttentionTraceResult]
    created_at: str


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def get_answer_direction(lm_head_weight: np.ndarray, token_id: int) -> np.ndarray:
    """Extract and normalize the lm_head row for a token ID."""
    direction = lm_head_weight[token_id].astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return direction
    return direction / norm


def decompose_attention_heads(
    pre_proj: np.ndarray,
    o_proj_weight: np.ndarray,
    num_heads: int,
) -> list[np.ndarray]:
    """Decompose concatenated head outputs into per-head residual contributions.

    Args:
        pre_proj: input to o_proj at last position, shape [hidden_dim]
        o_proj_weight: o_proj.weight, shape [hidden_dim, hidden_dim]
        num_heads: number of attention heads

    Returns:
        List of per-head contribution vectors, each shape [hidden_dim]
    """
    hidden_dim = pre_proj.shape[0]
    head_dim = hidden_dim // num_heads
    head_vectors = pre_proj.reshape(num_heads, head_dim)

    contributions = []
    for i in range(num_heads):
        head_vec = head_vectors[i]
        w_o_slice = o_proj_weight[:, i * head_dim : (i + 1) * head_dim]
        contrib = w_o_slice @ head_vec
        contributions.append(contrib)
    return contributions


def compute_head_entries(
    contributions: list[np.ndarray],
    answer_direction: np.ndarray,
    prompt: str,
    layer: int,
) -> list[AttentionTraceEntry]:
    """Compute answer projection and magnitude for each head's contribution."""
    entries = []
    for head_idx, contrib in enumerate(contributions):
        proj = float(np.dot(contrib, answer_direction))
        mag = float(np.linalg.norm(contrib))
        entries.append(AttentionTraceEntry(
            prompt=prompt,
            layer=layer,
            head_idx=head_idx,
            answer_projection=proj,
            magnitude=mag,
        ))
    return entries


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------


def run_attention_trace_local(
    model,
    tokenizer,
    arch,
    prompt: str,
    answer: str,
    layers: list[int],
    lm_head_weight: np.ndarray,
    seed: int = 42,
) -> AttentionTraceResult:
    """Extract per-head attention contributions locally."""
    import torch

    torch.manual_seed(seed)

    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_token_id = answer_ids[0] if answer_ids else None

    answer_dir = None
    if answer_token_id is not None:
        answer_dir = get_answer_direction(lm_head_weight, answer_token_id)

    model_layers = arch.get_layers(model)
    num_heads = model.config.num_attention_heads

    # Hook o_proj with register_forward_pre_hook to capture input to o_proj
    captured_pre_proj: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_idx: int):
        def hook_fn(_module, args):
            inp = args[0] if isinstance(args, tuple) else args
            # inp shape: [batch, seq_len, hidden_dim]
            captured_pre_proj[layer_idx] = inp[0, -1, :].detach().cpu().float()
        return hook_fn

    try:
        for layer_idx in layers:
            if layer_idx < len(model_layers):
                attn = arch.get_attention(model_layers[layer_idx])
                o_proj = attn.o_proj
                hooks.append(o_proj.register_forward_pre_hook(_make_hook(layer_idx)))

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in hooks:
            h.remove()

    # Build entries
    all_entries: list[AttentionTraceEntry] = []
    for layer_idx in layers:
        if layer_idx not in captured_pre_proj:
            continue
        pre_proj = captured_pre_proj[layer_idx].numpy()

        attn = arch.get_attention(model_layers[layer_idx])
        o_proj_weight = attn.o_proj.weight.data.cpu().float().numpy()

        contributions = decompose_attention_heads(pre_proj, o_proj_weight, num_heads)

        if answer_dir is not None:
            entries = compute_head_entries(contributions, answer_dir, prompt, layer_idx)
        else:
            entries = [
                AttentionTraceEntry(
                    prompt=prompt,
                    layer=layer_idx,
                    head_idx=i,
                    answer_projection=0.0,
                    magnitude=float(np.linalg.norm(c)),
                )
                for i, c in enumerate(contributions)
            ]
        all_entries.extend(entries)

    return AttentionTraceResult(prompt=prompt, answer=answer, entries=all_entries)


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


def run_attention_trace_remote(
    layer_contributions: dict[int, np.ndarray],
    tokenizer,
    prompt: str,
    answer: str,
    layers: list[int],
    lm_head_weight: np.ndarray,
) -> AttentionTraceResult:
    """Build attention-trace result from remotely-fetched contributions.

    Args:
        layer_contributions: {layer: np.ndarray of shape [num_heads, hidden_dim]}
    """
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_token_id = answer_ids[0] if answer_ids else None

    answer_dir = None
    if answer_token_id is not None:
        answer_dir = get_answer_direction(lm_head_weight, answer_token_id)

    all_entries: list[AttentionTraceEntry] = []
    for layer_idx in layers:
        if layer_idx not in layer_contributions:
            continue
        contribs = layer_contributions[layer_idx]  # [num_heads, hidden_dim]
        num_heads = contribs.shape[0]

        if answer_dir is not None:
            entries = compute_head_entries(
                [contribs[i] for i in range(num_heads)],
                answer_dir, prompt, layer_idx,
            )
        else:
            entries = [
                AttentionTraceEntry(
                    prompt=prompt,
                    layer=layer_idx,
                    head_idx=i,
                    answer_projection=0.0,
                    magnitude=float(np.linalg.norm(contribs[i])),
                )
                for i in range(num_heads)
            ]
        all_entries.extend(entries)

    return AttentionTraceResult(prompt=prompt, answer=answer, entries=all_entries)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def attention_trace_run_to_dict(run: AttentionTraceRun) -> dict:
    """Convert an AttentionTraceRun to a JSON-serializable dict."""
    return {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "model_name": run.model_name,
        "layers": run.layers,
        "prompt_count": run.prompt_count,
        "created_at": run.created_at,
        "results": [
            {
                "prompt": r.prompt,
                "answer": r.answer,
                "entries": [
                    {
                        "layer": e.layer,
                        "head_idx": e.head_idx,
                        "answer_projection": e.answer_projection,
                        "magnitude": e.magnitude,
                    }
                    for e in r.entries
                ],
            }
            for r in run.results
        ],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_BG = "#0a0e14"
_BG2 = "#0f1923"
_BG3 = "#1a2332"
_TEXT = "#c8ccd4"
_DIM = "#6e7a8a"
_ACCENT = "#e8956a"
_BLUE = "#4db8ff"
_GREEN = "#4caf50"
_RED = "#f44336"
_WHITE = "#ffffff"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _value_to_color(value: float, max_abs: float) -> str:
    """Map value to green (positive) / red (negative) / white (zero)."""
    if max_abs < 1e-10:
        return _WHITE
    ratio = value / max_abs
    ratio = max(-1.0, min(1.0, ratio))
    if ratio > 0:
        r = int(255 * (1 - ratio) + 76 * ratio)
        g = int(255 * (1 - ratio) + 175 * ratio)
        b = int(255 * (1 - ratio) + 80 * ratio)
    elif ratio < 0:
        neg = -ratio
        r = int(255 * (1 - neg) + 244 * neg)
        g = int(255 * (1 - neg) + 67 * neg)
        b = int(255 * (1 - neg) + 54 * neg)
    else:
        r, g, b = 255, 255, 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _attention_trace_css() -> str:
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
    .chart-container {{
        background: var(--bg2); border-radius: 8px; padding: 1rem;
        margin: 1rem 0; overflow-x: auto;
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
    .side-by-side {{
        display: grid; gap: 2rem;
        grid-template-columns: 1fr 1fr;
    }}
    @media (max-width: 1200px) {{
        .side-by-side {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .chart-container {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _render_heads_heatmap_svg(
    entries: list[AttentionTraceEntry],
    layers: list[int],
    num_heads: int,
    title: str = "Answer Projection",
) -> str:
    """Render a layers (y) x heads (x) heatmap SVG."""
    n_layers = len(layers)
    if n_layers == 0 or num_heads == 0:
        return ""

    layer_set = set(layers)
    matrix: dict[tuple[int, int], float] = {}
    for e in entries:
        if e.layer in layer_set:
            matrix[(e.layer, e.head_idx)] = e.answer_projection

    all_vals = list(matrix.values())
    if not all_vals:
        return ""
    max_abs = max(abs(v) for v in all_vals) if all_vals else 1.0

    cell_w = max(18, min(30, 960 // max(num_heads, 1)))
    cell_h = max(18, min(28, 600 // max(n_layers, 1)))
    margin_left = 50
    margin_top = 50
    grid_w = num_heads * cell_w
    grid_h = n_layers * cell_h
    svg_w = margin_left + grid_w + 20
    svg_h = margin_top + grid_h + 30

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_BG2};border-radius:8px;">'
    ]

    parts.append(
        f'<text x="{svg_w / 2}" y="16" text-anchor="middle" '
        f'fill="{_TEXT}" font-size="12" font-weight="bold">{_esc(title)}</text>'
    )

    # Head labels (x-axis)
    for h_idx in range(num_heads):
        x = margin_left + h_idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{margin_top - 6}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="8">H{h_idx}</text>'
        )

    # Layer labels (y-axis) + cells
    for l_idx, layer in enumerate(layers):
        y = margin_top + l_idx * cell_h
        parts.append(
            f'<text x="{margin_left - 6}" y="{y + cell_h / 2 + 3}" '
            f'text-anchor="end" fill="{_DIM}" font-size="9">L{layer}</text>'
        )
        for h_idx in range(num_heads):
            x = margin_left + h_idx * cell_w
            val = matrix.get((layer, h_idx), 0.0)
            color = _value_to_color(val, max_abs)
            tooltip = f"L{layer}.H{h_idx}: {val:+.4f}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 1}" '
                f'height="{cell_h - 1}" fill="{color}" rx="1">'
                f'<title>{tooltip}</title></rect>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_attention_trace_html_single(
    result: AttentionTraceResult, run: AttentionTraceRun,
    mlp_total: float | None = None,
) -> str:
    """Generate HTML report for a single-prompt attention-trace."""
    layers = run.layers
    num_heads = max((e.head_idx for e in result.entries), default=0) + 1

    attn_total = sum(e.answer_projection for e in result.entries)

    header = f"""
    <h1>Attention-Trace</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Prompt</span><br>
                <span class="meta-value">{_esc(result.prompt)}</span></div>
            <div><span class="meta-label">Answer</span><br>
                <span class="meta-value">{_esc(result.answer)}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Layers</span><br>
                <span class="meta-value">{len(layers)}</span></div>
            <div><span class="meta-label">Heads/Layer</span><br>
                <span class="meta-value">{num_heads}</span></div>
            <div><span class="meta-label">Total Attn Proj</span><br>
                <span class="meta-value">{attn_total:+.2f}</span></div>
        </div>
    </div>
    """

    heatmap = _render_heads_heatmap_svg(
        result.entries, layers, num_heads,
        title=f"Per-Head Answer Projection → {_esc(result.answer)}",
    )

    # Top 10 heads table
    sorted_entries = sorted(
        result.entries, key=lambda e: abs(e.answer_projection), reverse=True,
    )
    top_rows = ""
    for e in sorted_entries[:10]:
        proj_color = _GREEN if e.answer_projection > 0 else _RED
        top_rows += (
            f"<tr>"
            f"<td>L{e.layer}.H{e.head_idx}</td>"
            f'<td style="color:{proj_color}">{e.answer_projection:+.4f}</td>'
            f"<td>{e.magnitude:.4f}</td>"
            f"</tr>"
        )

    top_table = (
        '<div style="overflow-x:auto;"><table>'
        "<tr><th>Head</th><th>Answer Proj</th><th>Magnitude</th></tr>"
        f"{top_rows}</table></div>"
    )

    # Summary line
    summary_parts = [f"Total attention: {attn_total:+.2f}"]
    if mlp_total is not None:
        net = attn_total + mlp_total
        summary_parts.append(f"Total MLP (from token-trace): {mlp_total:+.2f}")
        summary_parts.append(f"Net: {net:+.2f}")
    summary = (
        '<div class="finding">' + " | ".join(summary_parts) + "</div>"
    )

    body = "\n".join([
        header,
        "<h2>Head Contribution Heatmap</h2>",
        f'<div class="chart-container">{heatmap}</div>',
        "<h2>Top 10 Heads</h2>",
        top_table,
        summary,
    ])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Attention-Trace: {_esc(result.prompt[:60])}</title>\n"
        f"<style>{_attention_trace_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def generate_attention_trace_html_batch(
    run: AttentionTraceRun,
    vulnerable_prompts: set[str] | None = None,
) -> str:
    """Generate HTML report for a batch attention-trace run."""
    layers = run.layers
    num_heads = 0
    if run.results and run.results[0].entries:
        num_heads = max(e.head_idx for e in run.results[0].entries) + 1

    # Split robust vs vulnerable
    has_split = vulnerable_prompts is not None and len(vulnerable_prompts or set()) > 0
    robust_results = []
    vuln_results = []
    for r in run.results:
        if (
            has_split
            and vulnerable_prompts is not None
            and r.prompt in vulnerable_prompts
        ):
            vuln_results.append(r)
        else:
            robust_results.append(r)

    header = f"""
    <h1>Attention-Trace Batch</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(run.dataset or 'custom')}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{run.prompt_count}</span></div>
            <div><span class="meta-label">Layers</span><br>
                <span class="meta-value">{len(layers)}</span></div>
            <div><span class="meta-label">Heads/Layer</span><br>
                <span class="meta-value">{num_heads}</span></div>
    """
    if has_split:
        header += f"""
            <div><span class="meta-label">Robust</span><br>
                <span class="meta-value">{len(robust_results)}</span></div>
            <div><span class="meta-label">Vulnerable</span><br>
                <span class="meta-value">{len(vuln_results)}</span></div>
        """
    header += "</div></div>"

    # Compute mean projections per head across results
    def _mean_entries(results: list[AttentionTraceResult]) -> list[AttentionTraceEntry]:
        accum: dict[tuple[int, int], list[float]] = {}
        mag_accum: dict[tuple[int, int], list[float]] = {}
        for r in results:
            for e in r.entries:
                key = (e.layer, e.head_idx)
                accum.setdefault(key, []).append(e.answer_projection)
                mag_accum.setdefault(key, []).append(e.magnitude)
        mean_entries = []
        for (layer, head_idx), projs in accum.items():
            mean_entries.append(AttentionTraceEntry(
                prompt="(mean)",
                layer=layer,
                head_idx=head_idx,
                answer_projection=float(np.mean(projs)),
                magnitude=float(np.mean(mag_accum[(layer, head_idx)])),
            ))
        return mean_entries

    body_parts = [header]

    if has_split and vuln_results:
        robust_mean = _mean_entries(robust_results) if robust_results else []
        vuln_mean = _mean_entries(vuln_results)

        robust_svg = _render_heads_heatmap_svg(
            robust_mean, layers, num_heads,
            title=f"Robust ({len(robust_results)} prompts)",
        )
        vuln_svg = _render_heads_heatmap_svg(
            vuln_mean, layers, num_heads,
            title=f"Vulnerable ({len(vuln_results)} prompts)",
        )

        body_parts.append("<h2>Robust vs Vulnerable Heatmaps</h2>")
        body_parts.append(
            '<div class="side-by-side">'
            f'<div class="chart-container">{robust_svg}</div>'
            f'<div class="chart-container">{vuln_svg}</div>'
            "</div>"
        )
    else:
        all_mean = _mean_entries(run.results)
        all_svg = _render_heads_heatmap_svg(
            all_mean, layers, num_heads,
            title=f"Mean Answer Projection ({run.prompt_count} prompts)",
        )
        body_parts.append("<h2>Mean Head Contribution Heatmap</h2>")
        body_parts.append(f'<div class="chart-container">{all_svg}</div>')

    # Top heads table
    all_mean_entries = _mean_entries(run.results)
    sorted_heads = sorted(
        all_mean_entries, key=lambda e: abs(e.answer_projection), reverse=True,
    )

    top_rows = ""
    for e in sorted_heads[:10]:
        proj_color = _GREEN if e.answer_projection > 0 else _RED
        robust_proj = ""
        vuln_proj = ""
        if has_split:
            # Compute per-group means for this head
            r_vals = [
                ent.answer_projection
                for r in robust_results for ent in r.entries
                if ent.layer == e.layer and ent.head_idx == e.head_idx
            ]
            v_vals = [
                ent.answer_projection
                for r in vuln_results for ent in r.entries
                if ent.layer == e.layer and ent.head_idx == e.head_idx
            ]
            robust_proj = f"<td>{np.mean(r_vals):+.4f}</td>" if r_vals else "<td>—</td>"
            vuln_proj = f"<td>{np.mean(v_vals):+.4f}</td>" if v_vals else "<td>—</td>"

        top_rows += (
            f"<tr>"
            f"<td>L{e.layer}.H{e.head_idx}</td>"
            f'<td style="color:{proj_color}">{e.answer_projection:+.4f}</td>'
            f"{robust_proj}{vuln_proj}"
            f"</tr>"
        )

    extra_cols = ""
    if has_split:
        extra_cols = (
            f"<th>Robust ({len(robust_results)})</th>"
            f"<th>Vulnerable ({len(vuln_results)})</th>"
        )

    top_table = (
        '<div style="overflow-x:auto;"><table>'
        f"<tr><th>Head</th><th>Mean Proj</th>{extra_cols}</tr>"
        f"{top_rows}</table></div>"
    )

    body_parts.append("<h2>Top 10 Heads by Mean Answer Projection</h2>")
    body_parts.append(top_table)

    # Aggregate stats
    if has_split and vuln_results:
        robust_totals = [
            sum(e.answer_projection for e in r.entries) for r in robust_results
        ] if robust_results else [0.0]
        vuln_totals = [
            sum(e.answer_projection for e in r.entries) for r in vuln_results
        ]
        robust_head_counts = [
            sum(1 for e in r.entries if e.answer_projection > 0.5)
            for r in robust_results
        ] if robust_results else [0]
        vuln_head_counts = [
            sum(1 for e in r.entries if e.answer_projection > 0.5)
            for r in vuln_results
        ]

        agg_table = (
            '<div style="overflow-x:auto;"><table>'
            f"<tr><th>Metric</th>"
            f"<th>Robust ({len(robust_results)})</th>"
            f"<th>Vulnerable ({len(vuln_results)})</th></tr>"
            f"<tr><td>Total attn proj</td>"
            f"<td>{np.mean(robust_totals):+.2f}</td>"
            f"<td>{np.mean(vuln_totals):+.2f}</td></tr>"
            f"<tr><td>Num heads &gt; +0.5</td>"
            f"<td>{np.mean(robust_head_counts):.1f}</td>"
            f"<td>{np.mean(vuln_head_counts):.1f}</td></tr>"
            "</table></div>"
        )
        body_parts.append("<h2>Aggregate</h2>")
        body_parts.append(agg_table)

    body = "\n".join(body_parts)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Attention-Trace: {_esc(run.dataset or 'batch')}</title>\n"
        f"<style>{_attention_trace_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
