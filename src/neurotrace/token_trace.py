"""Token-Trace — full position x layer MLP delta analysis."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TokenTraceEntry:
    """MLP delta metrics for one token position at one layer."""

    layer: int
    position: int
    token: str
    is_subject: bool
    is_last: bool
    answer_projection: float
    competitor_projection: float | None
    delta_magnitude: float


@dataclass
class TokenTraceResult:
    """Complete token-trace for a single prompt."""

    prompt: str
    answer: str
    subject: str | None
    entries: list[TokenTraceEntry]


@dataclass
class TokenTraceRun:
    """A complete token-trace run (one or more prompts)."""

    run_id: str
    dataset: str | None
    model_name: str
    layers: list[int]
    prompt_count: int
    results: list[TokenTraceResult]
    created_at: str


# ---------------------------------------------------------------------------
# Subject token detection
# ---------------------------------------------------------------------------


def find_subject_token_positions(
    prompt: str, subject: str, tokenizer,
) -> tuple[int, int]:
    """Find the start and end token positions of the subject in the prompt.

    Returns (start_pos, end_pos) inclusive, 0-indexed.
    """
    prompt_ids = tokenizer.encode(prompt)
    subject_ids = tokenizer.encode(subject, add_special_tokens=False)

    for i in range(len(prompt_ids) - len(subject_ids), -1, -1):
        if prompt_ids[i : i + len(subject_ids)] == subject_ids:
            return (i, i + len(subject_ids) - 1)

    raise ValueError(f"Subject '{subject}' not found in tokenized prompt")


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------


def get_answer_direction(lm_head_weight: np.ndarray, token_id: int) -> np.ndarray:
    """Extract and normalize the lm_head row for a token ID."""
    direction = lm_head_weight[token_id].astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return direction
    return direction / norm


def compute_answer_projection(
    delta: np.ndarray, answer_direction: np.ndarray,
) -> float:
    """Dot product of delta with answer direction.

    Uses dot product (not cosine) so magnitude matters — a large positive
    value means the MLP is actively pushing toward the answer.
    """
    return float(np.dot(delta, answer_direction))


# ---------------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------------


def run_token_trace_local(
    model,
    tokenizer,
    arch,
    prompt: str,
    answer: str,
    subject: str | None,
    layers: list[int],
    lm_head_weight: np.ndarray,
    commitment_data: dict[str, dict] | None = None,
    seed: int = 42,
) -> TokenTraceResult:
    """Extract MLP deltas at all token positions across specified layers (local)."""
    import torch

    torch.manual_seed(seed)

    tokens_ids = tokenizer.encode(prompt)
    token_texts = [tokenizer.decode([t]) for t in tokens_ids]
    num_positions = len(tokens_ids)

    # Find subject positions
    subject_start, subject_end = None, None
    if subject:
        try:
            subject_start, subject_end = find_subject_token_positions(
                prompt, subject, tokenizer,
            )
        except ValueError:
            pass

    # Get answer token direction
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_token_id = answer_ids[0] if answer_ids else None

    answer_dir = None
    if answer_token_id is not None:
        answer_dir = get_answer_direction(lm_head_weight, answer_token_id)

    # Get competitor direction
    competitor_dir = None
    if commitment_data and prompt in commitment_data:
        comp_str = commitment_data[prompt].get("competitor_token", "")
        if comp_str:
            comp_ids = tokenizer.encode(" " + comp_str, add_special_tokens=False)
            if not comp_ids:
                comp_ids = tokenizer.encode(comp_str, add_special_tokens=False)
            if comp_ids:
                competitor_dir = get_answer_direction(lm_head_weight, comp_ids[0])

    # Hook MLP modules — capture ALL positions
    model_layers = arch.get_layers(model)
    captured_in: dict[int, torch.Tensor] = {}
    captured_out: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output if not isinstance(output, tuple) else output[0]
            # Capture all positions: [batch, seq_len, hidden_dim]
            captured_in[layer_idx] = inp[0].detach().cpu().float()
            captured_out[layer_idx] = out[0].detach().cpu().float()
        return hook_fn

    try:
        for layer_idx in layers:
            if layer_idx < len(model_layers):
                mlp = arch.get_mlp(model_layers[layer_idx])
                hooks.append(mlp.register_forward_hook(_make_hook(layer_idx)))

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
    entries: list[TokenTraceEntry] = []
    for layer_idx in layers:
        if layer_idx not in captured_in:
            continue
        mlp_in = captured_in[layer_idx].numpy()  # [seq_len, hidden_dim]
        mlp_out = captured_out[layer_idx].numpy()

        for pos in range(num_positions):
            delta = mlp_out[pos] - mlp_in[pos]

            ans_proj = 0.0
            if answer_dir is not None:
                ans_proj = compute_answer_projection(delta, answer_dir)

            comp_proj = None
            if competitor_dir is not None:
                comp_proj = compute_answer_projection(delta, competitor_dir)

            is_subj = (
                subject_start is not None
                and subject_end is not None
                and subject_start <= pos <= subject_end
            )

            entries.append(TokenTraceEntry(
                layer=layer_idx,
                position=pos,
                token=token_texts[pos],
                is_subject=is_subj,
                is_last=(pos == num_positions - 1),
                answer_projection=ans_proj,
                competitor_projection=comp_proj,
                delta_magnitude=float(np.linalg.norm(delta)),
            ))

    return TokenTraceResult(
        prompt=prompt,
        answer=answer,
        subject=subject,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


def run_token_trace_remote(
    all_position_deltas: dict[int, np.ndarray],
    tokenizer,
    prompt: str,
    answer: str,
    subject: str | None,
    layers: list[int],
    lm_head_weight: np.ndarray,
    commitment_data: dict[str, dict] | None = None,
) -> TokenTraceResult:
    """Build token-trace result from remotely-fetched all-position MLP deltas.

    Args:
        all_position_deltas: {layer: np.ndarray of shape [num_positions, hidden_dim]}
    """
    tokens_ids = tokenizer.encode(prompt)
    token_texts = [tokenizer.decode([t]) for t in tokens_ids]
    num_positions = len(tokens_ids)

    subject_start, subject_end = None, None
    if subject:
        try:
            subject_start, subject_end = find_subject_token_positions(
                prompt, subject, tokenizer,
            )
        except ValueError:
            pass

    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_token_id = answer_ids[0] if answer_ids else None

    answer_dir = None
    if answer_token_id is not None:
        answer_dir = get_answer_direction(lm_head_weight, answer_token_id)

    competitor_dir = None
    if commitment_data and prompt in commitment_data:
        comp_str = commitment_data[prompt].get("competitor_token", "")
        if comp_str:
            comp_ids = tokenizer.encode(" " + comp_str, add_special_tokens=False)
            if not comp_ids:
                comp_ids = tokenizer.encode(comp_str, add_special_tokens=False)
            if comp_ids:
                competitor_dir = get_answer_direction(lm_head_weight, comp_ids[0])

    entries: list[TokenTraceEntry] = []
    for layer_idx in layers:
        if layer_idx not in all_position_deltas:
            continue
        deltas = all_position_deltas[layer_idx]  # [num_positions, hidden_dim]

        for pos in range(min(num_positions, deltas.shape[0])):
            delta = deltas[pos]

            ans_proj = 0.0
            if answer_dir is not None:
                ans_proj = compute_answer_projection(delta, answer_dir)

            comp_proj = None
            if competitor_dir is not None:
                comp_proj = compute_answer_projection(delta, competitor_dir)

            is_subj = (
                subject_start is not None
                and subject_end is not None
                and subject_start <= pos <= subject_end
            )

            entries.append(TokenTraceEntry(
                layer=layer_idx,
                position=pos,
                token=token_texts[pos],
                is_subject=is_subj,
                is_last=(pos == num_positions - 1),
                answer_projection=ans_proj,
                competitor_projection=comp_proj,
                delta_magnitude=float(np.linalg.norm(delta)),
            ))

    return TokenTraceResult(
        prompt=prompt,
        answer=answer,
        subject=subject,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def classify_token_position(
    token: str,
    is_subject: bool,
    is_last: bool,
    relation_keywords: set[str] | None = None,
) -> str:
    """Classify a token as 'subject', 'last', 'relation', or 'other'."""
    if is_subject:
        return "subject"
    if is_last:
        return "last"
    if relation_keywords:
        stripped = token.strip().lower()
        if stripped in relation_keywords:
            return "relation"
    return "other"


def aggregate_by_position_type(
    results: list[TokenTraceResult],
    layers: list[int],
    relation_keywords: set[str] | None = None,
) -> dict[str, dict[int, dict[str, float]]]:
    """Aggregate answer projections by position type and layer.

    Returns: {position_type: {layer: {"mean": ..., "count": ...}}}
    """
    accum: dict[str, dict[int, list[float]]] = {}

    for result in results:
        for entry in result.entries:
            pos_type = classify_token_position(
                entry.token, entry.is_subject, entry.is_last,
                relation_keywords,
            )
            accum.setdefault(pos_type, {}).setdefault(entry.layer, []).append(
                entry.answer_projection,
            )

    out: dict[str, dict[int, dict[str, float]]] = {}
    for pos_type, layer_vals in accum.items():
        out[pos_type] = {}
        for layer, vals in layer_vals.items():
            out[pos_type][layer] = {
                "mean": float(np.mean(vals)),
                "count": float(len(vals)),
            }
    return out


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def token_trace_run_to_dict(run: TokenTraceRun) -> dict:
    """Convert a TokenTraceRun to a JSON-serializable dict."""
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
                "subject": r.subject,
                "entries": [
                    {
                        "layer": e.layer,
                        "position": e.position,
                        "token": e.token,
                        "is_subject": e.is_subject,
                        "is_last": e.is_last,
                        "answer_projection": e.answer_projection,
                        "competitor_projection": e.competitor_projection,
                        "delta_magnitude": e.delta_magnitude,
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
        # Green intensity
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


def _magnitude_to_color(value: float, max_val: float) -> str:
    """Map magnitude to blue intensity."""
    if max_val < 1e-10:
        return _BG2
    ratio = min(1.0, value / max_val)
    r = int(15 * (1 - ratio) + 77 * ratio)
    g = int(25 * (1 - ratio) + 184 * ratio)
    b = int(35 * (1 - ratio) + 255 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def _token_trace_css() -> str:
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
    .gallery {{
        display: grid; gap: 1rem;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    }}
    .gallery-item {{
        background: var(--bg2); border-radius: 8px; padding: 1rem;
        cursor: pointer;
    }}
    .gallery-item:hover {{ background: var(--bg3); }}
    details {{ margin: 0.5rem 0; }}
    summary {{
        cursor: pointer; color: {_BLUE}; padding: 0.5rem 0.75rem;
        background: var(--bg2); border-radius: 4px; font-size: 0.9rem;
    }}
    summary:hover {{ background: var(--bg3); }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .chart-container {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _render_heatmap_svg(
    entries: list[TokenTraceEntry],
    tokens: list[str],
    layers: list[int],
    value_key: str = "answer_projection",
    title: str = "Answer Projection",
) -> str:
    """Render a tokens (x) x layers (y) heatmap SVG."""
    n_tokens = len(tokens)
    n_layers = len(layers)
    if n_tokens == 0 or n_layers == 0:
        return ""

    layer_set = set(layers)
    # Build matrix
    matrix: dict[tuple[int, int], float] = {}
    for e in entries:
        if e.layer in layer_set:
            val = getattr(e, value_key) if hasattr(e, value_key) else 0.0
            if val is None:
                val = 0.0
            matrix[(e.layer, e.position)] = val

    all_vals = list(matrix.values())
    if not all_vals:
        return ""
    max_abs = max(abs(v) for v in all_vals) if all_vals else 1.0

    cell_w = max(30, min(70, 700 // max(n_tokens, 1)))
    cell_h = max(18, min(28, 500 // max(n_layers, 1)))
    margin_left = 50
    margin_top = 60
    grid_w = n_tokens * cell_w
    grid_h = n_layers * cell_h
    svg_w = margin_left + grid_w + 20
    svg_h = margin_top + grid_h + 30

    use_magnitude = value_key == "delta_magnitude"

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_BG2};border-radius:8px;">'
    ]

    # Title
    parts.append(
        f'<text x="{svg_w / 2}" y="16" text-anchor="middle" '
        f'fill="{_TEXT}" font-size="12" font-weight="bold">{_esc(title)}</text>'
    )

    # Token labels (x-axis, top)
    for t_idx, tok in enumerate(tokens):
        x = margin_left + t_idx * cell_w + cell_w / 2
        label = tok.strip()[:8] or "▁"
        parts.append(
            f'<text x="{x}" y="{margin_top - 6}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="9" '
            f'transform="rotate(-35, {x}, {margin_top - 6})">'
            f'{_esc(label)}</text>'
        )

    # Layer labels (y-axis) + cells
    for l_idx, layer in enumerate(layers):
        y = margin_top + l_idx * cell_h
        parts.append(
            f'<text x="{margin_left - 6}" y="{y + cell_h / 2 + 3}" '
            f'text-anchor="end" fill="{_DIM}" font-size="9">L{layer}</text>'
        )
        for t_idx in range(n_tokens):
            x = margin_left + t_idx * cell_w
            val = matrix.get((layer, t_idx), 0.0)
            if use_magnitude:
                color = _magnitude_to_color(val, max_abs)
            else:
                color = _value_to_color(val, max_abs)
            tooltip = f"L{layer} / {_esc(tokens[t_idx].strip()[:12])}: {val:+.4f}"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 1}" '
                f'height="{cell_h - 1}" fill="{color}" rx="1">'
                f'<title>{tooltip}</title></rect>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_token_trace_html_single(
    result: TokenTraceResult, run: TokenTraceRun,
) -> str:
    """Generate HTML report for a single prompt token-trace."""
    tokens = []
    seen_pos = set()
    for e in result.entries:
        if e.position not in seen_pos:
            tokens.append(e.token)
            seen_pos.add(e.position)

    layers = run.layers

    header = f"""
    <h1>Token-Trace</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Prompt</span><br>
                <span class="meta-value">{_esc(result.prompt)}</span></div>
            <div><span class="meta-label">Answer</span><br>
                <span class="meta-value">{_esc(result.answer)}</span></div>
            <div><span class="meta-label">Subject</span><br>
                <span class="meta-value">{_esc(result.subject or '—')}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Layers</span><br>
                <span class="meta-value">{len(layers)}</span></div>
        </div>
    </div>
    """

    # Answer projection heatmap
    answer_heatmap = _render_heatmap_svg(
        result.entries, tokens, layers,
        value_key="answer_projection",
        title=f"Answer Projection ({_esc(result.answer)})",
    )

    # Competitor heatmap (if data exists)
    has_competitor = any(
        e.competitor_projection is not None for e in result.entries
    )
    competitor_heatmap = ""
    if has_competitor:
        competitor_heatmap = (
            "<h2>Competitor Projection</h2>"
            '<div class="chart-container">'
            + _render_heatmap_svg(
                result.entries, tokens, layers,
                value_key="competitor_projection",
                title="Competitor Projection",
            )
            + "</div>"
        )

    # Delta magnitude heatmap
    magnitude_heatmap = _render_heatmap_svg(
        result.entries, tokens, layers,
        value_key="delta_magnitude",
        title="MLP Delta Magnitude",
    )

    # Find peak/suppression
    peak_entry = max(result.entries, key=lambda e: e.answer_projection, default=None)
    min_entry = min(result.entries, key=lambda e: e.answer_projection, default=None)

    findings = ""
    if peak_entry and min_entry:
        findings = f"""
        <div class="finding">
            Peak answer signal: "{_esc(peak_entry.token.strip())}" @ L{peak_entry.layer}
            ({peak_entry.answer_projection:+.4f})
        </div>
        <div class="finding">
            Peak suppression: "{_esc(min_entry.token.strip())}" @ L{min_entry.layer}
            ({min_entry.answer_projection:+.4f})
        </div>
        """

    body = "\n".join([
        header,
        "<h2>Answer Projection Heatmap</h2>",
        f'<div class="chart-container">{answer_heatmap}</div>',
        competitor_heatmap,
        "<h2>Delta Magnitude Heatmap</h2>",
        f'<div class="chart-container">{magnitude_heatmap}</div>',
        "<h2>Findings</h2>",
        findings,
    ])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Token-Trace: {_esc(result.prompt[:60])}</title>\n"
        f"<style>{_token_trace_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def generate_token_trace_html_batch(
    run: TokenTraceRun,
    relation_keywords: set[str] | None = None,
) -> str:
    """Generate HTML report for a batch token-trace run."""

    # Aggregated heatmap by position type
    agg = aggregate_by_position_type(
        run.results, run.layers, relation_keywords,
    )

    pos_types = ["subject", "relation", "other", "last"]
    pos_types = [pt for pt in pos_types if pt in agg]

    header = f"""
    <h1>Token-Trace Batch</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(run.dataset or 'custom')}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{run.prompt_count}</span></div>
            <div><span class="meta-label">Layers</span><br>
                <span class="meta-value">{len(run.layers)}</span></div>
        </div>
    </div>
    """

    # Aggregated heatmap SVG
    n_pos_types = len(pos_types)
    n_layers = len(run.layers)
    if n_pos_types > 0 and n_layers > 0:
        all_means = [
            agg[pt][li]["mean"]
            for pt in pos_types
            for li in run.layers
            if li in agg[pt]
        ]
        max_abs = max((abs(v) for v in all_means), default=1.0)

        cell_w = max(25, min(50, 700 // max(n_layers, 1)))
        cell_h = 35
        margin_left = 80
        margin_top = 50
        grid_w = n_layers * cell_w
        grid_h = n_pos_types * cell_h
        svg_w = margin_left + grid_w + 20
        svg_h = margin_top + grid_h + 30

        svg_parts = [
            f'<svg viewBox="0 0 {svg_w} {svg_h}" '
            f'xmlns="http://www.w3.org/2000/svg" '
            f'style="background:{_BG2};border-radius:8px;">'
        ]

        svg_parts.append(
            f'<text x="{svg_w / 2}" y="16" text-anchor="middle" '
            f'fill="{_TEXT}" font-size="12" font-weight="bold">'
            f'Mean Answer Projection by Position Type</text>'
        )

        for l_idx, layer in enumerate(run.layers):
            x = margin_left + l_idx * cell_w + cell_w / 2
            svg_parts.append(
                f'<text x="{x}" y="{margin_top - 8}" text-anchor="middle" '
                f'fill="{_DIM}" font-size="9">L{layer}</text>'
            )

        for p_idx, pt in enumerate(pos_types):
            y = margin_top + p_idx * cell_h
            svg_parts.append(
                f'<text x="{margin_left - 6}" y="{y + cell_h / 2 + 3}" '
                f'text-anchor="end" fill="{_DIM}" font-size="10">'
                f'{_esc(pt)}</text>'
            )
            for l_idx, layer in enumerate(run.layers):
                x = margin_left + l_idx * cell_w
                stats = agg[pt].get(layer, {"mean": 0.0})
                val = stats["mean"]
                color = _value_to_color(val, max_abs)
                tooltip = f"{pt} @ L{layer}: {val:+.4f}"
                svg_parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 1}" '
                    f'height="{cell_h - 1}" fill="{color}" rx="1">'
                    f'<title>{tooltip}</title></rect>'
                )

        svg_parts.append("</svg>")
        agg_heatmap = "\n".join(svg_parts)
    else:
        agg_heatmap = ""

    # Position type summary table
    summary_rows = []
    for pt in pos_types:
        layer_data = agg[pt]
        if not layer_data:
            continue
        means = [
            layer_data[li]["mean"]
            for li in run.layers if li in layer_data
        ]
        if not means:
            continue
        avg_proj = float(np.mean(means))
        peak_layer = max(
            (li for li in run.layers if li in layer_data),
            key=lambda li: layer_data[li]["mean"],
        )
        peak_val = layer_data[peak_layer]["mean"]
        count = sum(
            layer_data[li]["count"]
            for li in run.layers if li in layer_data
        )

        role = "—"
        if avg_proj > 0.01:
            role = "Positive (factual signal)"
        elif avg_proj < -0.01:
            role = "Suppression"
        else:
            role = "Negligible"

        summary_rows.append(
            f"<tr>"
            f"<td>{_esc(pt)}</td>"
            f"<td>{avg_proj:+.4f}</td>"
            f"<td>L{peak_layer} ({peak_val:+.4f})</td>"
            f"<td>{int(count)}</td>"
            f"<td>{_esc(role)}</td>"
            f"</tr>"
        )

    summary_table = (
        '<div style="overflow-x:auto;"><table>'
        "<tr><th>Position Type</th><th>Avg Proj</th>"
        "<th>Peak Layer</th><th>Samples</th><th>Role</th></tr>"
        f"{''.join(summary_rows)}"
        "</table></div>"
    )

    # Per-prompt gallery (thumbnails)
    gallery_items = []
    for r in run.results[:20]:  # limit to 20 thumbnails
        thumb_tokens = []
        seen = set()
        for e in r.entries:
            if e.position not in seen:
                thumb_tokens.append(e.token)
                seen.add(e.position)
        thumb_svg = _render_heatmap_svg(
            r.entries, thumb_tokens, run.layers,
            value_key="answer_projection",
            title=f"{_esc(r.prompt[:30])} → {_esc(r.answer)}",
        )
        gallery_items.append(
            f'<div class="gallery-item">{thumb_svg}</div>'
        )

    gallery = ""
    if gallery_items:
        gallery = (
            "<h2>Per-Prompt Heatmaps</h2>"
            f'<div class="gallery">{"".join(gallery_items)}</div>'
        )

    body = "\n".join([
        header,
        "<h2>Aggregated Position-Type Analysis</h2>",
        f'<div class="chart-container">{agg_heatmap}</div>',
        summary_table,
        gallery,
    ])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Token-Trace: {_esc(run.dataset or 'batch')}</title>\n"
        f"<style>{_token_trace_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
