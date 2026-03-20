"""MLP ablation heatmap — data structures, runner, aggregation, and HTML generation."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HeatmapCell:
    """One cell in the heatmap grid: one prompt x one layer."""

    prompt_index: int
    prompt: str
    expected_answer: str
    layer: int
    baseline_token: str
    baseline_prob: float
    baseline_correct: bool
    ablated_token: str
    ablated_prob: float
    ablated_correct: bool
    delta_correct_prob: float  # change in correct answer's prob
    flipped: bool
    flip_direction: str  # "fixed" | "broke" | "changed" | "none"


@dataclass
class HeatmapLayerSummary:
    """Aggregate statistics for a single layer across all prompts."""

    layer: int
    total_prompts: int
    fixes: int
    breaks: int
    changes: int
    unchanged: int
    avg_delta_correct_prob: float
    impact_score: float  # (fixes - breaks) / total


@dataclass
class HeatmapResult:
    """Complete result of a heatmap run."""

    run_id: str
    dataset_name: str
    model_name: str
    num_layers: int
    num_prompts: int
    cells: list[HeatmapCell]
    layer_summaries: list[HeatmapLayerSummary]
    created_at: str


# ---------------------------------------------------------------------------
# Correctness matching
# ---------------------------------------------------------------------------


def check_correct(prediction: str, answer: str) -> bool:
    """Check if *prediction* matches *answer* using prefix matching.

    Mirrors the logic in ``scan.py`` — either string may be a prefix of
    the other, after stripping and lower-casing.
    """
    pred = prediction.strip().lower()
    ans = answer.strip().lower()
    if not pred or not ans:
        return False
    return ans.startswith(pred) or pred.startswith(ans)


# ---------------------------------------------------------------------------
# flip_direction helper
# ---------------------------------------------------------------------------


def _compute_flip_direction(
    baseline_correct: bool,
    ablated_correct: bool,
    baseline_token: str,
    ablated_token: str,
) -> str:
    """Determine the flip direction between baseline and ablated runs."""
    if not baseline_correct and ablated_correct:
        return "fixed"
    if baseline_correct and not ablated_correct:
        return "broke"
    if baseline_token.strip() != ablated_token.strip():
        return "changed"
    return "none"


# ---------------------------------------------------------------------------
# Layer summary aggregation
# ---------------------------------------------------------------------------


def build_layer_summaries(
    cells: list[HeatmapCell],
    num_layers: int,
    num_prompts: int,
) -> list[HeatmapLayerSummary]:
    """Build per-layer summary statistics from a list of cells."""
    # Group cells by layer
    by_layer: dict[int, list[HeatmapCell]] = {}
    for cell in cells:
        by_layer.setdefault(cell.layer, []).append(cell)

    summaries: list[HeatmapLayerSummary] = []
    for layer_idx in range(num_layers):
        layer_cells = by_layer.get(layer_idx, [])
        fixes = sum(1 for c in layer_cells if c.flip_direction == "fixed")
        breaks = sum(1 for c in layer_cells if c.flip_direction == "broke")
        changes = sum(1 for c in layer_cells if c.flip_direction == "changed")
        unchanged = sum(1 for c in layer_cells if c.flip_direction == "none")
        total = len(layer_cells) if layer_cells else num_prompts
        avg_delta = (
            sum(c.delta_correct_prob for c in layer_cells) / len(layer_cells)
            if layer_cells
            else 0.0
        )
        impact = (fixes - breaks) / total if total > 0 else 0.0

        summaries.append(
            HeatmapLayerSummary(
                layer=layer_idx,
                total_prompts=total,
                fixes=fixes,
                breaks=breaks,
                changes=changes,
                unchanged=unchanged,
                avg_delta_correct_prob=avg_delta,
                impact_score=impact,
            )
        )

    return summaries


# ---------------------------------------------------------------------------
# Local heatmap runner (requires torch)
# ---------------------------------------------------------------------------


def run_heatmap_local(
    model,
    tokenizer,
    arch,  # ModelArchitecture
    prompts: list[dict],  # [{"prompt": "...", "answer": "..."}]
    seed: int = 42,
    progress_callback=None,  # callable(prompt_idx, layer_idx, num_prompts, num_layers)
) -> list[HeatmapCell]:
    """Run MLP-zeroing ablation across all prompts and layers.

    For each prompt:
      1. Run a baseline forward pass (no ablation) to get the final token
         and probability.
      2. For each layer, zero the MLP output and run a forward pass to get
         the ablated token and probability.
      3. Build a :class:`HeatmapCell` for every (prompt, layer) pair.

    Args:
        model: A loaded ``AutoModelForCausalLM`` instance.
        tokenizer: The matching tokenizer.
        arch: :class:`~neurotrace.models.ModelArchitecture` for the model.
        prompts: List of ``{"prompt": str, "answer": str}`` dicts.
        seed: Random seed for reproducibility.
        progress_callback: Optional ``(prompt_idx, layer_idx, num_prompts,
            num_layers)`` callable invoked before each ablated run.

    Returns:
        A list of :class:`HeatmapCell` objects (one per prompt per layer).
    """
    import torch

    torch.manual_seed(seed)

    layers = arch.get_layers(model)
    num_layers = len(layers)
    num_prompts = len(prompts)

    def zero_mlp_hook(module, input, output):
        return torch.zeros_like(output)

    def _get_prediction(prompt_text: str) -> tuple[str, float, dict[str, float]]:
        """Run a forward pass and return (top_token, top_prob, full_probs_by_token).

        ``full_probs_by_token`` is not returned literally (too large); instead
        we return the top-1 token string and probability so the caller can
        look up correctness.
        """
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[0, -1, :]  # last token position
        probs = torch.softmax(logits.float().cpu(), dim=-1)
        top_prob, top_idx = probs.topk(1)
        token = tokenizer.decode(top_idx[0])
        return token.strip(), float(top_prob[0].item()), probs

    def _correct_answer_prob(probs_tensor, answer: str) -> float:
        """Get probability assigned to the correct answer token."""
        encoded = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not encoded:
            return 0.0
        answer_id = encoded[0]
        if answer_id < len(probs_tensor):
            return float(probs_tensor[answer_id].item())
        return 0.0

    cells: list[HeatmapCell] = []

    for p_idx, entry in enumerate(prompts):
        prompt_text = entry["prompt"]
        answer = entry["answer"]

        # --- Baseline (no ablation) ---
        baseline_token, baseline_prob, baseline_probs = _get_prediction(prompt_text)
        baseline_correct = check_correct(baseline_token, answer)
        baseline_answer_prob = _correct_answer_prob(baseline_probs, answer)

        # --- Per-layer MLP ablation ---
        for l_idx in range(num_layers):
            if progress_callback is not None:
                progress_callback(p_idx, l_idx, num_prompts, num_layers)

            layer = layers[l_idx]
            mlp = arch.get_mlp(layer)
            handle = mlp.register_forward_hook(zero_mlp_hook)
            try:
                ablated_token, ablated_prob, ablated_probs = _get_prediction(
                    prompt_text
                )
            finally:
                handle.remove()

            ablated_correct = check_correct(ablated_token, answer)
            ablated_answer_prob = _correct_answer_prob(ablated_probs, answer)
            delta_correct_prob = ablated_answer_prob - baseline_answer_prob

            flip_dir = _compute_flip_direction(
                baseline_correct, ablated_correct, baseline_token, ablated_token
            )
            flipped = flip_dir in ("fixed", "broke")

            cells.append(
                HeatmapCell(
                    prompt_index=p_idx,
                    prompt=prompt_text,
                    expected_answer=answer,
                    layer=l_idx,
                    baseline_token=baseline_token,
                    baseline_prob=baseline_prob,
                    baseline_correct=baseline_correct,
                    ablated_token=ablated_token,
                    ablated_prob=ablated_prob,
                    ablated_correct=ablated_correct,
                    delta_correct_prob=delta_correct_prob,
                    flipped=flipped,
                    flip_direction=flip_dir,
                )
            )

    return cells


# ---------------------------------------------------------------------------
# HTML heatmap generation
# ---------------------------------------------------------------------------

# Color palette
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


def _cell_color(flip_direction: str) -> str:
    """Return the base color for a heatmap cell."""
    if flip_direction == "fixed":
        return _GREEN
    if flip_direction == "broke":
        return _RED
    if flip_direction == "changed":
        return _BLUE
    return _GRAY


def _cell_opacity(delta_correct_prob: float) -> float:
    """Compute cell opacity proportional to |delta|, clamped [0.3, 1.0]."""
    raw = abs(delta_correct_prob)
    return max(0.3, min(1.0, raw * 5.0 + 0.3))


def _heatmap_css() -> str:
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
    .chart-container {{
        background: var(--bg2); border-radius: 8px; padding: 1rem;
        margin: 1rem 0; overflow-x: auto;
    }}
    .legend {{
        display: flex; gap: 1.5rem; margin: 1rem 0; flex-wrap: wrap;
        align-items: center;
    }}
    .legend-item {{
        display: flex; align-items: center; gap: 0.4rem;
        font-size: 0.85rem;
    }}
    .legend-swatch {{
        width: 16px; height: 16px; border-radius: 3px;
        display: inline-block;
    }}
    svg {{ max-width: 100%; height: auto; }}
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
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .chart-container {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def generate_heatmap_html(result: HeatmapResult) -> str:
    """Generate a self-contained HTML heatmap visualization.

    Layout:
      - Header with dataset name, model, timestamp, dimensions
      - SVG grid: X-axis = layers, Y-axis = prompts (sorted by answer)
      - Cell colour encodes flip direction; opacity encodes |delta_correct_prob|
      - Summary bar at bottom showing per-layer stacked totals
      - Right margin shows per-prompt baseline correctness
      - Legend for colour coding
    """
    # Sort prompts by expected answer for visual grouping
    prompt_order: dict[int, int] = {}  # original prompt_index -> display row
    unique_prompts: list[dict] = []
    seen: set[int] = set()
    for cell in result.cells:
        if cell.prompt_index not in seen:
            seen.add(cell.prompt_index)
            unique_prompts.append(
                {
                    "index": cell.prompt_index,
                    "prompt": cell.prompt,
                    "answer": cell.expected_answer,
                    "baseline_correct": cell.baseline_correct,
                }
            )
    unique_prompts.sort(key=lambda p: p["answer"].lower())
    for row, p in enumerate(unique_prompts):
        prompt_order[p["index"]] = row

    num_prompts = result.num_prompts
    num_layers = result.num_layers

    # Build cell lookup: (prompt_index, layer) -> HeatmapCell
    cell_map: dict[tuple[int, int], HeatmapCell] = {}
    for cell in result.cells:
        cell_map[(cell.prompt_index, cell.layer)] = cell

    # --- SVG dimensions ---
    cell_w = max(10, min(24, 900 // max(num_layers, 1)))
    cell_h = max(8, min(20, 600 // max(num_prompts, 1)))
    margin_left = 40
    margin_top = 40
    margin_right = 60  # for baseline status marks
    margin_bottom = 80  # for summary bar + labels
    grid_w = num_layers * cell_w
    grid_h = num_prompts * cell_h
    svg_w = margin_left + grid_w + margin_right
    svg_h = margin_top + grid_h + margin_bottom

    parts: list[str] = []
    parts.append(
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_BG2};border-radius:8px;">'
    )

    # --- Grid cells ---
    for cell in result.cells:
        row = prompt_order[cell.prompt_index]
        col = cell.layer
        x = margin_left + col * cell_w
        y = margin_top + row * cell_h
        color = _cell_color(cell.flip_direction)
        opacity = _cell_opacity(cell.delta_correct_prob)

        tooltip = (
            f"Prompt {cell.prompt_index}: {cell.prompt[:40]}...\n"
            f"Layer {cell.layer} | Answer: {cell.expected_answer}\n"
            f"Baseline: {cell.baseline_token} ({cell.baseline_prob:.3f})\n"
            f"Ablated: {cell.ablated_token} ({cell.ablated_prob:.3f})\n"
            f"Delta correct prob: {cell.delta_correct_prob:+.4f}\n"
            f"Direction: {cell.flip_direction}"
        )

        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_w - 1}" height="{cell_h - 1}" '
            f'fill="{color}" opacity="{opacity:.2f}" rx="1">'
            f"<title>{_esc(tooltip)}</title></rect>"
        )

    # --- X-axis labels (layer indices) ---
    label_step = max(1, num_layers // 20)
    for l_idx in range(0, num_layers, label_step):
        x = margin_left + l_idx * cell_w + cell_w / 2
        y_label = margin_top - 6
        parts.append(
            f'<text x="{x:.1f}" y="{y_label}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="9">{l_idx}</text>'
        )

    # X-axis title
    parts.append(
        f'<text x="{margin_left + grid_w / 2}" y="{margin_top - 22}" '
        f'text-anchor="middle" fill="{_DIM}" font-size="11">Layer</text>'
    )

    # --- Y-axis: prompt indices (sparse labels if many) ---
    y_label_step = max(1, num_prompts // 20)
    for row in range(0, num_prompts, y_label_step):
        y = margin_top + row * cell_h + cell_h / 2 + 3
        # Find original prompt index for this display row
        orig_idx = None
        for pidx, r in prompt_order.items():
            if r == row:
                orig_idx = pidx
                break
        label = str(orig_idx) if orig_idx is not None else str(row)
        parts.append(
            f'<text x="{margin_left - 4}" y="{y:.1f}" text-anchor="end" '
            f'fill="{_DIM}" font-size="8">{label}</text>'
        )

    # --- Right margin: baseline correctness marks ---
    for pinfo in unique_prompts:
        row = prompt_order[pinfo["index"]]
        y = margin_top + row * cell_h + cell_h / 2 + 3
        x_mark = margin_left + grid_w + 8
        if pinfo["baseline_correct"]:
            # Checkmark
            parts.append(
                f'<text x="{x_mark}" y="{y:.1f}" fill="{_GREEN}" '
                f'font-size="10" font-weight="bold">&#x2713;</text>'
            )
        else:
            # X mark
            parts.append(
                f'<text x="{x_mark}" y="{y:.1f}" fill="{_RED}" '
                f'font-size="10" font-weight="bold">&#x2717;</text>'
            )

    # Label for right margin
    parts.append(
        f'<text x="{margin_left + grid_w + 20}" y="{margin_top - 6}" '
        f'text-anchor="middle" fill="{_DIM}" font-size="8">Base</text>'
    )

    # --- Summary bar at bottom: stacked bars per layer ---
    bar_top = margin_top + grid_h + 10
    bar_height = 30

    for summary in result.layer_summaries:
        x = margin_left + summary.layer * cell_w
        total = summary.total_prompts or 1

        # Stack order: fixed (green), broke (red), changed (blue), unchanged (gray)
        segments = [
            (summary.fixes, _GREEN),
            (summary.breaks, _RED),
            (summary.changes, _BLUE),
            (summary.unchanged, _GRAY),
        ]

        y_offset = 0.0
        for count, color in segments:
            if count <= 0:
                continue
            seg_h = (count / total) * bar_height
            parts.append(
                f'<rect x="{x}" y="{bar_top + y_offset:.1f}" '
                f'width="{cell_w - 1}" height="{seg_h:.1f}" '
                f'fill="{color}" opacity="0.85" rx="0">'
                f"<title>L{summary.layer}: {count}/{total}</title></rect>"
            )
            y_offset += seg_h

    # Summary bar label
    parts.append(
        f'<text x="{margin_left + grid_w / 2}" '
        f'y="{bar_top + bar_height + 14}" '
        f'text-anchor="middle" fill="{_DIM}" font-size="10">'
        f"Per-layer summary</text>"
    )

    parts.append("</svg>")
    svg_content = "\n".join(parts)

    # --- Legend ---
    legend_html = (
        '<div class="legend">'
        f'<div class="legend-item">'
        f'<span class="legend-swatch" style="background:{_GREEN};"></span>'
        f"Fixed (wrong &rarr; correct)</div>"
        f'<div class="legend-item">'
        f'<span class="legend-swatch" style="background:{_RED};"></span>'
        f"Broke (correct &rarr; wrong)</div>"
        f'<div class="legend-item">'
        f'<span class="legend-swatch" style="background:{_BLUE};"></span>'
        f"Changed (different token, same correctness)</div>"
        f'<div class="legend-item">'
        f'<span class="legend-swatch" style="background:{_GRAY};"></span>'
        f"No change</div>"
        "</div>"
    )

    # --- Header section ---
    header_html = f"""
    <h1>MLP Ablation Heatmap</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(result.dataset_name)}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(result.model_name)}</span></div>
            <div><span class="meta-label">Timestamp</span><br>
                <span class="meta-value">{_esc(result.created_at)}</span></div>
            <div><span class="meta-label">Dimensions</span><br>
                <span class="meta-value">{result.num_prompts} prompts \
&times; {result.num_layers} layers</span></div>
            <div><span class="meta-label">Run ID</span><br>
                <span class="meta-value">{_esc(result.run_id[:12])}</span></div>
        </div>
    </div>
    """

    # --- Impact summary table ---
    # Find most impactful layers
    sorted_summaries = sorted(
        result.layer_summaries, key=lambda s: abs(s.impact_score), reverse=True
    )
    top_layers = sorted_summaries[:10]

    table_rows = []
    for s in top_layers:
        if s.impact_score > 0:
            impact_color = _GREEN
        elif s.impact_score < 0:
            impact_color = _RED
        else:
            impact_color = _DIM
        table_rows.append(
            f"<tr>"
            f"<td>{s.layer}</td>"
            f"<td>{s.fixes}</td>"
            f"<td>{s.breaks}</td>"
            f"<td>{s.changes}</td>"
            f"<td>{s.unchanged}</td>"
            f"<td>{s.avg_delta_correct_prob:+.4f}</td>"
            f'<td style="color:{impact_color};">{s.impact_score:+.3f}</td>'
            f"</tr>"
        )

    impact_table = (
        "<h2>Top Impact Layers</h2>"
        '<div style="overflow-x:auto;">'
        "<table>"
        "<tr><th>Layer</th><th>Fixes</th><th>Breaks</th>"
        "<th>Changes</th><th>Unchanged</th>"
        "<th>Avg Delta Prob</th><th>Impact Score</th></tr>"
        f"{''.join(table_rows)}"
        "</table></div>"
    )

    # --- Assemble page ---
    body = "\n".join(
        [
            header_html,
            "<h2>Heatmap</h2>",
            legend_html,
            f'<div class="chart-container">{svg_content}</div>',
            impact_table,
        ]
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>MLP Ablation Heatmap - {_esc(result.dataset_name)}</title>\n"
        f"<style>{_heatmap_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------


def heatmap_result_to_dict(result: HeatmapResult) -> dict:
    """Convert a :class:`HeatmapResult` to a JSON-serializable dict."""
    return {
        "run_id": result.run_id,
        "dataset_name": result.dataset_name,
        "model_name": result.model_name,
        "num_layers": result.num_layers,
        "num_prompts": result.num_prompts,
        "created_at": result.created_at,
        "cells": [
            {
                "prompt_index": c.prompt_index,
                "prompt": c.prompt,
                "expected_answer": c.expected_answer,
                "layer": c.layer,
                "baseline_token": c.baseline_token,
                "baseline_prob": c.baseline_prob,
                "baseline_correct": c.baseline_correct,
                "ablated_token": c.ablated_token,
                "ablated_prob": c.ablated_prob,
                "ablated_correct": c.ablated_correct,
                "delta_correct_prob": c.delta_correct_prob,
                "flipped": c.flipped,
                "flip_direction": c.flip_direction,
            }
            for c in result.cells
        ],
        "layer_summaries": [
            {
                "layer": s.layer,
                "total_prompts": s.total_prompts,
                "fixes": s.fixes,
                "breaks": s.breaks,
                "changes": s.changes,
                "unchanged": s.unchanged,
                "avg_delta_correct_prob": s.avg_delta_correct_prob,
                "impact_score": s.impact_score,
            }
            for s in result.layer_summaries
        ],
    }


def heatmap_result_from_dict(data: dict) -> HeatmapResult:
    """Reconstruct a :class:`HeatmapResult` from a dict.

    Inverse of :func:`heatmap_result_to_dict`.
    """
    cells = [
        HeatmapCell(
            prompt_index=c["prompt_index"],
            prompt=c["prompt"],
            expected_answer=c["expected_answer"],
            layer=c["layer"],
            baseline_token=c["baseline_token"],
            baseline_prob=c["baseline_prob"],
            baseline_correct=c["baseline_correct"],
            ablated_token=c["ablated_token"],
            ablated_prob=c["ablated_prob"],
            ablated_correct=c["ablated_correct"],
            delta_correct_prob=c["delta_correct_prob"],
            flipped=c["flipped"],
            flip_direction=c["flip_direction"],
        )
        for c in data["cells"]
    ]
    layer_summaries = [
        HeatmapLayerSummary(
            layer=s["layer"],
            total_prompts=s["total_prompts"],
            fixes=s["fixes"],
            breaks=s["breaks"],
            changes=s["changes"],
            unchanged=s["unchanged"],
            avg_delta_correct_prob=s["avg_delta_correct_prob"],
            impact_score=s["impact_score"],
        )
        for s in data["layer_summaries"]
    ]
    return HeatmapResult(
        run_id=data["run_id"],
        dataset_name=data["dataset_name"],
        model_name=data["model_name"],
        num_layers=data["num_layers"],
        num_prompts=data["num_prompts"],
        cells=cells,
        layer_summaries=layer_summaries,
        created_at=data["created_at"],
    )
