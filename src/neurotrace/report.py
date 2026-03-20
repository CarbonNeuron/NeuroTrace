"""Self-contained HTML report generation for NeuroTrace traces."""

from __future__ import annotations

import html as _html
import json
from typing import Any

import numpy as np

from neurotrace.analyzer import DiffResult
from neurotrace.types import TraceResult

# --- Theme: Refined Ocean ---
_BG = "#0a0e14"
_BG2 = "#0f1923"
_BG3 = "#1a2332"
_TEXT = "#c8ccd4"
_DIM = "#6e7a8a"
_ACCENT = "#e8956a"
_BLUE = "#4db8ff"
_GREEN = "#7ec699"
_RED = "#ff6666"
_PURPLE = "#c792ea"

CHART_COLORS = [_ACCENT, _BLUE, _GREEN, _PURPLE, _RED]


# --- Color Utilities ---


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lerp_color(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    t = max(0.0, min(1.0, t))
    return (
        f"#{int(r1 + (r2 - r1) * t):02x}"
        f"{int(g1 + (g2 - g1) * t):02x}"
        f"{int(b1 + (b2 - b1) * t):02x}"
    )


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _sample_layers(num_layers: int) -> list[int]:
    """Pick ~6 representative layers including first and last."""
    if num_layers <= 6:
        return list(range(num_layers))
    step = (num_layers - 1) / 5
    indices = {0, num_layers - 1}
    for i in range(1, 5):
        indices.add(round(i * step))
    return sorted(indices)


# --- CSS ---


def _css() -> str:
    return f"""
    :root {{
        --bg: {_BG}; --bg2: {_BG2}; --bg3: {_BG3};
        --text: {_TEXT}; --dim: {_DIM};
        --accent: {_ACCENT}; --blue: {_BLUE};
        --green: {_GREEN}; --red: {_RED};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont,
            'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 14px; line-height: 1.6; padding: 2rem;
        max-width: 1400px; margin: 0 auto;
    }}
    h1 {{ color: var(--accent); font-size: 1.8rem; margin-bottom: 0.5rem; }}
    h2 {{
        color: var(--blue); font-size: 1.3rem; margin: 2rem 0 1rem;
        border-bottom: 1px solid var(--bg3); padding-bottom: 0.5rem;
    }}
    h3 {{ color: var(--text); font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }}
    h4 {{ color: var(--dim); font-size: 0.95rem; margin: 0.5rem 0; }}
    .meta {{
        background: var(--bg2); padding: 1.5rem;
        border-radius: 8px; margin-bottom: 2rem;
    }}
    .meta-grid {{
        display: grid; gap: 0.5rem 2rem;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    }}
    .meta-label {{ color: var(--dim); font-size: 0.85rem; }}
    .meta-value {{
        color: var(--text);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
    .prediction {{
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 1.1rem; color: var(--accent); font-weight: 600;
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
    .pred-table td {{ text-align: center; vertical-align: top; }}
    .pred-table th {{ text-align: center; }}
    .chart-container {{
        background: var(--bg2); border-radius: 8px; padding: 1rem; margin: 1rem 0;
        overflow-x: auto;
    }}
    svg {{ max-width: 100%; height: auto; }}
    details {{ margin: 0.5rem 0; }}
    summary {{
        cursor: pointer; color: var(--blue); padding: 0.5rem 0.75rem;
        background: var(--bg2); border-radius: 4px; font-size: 0.9rem;
    }}
    summary:hover {{ background: var(--bg3); }}
    .flagged {{ background: rgba(255, 102, 102, 0.1) !important; }}
    .side-by-side {{
        display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;
    }}
    .full-width {{ grid-column: 1 / -1; }}
    pre {{
        background: var(--bg2); padding: 1rem; border-radius: 8px;
        overflow-x: auto; font-size: 0.8rem; color: var(--dim);
        max-height: 600px; overflow-y: auto;
    }}
    .heads-grid {{
        display: grid; gap: 1rem;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    }}
    @media (max-width: 900px) {{ .side-by-side {{ grid-template-columns: 1fr; }} }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .chart-container, summary, pre {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
        .meta-label {{ color: #666; }} .meta-value {{ color: #222; }}
        details {{ break-inside: avoid; }}
    }}
    """


# --- SVG Generators ---


def _svg_line_chart(
    series: list[dict],
    width: int = 800,
    height: int = 300,
    y_min: float | None = None,
    y_max: float | None = None,
    x_title: str = "",
    y_title: str = "",
) -> str:
    """Render a line chart as inline SVG.

    series: [{"label": str, "data": list[float], "color": str, "dash": bool}]
    """
    ml, mr, mt, mb = 60, 120, 25, 35
    cw = width - ml - mr
    ch = height - mt - mb

    all_vals = [v for s in series for v in s["data"] if v is not None]
    if not all_vals:
        return '<svg viewBox="0 0 100 50" xmlns="http://www.w3.org/2000/svg"></svg>'

    if y_min is None:
        y_min = min(all_vals)
    if y_max is None:
        y_max = max(all_vals)
    y_range = (y_max or 0.0) - (y_min or 0.0)
    pad = y_range * 0.05 or 0.5
    y_min = (y_min or 0.0) - pad
    y_max = (y_max or 0.0) + pad

    n_pts = max(len(s["data"]) for s in series)

    def tx(i: int) -> float:
        return ml + (i / max(n_pts - 1, 1)) * cw

    def ty(v: float) -> float:
        return mt + ch - ((v - y_min) / (y_max - y_min)) * ch

    p = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']

    # Grid lines
    for gi in range(6):
        y = mt + (gi / 5) * ch
        val = y_max - (gi / 5) * (y_max - y_min)
        p.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}" '
            f'stroke="{_BG3}" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="{_DIM}" font-size="10">{val:.3g}</text>'
        )

    # X-axis labels
    step = max(1, n_pts // 10)
    for i in range(0, n_pts, step):
        x = tx(i)
        p.append(
            f'<text x="{x:.1f}" y="{mt + ch + 15}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="10">{i}</text>'
        )

    # Axis titles
    if x_title:
        p.append(
            f'<text x="{ml + cw / 2}" y="{height - 2}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="11">{_esc(x_title)}</text>'
        )
    if y_title:
        p.append(
            f'<text x="12" y="{mt + ch / 2}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="11" '
            f'transform="rotate(-90, 12, {mt + ch / 2})">{_esc(y_title)}</text>'
        )

    # Data lines
    for s in series:
        pts = []
        for i, v in enumerate(s["data"]):
            if v is not None:
                pts.append(f"{tx(i):.1f},{ty(v):.1f}")
        if pts:
            dash = ' stroke-dasharray="6,3"' if s.get("dash") else ""
            p.append(
                f'<polyline points="{" ".join(pts)}" fill="none" '
                f'stroke="{s["color"]}" stroke-width="2"{dash}/>'
            )

    # Legend
    lx = ml + cw + 10
    for i, s in enumerate(series):
        ly = mt + 12 + i * 18
        dash = ' stroke-dasharray="4,2"' if s.get("dash") else ""
        p.append(
            f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" '
            f'stroke="{s["color"]}" stroke-width="2"{dash}/>'
        )
        p.append(
            f'<text x="{lx + 23}" y="{ly + 4}" fill="{_TEXT}" '
            f'font-size="10">{_esc(s["label"])}</text>'
        )

    p.append("</svg>")
    return "\n".join(p)


def _svg_attention_heatmap(
    weights: np.ndarray,
    tokens: list[str],
    cell_size: int = 20,
) -> str:
    """Render a single attention head heatmap (seq x seq) as SVG."""
    rows, cols = weights.shape
    w = cols * cell_size + 60
    h = rows * cell_size + 50
    ox, oy = 55, 40  # offset for labels

    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    # Column labels (top)
    for c in range(cols):
        label = tokens[c][:4] if c < len(tokens) else str(c)
        p.append(
            f'<text x="{ox + c * cell_size + cell_size / 2}" y="{oy - 6}" '
            f'text-anchor="middle" fill="{_DIM}" font-size="8" '
            f'transform="rotate(-45, {ox + c * cell_size + cell_size / 2}, {oy - 6})">'
            f"{_esc(label)}</text>"
        )

    # Row labels (left) + cells
    for r in range(rows):
        label = tokens[r][:5] if r < len(tokens) else str(r)
        p.append(
            f'<text x="{ox - 4}" y="{oy + r * cell_size + cell_size / 2 + 3}" '
            f'text-anchor="end" fill="{_DIM}" font-size="8">{_esc(label)}</text>'
        )
        for c in range(cols):
            val = float(weights[r, c])
            color = _lerp_color(_BG2, _ACCENT, val)
            p.append(
                f'<rect x="{ox + c * cell_size}" y="{oy + r * cell_size}" '
                f'width="{cell_size - 1}" height="{cell_size - 1}" '
                f'fill="{color}" rx="1">'
                f"<title>({r},{c}): {val:.3f}</title></rect>"
            )

    p.append("</svg>")
    return "\n".join(p)


# --- HTML Sections ---


def _section_header(meta: Any, top_pred: str = "", top_prob: float = 0.0) -> str:
    pred_html = ""
    if top_pred:
        pred_html = (
            f'<div style="margin-top:1rem;">'
            f'<span class="meta-label">Top Prediction</span><br>'
            f'<span class="prediction">{_esc(repr(top_pred))} '
            f"({top_prob:.1%})</span></div>"
        )
    return f"""
    <h1>NeuroTrace Report</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(meta.model_name)}</span></div>
            <div><span class="meta-label">Prompt</span><br>
                <span class="meta-value">{_esc(meta.prompt)}</span></div>
            <div><span class="meta-label">Trace ID</span><br>
                <span class="meta-value">{_esc(meta.trace_id[:12])}</span></div>
            <div><span class="meta-label">Timestamp</span><br>
                <span class="meta-value">{_esc(meta.timestamp)}</span></div>
            <div><span class="meta-label">Layers/Heads</span><br>
                <span class="meta-value">\
{meta.num_layers} / {meta.num_heads}</span></div>
            <div><span class="meta-label">Tokens</span><br>
                <span class="meta-value">{len(meta.token_ids)}</span></div>
        </div>
        {pred_html}
    </div>"""


def _section_prediction_table(layer_preds: list[dict] | None) -> str:
    if not layer_preds:
        return (
            '<p style="color: var(--dim);">'
            "No prediction data (trace may lack residual streams).</p>"
        )

    top_k = len(layer_preds[0].get("top_k_strings", []))
    headers = "<th>Layer</th>" + "".join(f"<th>#{i + 1}</th>" for i in range(top_k))

    rows = []
    for lp in layer_preds:
        cells = [
            f'<td style="text-align:right;color:var(--dim);background:var(--bg);">'
            f"{lp['layer_index']}</td>"
        ]
        for i in range(min(top_k, len(lp["top_k_strings"]))):
            prob = lp["top_k_probs"][i]
            token = lp["top_k_strings"][i]
            ann = ""
            if i < len(lp.get("annotations", [])):
                ann = lp["annotations"][i]
            bg = _lerp_color(_BG2, _ACCENT, min(prob * 2, 1.0))
            text_c = _TEXT if prob < 0.4 else _BG
            ann_html = ""
            if ann and ann != "--":
                ann_html = (
                    f'<br><span style="font-size:0.65rem;color:{_DIM};">'
                    f"{_esc(ann)}</span>"
                )
            cells.append(
                f'<td style="background:{bg};color:{text_c};">'
                f"{_esc(repr(token))}"
                f'<br><span style="font-size:0.75rem;">{prob:.3f}</span>'
                f"{ann_html}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f'<div style="overflow-x:auto;">'
        f'<table class="pred-table"><tr>{headers}</tr>'
        f"{''.join(rows)}</table></div>"
    )


def _section_token_tracking(
    token_tracks: list[dict] | None,
) -> str:
    if not token_tracks:
        return '<p style="color: var(--dim);">No token tracking data available.</p>'

    series = []
    for i, track in enumerate(token_tracks):
        series.append(
            {
                "label": repr(track["token"]),
                "data": track["probs"],
                "color": CHART_COLORS[i % len(CHART_COLORS)],
            }
        )

    chart = _svg_line_chart(
        series,
        y_min=0.0,
        x_title="Layer",
        y_title="Probability",
    )
    return f'<div class="chart-container">{chart}</div>'


def _section_residual_norms(layer_stats: list[dict]) -> str:
    if not layer_stats:
        return ""
    series = [
        {
            "label": "Residual In",
            "data": [s["residual_in_norm"] for s in layer_stats],
            "color": _BLUE,
        },
        {
            "label": "Residual Out",
            "data": [s["residual_out_norm"] for s in layer_stats],
            "color": _ACCENT,
        },
    ]
    chart = _svg_line_chart(series, x_title="Layer", y_title="L2 Norm")
    return f'<div class="chart-container">{chart}</div>'


def _section_entropy(layer_stats: list[dict]) -> str:
    if not layer_stats:
        return ""
    avg_entropies = []
    for s in layer_stats:
        ent = s.get("attention_entropy") or []
        avg_entropies.append(sum(ent) / len(ent) if ent else 0.0)

    series = [{"label": "Avg Entropy", "data": avg_entropies, "color": _GREEN}]
    chart = _svg_line_chart(series, x_title="Layer", y_title="Entropy")
    return f'<div class="chart-container">{chart}</div>'


def _section_attention(trace: TraceResult, sampled_layers: list[int]) -> str:
    tokens = trace.metadata.tokens
    parts = []

    for snap in trace.layer_snapshots:
        if snap.layer_index not in sampled_layers:
            continue
        if snap.attention_weights is None:
            continue

        num_heads = snap.attention_weights.shape[0]
        seq_len = snap.attention_weights.shape[1]
        cell = max(10, min(22, 220 // max(seq_len, 1)))

        heads_html = []
        for h in range(num_heads):
            w = snap.attention_weights[h]
            svg = _svg_attention_heatmap(w, tokens, cell_size=cell)
            heads_html.append(f"<div><h4>Head {h}</h4>{svg}</div>")

        grid = f'<div class="heads-grid">{"".join(heads_html)}</div>'
        parts.append(
            f"<details><summary>Layer {snap.layer_index} "
            f"({num_heads} heads, {seq_len} tokens)</summary>"
            f'<div style="padding:1rem;">{grid}</div></details>'
        )

    if not parts:
        return '<p style="color: var(--dim);">No attention data available.</p>'
    return "\n".join(parts)


def _section_raw_data(trace: TraceResult, layer_stats: list[dict]) -> str:
    meta = trace.metadata
    raw = {
        "trace_id": meta.trace_id,
        "model_name": meta.model_name,
        "prompt": meta.prompt,
        "tokens": meta.tokens,
        "num_layers": meta.num_layers,
        "num_heads": meta.num_heads,
        "hidden_size": meta.hidden_size,
        "timestamp": meta.timestamp,
        "layer_stats": [
            {
                "layer": s["layer_index"],
                "res_in_norm": round(s["residual_in_norm"], 4),
                "res_out_norm": round(s["residual_out_norm"], 4),
                "mlp_mag": round(s["mlp_activation_mag"], 4),
                "top1_token": s["top1_token"],
                "top1_prob": round(s["top1_prob"], 4),
            }
            for s in layer_stats
        ],
    }
    json_str = json.dumps(raw, indent=2, default=str)
    return (
        f"<details><summary>Raw JSON Data</summary>"
        f"<pre>{_esc(json_str)}</pre></details>"
    )


# --- Comparison-specific sections ---


def _section_diff_table(diff: DiffResult, token_lookup: dict[int, str]) -> str:
    headers = (
        "<th>Layer</th><th>Cosine Sim</th><th>Top-1 Changed</th>"
        "<th>KL Div</th><th>Trace A Top-1</th><th>Trace B Top-1</th>"
    )
    rows = []
    for m in diff.layer_metrics:
        cos_color = _lerp_color(_RED, _GREEN, m.cosine_similarity)
        flag_class = ' class="flagged"' if m.flagged else ""
        a_str = token_lookup.get(m.trace_a_top1, "?")
        b_str = token_lookup.get(m.trace_b_top1, "?")
        changed = (
            '<span style="color:var(--red);">YES</span>' if m.top1_changed else "no"
        )
        rows.append(
            f"<tr{flag_class}>"
            f'<td style="text-align:right;">{m.layer_index}</td>'
            f'<td style="color:{cos_color};">{m.cosine_similarity:.4f}</td>'
            f"<td>{changed}</td>"
            f"<td>{m.kl_divergence:.4f}</td>"
            f"<td>{_esc(repr(a_str))} ({m.trace_a_top1_prob:.2f})</td>"
            f"<td>{_esc(repr(b_str))} ({m.trace_b_top1_prob:.2f})</td>"
            f"</tr>"
        )

    flagged_count = sum(1 for m in diff.layer_metrics if m.flagged)
    summary = (
        f'<p style="margin-top:0.5rem;color:var(--dim);">'
        f"Flagged: {flagged_count}/{len(diff.layer_metrics)} layers"
    )
    if diff.first_divergence_layer is not None:
        summary += f" | First divergence: layer {diff.first_divergence_layer}"
    summary += "</p>"

    return (
        f'<div style="overflow-x:auto;">'
        f"<table><tr>{headers}</tr>{''.join(rows)}</table></div>"
        f"{summary}"
    )


def _section_overlay_chart(
    tracks_a: list[dict] | None,
    tracks_b: list[dict] | None,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    if not tracks_a and not tracks_b:
        return '<p style="color: var(--dim);">No tracking data.</p>'

    series = []
    for i, track in enumerate(tracks_a or []):
        series.append(
            {
                "label": f"{label_a}: {repr(track['token'])}",
                "data": track["probs"],
                "color": CHART_COLORS[i % len(CHART_COLORS)],
            }
        )
    for i, track in enumerate(tracks_b or []):
        series.append(
            {
                "label": f"{label_b}: {repr(track['token'])}",
                "data": track["probs"],
                "color": CHART_COLORS[i % len(CHART_COLORS)],
                "dash": True,
            }
        )

    chart = _svg_line_chart(series, y_min=0.0, x_title="Layer", y_title="Probability")
    return f'<div class="chart-container">{chart}</div>'


def _section_divergence_heatmap(diff: DiffResult) -> str:
    """Layers x metrics divergence grid."""
    if not diff.layer_metrics:
        return ""

    metrics_names = ["Cosine Sim", "KL Div", "Top-1 Changed"]
    n_layers = len(diff.layer_metrics)

    # Build matrix: rows=layers, cols=metrics, values normalized 0-1
    max_kl = max((m.kl_divergence for m in diff.layer_metrics), default=1.0)
    max_kl = max(max_kl, 0.01)

    cell_w, cell_h = 80, 22
    w = 80 + len(metrics_names) * cell_w + 10
    h = 40 + n_layers * cell_h + 10

    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']

    # Column headers
    for ci, name in enumerate(metrics_names):
        p.append(
            f'<text x="{80 + ci * cell_w + cell_w / 2}" y="25" '
            f'text-anchor="middle" fill="{_DIM}" font-size="10">'
            f"{_esc(name)}</text>"
        )

    # Rows
    for ri, m in enumerate(diff.layer_metrics):
        y = 35 + ri * cell_h
        p.append(
            f'<text x="70" y="{y + cell_h / 2 + 3}" text-anchor="end" '
            f'fill="{_DIM}" font-size="9">L{m.layer_index}</text>'
        )
        # Cosine sim: 1.0 = similar (cool), 0 = divergent (hot)
        cos_val = 1.0 - m.cosine_similarity
        cos_color = _lerp_color(_BG2, _RED, cos_val)
        p.append(
            f'<rect x="80" y="{y}" width="{cell_w - 2}" '
            f'height="{cell_h - 2}" fill="{cos_color}" rx="2">'
            f"<title>Cosine: {m.cosine_similarity:.4f}</title></rect>"
        )
        p.append(
            f'<text x="{80 + cell_w / 2}" y="{y + cell_h / 2 + 3}" '
            f'text-anchor="middle" fill="{_TEXT}" font-size="9">'
            f"{m.cosine_similarity:.3f}</text>"
        )

        # KL div
        kl_val = min(m.kl_divergence / max_kl, 1.0)
        kl_color = _lerp_color(_BG2, _RED, kl_val)
        kx = 80 + cell_w
        p.append(
            f'<rect x="{kx}" y="{y}" width="{cell_w - 2}" '
            f'height="{cell_h - 2}" fill="{kl_color}" rx="2">'
            f"<title>KL: {m.kl_divergence:.4f}</title></rect>"
        )
        p.append(
            f'<text x="{kx + cell_w / 2}" y="{y + cell_h / 2 + 3}" '
            f'text-anchor="middle" fill="{_TEXT}" font-size="9">'
            f"{m.kl_divergence:.3f}</text>"
        )

        # Top-1 changed
        t1_color = _RED if m.top1_changed else _BG2
        tx_pos = 80 + 2 * cell_w
        p.append(
            f'<rect x="{tx_pos}" y="{y}" width="{cell_w - 2}" '
            f'height="{cell_h - 2}" fill="{t1_color}" rx="2"/>'
        )
        t1_label = "YES" if m.top1_changed else "no"
        p.append(
            f'<text x="{tx_pos + cell_w / 2}" y="{y + cell_h / 2 + 3}" '
            f'text-anchor="middle" fill="{_TEXT}" font-size="9">'
            f"{t1_label}</text>"
        )

    p.append("</svg>")
    svg = "\n".join(p)
    return f'<div class="chart-container">{svg}</div>'


# --- Page Assembly ---


def _html_page(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html>\n"
        f'<html lang="en">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def generate_report(
    trace: TraceResult,
    layer_stats: list[dict],
    layer_predictions: list[dict] | None = None,
    token_tracks: list[dict] | None = None,
    full_attention: bool = False,
    no_attention: bool = False,
) -> str:
    """Generate a single-trace HTML report."""
    meta = trace.metadata
    top_pred = ""
    top_prob = 0.0
    if trace.token_predictions:
        last = trace.token_predictions[-1]
        if last.top_k_strings:
            top_pred = last.top_k_strings[0]
            top_prob = last.top_k_probs[0]

    sampled = (
        list(range(meta.num_layers))
        if full_attention
        else _sample_layers(meta.num_layers)
    )

    sections = [
        _section_header(meta, top_pred, top_prob),
        "<h2>Token Prediction Evolution</h2>",
        _section_prediction_table(layer_predictions),
        "<h2>Token Tracking</h2>",
        _section_token_tracking(token_tracks),
        "<h2>Residual Stream Norms</h2>",
        _section_residual_norms(layer_stats),
        "<h2>Attention Entropy</h2>",
        _section_entropy(layer_stats),
    ]

    if not no_attention:
        sections.append("<h2>Attention Heatmaps</h2>")
        sections.append(_section_attention(trace, sampled))

    sections.append("<h2>Raw Data</h2>")
    sections.append(_section_raw_data(trace, layer_stats))

    return _html_page(f"NeuroTrace: {meta.prompt[:60]}", "\n".join(sections))


def generate_comparison_report(
    trace_a: TraceResult,
    trace_b: TraceResult,
    stats_a: list[dict],
    stats_b: list[dict],
    preds_a: list[dict] | None,
    preds_b: list[dict] | None,
    tracks_a: list[dict] | None,
    tracks_b: list[dict] | None,
    diff_result: DiffResult,
    token_lookup: dict[int, str],
    full_attention: bool = False,
    no_attention: bool = False,
) -> str:
    """Generate a two-trace comparison HTML report."""
    meta_a = trace_a.metadata
    meta_b = trace_b.metadata

    top_pred_a, top_prob_a = "", 0.0
    if trace_a.token_predictions:
        last = trace_a.token_predictions[-1]
        if last.top_k_strings:
            top_pred_a = last.top_k_strings[0]
            top_prob_a = last.top_k_probs[0]

    top_pred_b, top_prob_b = "", 0.0
    if trace_b.token_predictions:
        last = trace_b.token_predictions[-1]
        if last.top_k_strings:
            top_pred_b = last.top_k_strings[0]
            top_prob_b = last.top_k_probs[0]

    sampled_a = (
        list(range(meta_a.num_layers))
        if full_attention
        else _sample_layers(meta_a.num_layers)
    )
    sampled_b = (
        list(range(meta_b.num_layers))
        if full_attention
        else _sample_layers(meta_b.num_layers)
    )

    label_a = meta_a.label or meta_a.trace_id[:8]
    label_b = meta_b.label or meta_b.trace_id[:8]

    sections = [
        "<h1>NeuroTrace Comparison Report</h1>",
        # Side-by-side headers
        '<div class="side-by-side">',
        f"<div>{_section_header(meta_a, top_pred_a, top_prob_a)}</div>",
        f"<div>{_section_header(meta_b, top_pred_b, top_prob_b)}</div>",
        "</div>",
        # Diff table (full width)
        "<h2>Divergence Summary</h2>",
        _section_diff_table(diff_result, token_lookup),
        # Overlay chart
        "<h2>Token Tracking Overlay</h2>",
        _section_overlay_chart(tracks_a, tracks_b, label_a, label_b),
        # Divergence heatmap
        "<h2>Divergence Heatmap</h2>",
        _section_divergence_heatmap(diff_result),
        # Side-by-side prediction tables
        "<h2>Token Prediction Evolution</h2>",
        '<div class="side-by-side">',
        f"<div><h3>{_esc(label_a)}</h3>{_section_prediction_table(preds_a)}</div>",
        f"<div><h3>{_esc(label_b)}</h3>{_section_prediction_table(preds_b)}</div>",
        "</div>",
        # Side-by-side norms and entropy
        "<h2>Residual Stream Norms</h2>",
        '<div class="side-by-side">',
        f"<div><h3>{_esc(label_a)}</h3>{_section_residual_norms(stats_a)}</div>",
        f"<div><h3>{_esc(label_b)}</h3>{_section_residual_norms(stats_b)}</div>",
        "</div>",
        "<h2>Attention Entropy</h2>",
        '<div class="side-by-side">',
        f"<div><h3>{_esc(label_a)}</h3>{_section_entropy(stats_a)}</div>",
        f"<div><h3>{_esc(label_b)}</h3>{_section_entropy(stats_b)}</div>",
        "</div>",
    ]

    if not no_attention:
        sections.extend(
            [
                "<h2>Attention Heatmaps</h2>",
                '<div class="side-by-side">',
                f"<div><h3>{_esc(label_a)}</h3>"
                f"{_section_attention(trace_a, sampled_a)}</div>",
                f"<div><h3>{_esc(label_b)}</h3>"
                f"{_section_attention(trace_b, sampled_b)}</div>",
                "</div>",
            ]
        )

    # Raw data
    sections.extend(
        [
            "<h2>Raw Data</h2>",
            '<div class="side-by-side">',
            f"<div><h3>{_esc(label_a)}</h3>{_section_raw_data(trace_a, stats_a)}</div>",
            f"<div><h3>{_esc(label_b)}</h3>{_section_raw_data(trace_b, stats_b)}</div>",
            "</div>",
        ]
    )

    title = f"NeuroTrace: {meta_a.prompt[:30]} vs {meta_b.prompt[:30]}"
    return _html_page(title, "\n".join(sections))
