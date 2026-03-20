"""Diagnose — combined attention-trace + token-trace circuit diagnosis."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

from neurotrace.attention_trace import AttentionTraceEntry, AttentionTraceResult
from neurotrace.token_trace import TokenTraceEntry, TokenTraceResult

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CircuitAnalysis:
    active_heads: list[dict]  # [{layer, head, projection}, ...] sorted desc
    top_head: dict | None  # {layer, head, projection}
    top3_share: float
    circuit_type: str  # "concentrated" | "distributed" | "mixed"
    total_attention: float
    gatekeeper_present: bool


@dataclass
class SuppressionAnalysis:
    total_mlp: float
    subject_mlp: float
    worst_layer: int | None
    worst_layer_value: float
    suppression_ratio: float


@dataclass
class Verdict:
    signal_strength: str  # "strong" | "moderate" | "weak" | "absent"
    vulnerability: str  # "robust" | "moderate" | "vulnerable" | "absent"
    confidence: str  # "high" | "low"
    reason: str


@dataclass
class RepairPrescription:
    target_heads: list[dict]  # [{layer, head}, ...]
    target_layers: list[int]
    suggested_fix: str
    suggested_modules: list[str]
    alternative_fix: str
    alternative_layers: list[int]
    alternative_modules: list[str]


@dataclass
class DiagnosisResult:
    prompt: str
    answer: str
    circuit: CircuitAnalysis
    suppression: SuppressionAnalysis
    verdict: Verdict
    repair: RepairPrescription | None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

ACTIVE_HEAD_THRESHOLD = 0.1


def analyze_circuit(attn_result: AttentionTraceResult) -> CircuitAnalysis:
    """Compute circuit analysis from attention-trace entries."""
    active = [
        {"layer": e.layer, "head": e.head_idx, "projection": e.answer_projection}
        for e in attn_result.entries
        if e.answer_projection > ACTIVE_HEAD_THRESHOLD
    ]
    active.sort(key=lambda h: h["projection"], reverse=True)

    total_attention = sum(h["projection"] for h in active)
    top_head = active[0] if active else None

    top3_proj = sum(h["projection"] for h in active[:3])
    top3_share = top3_proj / total_attention if total_attention > 0 else 0.0

    if top3_share > 0.7:
        circuit_type = "concentrated"
    elif top3_share < 0.4:
        circuit_type = "distributed"
    else:
        circuit_type = "mixed"

    gatekeeper_present = any(
        h["layer"] == 21 and h["head"] == 29 for h in active
    )

    return CircuitAnalysis(
        active_heads=active,
        top_head=top_head,
        top3_share=top3_share,
        circuit_type=circuit_type,
        total_attention=total_attention,
        gatekeeper_present=gatekeeper_present,
    )


def analyze_suppression(
    token_result: TokenTraceResult,
    total_attention: float,
) -> SuppressionAnalysis:
    """Compute suppression analysis from token-trace entries."""
    # MLP projections at last position across all layers
    last_entries = [e for e in token_result.entries if e.is_last]
    total_mlp = sum(e.answer_projection for e in last_entries)

    # MLP projections at subject positions across all layers
    subject_entries = [e for e in token_result.entries if e.is_subject]
    subject_mlp = sum(e.answer_projection for e in subject_entries)

    # Worst layer = most negative MLP at last position
    worst_layer = None
    worst_value = 0.0
    for e in last_entries:
        if e.answer_projection < worst_value:
            worst_value = e.answer_projection
            worst_layer = e.layer

    suppression_ratio = (
        abs(total_mlp) / total_attention if total_attention > 0 else 0.0
    )

    return SuppressionAnalysis(
        total_mlp=total_mlp,
        subject_mlp=subject_mlp,
        worst_layer=worst_layer,
        worst_layer_value=worst_value,
        suppression_ratio=suppression_ratio,
    )


def compute_verdict(
    circuit: CircuitAnalysis,
    suppression: SuppressionAnalysis,
) -> Verdict:
    """Compute vulnerability verdict."""
    total = circuit.total_attention

    if total > 2.5:
        signal_strength = "strong"
    elif total >= 1.5:
        signal_strength = "moderate"
    elif total >= 1.0:
        signal_strength = "weak"
    else:
        signal_strength = "absent"

    if signal_strength == "strong":
        vulnerability = "robust"
    elif signal_strength == "weak":
        vulnerability = "vulnerable"
    elif signal_strength == "absent":
        vulnerability = "absent"
    else:
        vulnerability = "moderate"

    # Confidence: high if circuit_type matches vulnerability prediction
    confident = (
        (circuit.circuit_type == "concentrated" and vulnerability == "vulnerable")
        or (circuit.circuit_type == "distributed" and vulnerability == "robust")
    )
    confidence = "high" if confident else "low"

    reason = f"{signal_strength.capitalize()} signal ({total:.2f})"
    if circuit.circuit_type != "mixed":
        reason += f" + {circuit.circuit_type} circuit"

    return Verdict(
        signal_strength=signal_strength,
        vulnerability=vulnerability,
        confidence=confidence,
        reason=reason,
    )


def compute_repair(
    circuit: CircuitAnalysis,
    suppression: SuppressionAnalysis,
    verdict: Verdict,
) -> RepairPrescription | None:
    """Compute repair prescription (only if not robust)."""
    if verdict.vulnerability == "robust":
        return None

    target_heads = [
        {"layer": h["layer"], "head": h["head"]}
        for h in circuit.active_heads[:3]
    ]
    target_layers = sorted({h["layer"] for h in target_heads})

    alt_layers = (
        [suppression.worst_layer]
        if suppression.worst_layer is not None
        else []
    )

    return RepairPrescription(
        target_heads=target_heads,
        target_layers=target_layers,
        suggested_fix="attention-lora",
        suggested_modules=["q_proj", "k_proj", "v_proj"],
        alternative_fix="mlp-lora",
        alternative_layers=alt_layers,
        alternative_modules=["gate_proj", "up_proj", "down_proj"],
    )


def run_diagnosis(
    attn_result: AttentionTraceResult,
    token_result: TokenTraceResult,
) -> DiagnosisResult:
    """Run full diagnosis from attention-trace + token-trace results."""
    circuit = analyze_circuit(attn_result)
    suppression = analyze_suppression(token_result, circuit.total_attention)
    verdict = compute_verdict(circuit, suppression)
    repair = compute_repair(circuit, suppression, verdict)

    return DiagnosisResult(
        prompt=attn_result.prompt,
        answer=attn_result.answer,
        circuit=circuit,
        suppression=suppression,
        verdict=verdict,
        repair=repair,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def diagnosis_to_dict(d: DiagnosisResult) -> dict:
    """Convert a DiagnosisResult to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "prompt": d.prompt,
        "answer": d.answer,
        "circuit": {
            "active_heads": d.circuit.active_heads,
            "top3_share": d.circuit.top3_share,
            "circuit_type": d.circuit.circuit_type,
            "total_attention": d.circuit.total_attention,
        },
        "suppression": {
            "total_mlp": d.suppression.total_mlp,
            "subject_mlp": d.suppression.subject_mlp,
            "worst_layer": d.suppression.worst_layer,
            "suppression_ratio": d.suppression.suppression_ratio,
        },
        "verdict": {
            "signal_strength": d.verdict.signal_strength,
            "vulnerability": d.verdict.vulnerability,
            "confidence": d.verdict.confidence,
        },
    }
    if d.repair is not None:
        result["repair"] = {
            "target_heads": d.repair.target_heads,
            "target_layers": d.repair.target_layers,
            "suggested_fix": d.repair.suggested_fix,
            "suggested_modules": d.repair.suggested_modules,
            "alternative_fix": d.repair.alternative_fix,
            "alternative_layers": d.repair.alternative_layers,
            "alternative_modules": d.repair.alternative_modules,
        }
    return result


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
_YELLOW = "#ffca28"
_ORANGE = "#ff9800"
_WHITE = "#ffffff"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _verdict_color(vulnerability: str) -> str:
    return {
        "robust": _GREEN,
        "moderate": _YELLOW,
        "vulnerable": _ORANGE,
        "absent": _RED,
    }.get(vulnerability, _DIM)


def _value_to_color(value: float, max_abs: float) -> str:
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


def _diagnose_css() -> str:
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
    .panel-grid {{
        display: grid; gap: 2rem;
        grid-template-columns: 1fr 1fr 1fr;
    }}
    @media (max-width: 1200px) {{
        .panel-grid {{ grid-template-columns: 1fr; }}
    }}
    .panel {{
        background: var(--bg2); border-radius: 8px; padding: 1.5rem;
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
    .verdict-card {{
        background: var(--bg2); border-radius: 8px; padding: 1.5rem;
        margin: 1rem 0;
    }}
    .verdict-badge {{
        display: inline-block; padding: 0.3rem 0.8rem;
        border-radius: 4px; font-weight: bold; font-size: 0.9rem;
    }}
    .finding {{
        background: var(--bg2); padding: 1rem 1.5rem;
        border-left: 3px solid {_ACCENT}; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.95rem;
    }}
    .bar {{
        height: 20px; border-radius: 3px; display: inline-block;
        vertical-align: middle;
    }}
    details {{ margin: 0.5rem 0; }}
    summary {{
        cursor: pointer; color: {_BLUE}; padding: 0.5rem 0.75rem;
        background: var(--bg2); border-radius: 4px; font-size: 0.9rem;
    }}
    summary:hover {{ background: var(--bg3); }}
    .batch-row {{ cursor: pointer; }}
    .batch-row:hover td {{ background: var(--bg3); }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .panel, .chart-container, .verdict-card {{
            background: #f5f5f5; color: #222;
        }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _render_circuit_heatmap_svg(
    entries: list[AttentionTraceEntry],
    layers: list[int],
    num_heads: int,
) -> str:
    """Render layers x heads heatmap for circuit analysis."""
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
    max_abs = max(abs(v) for v in all_vals)

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
        f'fill="{_TEXT}" font-size="12" font-weight="bold">'
        f'Per-Head Answer Projection</text>'
    )

    for h_idx in range(num_heads):
        x = margin_left + h_idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x}" y="{margin_top - 6}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="8">H{h_idx}</text>'
        )

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


def _render_suppression_bars_svg(
    token_entries: list[TokenTraceEntry],
    layers: list[int],
) -> str:
    """Render per-layer MLP bar chart at last position."""
    last_by_layer: dict[int, float] = {}
    for e in token_entries:
        if e.is_last and e.layer in set(layers):
            last_by_layer[e.layer] = e.answer_projection

    if not last_by_layer:
        return ""

    bar_h = 22
    bar_gap = 4
    margin_left = 50
    margin_top = 30
    chart_w = 400
    n = len(layers)
    svg_h = margin_top + n * (bar_h + bar_gap) + 20
    svg_w = margin_left + chart_w + 80

    vals = [last_by_layer.get(ly, 0.0) for ly in layers]
    max_abs = max(abs(v) for v in vals) if vals else 1.0

    parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_BG2};border-radius:8px;">'
    ]
    parts.append(
        f'<text x="{svg_w / 2}" y="16" text-anchor="middle" '
        f'fill="{_TEXT}" font-size="12" font-weight="bold">'
        f'Per-Layer MLP at Last Position</text>'
    )

    center_x = margin_left + chart_w / 2
    # center line
    parts.append(
        f'<line x1="{center_x}" y1="{margin_top}" '
        f'x2="{center_x}" y2="{svg_h - 10}" '
        f'stroke="{_DIM}" stroke-width="1" stroke-dasharray="3,3"/>'
    )

    for i, layer in enumerate(layers):
        y = margin_top + i * (bar_h + bar_gap)
        val = last_by_layer.get(layer, 0.0)
        bar_width = abs(val) / max_abs * (chart_w / 2) if max_abs > 0 else 0

        parts.append(
            f'<text x="{margin_left - 6}" y="{y + bar_h / 2 + 4}" '
            f'text-anchor="end" fill="{_DIM}" font-size="9">L{layer}</text>'
        )

        color = _GREEN if val > 0 else _RED
        if val >= 0:
            bx = center_x
        else:
            bx = center_x - bar_width

        parts.append(
            f'<rect x="{bx}" y="{y}" width="{bar_width}" '
            f'height="{bar_h}" fill="{color}" rx="2" opacity="0.8">'
            f'<title>L{layer}: {val:+.4f}</title></rect>'
        )
        parts.append(
            f'<text x="{margin_left + chart_w + 5}" y="{y + bar_h / 2 + 4}" '
            f'fill="{_DIM}" font-size="9">{val:+.2f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_diagnose_html_single(
    diag: DiagnosisResult,
    attn_result: AttentionTraceResult,
    token_result: TokenTraceResult,
    layers: list[int],
    model_name: str,
) -> str:
    """Generate HTML report for a single-prompt diagnosis."""
    num_heads = max((e.head_idx for e in attn_result.entries), default=0) + 1
    vc = _verdict_color(diag.verdict.vulnerability)

    header = f"""
    <h1>Diagnosis</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Prompt</span><br>
                <span class="meta-value">{_esc(diag.prompt)}</span></div>
            <div><span class="meta-label">Answer</span><br>
                <span class="meta-value">{_esc(diag.answer)}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(model_name)}</span></div>
            <div><span class="meta-label">Verdict</span><br>
                <span class="verdict-badge"
                      style="background:{vc};color:#000;">
                    {_esc(diag.verdict.vulnerability.upper())}
                </span></div>
        </div>
    </div>
    """

    # Circuit heatmap
    circuit_svg = _render_circuit_heatmap_svg(
        attn_result.entries, layers, num_heads,
    )

    # Suppression bars
    suppression_svg = _render_suppression_bars_svg(
        token_result.entries, layers,
    )

    # Verdict card
    repair_html = ""
    if diag.repair:
        heads_str = ", ".join(
            f"L{h['layer']}.H{h['head']}" for h in diag.repair.target_heads
        )
        repair_html = f"""
        <h3>Repair Prescription</h3>
        <div class="finding">
            Target: {_esc(', '.join(f'L{ly}' for ly in diag.repair.target_layers))}
            attention ({_esc(', '.join(diag.repair.suggested_modules))})<br>
            Heads to strengthen: {_esc(heads_str)}<br>
            Alternative: L{diag.suppression.worst_layer or '?'} MLP
            ({_esc(', '.join(diag.repair.alternative_modules))})
        </div>
        """

    verdict_card = f"""
    <div class="verdict-card">
        <h3>Verdict</h3>
        <table>
            <tr><td>Vulnerability</td>
                <td><span class="verdict-badge"
                    style="background:{vc};color:#000;">
                    {_esc(diag.verdict.vulnerability.upper())}</span></td></tr>
            <tr><td>Reason</td><td>{_esc(diag.verdict.reason)}</td></tr>
            <tr><td>Confidence</td><td>{_esc(diag.verdict.confidence.capitalize())}</td></tr>
            <tr><td>Signal Strength</td>
                <td>{_esc(diag.verdict.signal_strength.capitalize())}
                    ({diag.circuit.total_attention:+.2f})</td></tr>
        </table>
        {repair_html}
    </div>
    """

    # Circuit table
    circuit_table = """
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
    """
    active_str = f"{len(diag.circuit.active_heads)}"
    if diag.circuit.active_heads:
        top_entries = ", ".join(
            f"L{h['layer']}.H{h['head']} {h['projection']:+.2f}"
            for h in diag.circuit.active_heads[:3]
        )
        if len(diag.circuit.active_heads) > 3:
            active_str += f" ({top_entries}, ...)"
        else:
            active_str += f" ({top_entries})"
    circuit_table += f"""
        <tr><td>Active heads</td><td>{_esc(active_str)}</td></tr>
        <tr><td>Circuit type</td>
            <td>{_esc(diag.circuit.circuit_type.capitalize())}
                (top3: {diag.circuit.top3_share:.0%})</td></tr>
        <tr><td>Total attention</td>
            <td>{diag.circuit.total_attention:+.2f}</td></tr>
        <tr><td>Signal strength</td>
            <td>{_esc(diag.verdict.signal_strength.capitalize())}</td></tr>
    </table>
    """

    # Suppression table
    suppression_table = f"""
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total MLP</td><td>{diag.suppression.total_mlp:+.2f}</td></tr>
        <tr><td>Subject MLP</td><td>{diag.suppression.subject_mlp:+.2f}</td></tr>
        <tr><td>Worst layer</td>
            <td>L{diag.suppression.worst_layer}
                ({diag.suppression.worst_layer_value:+.2f})</td></tr>
        <tr><td>Suppression ratio</td>
            <td>{diag.suppression.suppression_ratio:.1f}x</td></tr>
    </table>
    """

    body = "\n".join([
        header,
        '<div class="panel-grid">',
        f'<div class="panel"><h3>Circuit</h3>{circuit_table}'
        f'<div class="chart-container">{circuit_svg}</div></div>',
        f'<div class="panel"><h3>Suppression</h3>{suppression_table}'
        f'<div class="chart-container">{suppression_svg}</div></div>',
        f'<div class="panel">{verdict_card}</div>',
        '</div>',
    ])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Diagnosis: {_esc(diag.prompt[:60])}</title>\n"
        f"<style>{_diagnose_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )


def generate_diagnose_html_batch(
    diagnoses: list[DiagnosisResult],
    dataset_name: str | None,
    model_name: str,
) -> str:
    """Generate HTML report for a batch diagnosis run."""
    counts = {"robust": 0, "moderate": 0, "vulnerable": 0, "absent": 0}
    for d in diagnoses:
        counts[d.verdict.vulnerability] = counts.get(d.verdict.vulnerability, 0) + 1

    # Most common repair target layers
    layer_counts: dict[int, int] = {}
    for d in diagnoses:
        if d.repair:
            for ly in d.repair.target_layers:
                layer_counts[ly] = layer_counts.get(ly, 0) + 1
    top_layers = sorted(
        layer_counts.items(), key=lambda x: x[1], reverse=True,
    )[:5]

    header = f"""
    <h1>Diagnosis Summary</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(dataset_name or 'custom')}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{len(diagnoses)}</span></div>
        </div>
    </div>
    """

    # Verdict summary table
    summary = """
    <h2>Verdict Summary</h2>
    <table>
        <tr><th>Verdict</th><th>Count</th></tr>
    """
    for v in ["robust", "moderate", "vulnerable", "absent"]:
        vc = _verdict_color(v)
        summary += (
            f'<tr><td><span class="verdict-badge" '
            f'style="background:{vc};color:#000;">'
            f'{_esc(v.capitalize())}</span></td>'
            f'<td>{counts.get(v, 0)}</td></tr>'
        )
    summary += "</table>"

    # Top repair targets
    targets_html = ""
    if top_layers:
        targets_html = "<h3>Most common repair targets</h3><div class=\"finding\">"
        for layer, count in top_layers:
            targets_html += f"L{layer} attention: {count} prompts<br>"
        targets_html += "</div>"

    # Per-prompt table
    rows = ""
    for d in diagnoses:
        vc = _verdict_color(d.verdict.vulnerability)
        rows += (
            f"<tr>"
            f"<td>{_esc(d.prompt[:50])}</td>"
            f"<td>{_esc(d.answer)}</td>"
            f"<td>{d.circuit.total_attention:+.2f}</td>"
            f"<td>{d.suppression.total_mlp:+.2f}</td>"
            f"<td>{_esc(d.circuit.circuit_type)}</td>"
            f'<td><span class="verdict-badge" '
            f'style="background:{vc};color:#000;">'
            f'{_esc(d.verdict.vulnerability.capitalize())}</span></td>'
            f"</tr>"
        )

    detail_table = f"""
    <h2>Per-Prompt Results</h2>
    <table>
        <tr>
            <th>Prompt</th><th>Answer</th><th>Total Attn</th>
            <th>Total MLP</th><th>Circuit</th><th>Verdict</th>
        </tr>
        {rows}
    </table>
    """

    # Per-prompt expandable details
    details_html = "<h2>Details</h2>"
    for d in diagnoses:
        vc = _verdict_color(d.verdict.vulnerability)
        repair_text = ""
        if d.repair:
            heads_str = ", ".join(
                f"L{h['layer']}.H{h['head']}" for h in d.repair.target_heads
            )
            repair_text = (
                f"<br>Repair: {d.repair.suggested_fix} → "
                f"{', '.join(f'L{ly}' for ly in d.repair.target_layers)} "
                f"({heads_str})"
            )
        details_html += (
            f'<details><summary>'
            f'<span class="verdict-badge" '
            f'style="background:{vc};color:#000;font-size:0.75rem;">'
            f'{_esc(d.verdict.vulnerability.upper())}</span> '
            f'{_esc(d.prompt[:60])} → {_esc(d.answer)}'
            f'</summary>'
            f'<div class="finding">'
            f'Signal: {d.circuit.total_attention:+.2f} '
            f'({d.verdict.signal_strength}) | '
            f'MLP: {d.suppression.total_mlp:+.2f} | '
            f'Circuit: {d.circuit.circuit_type} '
            f'(top3: {d.circuit.top3_share:.0%}) | '
            f'Suppression: {d.suppression.suppression_ratio:.1f}x'
            f'{repair_text}'
            f'</div></details>'
        )

    body = "\n".join([
        header, summary, targets_html, detail_table, details_html,
    ])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Diagnosis: {_esc(dataset_name or 'batch')}</title>\n"
        f"<style>{_diagnose_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
