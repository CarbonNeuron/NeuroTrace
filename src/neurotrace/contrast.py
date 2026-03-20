"""Domain geometry contrast — MLP delta analysis across domains."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

import numpy as np  # noqa: I001

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContrastCell:
    """Per-prompt, per-layer MLP delta metrics."""

    domain: str
    prompt: str
    answer: str
    layer: int
    mlp_delta_norm: float
    answer_projection: float
    competitor_projection: float
    competitor_token: str


@dataclass
class ContrastDomainSummary:
    """Aggregate MLP delta stats for one domain at one layer."""

    domain: str
    layer: int
    mean_delta_norm: float
    std_delta_norm: float
    mean_answer_proj: float
    mean_competitor_proj: float
    cosine_similarities: dict[str, float]  # domain_name -> mean cosine sim


@dataclass
class ContrastResult:
    """Complete result of a contrast analysis run."""

    run_id: str
    domains: list[str]
    layers: list[int]
    model_name: str
    cells: list[ContrastCell]
    summaries: list[ContrastDomainSummary]
    vulnerable_vs_robust: dict | None  # Analysis 5 results
    created_at: str


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_mlp_delta(mlp_input: np.ndarray, mlp_output: np.ndarray) -> np.ndarray:
    """Compute MLP delta vector (what the MLP adds)."""
    return mlp_output - mlp_input


def compute_delta_norm(delta: np.ndarray) -> float:
    """L2 norm of an MLP delta vector."""
    return float(np.linalg.norm(delta))


def compute_answer_projection(
    delta: np.ndarray,
    answer_direction: np.ndarray,
) -> float:
    """Project MLP delta onto answer direction (normalized dot product).

    Positive = pushing toward answer, negative = pushing away.
    """
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-10:
        return 0.0
    normalized_delta = delta / delta_norm
    return float(np.dot(normalized_delta, answer_direction))


def compute_domain_centroid(deltas: list[np.ndarray]) -> np.ndarray:
    """Compute the mean MLP delta vector for a domain at a layer."""
    if not deltas:
        return np.zeros(1)
    return np.mean(np.stack(deltas), axis=0)


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_answer_direction(lm_head_weight: np.ndarray, token_id: int) -> np.ndarray:
    """Extract and normalize the lm_head row for a token ID."""
    direction = lm_head_weight[token_id].astype(np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return direction
    return direction / norm


def build_domain_summaries(
    cells: list[ContrastCell],
    domains: list[str],
    layers: list[int],
    domain_centroids: dict[tuple[str, int], np.ndarray],
) -> list[ContrastDomainSummary]:
    """Build per-domain, per-layer summary statistics."""
    # Group cells by (domain, layer)
    by_key: dict[tuple[str, int], list[ContrastCell]] = {}
    for cell in cells:
        key = (cell.domain, cell.layer)
        by_key.setdefault(key, []).append(cell)

    summaries: list[ContrastDomainSummary] = []
    for domain in domains:
        for layer in layers:
            key = (domain, layer)
            group = by_key.get(key, [])
            if not group:
                summaries.append(ContrastDomainSummary(
                    domain=domain,
                    layer=layer,
                    mean_delta_norm=0.0,
                    std_delta_norm=0.0,
                    mean_answer_proj=0.0,
                    mean_competitor_proj=0.0,
                    cosine_similarities={},
                ))
                continue

            norms = [c.mlp_delta_norm for c in group]
            answer_projs = [c.answer_projection for c in group]
            competitor_projs = [c.competitor_projection for c in group]

            # Pairwise cosine similarities with other domains
            cosines: dict[str, float] = {}
            centroid_a = domain_centroids.get((domain, layer))
            for other_domain in domains:
                centroid_b = domain_centroids.get((other_domain, layer))
                if centroid_a is not None and centroid_b is not None:
                    cosines[other_domain] = compute_cosine_similarity(
                        centroid_a, centroid_b
                    )
                else:
                    cosines[other_domain] = 0.0

            summaries.append(ContrastDomainSummary(
                domain=domain,
                layer=layer,
                mean_delta_norm=float(np.mean(norms)),
                std_delta_norm=float(np.std(norms)),
                mean_answer_proj=float(np.mean(answer_projs)),
                mean_competitor_proj=float(np.mean(competitor_projs)),
                cosine_similarities=cosines,
            ))

    return summaries


def analyze_vulnerable_vs_robust(
    cells: list[ContrastCell],
    heatmap_cells_json: str,
    domain: str,
    layers: list[int],
    domain_deltas: dict[tuple[str, int, str], np.ndarray],
) -> dict:
    """Compare MLP deltas for vulnerable vs robust prompts within a domain.

    Args:
        cells: All contrast cells.
        heatmap_cells_json: JSON string of heatmap cells for this domain.
        domain: Which domain to analyze.
        layers: Which layers to compare.
        domain_deltas: Map of (domain, layer, prompt) -> delta vector.

    Returns:
        Dict with per-layer vulnerable vs robust comparison.
    """
    import json

    if isinstance(heatmap_cells_json, str):
        heatmap_cells = json.loads(heatmap_cells_json)
    else:
        heatmap_cells = heatmap_cells_json

    # Find which prompts have breaks
    vulnerable_prompts: set[str] = set()
    all_prompts: set[str] = set()
    for cell in heatmap_cells:
        all_prompts.add(cell["prompt"])
        if cell["flip_direction"] == "broke":
            vulnerable_prompts.add(cell["prompt"])

    robust_prompts = all_prompts - vulnerable_prompts

    if not vulnerable_prompts or not robust_prompts:
        return {"status": "insufficient_data", "layers": {}}

    result: dict[str, Any] = {"status": "ok", "layers": {}}
    for layer in layers:
        vuln_deltas = []
        robust_deltas = []
        for prompt in vulnerable_prompts:
            key = (domain, layer, prompt)
            if key in domain_deltas:
                vuln_deltas.append(domain_deltas[key])
        for prompt in robust_prompts:
            key = (domain, layer, prompt)
            if key in domain_deltas:
                robust_deltas.append(domain_deltas[key])

        if not vuln_deltas or not robust_deltas:
            continue

        vuln_centroid = compute_domain_centroid(vuln_deltas)
        robust_centroid = compute_domain_centroid(robust_deltas)
        cos_sim = compute_cosine_similarity(vuln_centroid, robust_centroid)

        vuln_norms = [float(np.linalg.norm(d)) for d in vuln_deltas]
        robust_norms = [float(np.linalg.norm(d)) for d in robust_deltas]

        result["layers"][str(layer)] = {
            "cosine_similarity": cos_sim,
            "vulnerable_mean_norm": float(np.mean(vuln_norms)),
            "robust_mean_norm": float(np.mean(robust_norms)),
            "n_vulnerable": len(vuln_deltas),
            "n_robust": len(robust_deltas),
        }

    return result


# ---------------------------------------------------------------------------
# Local runner (requires torch)
# ---------------------------------------------------------------------------


def run_contrast_local(
    model,
    tokenizer,
    arch,
    domain_prompts: dict[str, list[dict]],
    layers: list[int],
    commitment_data: dict[str, dict] | None = None,
    seed: int = 42,
    progress_callback=None,
) -> tuple[list[ContrastCell], dict[tuple[str, int, str], np.ndarray]]:
    """Run MLP delta extraction locally.

    Args:
        model: Loaded model.
        tokenizer: Tokenizer.
        arch: ModelArchitecture.
        domain_prompts: {domain_name: [{"prompt": ..., "answer": ...}]}.
        layers: Which layers to analyze.
        commitment_data: Optional commitment data per prompt.
        seed: Random seed.
        progress_callback: (current, total) callback.

    Returns:
        (cells, domain_deltas) tuple.
    """
    import torch

    from neurotrace.models import get_lm_head_and_norm

    torch.manual_seed(seed)

    model_layers = arch.get_layers(model)
    lm_head, final_ln = get_lm_head_and_norm(model)
    lm_head_weight = lm_head.weight.data.cpu().float().numpy()

    cells: list[ContrastCell] = []
    domain_deltas: dict[tuple[str, int, str], np.ndarray] = {}

    total_prompts = sum(len(ps) for ps in domain_prompts.values())
    current = 0

    for domain, prompts in domain_prompts.items():
        for entry in prompts:
            if progress_callback:
                progress_callback(current, total_prompts)
            current += 1

            prompt_text = entry["prompt"]
            answer = entry["answer"]

            # Get answer token id
            answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
            if not answer_ids:
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_token_id = answer_ids[0] if answer_ids else None

            # Get competitor token id from commitment data
            competitor_token_id = None
            competitor_token_str = ""
            if commitment_data and prompt_text in commitment_data:
                competitor_token_str = commitment_data[prompt_text].get(
                    "competitor_token", ""
                )
                if competitor_token_str:
                    comp_ids = tokenizer.encode(
                        " " + competitor_token_str, add_special_tokens=False
                    )
                    if not comp_ids:
                        comp_ids = tokenizer.encode(
                            competitor_token_str, add_special_tokens=False
                        )
                    competitor_token_id = comp_ids[0] if comp_ids else None

            # Hook MLP modules at specified layers
            captured_input: dict[int, torch.Tensor] = {}
            captured_output: dict[int, torch.Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHandle] = []

            def _make_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    inp = input[0] if isinstance(input, tuple) else input
                    captured_input[layer_idx] = inp[:, -1, :].detach().cpu().float()
                    out = output if not isinstance(output, tuple) else output[0]
                    captured_output[layer_idx] = out[:, -1, :].detach().cpu().float()
                return hook_fn

            try:
                for layer_idx in layers:
                    if layer_idx < len(model_layers):
                        mlp = arch.get_mlp(model_layers[layer_idx])
                        hooks.append(mlp.register_forward_hook(_make_hook(layer_idx)))

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

            # Process each layer
            for layer_idx in layers:
                if layer_idx not in captured_input or layer_idx not in captured_output:
                    continue

                mlp_in = captured_input[layer_idx][0].numpy()
                mlp_out = captured_output[layer_idx][0].numpy()
                delta = compute_mlp_delta(mlp_in, mlp_out)
                delta_norm = compute_delta_norm(delta)

                # Answer projection
                answer_proj = 0.0
                if answer_token_id is not None:
                    answer_dir = get_answer_direction(lm_head_weight, answer_token_id)
                    answer_proj = compute_answer_projection(delta, answer_dir)

                # Competitor projection
                competitor_proj = 0.0
                if competitor_token_id is not None:
                    comp_dir = get_answer_direction(lm_head_weight, competitor_token_id)
                    competitor_proj = compute_answer_projection(delta, comp_dir)

                cells.append(ContrastCell(
                    domain=domain,
                    prompt=prompt_text,
                    answer=answer,
                    layer=layer_idx,
                    mlp_delta_norm=delta_norm,
                    answer_projection=answer_proj,
                    competitor_projection=competitor_proj,
                    competitor_token=competitor_token_str,
                ))
                domain_deltas[(domain, layer_idx, prompt_text)] = delta

    return cells, domain_deltas


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


def run_contrast_remote(
    mlp_data_by_prompt: list[dict],
    lm_head_weight: np.ndarray,
    tokenizer,
    domain_prompts: dict[str, list[dict]],
    layers: list[int],
    commitment_data: dict[str, dict] | None = None,
) -> tuple[list[ContrastCell], dict[tuple[str, int, str], np.ndarray]]:
    """Compute contrast from remotely-fetched MLP activations.

    Args:
        mlp_data_by_prompt: List of dicts with 'mlp_input' and 'mlp_output'
            numpy arrays keyed by layer.
        lm_head_weight: lm_head weight matrix as numpy.
        tokenizer: Tokenizer for encoding tokens.
        domain_prompts: {domain: [{"prompt": ..., "answer": ...}]}.
        layers: Which layers to analyze.
        commitment_data: Optional {prompt: {"competitor_token": str}}.

    Returns:
        (cells, domain_deltas).
    """
    cells: list[ContrastCell] = []
    domain_deltas: dict[tuple[str, int, str], np.ndarray] = {}

    prompt_idx = 0
    for domain, prompts in domain_prompts.items():
        for entry in prompts:
            prompt_text = entry["prompt"]
            answer = entry["answer"]

            answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
            if not answer_ids:
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_token_id = answer_ids[0] if answer_ids else None

            competitor_token_id = None
            competitor_token_str = ""
            if commitment_data and prompt_text in commitment_data:
                competitor_token_str = commitment_data[prompt_text].get(
                    "competitor_token", ""
                )
                if competitor_token_str:
                    comp_ids = tokenizer.encode(
                        " " + competitor_token_str, add_special_tokens=False
                    )
                    if not comp_ids:
                        comp_ids = tokenizer.encode(
                            competitor_token_str, add_special_tokens=False
                        )
                    competitor_token_id = comp_ids[0] if comp_ids else None

            mlp_data = mlp_data_by_prompt[prompt_idx]
            prompt_idx += 1

            for layer_idx in layers:
                layer_key = str(layer_idx)
                if layer_key not in mlp_data:
                    continue

                mlp_in = mlp_data[layer_key]["mlp_input"]
                mlp_out = mlp_data[layer_key]["mlp_output"]
                delta = compute_mlp_delta(mlp_in, mlp_out)
                delta_norm = compute_delta_norm(delta)

                answer_proj = 0.0
                if answer_token_id is not None:
                    answer_dir = get_answer_direction(lm_head_weight, answer_token_id)
                    answer_proj = compute_answer_projection(delta, answer_dir)

                competitor_proj = 0.0
                if competitor_token_id is not None:
                    comp_dir = get_answer_direction(lm_head_weight, competitor_token_id)
                    competitor_proj = compute_answer_projection(delta, comp_dir)

                cells.append(ContrastCell(
                    domain=domain,
                    prompt=prompt_text,
                    answer=answer,
                    layer=layer_idx,
                    mlp_delta_norm=delta_norm,
                    answer_projection=answer_proj,
                    competitor_projection=competitor_proj,
                    competitor_token=competitor_token_str,
                ))
                domain_deltas[(domain, layer_idx, prompt_text)] = delta

    return cells, domain_deltas


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def contrast_result_to_dict(result: ContrastResult) -> dict:
    """Convert a ContrastResult to a JSON-serializable dict."""
    return {
        "run_id": result.run_id,
        "domains": result.domains,
        "layers": result.layers,
        "model_name": result.model_name,
        "created_at": result.created_at,
        "cells": [
            {
                "domain": c.domain,
                "prompt": c.prompt,
                "answer": c.answer,
                "layer": c.layer,
                "mlp_delta_norm": c.mlp_delta_norm,
                "answer_projection": c.answer_projection,
                "competitor_projection": c.competitor_projection,
                "competitor_token": c.competitor_token,
            }
            for c in result.cells
        ],
        "summaries": [
            {
                "domain": s.domain,
                "layer": s.layer,
                "mean_delta_norm": s.mean_delta_norm,
                "std_delta_norm": s.std_delta_norm,
                "mean_answer_proj": s.mean_answer_proj,
                "mean_competitor_proj": s.mean_competitor_proj,
                "cosine_similarities": s.cosine_similarities,
            }
            for s in result.summaries
        ],
        "vulnerable_vs_robust": result.vulnerable_vs_robust,
    }


def contrast_result_from_dict(data: dict) -> ContrastResult:
    """Reconstruct a ContrastResult from a dict."""
    cells = [
        ContrastCell(
            domain=c["domain"],
            prompt=c["prompt"],
            answer=c["answer"],
            layer=c["layer"],
            mlp_delta_norm=c["mlp_delta_norm"],
            answer_projection=c["answer_projection"],
            competitor_projection=c["competitor_projection"],
            competitor_token=c["competitor_token"],
        )
        for c in data["cells"]
    ]
    summaries = [
        ContrastDomainSummary(
            domain=s["domain"],
            layer=s["layer"],
            mean_delta_norm=s["mean_delta_norm"],
            std_delta_norm=s["std_delta_norm"],
            mean_answer_proj=s["mean_answer_proj"],
            mean_competitor_proj=s["mean_competitor_proj"],
            cosine_similarities=s["cosine_similarities"],
        )
        for s in data["summaries"]
    ]
    return ContrastResult(
        run_id=data["run_id"],
        domains=data["domains"],
        layers=data["layers"],
        model_name=data["model_name"],
        cells=cells,
        summaries=summaries,
        vulnerable_vs_robust=data.get("vulnerable_vs_robust"),
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


def _contrast_css() -> str:
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
    .summary-block {{ margin: 1.5rem 0; }}
    .finding {{
        background: var(--bg2); padding: 1rem 1.5rem;
        border-left: 3px solid {_ACCENT}; border-radius: 0 8px 8px 0;
        margin: 0.5rem 0; font-size: 0.95rem;
    }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .chart-container {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _norm_color(value: float, max_value: float) -> str:
    """Color intensity based on relative magnitude."""
    if max_value <= 0:
        return _DIM
    ratio = min(1.0, value / max_value)
    if ratio > 0.7:
        return _RED
    if ratio > 0.4:
        return _YELLOW
    return _DIM


def _proj_color(value: float) -> str:
    """Color for projections: green positive, red negative."""
    if value > 0.1:
        return _GREEN
    if value < -0.1:
        return _RED
    return _DIM


def _heatmap_cell_svg(
    x: float, y: float, w: float, h: float,
    value: float, max_val: float,
    tooltip: str,
) -> str:
    """Generate an SVG rect for a heatmap cell."""
    if max_val <= 0:
        opacity = 0.3
    else:
        opacity = max(0.2, min(1.0, value / max_val))
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{_ACCENT}" opacity="{opacity:.2f}" rx="1">'
        f'<title>{_esc(tooltip)}</title></rect>'
    )


def generate_contrast_html(result: ContrastResult) -> str:
    """Generate a self-contained HTML contrast report."""
    domains = result.domains
    layers = result.layers

    # --- Header ---
    header = f"""
    <h1>Domain Geometry Contrast</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Domains</span><br>
                <span class="meta-value">{_esc(', '.join(domains))}</span></div>
            <div><span class="meta-label">Layers</span><br>
                <span class="meta-value">{_esc(
                    ', '.join(str(x) for x in layers)
                )}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(result.model_name)}</span></div>
            <div><span class="meta-label">Run ID</span><br>
                <span class="meta-value">{_esc(result.run_id[:12])}</span></div>
        </div>
    </div>
    """

    # --- Per-layer tables ---
    layer_tables = []
    for layer in layers:
        layer_summaries = [s for s in result.summaries if s.layer == layer]
        if not layer_summaries:
            continue

        max_norm = max((s.mean_delta_norm for s in layer_summaries), default=1.0)

        rows = []
        for s in layer_summaries:
            norm_c = _norm_color(s.mean_delta_norm, max_norm)
            ans_c = _proj_color(s.mean_answer_proj)
            comp_c = _proj_color(s.mean_competitor_proj)

            cos_cells = []
            for d in domains:
                cos_val = s.cosine_similarities.get(d, 0.0)
                cos_cells.append(f"<td>{cos_val:.3f}</td>")

            rows.append(
                f"<tr>"
                f"<td>{_esc(s.domain)}</td>"
                f'<td style="color:{norm_c}">{s.mean_delta_norm:.2f}</td>'
                f"<td>{s.std_delta_norm:.2f}</td>"
                f'<td style="color:{ans_c}">{s.mean_answer_proj:+.3f}</td>'
                f'<td style="color:{comp_c}">{s.mean_competitor_proj:+.3f}</td>'
                f"{''.join(cos_cells)}"
                f"</tr>"
            )

        cos_headers = "".join(
            f"<th>cos({_esc(d[:6])})</th>" for d in domains
        )

        layer_tables.append(
            f"<h2>Layer {layer}</h2>"
            f'<div style="overflow-x:auto;"><table>'
            f"<tr><th>Domain</th><th>Mean|&Delta;|</th><th>Std|&Delta;|</th>"
            f"<th>&rarr;Answer</th><th>&rarr;Competitor</th>"
            f"{cos_headers}</tr>"
            f"{''.join(rows)}"
            f"</table></div>"
        )

    # --- Heatmap SVG: domains × layers colored by delta magnitude ---
    summary_by_key = {
        (s.domain, s.layer): s for s in result.summaries
    }
    max_norm_all = max(
        (s.mean_delta_norm for s in result.summaries), default=1.0
    )

    cell_w = max(30, min(60, 600 // max(len(layers), 1)))
    cell_h = 30
    margin_left = 100
    margin_top = 40
    grid_w = len(layers) * cell_w
    grid_h = len(domains) * cell_h
    svg_w = margin_left + grid_w + 20
    svg_h = margin_top + grid_h + 40

    svg_parts = [
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{_BG2};border-radius:8px;">'
    ]

    for d_idx, domain in enumerate(domains):
        y = margin_top + d_idx * cell_h
        svg_parts.append(
            f'<text x="{margin_left - 6}" y="{y + cell_h / 2 + 4}" '
            f'text-anchor="end" fill="{_DIM}" font-size="11">'
            f'{_esc(domain[:12])}</text>'
        )
        for l_idx, layer in enumerate(layers):
            x = margin_left + l_idx * cell_w
            s = summary_by_key.get((domain, layer))
            val = s.mean_delta_norm if s else 0.0
            tooltip = (
                f"{domain} @ L{layer}\n"
                f"|Δ|={val:.2f}"
            )
            svg_parts.append(_heatmap_cell_svg(
                x, y, cell_w - 2, cell_h - 2,
                val, max_norm_all, tooltip,
            ))

    # X-axis labels
    for l_idx, layer in enumerate(layers):
        x = margin_left + l_idx * cell_w + cell_w / 2
        svg_parts.append(
            f'<text x="{x}" y="{margin_top - 8}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="10">L{layer}</text>'
        )

    svg_parts.append("</svg>")

    heatmap_section = (
        "<h2>MLP Delta Magnitude Heatmap</h2>"
        f'<div class="chart-container">{"".join(svg_parts)}</div>'
    )

    # --- Vulnerable vs Robust section ---
    vuln_section = ""
    vr = result.vulnerable_vs_robust
    if vr and vr.get("status") == "ok":
        vuln_rows = []
        for layer_str, info in vr.get("layers", {}).items():
            cos_sim = info["cosine_similarity"]
            vuln_rows.append(
                f"<tr>"
                f"<td>L{_esc(layer_str)}</td>"
                f"<td>{cos_sim:.3f}</td>"
                f"<td>{info['vulnerable_mean_norm']:.2f}</td>"
                f"<td>{info['robust_mean_norm']:.2f}</td>"
                f"<td>{info['n_vulnerable']}</td>"
                f"<td>{info['n_robust']}</td>"
                f"</tr>"
            )
        if vuln_rows:
            vuln_section = (
                "<h2>Vulnerable vs Robust (within domain)</h2>"
                '<div style="overflow-x:auto;"><table>'
                "<tr><th>Layer</th><th>Cosine Sim</th>"
                "<th>Vuln Mean|&Delta;|</th><th>Robust Mean|&Delta;|</th>"
                "<th>N Vuln</th><th>N Robust</th></tr>"
                f"{''.join(vuln_rows)}"
                "</table></div>"
            )

    body = "\n".join([header, heatmap_section] + layer_tables + [vuln_section])

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Domain Contrast - {_esc(', '.join(domains))}</title>\n"
        f"<style>{_contrast_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
