"""Logit Prism decomposition — exact additive logit attribution."""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerContribution:
    """Per-layer attention + MLP contribution to a token's logit."""

    layer: int
    attention: float
    mlp: float


@dataclass
class TokenDecomposition:
    """Full decomposition of a single token's final logit."""

    token_str: str
    token_id: int
    final_logit: float
    embedding: float
    layers: list[LayerContribution]
    reconstruction_error: float
    norm_scale: float


@dataclass
class DecomposeResult:
    """Per-prompt decomposition: answer vs competitor."""

    prompt: str
    answer: str
    competitor: str
    answer_logit: float
    competitor_logit: float
    margin: float
    embedding_margin: float
    component_margins: list[dict]  # [{layer, attn_margin, mlp_margin}]
    reconstruction_error: float
    answer_decomposition: TokenDecomposition
    competitor_decomposition: TokenDecomposition


@dataclass
class DecomposeRun:
    """Complete decompose run."""

    run_id: str
    dataset: str | None
    model_name: str
    num_prompts: int
    results: list[DecomposeResult]
    created_at: str


# ---------------------------------------------------------------------------
# Core decomposition (local)
# ---------------------------------------------------------------------------


def decompose_logits(
    model,
    tokenizer,
    prompt: str,
    target_tokens: list[str],
    seed: int = 42,
) -> dict[str, TokenDecomposition]:
    """Run the Logit Prism decomposition for target tokens.

    Hooks self_attn and mlp on each layer to capture pre-skip-connection
    outputs. Then projects each component through w_unembed * diag(w_norm) / s
    to get exact additive contributions.
    """
    import torch

    torch.manual_seed(seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Resolve target token IDs
    target_ids = []
    for t in target_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(" " + t, add_special_tokens=False)
        target_ids.append(ids[0] if ids else 0)

    # Hook attention and MLP outputs
    components: dict[int, dict[str, Any]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_attn_hook(layer_idx: int):
        def hook(module, input, output):
            components[layer_idx]["attn"] = output[0][0, -1, :].detach()
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(module, input, output):
            out = output[0, -1, :] if output.dim() == 3 else output[-1, :]
            components[layer_idx]["mlp"] = out.detach()
        return hook

    layers = model.model.layers
    for i, layer in enumerate(layers):
        components[i] = {}
        hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

    try:
        with torch.no_grad():
            outputs = model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    # Embedding at last position
    embed = model.model.embed_tokens(input_ids)[0, -1, :].detach()

    # Final layer norm weight
    ln_weight = model.model.norm.weight.detach()

    # Reconstruct residual stream
    residual = embed.clone()
    for i in range(len(layers)):
        residual = residual + components[i]["attn"] + components[i]["mlp"]

    # RMS norm scale factor
    variance = residual.pow(2).mean(-1, keepdim=False)
    s = torch.sqrt(variance + model.model.norm.variance_epsilon)

    # Unembedding matrix
    w_unembed = model.lm_head.weight

    results: dict[str, TokenDecomposition] = {}
    for token_str, token_id in zip(target_tokens, target_ids):
        unembed_vec = w_unembed[token_id]
        proj = (unembed_vec * ln_weight) / s

        embed_contrib = (proj * embed).sum().item()
        layer_contribs = []
        total = embed_contrib

        for i in range(len(layers)):
            attn_c = (proj * components[i]["attn"]).sum().item()
            mlp_c = (proj * components[i]["mlp"]).sum().item()
            layer_contribs.append(LayerContribution(
                layer=i, attention=attn_c, mlp=mlp_c,
            ))
            total += attn_c + mlp_c

        actual_logit = outputs.logits[0, -1, token_id].item()

        results[token_str] = TokenDecomposition(
            token_str=token_str,
            token_id=token_id,
            final_logit=actual_logit,
            embedding=embed_contrib,
            layers=layer_contribs,
            reconstruction_error=abs(total - actual_logit),
            norm_scale=s.item(),
        )

    return results


def get_top_competitors(
    model, tokenizer, prompt: str, answer: str, n: int = 3, seed: int = 42,
) -> list[str]:
    """Run a forward pass and return top-N predicted tokens excluding answer."""
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    _top_vals, top_ids = torch.topk(probs, n + 5)

    answer_ids_sp = tokenizer.encode(" " + answer, add_special_tokens=False)
    answer_ids_raw = tokenizer.encode(answer, add_special_tokens=False)
    exclude = set(answer_ids_sp + answer_ids_raw)

    competitors = []
    for tid in top_ids.tolist():
        if tid not in exclude:
            tok = tokenizer.decode([tid]).strip()
            if tok:
                competitors.append(tok)
            if len(competitors) >= n:
                break
    return competitors


def build_decompose_result(
    prompt: str,
    answer: str,
    competitor: str,
    decompositions: dict[str, TokenDecomposition],
) -> DecomposeResult:
    """Build a DecomposeResult comparing answer vs competitor."""
    a = decompositions[answer]
    c = decompositions[competitor]

    embedding_margin = a.embedding - c.embedding
    component_margins = []
    for al, cl in zip(a.layers, c.layers):
        component_margins.append({
            "layer": al.layer,
            "attn_margin": al.attention - cl.attention,
            "mlp_margin": al.mlp - cl.mlp,
        })

    return DecomposeResult(
        prompt=prompt,
        answer=answer,
        competitor=competitor,
        answer_logit=a.final_logit,
        competitor_logit=c.final_logit,
        margin=a.final_logit - c.final_logit,
        embedding_margin=embedding_margin,
        component_margins=component_margins,
        reconstruction_error=max(a.reconstruction_error, c.reconstruction_error),
        answer_decomposition=a,
        competitor_decomposition=c,
    )


def run_decompose_local(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    competitors: list[str] | None = None,
    seed: int = 42,
) -> list[DecomposeResult]:
    """Run decomposition locally for a single prompt.

    If competitors is None, auto-detect top-3 competitors.
    Returns one DecomposeResult per competitor.
    """
    if competitors is None or not competitors:
        competitors = get_top_competitors(
            model, tokenizer, prompt, answer, n=3, seed=seed,
        )

    all_tokens = [answer] + competitors
    decompositions = decompose_logits(model, tokenizer, prompt, all_tokens, seed)

    results = []
    for comp in competitors:
        if comp in decompositions and answer in decompositions:
            results.append(build_decompose_result(
                prompt, answer, comp, decompositions,
            ))
    return results


# ---------------------------------------------------------------------------
# Remote runner
# ---------------------------------------------------------------------------


def run_decompose_remote(
    remote_data: dict,
    prompt: str,
    answer: str,
    competitor: str,
) -> DecomposeResult:
    """Build DecomposeResult from remote worker response data."""
    a_data = remote_data[answer]
    c_data = remote_data[competitor]

    a_layers = [
        LayerContribution(layer=lc["layer"], attention=lc["attention"], mlp=lc["mlp"])
        for lc in a_data["layers"]
    ]
    c_layers = [
        LayerContribution(layer=lc["layer"], attention=lc["attention"], mlp=lc["mlp"])
        for lc in c_data["layers"]
    ]

    a_decomp = TokenDecomposition(
        token_str=answer,
        token_id=a_data["token_id"],
        final_logit=a_data["final_logit"],
        embedding=a_data["embedding"],
        layers=a_layers,
        reconstruction_error=a_data["reconstruction_error"],
        norm_scale=a_data["norm_scale"],
    )
    c_decomp = TokenDecomposition(
        token_str=competitor,
        token_id=c_data["token_id"],
        final_logit=c_data["final_logit"],
        embedding=c_data["embedding"],
        layers=c_layers,
        reconstruction_error=c_data["reconstruction_error"],
        norm_scale=c_data["norm_scale"],
    )

    embedding_margin = a_decomp.embedding - c_decomp.embedding
    component_margins = []
    for al, cl in zip(a_layers, c_layers):
        component_margins.append({
            "layer": al.layer,
            "attn_margin": al.attention - cl.attention,
            "mlp_margin": al.mlp - cl.mlp,
        })

    return DecomposeResult(
        prompt=prompt,
        answer=answer,
        competitor=competitor,
        answer_logit=a_decomp.final_logit,
        competitor_logit=c_decomp.final_logit,
        margin=a_decomp.final_logit - c_decomp.final_logit,
        embedding_margin=embedding_margin,
        component_margins=component_margins,
        reconstruction_error=max(
            a_decomp.reconstruction_error, c_decomp.reconstruction_error,
        ),
        answer_decomposition=a_decomp,
        competitor_decomposition=c_decomp,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _decomp_to_dict(d: TokenDecomposition) -> dict:
    return {
        "token_str": d.token_str,
        "token_id": d.token_id,
        "final_logit": d.final_logit,
        "embedding": d.embedding,
        "layers": [
            {"layer": lc.layer, "attention": lc.attention, "mlp": lc.mlp}
            for lc in d.layers
        ],
        "reconstruction_error": d.reconstruction_error,
        "norm_scale": d.norm_scale,
    }


def decompose_result_to_dict(r: DecomposeResult) -> dict:
    return {
        "prompt": r.prompt,
        "answer": r.answer,
        "competitor": r.competitor,
        "answer_logit": r.answer_logit,
        "competitor_logit": r.competitor_logit,
        "margin": r.margin,
        "embedding_margin": r.embedding_margin,
        "component_margins": r.component_margins,
        "reconstruction_error": r.reconstruction_error,
        "answer_decomposition": _decomp_to_dict(r.answer_decomposition),
        "competitor_decomposition": _decomp_to_dict(r.competitor_decomposition),
    }


def decompose_run_to_dict(run: DecomposeRun) -> dict:
    return {
        "run_id": run.run_id,
        "dataset": run.dataset,
        "model_name": run.model_name,
        "num_prompts": run.num_prompts,
        "created_at": run.created_at,
        "results": [decompose_result_to_dict(r) for r in run.results],
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


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _decompose_css() -> str:
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
    .bar-container {{
        display: flex; align-items: center; height: 16px;
        background: var(--bg3); border-radius: 3px; overflow: hidden;
    }}
    .bar-fill {{
        height: 100%; border-radius: 3px;
    }}
    .prompt-section {{
        background: var(--bg2); border-radius: 8px;
        padding: 1.5rem; margin-bottom: 2rem;
    }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, .prompt-section {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def _margin_class(val: float) -> str:
    return "positive" if val >= 0 else "negative"


def _render_single_table(result: DecomposeResult) -> str:
    """Render a component breakdown table for a single prompt."""
    a = result.answer_decomposition
    c = result.competitor_decomposition
    rows = []

    # Embedding row
    em = result.embedding_margin
    rows.append(
        f"<tr><td>Embedding</td>"
        f"<td>{a.embedding:+.4f}</td>"
        f"<td>{c.embedding:+.4f}</td>"
        f'<td class="{_margin_class(em)}">{em:+.4f}</td></tr>'
    )

    # Per-layer rows
    for al, cl, cm in zip(a.layers, c.layers, result.component_margins):
        attn_m = cm["attn_margin"]
        mlp_m = cm["mlp_margin"]
        rows.append(
            f"<tr><td>L{al.layer} attn</td>"
            f"<td>{al.attention:+.4f}</td>"
            f"<td>{cl.attention:+.4f}</td>"
            f'<td class="{_margin_class(attn_m)}">{attn_m:+.4f}</td></tr>'
        )
        rows.append(
            f"<tr><td>L{al.layer} mlp</td>"
            f"<td>{al.mlp:+.4f}</td>"
            f"<td>{cl.mlp:+.4f}</td>"
            f'<td class="{_margin_class(mlp_m)}">{mlp_m:+.4f}</td></tr>'
        )

    # Total row
    rows.append(
        f'<tr style="border-top:2px solid var(--bg3);font-weight:bold">'
        f"<td>TOTAL</td>"
        f"<td>{a.final_logit:+.4f}</td>"
        f"<td>{c.final_logit:+.4f}</td>"
        f'<td class="{_margin_class(result.margin)}">'
        f"{result.margin:+.4f}</td></tr>"
    )

    return (
        "<table>"
        "<thead><tr>"
        "<th>Component</th>"
        f"<th>{_esc(result.answer)}</th>"
        f"<th>{_esc(result.competitor)}</th>"
        "<th>Margin</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _render_batch_heatmap(results: list[DecomposeResult]) -> str:
    """Render a heatmap of margin-per-component across all prompts."""
    if not results:
        return ""

    # Collect all component names
    sample = results[0]
    components = ["embed"]
    for lc in sample.component_margins:
        components.append(f"L{lc['layer']}a")
        components.append(f"L{lc['layer']}m")

    # Header
    header = "<th>Prompt</th><th>Answer</th>"
    for comp in components:
        header += f"<th>{_esc(comp)}</th>"
    header += "<th>Total</th>"

    # Rows
    rows = []
    for r in results:
        margins = [r.embedding_margin]
        for cm in r.component_margins:
            margins.append(cm["attn_margin"])
            margins.append(cm["mlp_margin"])

        cells = ""
        max_abs = max(abs(m) for m in margins) if margins else 1.0
        for m in margins:
            intensity = min(abs(m) / max(max_abs, 0.01), 1.0) * 0.8
            cells += (
                f'<td style="background:rgba('
                f'{"76,175,80" if m >= 0 else "244,67,54"}'
                f',{intensity:.2f});text-align:center">'
                f"{m:+.2f}</td>"
            )

        mc = _margin_class(r.margin)
        rows.append(
            f"<tr><td>{_esc(r.prompt[:50])}</td>"
            f"<td>{_esc(r.answer)}</td>"
            f"{cells}"
            f'<td class="{mc}">{r.margin:+.2f}</td></tr>'
        )

    return (
        '<div style="overflow-x:auto;">'
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )


def generate_decompose_html(run: DecomposeRun) -> str:
    """Generate self-contained HTML report for decompose run."""
    header = f"""
    <h1>Logit Prism Decomposition</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Dataset</span><br>
                <span class="meta-value">{_esc(run.dataset or 'single')}</span></div>
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Prompts</span><br>
                <span class="meta-value">{run.num_prompts}</span></div>
            <div><span class="meta-label">Run ID</span><br>
                <span class="meta-value">{_esc(run.run_id[:12])}</span></div>
        </div>
    </div>
    """

    body_parts = [header]

    if run.num_prompts == 1 and run.results:
        # Single prompt mode: detailed table per competitor
        seen_prompts: set[str] = set()
        for r in run.results:
            if r.prompt not in seen_prompts:
                seen_prompts.add(r.prompt)
                body_parts.append(
                    f'<div class="prompt-section">'
                    f"<h3>{_esc(r.prompt)}</h3>"
                    f"<p>Reconstruction error: "
                    f"{r.reconstruction_error:.4f} "
                    f"{'✓' if r.reconstruction_error < 0.01 else '⚠'}</p>"
                )
            body_parts.append(
                f"<h3>{_esc(r.answer)} ({r.answer_logit:.2f}) vs "
                f"{_esc(r.competitor)} ({r.competitor_logit:.2f})</h3>"
            )
            body_parts.append(_render_single_table(r))
        body_parts.append("</div>")
    else:
        # Batch mode: heatmap
        body_parts.append("<h2>Margin Heatmap</h2>")
        body_parts.append(_render_batch_heatmap(run.results))

    body = "\n".join(body_parts)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1">\n'
        f"<title>Logit Prism - {_esc(run.dataset or 'decompose')}</title>\n"
        f"<style>{_decompose_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
