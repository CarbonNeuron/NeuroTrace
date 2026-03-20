"""Mechanistic MLP decomposition: direction -> MLP -> tokens."""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ForwardResult:
    """Forward circuit: direction -> MLP -> token logits."""

    mlp_output: np.ndarray  # (hidden_dim,)
    logit_contribution: np.ndarray  # (vocab_size,)
    top_boosted: list[dict]  # [{token_id, token, logit}, ...]
    top_suppressed: list[dict]  # [{token_id, token, logit}, ...]
    pre_mlp_logits: np.ndarray  # (vocab_size,)
    pre_mlp_top: list[dict]  # [{token_id, token, logit}, ...]


@dataclass
class ReverseResult:
    """Reverse circuit: token -> MLP -> required input direction."""

    token: str
    token_id: int
    input_direction: np.ndarray  # (hidden_dim,) unit vector
    cosine_sim_with_probe: float


@dataclass
class PromptComparison:
    """Side-by-side MLP behavior for a specific prompt."""

    prompt: str
    top_boosted: list[dict]  # [{token_id, token, logit}, ...]
    top_suppressed: list[dict]  # [{token_id, token, logit}, ...]
    activation_norm: float
    cosine_sim_with_direction: float


@dataclass
class RealPromptMLP:
    """Real MLP analysis for a single prompt."""

    prompt: str
    mlp_output_norm: float
    sabotage_projection: float
    top_boosted: list[dict]  # [{token_id, token, logit}, ...]
    top_suppressed: list[dict]  # [{token_id, token, logit}, ...]
    final_top_tokens: list[dict]  # full model prediction


@dataclass
class PairwiseDiff:
    """Differential MLP analysis between two prompts."""

    prompt_a: str
    prompt_b: str
    cosine_similarity: float
    norm_ratio: float
    diff_boosted: list[dict]  # tokens MORE boosted for prompt_a
    diff_suppressed: list[dict]  # tokens MORE boosted for prompt_b


@dataclass
class RealComparisonResult:
    """Full real MLP comparison result."""

    prompts: list[RealPromptMLP] = field(default_factory=list)
    pairwise: list[PairwiseDiff] = field(default_factory=list)


@dataclass
class CircuitResult:
    """Full circuit analysis result."""

    probe_dir: str
    model_name: str
    layer: int
    forward: ForwardResult
    reverse: list[ReverseResult]
    comparisons: list[PromptComparison]
    real_comparison: RealComparisonResult | None = None


def load_direction(probe_dir: str) -> np.ndarray:
    """Load the mean_direction.npy from a probe output directory."""
    path = os.path.join(probe_dir, "mean_direction.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No mean_direction.npy found in {probe_dir}")
    direction = np.load(path)
    return direction


def detect_layer_from_probe(probe_dir: str) -> int | None:
    """Try to detect the layer from the probe's report.md."""
    report_path = os.path.join(probe_dir, "report.md")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        for line in f:
            if "- Layer:" in line:
                try:
                    return int(line.split(":")[-1].strip())
                except ValueError:
                    pass
    return None


def forward_circuit(
    model: torch.nn.Module,
    tokenizer,
    direction: np.ndarray,
    layer: int,
    top_k: int = 30,
) -> ForwardResult:
    """Trace direction through MLP to token logits.

    Linear approximation: treats direction as a single input activation vector.
    """
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    layers = arch.get_layers(model)
    mlp = arch.get_mlp(layers[layer])

    model_dtype = next(model.parameters()).dtype
    direction_t = torch.tensor(direction, dtype=model_dtype)

    with torch.no_grad():
        # Extract MLP weight matrices
        gate_proj = mlp.gate_proj.weight  # (intermediate, hidden)
        up_proj = mlp.up_proj.weight  # (intermediate, hidden)
        down_proj = mlp.down_proj.weight  # (hidden, intermediate)

        # Forward pass through SwiGLU MLP
        gate_out = gate_proj @ direction_t
        up_out = up_proj @ direction_t
        hidden = torch.nn.functional.silu(gate_out) * up_out
        mlp_output = down_proj @ hidden

        # Project to token logits via lm_head
        lm_head_weight = model.lm_head.weight  # (vocab_size, hidden)
        lm_head_bias = getattr(model.lm_head, "bias", None)

        logit_contribution = lm_head_weight @ mlp_output
        if lm_head_bias is not None:
            logit_contribution = logit_contribution + lm_head_bias

        # Top boosted tokens
        top_vals, top_ids = logit_contribution.topk(top_k)
        top_boosted = []
        for i in range(top_k):
            tid = top_ids[i].item()
            top_boosted.append({
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "logit": float(top_vals[i].item()),
            })

        # Top suppressed tokens
        bot_vals, bot_ids = logit_contribution.topk(top_k, largest=False)
        top_suppressed = []
        for i in range(top_k):
            tid = bot_ids[i].item()
            top_suppressed.append({
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "logit": float(bot_vals[i].item()),
            })

        # Pre-MLP prediction (direct logit lens)
        pre_mlp_logits = lm_head_weight @ direction_t
        if lm_head_bias is not None:
            pre_mlp_logits = pre_mlp_logits + lm_head_bias

        pre_top_vals, pre_top_ids = pre_mlp_logits.topk(top_k)
        pre_mlp_top = []
        for i in range(top_k):
            tid = pre_top_ids[i].item()
            pre_mlp_top.append({
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "logit": float(pre_top_vals[i].item()),
            })

    return ForwardResult(
        mlp_output=mlp_output.cpu().numpy(),
        logit_contribution=logit_contribution.cpu().numpy(),
        top_boosted=top_boosted,
        top_suppressed=top_suppressed,
        pre_mlp_logits=pre_mlp_logits.cpu().numpy(),
        pre_mlp_top=pre_mlp_top,
    )


def reverse_circuit(
    model: torch.nn.Module,
    tokenizer,
    direction: np.ndarray,
    layer: int,
    tokens: list[str],
) -> list[ReverseResult]:
    """Trace tokens backwards through MLP to find required input directions."""
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    layers = arch.get_layers(model)
    mlp = arch.get_mlp(layers[layer])

    model_dtype = next(model.parameters()).dtype
    direction_t = torch.tensor(direction, dtype=model_dtype)

    results = []
    with torch.no_grad():
        gate_proj = mlp.gate_proj.weight  # (intermediate, hidden)
        down_proj = mlp.down_proj.weight  # (hidden, intermediate)
        lm_head_weight = model.lm_head.weight  # (vocab_size, hidden)

        for token_str in tokens:
            # Encode the token
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if not token_ids:
                continue
            token_id = token_ids[0]
            token_embedding = lm_head_weight[token_id]  # (hidden,)

            # Reverse through down_proj: find hidden that maps to token_embedding
            # down_proj @ hidden = token_embedding
            hidden_target = torch.linalg.lstsq(
                down_proj, token_embedding
            ).solution  # (intermediate,)

            # Reverse through gate_proj: find input direction
            # gate_proj @ input = hidden_target
            input_dir = torch.linalg.lstsq(
                gate_proj, hidden_target
            ).solution  # (hidden,)

            # Normalize
            norm = input_dir.norm()
            if norm > 0:
                input_dir = input_dir / norm

            # Cosine similarity with probe direction
            cos_sim = float(torch.dot(input_dir, direction_t).item())

            decoded = tokenizer.decode([token_id])
            results.append(ReverseResult(
                token=decoded,
                token_id=token_id,
                input_direction=input_dir.cpu().numpy(),
                cosine_sim_with_probe=cos_sim,
            ))

    return results


def compare_prompt(
    model: torch.nn.Module,
    tokenizer,
    direction: np.ndarray,
    layer: int,
    prompt: str,
    top_k: int = 30,
    seed: int = 42,
) -> PromptComparison:
    """Capture real MLP behavior on a prompt and compare with the probe direction."""
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    layers = arch.get_layers(model)
    target_layer = layers[layer]
    mlp = arch.get_mlp(target_layer)

    model_dtype = next(model.parameters()).dtype
    direction_t = torch.tensor(direction, dtype=model_dtype)

    captured = {}

    def mlp_hook(_module, input, output):
        captured["mlp_input"] = input[0].detach()
        out_tensor = output[0] if isinstance(output, tuple) else output
        captured["mlp_output"] = out_tensor.detach()

    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(next(model.parameters()).device)

    handle = mlp.register_forward_hook(mlp_hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    # Last token position
    mlp_input = captured["mlp_input"][0, -1, :]  # (hidden,)
    mlp_out = captured["mlp_output"][0, -1, :]  # (hidden,)

    # Project MLP output through lm_head
    lm_head_weight = model.lm_head.weight
    lm_head_bias = getattr(model.lm_head, "bias", None)

    with torch.no_grad():
        logits = lm_head_weight @ mlp_out
        if lm_head_bias is not None:
            logits = logits + lm_head_bias

        top_vals, top_ids = logits.topk(top_k)
        top_boosted = []
        for i in range(top_k):
            tid = top_ids[i].item()
            top_boosted.append({
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "logit": float(top_vals[i].item()),
            })

        bot_vals, bot_ids = logits.topk(top_k, largest=False)
        top_suppressed = []
        for i in range(top_k):
            tid = bot_ids[i].item()
            top_suppressed.append({
                "token_id": tid,
                "token": tokenizer.decode([tid]),
                "logit": float(bot_vals[i].item()),
            })

        # Cosine similarity between real activation and probe direction
        cos_sim = float(torch.nn.functional.cosine_similarity(
            mlp_input.unsqueeze(0), direction_t.unsqueeze(0)
        ).item())

    return PromptComparison(
        prompt=prompt,
        top_boosted=top_boosted,
        top_suppressed=top_suppressed,
        activation_norm=float(mlp_input.norm().item()),
        cosine_sim_with_direction=cos_sim,
    )


def _capture_mlp_io(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer: int,
) -> dict:
    """Capture MLP input and output at a specific layer for a prompt."""
    from neurotrace.models import get_architecture

    arch = get_architecture(model.config.model_type)
    layers = arch.get_layers(model)
    target_mlp = arch.get_mlp(layers[layer])

    pre_mlp_input = {}
    mlp_out = {}

    def mlp_pre_hook(module, args):
        pre_mlp_input["val"] = args[0][:, -1, :].detach().clone()
        return args

    def mlp_post_hook(module, args, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        mlp_out["val"] = out_tensor[:, -1, :].detach().clone()
        return output

    h1 = target_mlp.register_forward_pre_hook(mlp_pre_hook)
    h2 = target_mlp.register_forward_hook(mlp_post_hook)

    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens["input_ids"].to(next(model.parameters()).device)

    with torch.no_grad():
        outputs = model(input_ids)

    h1.remove()
    h2.remove()

    final_logits = outputs.logits[0, -1, :]  # (vocab_size,)

    return {
        "mlp_input": pre_mlp_input["val"].squeeze(0),  # (hidden_dim,)
        "mlp_output": mlp_out["val"].squeeze(0),  # (hidden_dim,)
        "final_logits": final_logits.detach(),  # (vocab_size,)
    }


def _top_k_tokens(
    logits: torch.Tensor, tokenizer, top_k: int, largest: bool = True,
) -> list[dict]:
    """Extract top-k tokens from logit vector."""
    vals, ids = logits.topk(top_k, largest=largest)
    results = []
    for i in range(top_k):
        tid = ids[i].item()
        results.append({
            "token_id": tid,
            "token": tokenizer.decode([tid]),
            "logit": float(vals[i].item()),
        })
    return results


def compare_real_mlp(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    layer: int,
    direction: np.ndarray,
    top_k: int = 30,
) -> RealComparisonResult:
    """Compare real MLP behavior across prompts.

    For each prompt, runs a forward pass, captures the MLP's additive
    contribution at the target layer, and projects it through lm_head.
    For pairs of prompts, computes differential MLP contributions.
    """
    model_dtype = next(model.parameters()).dtype
    direction_t = torch.tensor(direction, dtype=model_dtype).to(
        next(model.parameters()).device
    )
    lm_head_weight = model.lm_head.weight
    lm_head_bias = getattr(model.lm_head, "bias", None)

    captures = []
    prompt_results = []

    for prompt in prompts:
        cap = _capture_mlp_io(model, tokenizer, prompt, layer)
        captures.append(cap)

        mlp_output = cap["mlp_output"]

        # MLP's token-level contribution
        with torch.no_grad():
            mlp_logits = lm_head_weight @ mlp_output
            if lm_head_bias is not None:
                mlp_logits = mlp_logits + lm_head_bias

        output_norm = float(mlp_output.norm().item())
        projection = float(torch.dot(mlp_output, direction_t).item())

        # Full model top tokens
        final_top = _top_k_tokens(cap["final_logits"], tokenizer, top_k)

        prompt_results.append(RealPromptMLP(
            prompt=prompt,
            mlp_output_norm=output_norm,
            sabotage_projection=projection,
            top_boosted=_top_k_tokens(mlp_logits, tokenizer, top_k, largest=True),
            top_suppressed=_top_k_tokens(mlp_logits, tokenizer, top_k, largest=False),
            final_top_tokens=final_top,
        ))

    # Pairwise diffs
    pairwise = []
    for i in range(len(captures)):
        for j in range(i + 1, len(captures)):
            mlp_a = captures[i]["mlp_output"]
            mlp_b = captures[j]["mlp_output"]

            cos_sim = float(torch.nn.functional.cosine_similarity(
                mlp_a.unsqueeze(0), mlp_b.unsqueeze(0)
            ).item())

            norm_a = float(mlp_a.norm().item())
            norm_b = float(mlp_b.norm().item())
            norm_ratio = norm_a / norm_b if norm_b > 0 else float("inf")

            diff = mlp_a - mlp_b
            with torch.no_grad():
                diff_logits = lm_head_weight @ diff
                if lm_head_bias is not None:
                    diff_logits = diff_logits  # bias cancels in difference

            pairwise.append(PairwiseDiff(
                prompt_a=prompts[i],
                prompt_b=prompts[j],
                cosine_similarity=cos_sim,
                norm_ratio=norm_ratio,
                diff_boosted=_top_k_tokens(diff_logits, tokenizer, top_k, largest=True),
                diff_suppressed=_top_k_tokens(
                    diff_logits, tokenizer, top_k, largest=False,
                ),
            ))

    return RealComparisonResult(prompts=prompt_results, pairwise=pairwise)


def run_circuit(
    model: torch.nn.Module,
    tokenizer,
    probe_dir: str,
    layer: int,
    top_k: int = 30,
    reverse_tokens: list[str] | None = None,
    compare_prompts: list[str] | None = None,
    seed: int = 42,
) -> CircuitResult:
    """Run the full circuit analysis pipeline."""
    direction = load_direction(probe_dir)
    model_name = model.config._name_or_path

    forward = forward_circuit(model, tokenizer, direction, layer, top_k=top_k)

    reverse = []
    if reverse_tokens:
        reverse = reverse_circuit(model, tokenizer, direction, layer, reverse_tokens)

    comparisons = []
    real_comparison = None
    if compare_prompts:
        for prompt in compare_prompts:
            comp = compare_prompt(
                model, tokenizer, direction, layer, prompt,
                top_k=top_k, seed=seed,
            )
            comparisons.append(comp)

        # Real MLP comparison (always run when compare prompts are provided)
        real_comparison = compare_real_mlp(
            model, tokenizer, compare_prompts, layer, direction, top_k=top_k,
        )

    return CircuitResult(
        probe_dir=probe_dir,
        model_name=model_name,
        layer=layer,
        forward=forward,
        reverse=reverse,
        comparisons=comparisons,
        real_comparison=real_comparison,
    )


def save_circuit_outputs(result: CircuitResult, output_dir: str) -> None:
    """Save all circuit outputs to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # forward_analysis.json
    forward_data = {
        "layer": result.layer,
        "top_boosted_tokens": result.forward.top_boosted,
        "top_suppressed_tokens": result.forward.top_suppressed,
        "pre_mlp_top_tokens": result.forward.pre_mlp_top,
    }
    with open(os.path.join(output_dir, "forward_analysis.json"), "w") as f:
        json.dump(forward_data, f, indent=2)

    # reverse_analysis.json
    if result.reverse:
        reverse_data = []
        for r in result.reverse:
            reverse_data.append({
                "token": r.token,
                "token_id": r.token_id,
                "cosine_sim_with_probe": r.cosine_sim_with_probe,
            })
        with open(os.path.join(output_dir, "reverse_analysis.json"), "w") as f:
            json.dump(reverse_data, f, indent=2)

    # prompt_comparison.json
    if result.comparisons:
        comp_data = []
        for c in result.comparisons:
            comp_data.append({
                "prompt": c.prompt,
                "activation_norm": c.activation_norm,
                "cosine_sim_with_direction": c.cosine_sim_with_direction,
                "top_boosted_tokens": c.top_boosted,
                "top_suppressed_tokens": c.top_suppressed,
            })
        with open(os.path.join(output_dir, "prompt_comparison.json"), "w") as f:
            json.dump(comp_data, f, indent=2)

    # real_comparison.json
    if result.real_comparison:
        rc = result.real_comparison
        rc_data = {
            "prompts": [
                {
                    "text": p.prompt,
                    "mlp_output_norm": p.mlp_output_norm,
                    "sabotage_projection": p.sabotage_projection,
                    "top_boosted": p.top_boosted,
                    "top_suppressed": p.top_suppressed,
                    "final_top_tokens": p.final_top_tokens,
                }
                for p in rc.prompts
            ],
            "pairwise": [
                {
                    "prompt_a": pw.prompt_a,
                    "prompt_b": pw.prompt_b,
                    "cosine_similarity": pw.cosine_similarity,
                    "norm_ratio": pw.norm_ratio,
                    "diff_boosted": pw.diff_boosted,
                    "diff_suppressed": pw.diff_suppressed,
                }
                for pw in rc.pairwise
            ],
        }
        with open(os.path.join(output_dir, "real_comparison.json"), "w") as f:
            json.dump(rc_data, f, indent=2)

    # report.md
    report = generate_report(result)
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report)


def generate_report(result: CircuitResult) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        f"# Circuit Analysis: {os.path.basename(result.probe_dir)}"
        f" -> Layer {result.layer} MLP",
        "",
        "## Sabotage Direction -> MLP -> Token Output",
        "",
        "### Tokens BOOSTED by MLP (what interference looks like)",
        "| Rank | Token | Logit Contribution |",
        "|------|-------|-------------------|",
    ]
    for i, t in enumerate(result.forward.top_boosted, 1):
        lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")

    lines.extend([
        "",
        "### Tokens SUPPRESSED by MLP",
        "| Rank | Token | Logit Contribution |",
        "|------|-------|-------------------|",
    ])
    for i, t in enumerate(result.forward.top_suppressed, 1):
        lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")

    lines.extend([
        "",
        "### Pre-MLP Prediction (residual stream without MLP)",
        "| Rank | Token | Logit |",
        "|------|-------|-------|",
    ])
    for i, t in enumerate(result.forward.pre_mlp_top, 1):
        lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")

    if result.reverse:
        lines.extend([
            "",
            "## Reverse Circuit: Token -> MLP -> Required Input",
            "| Token | Cosine Sim with Sabotage Direction |",
            "|-------|------------------------------------|",
        ])
        for r in result.reverse:
            lines.append(f"| {r.token} | {r.cosine_sim_with_probe:.4f} |")

    if result.comparisons:
        lines.extend(["", "## Prompt Comparison"])
        for comp in result.comparisons:
            lines.extend([
                f"### \"{comp.prompt}\"",
                f"- Activation norm: {comp.activation_norm:.4f}",
                f"- Cosine sim with sabotage direction:"
                f" {comp.cosine_sim_with_direction:.4f}",
                "",
                "Top MLP-boosted tokens:",
                "| Rank | Token | Logit |",
                "|------|-------|-------|",
            ])
            for i, t in enumerate(comp.top_boosted[:10], 1):
                lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")
            lines.append("")

    if result.real_comparison:
        rc = result.real_comparison
        lines.extend(["", "## Real MLP Comparison", ""])
        lines.append("### Per-Prompt MLP Analysis")

        for p in rc.prompts:
            lines.extend([
                "",
                f"#### \"{p.prompt}\"",
                f"MLP output norm: {p.mlp_output_norm:.2f}"
                f" | Projection onto sabotage direction: {p.sabotage_projection:.4f}",
                "",
                "Top MLP-boosted tokens:",
                "| Rank | Token | Logit Contribution |",
                "|------|-------|--------------------|",
            ])
            for i, t in enumerate(p.top_boosted[:15], 1):
                lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")
            lines.extend([
                "",
                "Top MLP-suppressed tokens:",
                "| Rank | Token | Logit Contribution |",
                "|------|-------|--------------------|",
            ])
            for i, t in enumerate(p.top_suppressed[:15], 1):
                lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")

            lines.extend([
                "",
                "Full model top predictions:",
                "| Rank | Token | Logit |",
                "|------|-------|-------|",
            ])
            for i, t in enumerate(p.final_top_tokens[:10], 1):
                lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")

        if rc.pairwise:
            lines.extend(["", "### Differential Analysis"])
            for pw in rc.pairwise:
                lines.extend([
                    "",
                    f"#### {pw.prompt_a} vs {pw.prompt_b}",
                    f"MLP output cosine similarity: {pw.cosine_similarity:.4f}",
                    f"Norm ratio: {pw.norm_ratio:.4f}",
                    "",
                    f"Tokens MORE boosted for \"{pw.prompt_a}\":",
                    "| Rank | Token | Differential Logit |",
                    "|------|-------|-------------------|",
                ])
                for i, t in enumerate(pw.diff_boosted[:15], 1):
                    lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")
                lines.extend([
                    "",
                    f"Tokens MORE boosted for \"{pw.prompt_b}\":",
                    "| Rank | Token | Differential Logit |",
                    "|------|-------|-------------------|",
                ])
                for i, t in enumerate(pw.diff_suppressed[:15], 1):
                    lines.append(f"| {i} | {t['token']} | {t['logit']:+.4f} |")

    return "\n".join(lines)
