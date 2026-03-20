"""Ablation engine — run inference with targeted components zeroed/scaled."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from neurotrace.models import ModelArchitecture, get_architecture
from neurotrace.tracer import Tracer
from neurotrace.types import TraceResult


@dataclass
class AblationSpec:
    """Describes which interventions to apply."""

    zero_layers: list[int]
    zero_heads: list[tuple[int, int]]  # (layer, head)
    scale_layers: list[tuple[int, float]]  # (layer, factor)
    zero_mlp: list[int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.zero_mlp is None:
            self.zero_mlp = []

    def to_json(self) -> str:
        return json.dumps({
            "zero_layers": self.zero_layers,
            "zero_heads": [[l, h] for l, h in self.zero_heads],
            "scale_layers": [[l, f] for l, f in self.scale_layers],
            "zero_mlp": self.zero_mlp,
        })

    def describe(self) -> str:
        parts = []
        if self.zero_layers:
            parts.append(f"zero-layers={','.join(map(str, self.zero_layers))}")
        if self.zero_heads:
            parts.append(
                "zero-heads="
                + ",".join(f"{l}:{h}" for l, h in self.zero_heads)
            )
        if self.scale_layers:
            parts.append(
                "scale-layers="
                + ",".join(f"{l}:{f}" for l, f in self.scale_layers)
            )
        if self.zero_mlp:
            parts.append(f"zero-mlp={','.join(map(str, self.zero_mlp))}")
        return "; ".join(parts)


@dataclass
class LayerComparison:
    """Per-layer comparison between baseline and ablated traces."""

    layer_index: int
    baseline_top1: str
    baseline_top1_prob: float
    ablated_top1: str
    ablated_top1_prob: float
    cosine_similarity: float
    changed: bool


@dataclass
class AblationResult:
    """Complete ablation comparison result."""

    baseline_trace: TraceResult
    ablated_trace: TraceResult
    spec: AblationSpec
    baseline_final_token: str
    baseline_final_prob: float
    ablated_final_token: str
    ablated_final_prob: float
    layer_comparisons: list[LayerComparison]


def parse_zero_layers(value: str) -> list[int]:
    """Parse '20' or '20,21,22' into list of ints."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_zero_heads(value: str) -> list[tuple[int, int]]:
    """Parse '20:7,20:12' into list of (layer, head) tuples."""
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        result.append((int(parts[0]), int(parts[1])))
    return result


def parse_scale_layers(value: str) -> list[tuple[int, float]]:
    """Parse '20:0.5,21:2.0' into list of (layer, factor) tuples."""
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        result.append((int(parts[0]), float(parts[1])))
    return result


class AblationHookManager:
    """Registers forward hooks that apply ablation interventions."""

    def __init__(
        self,
        model: torch.nn.Module,
        architecture: ModelArchitecture,
        spec: AblationSpec,
    ):
        self._model = model
        self._arch = architecture
        self._spec = spec
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        layers = self._arch.get_layers(self._model)

        # Build lookup sets for fast checking
        zero_layer_set = set(self._spec.zero_layers)
        zero_head_map: dict[int, set[int]] = {}
        for layer_idx, head_idx in self._spec.zero_heads:
            zero_head_map.setdefault(layer_idx, set()).add(head_idx)
        scale_layer_map = dict(self._spec.scale_layers)
        zero_mlp_set = set(self._spec.zero_mlp)

        for i, layer in enumerate(layers):
            # Full layer zeroing or scaling
            if i in zero_layer_set or i in scale_layer_map:
                factor = 0.0 if i in zero_layer_set else scale_layer_map.get(i, 1.0)
                handle = layer.register_forward_hook(
                    self._make_layer_hook(factor)
                )
                self._handles.append(handle)

            # Per-head zeroing
            if i in zero_head_map:
                attn = self._arch.get_attention(layer)
                heads_to_zero = zero_head_map[i]
                handle = attn.register_forward_hook(
                    self._make_head_zero_hook(heads_to_zero)
                )
                self._handles.append(handle)

            # MLP sublayer zeroing
            if i in zero_mlp_set:
                mlp = self._arch.get_mlp(layer)
                handle = mlp.register_forward_hook(self._make_mlp_zero_hook())
                self._handles.append(handle)

    @staticmethod
    def _make_layer_hook(factor: float):
        """Hook that scales the layer output before it's added to residual stream.

        For decoder layers, output is typically (hidden_states, ...).
        We need to scale the hidden_states (first element) which represents
        the layer's contribution to the residual stream.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                inp = input[0] if isinstance(input, tuple) else input
                # Scale the difference (layer contribution) rather than the absolute output
                # layer_output = residual_input + layer_contribution
                # We want: residual_input + factor * layer_contribution
                contribution = output[0] - inp
                scaled = inp + factor * contribution
                return (scaled,) + output[1:]
            return input[0] + factor * (output - input[0])

        return hook

    @staticmethod
    def _make_head_zero_hook(heads_to_zero: set[int]):
        """Hook that zeros specific attention head outputs.

        The attention module output[0] shape is (batch, seq_len, hidden_size).
        We reshape to (batch, seq_len, num_heads, head_dim), zero targeted
        heads, then reshape back.
        """
        def hook(module, input, output):
            attn_output = output[0]  # (batch, seq_len, hidden_size)
            num_heads = module.config.num_attention_heads
            head_dim = module.head_dim
            batch, seq_len, _ = attn_output.shape

            # Reshape to (batch, seq_len, num_heads, head_dim)
            reshaped = attn_output.view(batch, seq_len, num_heads, head_dim)
            for h in heads_to_zero:
                if h < num_heads:
                    reshaped[:, :, h, :] = 0.0
            modified = reshaped.view(batch, seq_len, -1)

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    @staticmethod
    def _make_mlp_zero_hook():
        """Hook that zeros the MLP sublayer output."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)

        return hook

    @property
    def num_hooks(self) -> int:
        return len(self._handles)

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def run_ablation(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    spec: AblationSpec,
    baseline: TraceResult | None = None,
    label: str | None = None,
    seed: int = 42,
) -> AblationResult:
    """Run ablated inference and compare to baseline.

    If baseline is None, runs a clean trace first.
    """
    arch = get_architecture(model.config.model_type)

    # Run baseline if needed
    tracer = Tracer(model, tokenizer)
    if baseline is None:
        baseline = tracer.trace(prompt, label=f"{label}-baseline" if label else None, seed=seed)

    # Run ablated trace with intervention hooks
    # We need to do the trace manually with our ablation hooks active
    ablated = _run_ablated_trace(model, tokenizer, arch, prompt, spec, label, seed)

    # Build per-layer comparisons
    layer_comparisons = _compare_traces(baseline, ablated, model, tokenizer)

    # Final predictions
    b_token, b_prob = _get_final_prediction(baseline)
    a_token, a_prob = _get_final_prediction(ablated)

    return AblationResult(
        baseline_trace=baseline,
        ablated_trace=ablated,
        spec=spec,
        baseline_final_token=b_token,
        baseline_final_prob=b_prob,
        ablated_final_token=a_token,
        ablated_final_prob=a_prob,
        layer_comparisons=layer_comparisons,
    )


def _run_ablated_trace(
    model: torch.nn.Module,
    tokenizer,
    arch: ModelArchitecture,
    prompt: str,
    spec: AblationSpec,
    label: str | None,
    seed: int,
) -> TraceResult:
    """Run a trace with ablation hooks active."""
    ablation_hooks = AblationHookManager(model, arch, spec)
    try:
        tracer = Tracer(model, tokenizer)
        result = tracer.trace(prompt, label=label, seed=seed)
    finally:
        ablation_hooks.cleanup()
    return result


def _get_final_prediction(trace: TraceResult) -> tuple[str, float]:
    """Get the final (last position) top-1 prediction."""
    if trace.token_predictions:
        last = trace.token_predictions[-1]
        if last.top_k_strings:
            return last.top_k_strings[0], last.top_k_probs[0]
    return "", 0.0


def _compare_traces(
    baseline: TraceResult,
    ablated: TraceResult,
    model: torch.nn.Module,
    tokenizer,
) -> list[LayerComparison]:
    """Compare per-layer predictions and residual cosine similarity."""
    comparisons = []

    lm_head = model.lm_head
    final_ln = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm

    for snap_b, snap_a in zip(
        baseline.layer_snapshots, ablated.layer_snapshots
    ):
        # Cosine similarity of residual streams
        cos_sim = 1.0
        if snap_b.residual_out is not None and snap_a.residual_out is not None:
            b_flat = snap_b.residual_out.flatten()
            a_flat = snap_a.residual_out.flatten()
            if b_flat.shape == a_flat.shape:
                denom = np.linalg.norm(b_flat) * np.linalg.norm(a_flat)
                if denom > 1e-10:
                    cos_sim = float(np.dot(b_flat, a_flat) / denom)

        # Get top-1 predictions via lm_head projection
        b_tok, b_prob = _project_top1(snap_b.residual_out, lm_head, final_ln, tokenizer)
        a_tok, a_prob = _project_top1(snap_a.residual_out, lm_head, final_ln, tokenizer)

        comparisons.append(LayerComparison(
            layer_index=snap_b.layer_index,
            baseline_top1=b_tok,
            baseline_top1_prob=b_prob,
            ablated_top1=a_tok,
            ablated_top1_prob=a_prob,
            cosine_similarity=cos_sim,
            changed=b_tok != a_tok,
        ))

    return comparisons


def _project_top1(
    residual_out: np.ndarray | None,
    lm_head: torch.nn.Module,
    final_ln: torch.nn.Module | None,
    tokenizer,
) -> tuple[str, float]:
    """Project residual_out through lm_head to get top-1 token string and prob."""
    if residual_out is None:
        return "", 0.0

    with torch.no_grad():
        res = torch.tensor(residual_out, dtype=torch.float32).unsqueeze(0)
        if final_ln is not None:
            res = final_ln(res)
        logits = lm_head(res.squeeze(0))
        probs = torch.softmax(logits[-1], dim=-1)
        top1_idx = int(torch.argmax(probs).item())
        top1_prob = float(probs[top1_idx].item())
        top1_str = tokenizer.decode(top1_idx)

    return top1_str, top1_prob
