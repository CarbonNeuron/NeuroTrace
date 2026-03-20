"""Neuron-level MLP attribution — profile and ablate individual neurons."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch

from neurotrace.models import ModelArchitecture, get_architecture, get_final_prediction
from neurotrace.tracer import Tracer
from neurotrace.types import TraceResult


@dataclass
class NeuronProfile:
    """Activation profile for neurons in one MLP layer."""

    profile_id: str
    trace_id: str | None
    contrast_trace_id: str | None
    layer: int
    component: str  # "mlp"
    prompt: str
    contrast_prompt: str | None
    model_name: str
    neuron_indices: list[int]
    target_activations: list[float]
    contrast_activations: list[float] | None
    diff_activations: list[float] | None
    label: str | None
    created_at: str


@dataclass
class NeuronAblationResult:
    """Result of ablating specific neurons."""

    neurons: list[int]
    baseline_top1: str
    baseline_top1_prob: float
    ablated_top1: str
    ablated_top1_prob: float
    changed: bool
    trace: TraceResult


def parse_neurons(value: str) -> list[int]:
    """Parse neuron indices from '0,1,2' or '100-200' or mixed '0,1,100-200'."""
    result = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            result.append(int(part))
    return result


def _capture_mlp_intermediate(
    model: torch.nn.Module,
    tokenizer,
    arch: ModelArchitecture,
    prompt: str,
    layer: int,
    seed: int = 42,
) -> tuple[np.ndarray, TraceResult]:
    """Run forward pass and capture MLP intermediate activations at a layer.

    Hooks into down_proj to capture its input (= SiLU(gate) * up, before down_proj).

    Returns (intermediate_activations, trace_result) where activations shape
    is [seq_len, intermediate_size].
    """
    layers = arch.get_layers(model)
    mlp = arch.get_mlp(layers[layer])
    down_proj = mlp.down_proj

    captured = {}

    def capture_hook(module, input, output):
        inp = input[0] if isinstance(input, tuple) else input
        captured["intermediate"] = inp.detach().cpu().float().numpy().squeeze(0)

    handle = down_proj.register_forward_hook(capture_hook)
    try:
        tracer = Tracer(model, tokenizer)
        trace = tracer.trace(prompt, seed=seed)
    finally:
        handle.remove()

    return captured["intermediate"], trace


def profile_neurons(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer: int,
    top_n: int = 20,
    contrast_prompt: str | None = None,
    seed: int = 42,
    label: str | None = None,
) -> NeuronProfile:
    """Profile MLP intermediate neurons by activation magnitude or differential.

    If contrast_prompt is provided, ranks by |target - contrast| activation.
    Otherwise ranks by absolute activation magnitude on target prompt.
    """
    arch = get_architecture(model.config.model_type)

    # Capture target activations
    target_acts, target_trace = _capture_mlp_intermediate(
        model, tokenizer, arch, prompt, layer, seed
    )
    # Use last token position
    target_last = target_acts[-1]  # [intermediate_size]

    contrast_last = None
    contrast_trace = None
    if contrast_prompt is not None:
        contrast_acts, contrast_trace = _capture_mlp_intermediate(
            model, tokenizer, arch, contrast_prompt, layer, seed
        )
        contrast_last = contrast_acts[-1]

    # Rank neurons
    if contrast_last is not None:
        diff = target_last - contrast_last
        ranking_metric = np.abs(diff)
    else:
        diff = None
        ranking_metric = np.abs(target_last)

    top_indices = np.argsort(ranking_metric)[::-1][:top_n].tolist()

    return NeuronProfile(
        profile_id=str(uuid.uuid4()),
        trace_id=target_trace.metadata.trace_id,
        contrast_trace_id=contrast_trace.metadata.trace_id if contrast_trace else None,
        layer=layer,
        component="mlp",
        prompt=prompt,
        contrast_prompt=contrast_prompt,
        model_name=model.config.name_or_path,
        neuron_indices=top_indices,
        target_activations=[float(target_last[i]) for i in top_indices],
        contrast_activations=(
            [float(contrast_last[i]) for i in top_indices]
            if contrast_last is not None
            else None
        ),
        diff_activations=(
            [float(diff[i]) for i in top_indices] if diff is not None else None
        ),
        label=label,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def ablate_neurons(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer: int,
    neuron_groups: list[list[int]],
    baseline: TraceResult | None = None,
    label_prefix: str | None = None,
    seed: int = 42,
) -> tuple[TraceResult | None, list[NeuronAblationResult]]:
    """Ablate groups of neurons and measure impact on prediction.

    Each group in neuron_groups is ablated independently.
    Returns (baseline_trace_if_created, list_of_results).
    """
    arch = get_architecture(model.config.model_type)

    # Run baseline if needed
    baseline_created = None
    if baseline is None:
        tracer = Tracer(model, tokenizer)
        baseline = tracer.trace(prompt, seed=seed)
        baseline_created = baseline

    # Get baseline final prediction
    b_token, b_prob = get_final_prediction(baseline.token_predictions)

    results = []
    for group in neuron_groups:
        # Register pre-hook on down_proj to zero neuron indices
        layers = arch.get_layers(model)
        mlp = arch.get_mlp(layers[layer])
        down_proj = mlp.down_proj

        def make_zero_hook(indices):
            def hook(module, input):
                inp = input[0] if isinstance(input, tuple) else input
                modified = inp.clone()
                modified[:, :, indices] = 0.0
                if isinstance(input, tuple):
                    return (modified,) + input[1:]
                return (modified,)

            return hook

        handle = down_proj.register_forward_pre_hook(make_zero_hook(group))
        try:
            group_label = None
            if label_prefix:
                group_label = f"{label_prefix}-n{'_'.join(map(str, group[:3]))}"
            tracer = Tracer(model, tokenizer)
            ablated_trace = tracer.trace(prompt, label=group_label, seed=seed)
        finally:
            handle.remove()

        a_token, a_prob = get_final_prediction(ablated_trace.token_predictions)

        results.append(
            NeuronAblationResult(
                neurons=group,
                baseline_top1=b_token,
                baseline_top1_prob=b_prob,
                ablated_top1=a_token,
                ablated_top1_prob=a_prob,
                changed=b_token != a_token,
                trace=ablated_trace,
            )
        )

    return baseline_created, results
