"""MLP key vector fingerprinting for analytical regression checking."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Tensor serialization (float16)
# ---------------------------------------------------------------------------


def serialize_f16_tensor(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes with shape header, stored as float16."""
    header = struct.pack("<i", arr.ndim)
    for dim in arr.shape:
        header += struct.pack("<i", dim)
    return header + arr.astype(np.float16).tobytes()


def deserialize_f16_tensor(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy float16 array."""
    offset = 0
    ndim = struct.unpack_from("<i", blob, offset)[0]
    offset += 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<i", blob, offset)[0])
        offset += 4
    data = np.frombuffer(blob, dtype=np.float16, offset=offset)
    return data.reshape(shape)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Fingerprint:
    prompt: str
    answer: str
    competitor: str
    answer_logit: float
    competitor_logit: float
    margin: float
    key_vectors: np.ndarray  # [num_layers, intermediate_size] float16
    p_answer: np.ndarray     # [hidden_size] float16
    p_competitor: np.ndarray  # [hidden_size] float16


@dataclass
class FingerprintRun:
    run_id: str
    dataset: str
    model_name: str
    prompt_count: int
    fingerprints: list[Fingerprint]


@dataclass
class AnalyticalRegression:
    prompt: str
    answer: str
    current_margin: float
    margin_change: float
    new_margin: float
    alignment: float
    regression: bool


# ---------------------------------------------------------------------------
# Local fingerprinting
# ---------------------------------------------------------------------------


def compute_fingerprint_local(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    seed: int = 42,
) -> Fingerprint:
    """Compute a fingerprint for a single prompt locally."""
    import torch

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Resolve answer token ID
    answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    answer_id = answer_ids[0] if answer_ids else 0

    # Capture MLP key vectors (input to down_proj) at every layer
    key_vectors: dict[int, torch.Tensor] = {}

    def make_down_proj_pre_hook(layer_idx):
        def hook(_module, args):
            inp = args[0] if isinstance(args, tuple) else args
            if inp.dim() == 3:
                key_vectors[layer_idx] = inp[0, -1, :].detach().cpu().half()
            elif inp.dim() == 2:
                key_vectors[layer_idx] = inp[-1, :].detach().cpu().half()
            else:
                key_vectors[layer_idx] = inp.detach().cpu().half()
        return hook

    hooks = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        hooks.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                make_down_proj_pre_hook(i)
            )
        )

    # Single forward pass with hidden states
    try:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()

    # Get logits and find competitor
    logits = outputs.logits[0, -1, :]
    answer_logit = logits[answer_id].item()

    logits_masked = logits.clone()
    logits_masked[answer_id] = float("-inf")
    competitor_id = logits_masked.argmax().item()
    competitor_logit = logits[competitor_id].item()
    competitor = tokenizer.decode([competitor_id]).strip()
    if not competitor:
        competitor = tokenizer.decode([competitor_id])

    # Compute Logit Prism projection vectors
    # hidden_states[-1] is after all layers, before final norm
    last_hidden = outputs.hidden_states[-1][0, -1, :]
    ln_weight = model.model.norm.weight.detach()
    variance = last_hidden.pow(2).mean(-1, keepdim=False)
    s = torch.sqrt(variance + model.model.norm.variance_epsilon)

    w_unembed = model.lm_head.weight
    p_answer = (w_unembed[answer_id] * ln_weight / s).detach().cpu().half()
    p_competitor = (w_unembed[competitor_id] * ln_weight / s).detach().cpu().half()

    # Stack key vectors into tensor
    num_layers = len(layers)
    kv_tensor = torch.stack([key_vectors[i] for i in range(num_layers)])

    return Fingerprint(
        prompt=prompt,
        answer=answer,
        competitor=competitor,
        answer_logit=answer_logit,
        competitor_logit=competitor_logit,
        margin=answer_logit - competitor_logit,
        key_vectors=kv_tensor.numpy(),
        p_answer=p_answer.numpy(),
        p_competitor=p_competitor.numpy(),
    )


# ---------------------------------------------------------------------------
# Analytical regression checking
# ---------------------------------------------------------------------------


def check_regressions_fast(
    delta: np.ndarray,
    k_star: np.ndarray,
    layer: int,
    fingerprints: list[Fingerprint],
    threshold: float = 0.5,
) -> list[AnalyticalRegression]:
    """Check for regressions analytically using fingerprints.

    Given a repair edit (delta, k_star, layer), compute the margin change
    for each fingerprinted prompt using dot products instead of forward passes.

    Args:
        delta: Correction vector [hidden_size], float32
        k_star: Key vector at target layer [intermediate_size], float32
        layer: Layer index where the edit is applied
        fingerprints: List of pre-computed fingerprints
        threshold: Minimum absolute margin change to flag

    Returns:
        List of AnalyticalRegression results for affected prompts.
    """
    delta = delta.astype(np.float32)
    k_star = k_star.astype(np.float32)
    k_star_norm_sq = float(np.dot(k_star, k_star))
    if k_star_norm_sq < 1e-10:
        return []

    affected = []

    for fp in fingerprints:
        k_other = fp.key_vectors[layer].astype(np.float32)
        alignment = float(np.dot(k_star, k_other)) / k_star_norm_sq

        if abs(alignment) < 1e-6:
            continue

        # Output change for this prompt
        delta_output = delta * alignment

        # Margin change: how much does the answer-vs-competitor margin shift?
        p_margin = fp.p_answer.astype(np.float32) - fp.p_competitor.astype(np.float32)
        margin_change = float(np.dot(p_margin, delta_output))

        if abs(margin_change) > threshold:
            new_margin = fp.margin + margin_change
            affected.append(AnalyticalRegression(
                prompt=fp.prompt,
                answer=fp.answer,
                current_margin=fp.margin,
                margin_change=margin_change,
                new_margin=new_margin,
                alignment=alignment,
                regression=(new_margin < 0 and fp.margin >= 0),
            ))

    return affected


# ---------------------------------------------------------------------------
# Build fingerprint from remote response
# ---------------------------------------------------------------------------


def build_fingerprint_from_remote(data: dict) -> Fingerprint:
    """Build a Fingerprint from remote worker response data."""
    import base64

    kv_bytes = base64.b64decode(data["key_vectors_b64"])
    kv_shape = data["key_vectors_shape"]
    key_vectors = np.frombuffer(kv_bytes, dtype=np.float16).reshape(kv_shape)

    p_answer_bytes = base64.b64decode(data["p_answer_b64"])
    p_answer = np.frombuffer(p_answer_bytes, dtype=np.float16).copy()

    p_competitor_bytes = base64.b64decode(data["p_competitor_b64"])
    p_competitor = np.frombuffer(p_competitor_bytes, dtype=np.float16).copy()

    return Fingerprint(
        prompt=data["prompt"],
        answer=data["answer"],
        competitor=data["competitor"],
        answer_logit=data["answer_logit"],
        competitor_logit=data["competitor_logit"],
        margin=data["margin"],
        key_vectors=key_vectors.copy(),
        p_answer=p_answer,
        p_competitor=p_competitor,
    )


# ---------------------------------------------------------------------------
# Alignment analysis
# ---------------------------------------------------------------------------


def compute_alignment_stats(
    fingerprints: list[Fingerprint],
) -> dict:
    """Compute cross-prompt alignment statistics from fingerprints."""
    if len(fingerprints) < 2:
        return {
            "avg_alignment": 0.0,
            "max_pair": None,
            "min_pair": None,
        }

    # Use the mean key vector across all layers for alignment
    n = len(fingerprints)
    max_alignment = -float("inf")
    min_alignment = float("inf")
    max_pair = None
    min_pair = None
    total = 0.0
    count = 0

    for i in range(n):
        ki = fingerprints[i].key_vectors.astype(np.float32).mean(axis=0)
        ki_norm = np.linalg.norm(ki)
        if ki_norm < 1e-10:
            continue
        for j in range(i + 1, n):
            kj = fingerprints[j].key_vectors.astype(np.float32).mean(axis=0)
            kj_norm = np.linalg.norm(kj)
            if kj_norm < 1e-10:
                continue
            cosine = float(np.dot(ki, kj) / (ki_norm * kj_norm))
            alignment = abs(cosine)
            total += alignment
            count += 1

            if alignment > max_alignment:
                max_alignment = alignment
                max_pair = (fingerprints[i].answer, fingerprints[j].answer)
            if alignment < min_alignment:
                min_alignment = alignment
                min_pair = (fingerprints[i].answer, fingerprints[j].answer)

    avg = total / count if count > 0 else 0.0
    return {
        "avg_alignment": avg,
        "max_pair": max_pair,
        "max_alignment": max_alignment if max_pair else 0.0,
        "min_pair": min_pair,
        "min_alignment": min_alignment if min_pair else 0.0,
    }
