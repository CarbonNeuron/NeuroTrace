"""Trace comparison and divergence detection."""

from dataclasses import dataclass

import numpy as np

from neurotrace.types import TraceResult


@dataclass
class LayerDiffMetrics:
    """Per-layer divergence metrics between two traces."""

    layer_index: int
    cosine_similarity: float
    top1_changed: bool
    kl_divergence: float
    flagged: bool
    trace_a_top1: int
    trace_a_top1_prob: float
    trace_b_top1: int
    trace_b_top1_prob: float


@dataclass
class DiffResult:
    """Complete diff between two traces."""

    trace_a_id: str
    trace_b_id: str
    layer_metrics: list[LayerDiffMetrics]
    critical_heads: list[tuple[int, int, float]]  # (layer, head, js_div)
    first_divergence_layer: int | None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-10:
        return 1.0
    return float(np.dot(a_flat, b_flat) / denom)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with smoothing."""
    eps = 1e-10
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def compute_diff(
    trace_a: TraceResult,
    trace_b: TraceResult,
    cosine_threshold: float = 0.95,
    kl_threshold: float = 0.5,
) -> DiffResult:
    """Compare two traces and compute all divergence metrics."""
    layer_metrics = []
    first_divergence = None
    all_head_divergences: list[tuple[int, int, float]] = []

    for snap_a, snap_b in zip(trace_a.layer_snapshots, trace_b.layer_snapshots):
        # (A) Cosine similarity of residual streams
        cos_sim = 1.0
        if snap_a.residual_out is not None and snap_b.residual_out is not None:
            cos_sim = _cosine_sim(snap_a.residual_out, snap_b.residual_out)

        # (B) Top-1 prediction change
        top1_changed = snap_a.top1_token != snap_b.top1_token

        # (C) KL divergence of attention weight distributions
        kl_div = 0.0
        if snap_a.attention_weights is not None and snap_b.attention_weights is not None:
            a_avg = snap_a.attention_weights.mean(axis=0).flatten()
            b_avg = snap_b.attention_weights.mean(axis=0).flatten()
            kl_div = _kl_divergence(a_avg, b_avg)

        # (D) Combined flagging
        flagged = cos_sim < cosine_threshold or top1_changed or kl_div > kl_threshold

        if flagged and first_divergence is None:
            first_divergence = snap_a.layer_index

        layer_metrics.append(
            LayerDiffMetrics(
                layer_index=snap_a.layer_index,
                cosine_similarity=cos_sim,
                top1_changed=top1_changed,
                kl_divergence=kl_div,
                flagged=flagged,
                trace_a_top1=snap_a.top1_token,
                trace_a_top1_prob=snap_a.top1_prob,
                trace_b_top1=snap_b.top1_token,
                trace_b_top1_prob=snap_b.top1_prob,
            )
        )

        # Per-head JS divergence for critical head detection
        if snap_a.attention_weights is not None and snap_b.attention_weights is not None:
            num_heads = snap_a.attention_weights.shape[0]
            for h in range(num_heads):
                a_head = snap_a.attention_weights[h].flatten()
                b_head = snap_b.attention_weights[h].flatten()
                js = _js_divergence(a_head, b_head)
                all_head_divergences.append((snap_a.layer_index, h, js))

    # Top critical heads by JS divergence
    all_head_divergences.sort(key=lambda x: x[2], reverse=True)
    critical_heads = all_head_divergences[:5]

    return DiffResult(
        trace_a_id=trace_a.metadata.trace_id,
        trace_b_id=trace_b.metadata.trace_id,
        layer_metrics=layer_metrics,
        critical_heads=critical_heads,
        first_divergence_layer=first_divergence,
    )
