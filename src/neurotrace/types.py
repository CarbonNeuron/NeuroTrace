"""Shared types and dataclasses for NeuroTrace."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TraceMetadata:
    """Metadata for a single forward-pass trace."""

    trace_id: str
    model_name: str
    model_revision: str
    prompt: str
    token_ids: list[int]
    tokens: list[str]
    num_layers: int
    num_heads: int
    hidden_size: int
    param_count: int
    device: str
    dtype: str
    random_seed: int
    label: str | None
    capture_mode: str  # "full" or "light"
    layer_stride: int
    timestamp: str


@dataclass
class LayerSnapshot:
    """Captured activations and stats for a single transformer layer."""

    layer_index: int
    # Raw tensors (nullable for light mode / stride skips)
    residual_in: np.ndarray | None
    residual_out: np.ndarray | None
    attention_weights: np.ndarray | None  # [heads, seq, seq]
    attention_output: np.ndarray | None
    mlp_in: np.ndarray | None  # None in light mode
    mlp_out: np.ndarray | None  # None in light mode
    ln_values: np.ndarray | None
    # Pre-computed stats (always present)
    residual_in_norm: float
    residual_out_norm: float
    attention_entropy: list[float]  # per-head
    mlp_activation_mag: float
    top1_token: int
    top1_prob: float


@dataclass
class TokenPrediction:
    """Top-k predictions at a single token position."""

    position: int
    top_k_tokens: list[int]
    top_k_probs: list[float]
    top_k_strings: list[str]


@dataclass
class TraceResult:
    """Complete result of a forward-pass trace."""

    metadata: TraceMetadata
    layer_snapshots: list[LayerSnapshot]
    token_predictions: list[TokenPrediction]
    final_logits: np.ndarray  # [seq_len, vocab_size]
