"""Core forward-pass tracing engine."""

import uuid
from datetime import datetime, timezone

import numpy as np
import torch

from neurotrace.hooks import HookManager
from neurotrace.models import (
    ModelArchitecture,
    count_parameters,
    get_architecture,
    get_model_revision,
)
from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)


class Tracer:
    """Orchestrates a full forward-pass trace through a transformer model."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        capture_mode: str = "full",
        layer_stride: int = 1,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._capture_mode = capture_mode
        self._layer_stride = layer_stride
        self._arch = get_architecture(model.config.model_type)

    def trace(
        self,
        prompt: str,
        label: str | None = None,
        seed: int = 42,
        top_k: int = 50,
    ) -> TraceResult:
        """Run a forward pass and capture all activations."""
        torch.manual_seed(seed)

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        token_ids = inputs["input_ids"].squeeze(0).tolist()
        tokens = [self._tokenizer.decode(tid) for tid in token_ids]

        # Enable attention output on model config temporarily.
        # Must switch to eager attention implementation to support output_attentions.
        old_output_attentions = getattr(
            self._model.config, "output_attentions", False
        )
        old_attn_implementation = getattr(
            self._model.config, "_attn_implementation", None
        )
        self._model.config._attn_implementation = "eager"
        self._model.config.output_attentions = True

        # Register hooks
        manager = HookManager(
            self._model, self._arch, self._capture_mode, self._layer_stride
        )

        try:
            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs, output_attentions=True)

            logits = outputs.logits.squeeze(0)  # [seq_len, vocab]
            logits_np = logits.cpu().float().numpy()

            # Build per-position predictions
            token_predictions = self._extract_predictions(logits, top_k)

            # Build layer snapshots from captured data
            captured = manager.get_captured_data()
            layer_snapshots = self._build_snapshots(captured)

            # Build metadata
            model_name = self._model.config._name_or_path
            try:
                revision = get_model_revision(model_name)
            except Exception:
                revision = "unknown"

            metadata = TraceMetadata(
                trace_id=str(uuid.uuid4()),
                model_name=model_name,
                model_revision=revision,
                prompt=prompt,
                token_ids=token_ids,
                tokens=tokens,
                num_layers=self._model.config.num_hidden_layers,
                num_heads=self._model.config.num_attention_heads,
                hidden_size=self._model.config.hidden_size,
                param_count=count_parameters(self._model),
                device=str(self._model.device),
                dtype=str(self._model.dtype),
                random_seed=seed,
                label=label,
                capture_mode=self._capture_mode,
                layer_stride=self._layer_stride,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            return TraceResult(
                metadata=metadata,
                layer_snapshots=layer_snapshots,
                token_predictions=token_predictions,
                final_logits=logits_np,
            )
        finally:
            manager.cleanup()
            # Restore original config values
            self._model.config._attn_implementation = old_attn_implementation
            self._model.config.output_attentions = old_output_attentions

    def _extract_predictions(
        self, logits: torch.Tensor, top_k: int
    ) -> list[TokenPrediction]:
        """Extract top-k predictions at each token position."""
        probs = torch.softmax(logits, dim=-1)
        predictions = []
        for pos in range(logits.shape[0]):
            topk = torch.topk(probs[pos], k=top_k)
            token_ids = topk.indices.tolist()
            token_probs = topk.values.tolist()
            token_strings = [self._tokenizer.decode(tid) for tid in token_ids]
            predictions.append(
                TokenPrediction(
                    position=pos,
                    top_k_tokens=token_ids,
                    top_k_probs=token_probs,
                    top_k_strings=token_strings,
                )
            )
        return predictions

    def _build_snapshots(self, captured: dict) -> list[LayerSnapshot]:
        """Assemble LayerSnapshot objects from captured hook data."""
        num_layers = self._model.config.num_hidden_layers
        snapshots = []

        lm_head = self._model.lm_head
        # Get the final layer norm if it exists
        final_ln = None
        if hasattr(self._model, "model") and hasattr(self._model.model, "norm"):
            final_ln = self._model.model.norm

        for i in range(num_layers):
            residual_data = captured.get((i, "residual"), {})
            attn_data = captured.get((i, "attention"), {})
            mlp_data = captured.get((i, "mlp"), {})
            ln_data = captured.get((i, "ln"), {})

            # Compute per-layer top-1 by projecting residual_out through lm_head
            residual_out = residual_data.get("output_tensor")
            top1_token = 0
            top1_prob = 0.0
            if residual_out is not None and lm_head is not None:
                with torch.no_grad():
                    res_tensor = torch.tensor(
                        residual_out, dtype=torch.float32
                    ).unsqueeze(0)
                    if final_ln is not None:
                        res_tensor = final_ln(res_tensor)
                    layer_logits = lm_head(res_tensor.squeeze(0))
                    layer_probs = torch.softmax(layer_logits[-1], dim=-1)
                    top1_token = int(torch.argmax(layer_probs).item())
                    top1_prob = float(layer_probs[top1_token].item())

            # Get ln values
            ln_values_list = ln_data.get("values", [])
            ln_values = (
                ln_values_list[0]
                if ln_values_list and ln_values_list[0] is not None
                else None
            )

            snapshots.append(
                LayerSnapshot(
                    layer_index=i,
                    residual_in=residual_data.get("input_tensor"),
                    residual_out=residual_data.get("output_tensor"),
                    attention_weights=attn_data.get("weights"),
                    attention_output=attn_data.get("output_tensor"),
                    mlp_in=mlp_data.get("input_tensor"),
                    mlp_out=mlp_data.get("output_tensor"),
                    ln_values=ln_values,
                    residual_in_norm=residual_data.get("input_norm", 0.0),
                    residual_out_norm=residual_data.get("output_norm", 0.0),
                    attention_entropy=attn_data.get("entropy", []),
                    mlp_activation_mag=mlp_data.get("activation_mag", 0.0),
                    top1_token=top1_token,
                    top1_prob=top1_prob,
                )
            )

        return snapshots
