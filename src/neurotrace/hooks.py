"""PyTorch hook registration and management for activation capture."""

import numpy as np
import torch

from neurotrace.models import ModelArchitecture


class HookManager:
    """Registers forward hooks to capture activations during inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        architecture: ModelArchitecture,
        capture_mode: str = "full",
        layer_stride: int = 1,
    ):
        self._model = model
        self._arch = architecture
        self._capture_mode = capture_mode
        self._layer_stride = layer_stride
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._captured: dict[tuple[int, str], dict] = {}

        self._register_hooks()

    @property
    def num_hooks(self) -> int:
        return len(self._handles)

    def _should_store_tensors(self, layer_index: int) -> bool:
        return layer_index % self._layer_stride == 0

    def _register_hooks(self) -> None:
        layers = self._arch.get_layers(self._model)
        for i, layer in enumerate(layers):
            store_tensors = self._should_store_tensors(i)

            # Hook on the layer itself (residual stream)
            handle = layer.register_forward_hook(
                self._make_residual_hook(i, store_tensors)
            )
            self._handles.append(handle)

            # Hook on attention module
            attn = self._arch.get_attention(layer)
            handle = attn.register_forward_hook(
                self._make_attention_hook(i, store_tensors)
            )
            self._handles.append(handle)

            # Hook on MLP module
            mlp = self._arch.get_mlp(layer)
            handle = mlp.register_forward_hook(self._make_mlp_hook(i, store_tensors))
            self._handles.append(handle)

            # Hooks on layer norms
            for ln in self._arch.get_layer_norms(layer):
                handle = ln.register_forward_hook(self._make_ln_hook(i, store_tensors))
                self._handles.append(handle)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    def _make_residual_hook(self, layer_index: int, store_tensors: bool):
        def hook(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            inp_np = self._to_numpy(inp.squeeze(0))
            out_np = self._to_numpy(out.squeeze(0))
            self._captured[(layer_index, "residual")] = {
                "input_tensor": inp_np if store_tensors else None,
                "output_tensor": out_np if store_tensors else None,
                "input_norm": float(np.linalg.norm(inp_np)),
                "output_norm": float(np.linalg.norm(out_np)),
            }

        return hook

    def _make_attention_hook(self, layer_index: int, store_tensors: bool):
        def hook(module, input, output):
            attn_output = output[0].squeeze(0)
            attn_weights = (
                output[1].squeeze(0)
                if len(output) > 1 and output[1] is not None
                else None
            )
            out_np = self._to_numpy(attn_output)
            weights_np = (
                self._to_numpy(attn_weights) if attn_weights is not None else None
            )

            # Compute per-head entropy from attention weights
            entropy = []
            if weights_np is not None:
                for head_idx in range(weights_np.shape[0]):
                    head_weights = weights_np[head_idx]  # [seq, seq]
                    eps = 1e-10
                    h = -np.sum(head_weights * np.log(head_weights + eps), axis=-1)
                    entropy.append(float(np.mean(h)))

            self._captured[(layer_index, "attention")] = {
                "output_tensor": out_np if store_tensors else None,
                "weights": weights_np if store_tensors else None,
                "entropy": entropy,
            }

        return hook

    def _make_mlp_hook(self, layer_index: int, store_tensors: bool):
        is_full = self._capture_mode == "full"

        def hook(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            out_np = self._to_numpy(out.squeeze(0))
            inp_np = self._to_numpy(inp.squeeze(0))
            self._captured[(layer_index, "mlp")] = {
                "input_tensor": inp_np if (store_tensors and is_full) else None,
                "output_tensor": out_np if (store_tensors and is_full) else None,
                "activation_mag": float(np.linalg.norm(out_np)),
            }

        return hook

    def _make_ln_hook(self, layer_index: int, store_tensors: bool):
        is_full = self._capture_mode == "full"

        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            out_np = self._to_numpy(out.squeeze(0))
            key = (layer_index, "ln")
            if key not in self._captured:
                self._captured[key] = {"values": []}
            self._captured[key]["values"].append(
                out_np if (store_tensors and is_full) else None
            )

        return hook

    def get_captured_data(self) -> dict[tuple[int, str], dict]:
        return self._captured

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()
