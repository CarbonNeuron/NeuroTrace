"""Model loading and architecture detection for NeuroTrace."""

from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelArchitecture:
    """Describes how to find components in a transformer model."""

    layer_container: str  # e.g. "model.layers"
    attention_module: str  # e.g. "self_attn"
    mlp_module: str  # e.g. "mlp"
    layer_norm_modules: list[str] = field(default_factory=list)

    def get_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Resolve the layer container and return list of layers."""
        container = model
        for attr in self.layer_container.split("."):
            container = getattr(container, attr)
        return list(container)

    def get_attention(self, layer: torch.nn.Module) -> torch.nn.Module:
        return getattr(layer, self.attention_module)

    def get_mlp(self, layer: torch.nn.Module) -> torch.nn.Module:
        return getattr(layer, self.mlp_module)

    def get_layer_norms(self, layer: torch.nn.Module) -> list[torch.nn.Module]:
        return [getattr(layer, name) for name in self.layer_norm_modules]


# Registry: model_type string -> ModelArchitecture
ARCHITECTURE_REGISTRY: dict[str, ModelArchitecture] = {
    "llama": ModelArchitecture(
        layer_container="model.layers",
        attention_module="self_attn",
        mlp_module="mlp",
        layer_norm_modules=["input_layernorm", "post_attention_layernorm"],
    ),
}


def get_architecture(model_type: str) -> ModelArchitecture:
    """Look up architecture config for a model type."""
    if model_type not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unsupported model type: {model_type!r}. "
            f"Supported: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    return ARCHITECTURE_REGISTRY[model_type]


def load_model(
    model_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load a HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def get_model_revision(model_name: str) -> str:
    """Get the HuggingFace commit hash for the model."""
    from huggingface_hub import model_info
    info = model_info(model_name)
    return info.sha


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())
