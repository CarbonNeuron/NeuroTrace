import pytest
import torch

from neurotrace.hooks import HookManager
from neurotrace.models import get_architecture

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.slow
def test_hooks_register_on_correct_modules(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)
    manager = HookManager(model, arch, capture_mode="full", layer_stride=1)

    num_layers = model.config.num_hidden_layers
    # Each layer: residual (layer itself), attention, mlp, + layer norms
    expected_hooks = num_layers * (1 + 1 + 1 + len(arch.layer_norm_modules))
    assert manager.num_hooks == expected_hooks

    manager.cleanup()


@pytest.mark.slow
def test_hooks_capture_activations(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)
    manager = HookManager(model, arch, capture_mode="full", layer_stride=1)

    inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs, output_attentions=True)

    captured = manager.get_captured_data()
    num_layers = model.config.num_hidden_layers
    inputs["input_ids"].shape[1]

    # Check we captured data for every layer
    for i in range(num_layers):
        assert (i, "residual") in captured
        assert (i, "attention") in captured
        assert (i, "mlp") in captured

    manager.cleanup()


@pytest.mark.slow
def test_hooks_cleanup(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)

    # Record pre-existing hook counts (HF may install its own hooks)
    pre_hook_counts = {id(m): len(m._forward_hooks) for m in model.modules()}

    manager = HookManager(model, arch, capture_mode="full", layer_stride=1)
    assert manager.num_hooks > 0

    manager.cleanup()
    assert manager.num_hooks == 0

    # Verify our hooks were removed — counts should return to pre-existing levels
    for module in model.modules():
        assert len(module._forward_hooks) == pre_hook_counts[id(module)]


@pytest.mark.slow
def test_light_mode_skips_mlp_tensors(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)
    manager = HookManager(model, arch, capture_mode="light", layer_stride=1)

    inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs, output_attentions=True)

    captured = manager.get_captured_data()
    # In light mode, MLP captures should have stats but no raw tensors
    for i in range(model.config.num_hidden_layers):
        mlp_data = captured[(i, "mlp")]
        assert mlp_data["output_tensor"] is None
        assert mlp_data["input_tensor"] is None
        assert mlp_data["activation_mag"] is not None

    manager.cleanup()
