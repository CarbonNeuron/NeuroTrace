# tests/test_models.py
import pytest

from neurotrace.models import ModelArchitecture, get_architecture, load_model

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_llama_architecture_registered():
    arch = get_architecture("llama")
    assert isinstance(arch, ModelArchitecture)
    assert arch.layer_container == "model.layers"
    assert arch.attention_module == "self_attn"
    assert arch.mlp_module == "mlp"


@pytest.mark.slow
def test_load_model():
    model, tokenizer = load_model(TINYLLAMA)
    assert model.config.model_type == "llama"
    assert tokenizer is not None
    assert model.config.num_hidden_layers > 0


@pytest.mark.slow
def test_load_model_no_auth_warning(capsys):
    """load_model should not produce HF authentication warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(TINYLLAMA)
    auth_warnings = [w for w in caught if "unauthenticated" in str(w.message).lower()]
    assert len(auth_warnings) == 0


@pytest.mark.slow
def test_auto_detect_architecture():
    model, _ = load_model(TINYLLAMA)
    arch = get_architecture(model.config.model_type)
    # Verify we can resolve the layer container
    layers = arch.get_layers(model)
    assert len(layers) == model.config.num_hidden_layers
