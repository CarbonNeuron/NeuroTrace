# tests/conftest.py
import pytest

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="session")
def tinyllama_model():
    """Load TinyLlama once for the entire test session."""
    from neurotrace.models import load_model
    model, tokenizer = load_model(TINYLLAMA_MODEL)
    return model, tokenizer
