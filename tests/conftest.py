# tests/conftest.py
import os

import pytest

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def model_available():
    """Check if TinyLlama is cached locally."""
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    )
    return os.path.isdir(cache_dir)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that require a real model",
    )


@pytest.fixture(scope="session")
def tinyllama_model():
    """Load TinyLlama once for the entire test session."""
    if not model_available():
        pytest.skip("TinyLlama model not cached locally")

    from neurotrace.models import load_model

    model, tokenizer = load_model(TINYLLAMA_MODEL)
    return model, tokenizer
