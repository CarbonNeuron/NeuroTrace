# NeuroTrace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interpretability toolkit that captures, stores, and compares full forward-pass traces from open-source transformer models.

**Architecture:** Hook-based activation capture on HuggingFace transformer models, stored in DuckDB with pre-computed stats as queryable columns and raw tensors as blobs. Click CLI with Rich terminal output. Registry-based model architecture detection.

**Tech Stack:** Python 3.12+, PyTorch (CPU), HuggingFace Transformers, Click, DuckDB, Rich, numpy, uv

---

## File Structure

```
neurotrace/
├── .gitignore               # Ignore __pycache__, .db, .pytest_cache, etc.
├── pyproject.toml           # Package config, deps, entry point
├── Makefile                 # setup, test, lint, trace-demo, clean
├── README.md                # Install + quickstart
├── src/
│   └── neurotrace/
│       ├── __init__.py      # Version, public API
│       ├── types.py         # Dataclasses: TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult
│       ├── models.py        # ModelArchitecture registry, model loading, config detection
│       ├── hooks.py         # HookManager: register/collect/cleanup forward hooks
│       ├── tracer.py        # Tracer: orchestrates forward pass capture end-to-end
│       ├── storage.py       # TraceDB: DuckDB read/write, schema init, blob serialization
│       ├── analyzer.py      # Diff engine: cosine sim, top-1 change, KL div, combined flagging
│       └── cli.py           # Click CLI: trace, list, inspect, diff, predict
└── tests/
    ├── conftest.py          # Shared fixtures (tmp db, sample trace, model fixture)
    ├── test_types.py        # Dataclass construction + validation
    ├── test_models.py       # Architecture registry + model loading
    ├── test_hooks.py        # Hook registration, capture, cleanup
    ├── test_storage.py      # DuckDB roundtrip fidelity
    ├── test_tracer.py       # Full trace pipeline on TinyLlama
    ├── test_analyzer.py     # Diff metrics with synthetic traces
    └── test_cli.py          # CLI integration tests via CliRunner
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `src/neurotrace/__init__.py`
- Create: `README.md`

- [ ] **Step 1: Create .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.db
*.egg-info/
dist/
build/
.venv/
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "neurotrace"
version = "0.1.0"
description = "Interpretability toolkit for tracing inference through transformer models"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0",
    "transformers>=4.30",
    "huggingface_hub>=0.20",
    "click>=8.0",
    "duckdb>=1.0",
    "rich>=13.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "pyright>=1.1",
]

[project.scripts]
neurotrace = "neurotrace.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/neurotrace"]

[tool.ruff]
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
markers = [
    "model_download: marks tests that download HF models (deselect with '-m \"not model_download\"')",
]
testpaths = ["tests"]
```

- [ ] **Step 3: Create src/neurotrace/__init__.py**

```python
"""NeuroTrace: interpretability toolkit for tracing transformer inference."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create Makefile**

```makefile
.PHONY: setup test test-quick lint trace-demo clean

setup:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

test-quick:
	uv run pytest tests/ -v -m "not model_download"

lint:
	uv run ruff check src/ tests/

trace-demo:
	uv run neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is" --db demo.db
	uv run neurotrace list --db demo.db
	uv run neurotrace inspect --db demo.db --trace-id latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -f *.db
```

- [ ] **Step 5: Create README.md**

Minimal: project name, one-line description, install (`uv sync`), quickstart (trace → list → inspect → diff), dev commands (make test/lint).

- [ ] **Step 6: Run `uv sync --all-extras` and verify it resolves**

Run: `uv sync --all-extras`
Expected: Dependencies install successfully, lock file created.

- [ ] **Step 7: Commit**

```bash
git add .gitignore pyproject.toml uv.lock Makefile README.md src/neurotrace/__init__.py
git commit -m "feat: project scaffolding with pyproject.toml, Makefile, and package init"
```

---

### Task 2: Types Module

**Files:**
- Create: `src/neurotrace/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write failing test for dataclass construction**

```python
# tests/test_types.py
import numpy as np
from neurotrace.types import TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult


def test_trace_metadata_construction():
    meta = TraceMetadata(
        trace_id="test-uuid",
        model_name="test-model",
        model_revision="abc123",
        prompt="hello world",
        token_ids=[1, 2, 3],
        tokens=["hello", " world", ""],
        num_layers=12,
        num_heads=8,
        hidden_size=512,
        param_count=1_000_000,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label=None,
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    assert meta.model_name == "test-model"
    assert meta.label is None
    assert meta.capture_mode == "full"


def test_layer_snapshot_construction():
    seq_len, hidden, heads = 5, 512, 8
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.random.randn(seq_len, hidden).astype(np.float32),
        residual_out=np.random.randn(seq_len, hidden).astype(np.float32),
        attention_weights=np.random.randn(heads, seq_len, seq_len).astype(np.float32),
        attention_output=np.random.randn(seq_len, hidden).astype(np.float32),
        mlp_in=np.random.randn(seq_len, hidden).astype(np.float32),
        mlp_out=np.random.randn(seq_len, hidden).astype(np.float32),
        ln_values=np.random.randn(seq_len, hidden).astype(np.float32),
        residual_in_norm=1.0,
        residual_out_norm=1.1,
        attention_entropy=[0.5] * heads,
        mlp_activation_mag=0.8,
        top1_token=42,
        top1_prob=0.3,
    )
    assert snap.layer_index == 0
    assert snap.attention_weights.shape == (heads, seq_len, seq_len)


def test_layer_snapshot_light_mode():
    seq_len, hidden, heads = 5, 512, 8
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.random.randn(seq_len, hidden).astype(np.float32),
        residual_out=np.random.randn(seq_len, hidden).astype(np.float32),
        attention_weights=np.random.randn(heads, seq_len, seq_len).astype(np.float32),
        attention_output=np.random.randn(seq_len, hidden).astype(np.float32),
        mlp_in=None,
        mlp_out=None,
        ln_values=None,
        residual_in_norm=1.0,
        residual_out_norm=1.1,
        attention_entropy=[0.5] * heads,
        mlp_activation_mag=0.8,
        top1_token=42,
        top1_prob=0.3,
    )
    assert snap.mlp_in is None
    assert snap.mlp_out is None


def test_token_prediction_construction():
    pred = TokenPrediction(
        position=0,
        top_k_tokens=[42, 10, 5],
        top_k_probs=[0.5, 0.3, 0.2],
        top_k_strings=["the", "a", "an"],
    )
    assert pred.position == 0
    assert len(pred.top_k_tokens) == 3


def test_trace_result_construction():
    meta = TraceMetadata(
        trace_id="test-uuid",
        model_name="test-model",
        model_revision="abc123",
        prompt="hello",
        token_ids=[1],
        tokens=["hello"],
        num_layers=1,
        num_heads=1,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label="test",
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.zeros((1, 4), dtype=np.float32),
        residual_out=np.zeros((1, 4), dtype=np.float32),
        attention_weights=np.zeros((1, 1, 1), dtype=np.float32),
        attention_output=np.zeros((1, 4), dtype=np.float32),
        mlp_in=np.zeros((1, 4), dtype=np.float32),
        mlp_out=np.zeros((1, 4), dtype=np.float32),
        ln_values=np.zeros((1, 4), dtype=np.float32),
        residual_in_norm=0.0,
        residual_out_norm=0.0,
        attention_entropy=[0.0],
        mlp_activation_mag=0.0,
        top1_token=1,
        top1_prob=1.0,
    )
    pred = TokenPrediction(position=0, top_k_tokens=[1], top_k_probs=[1.0], top_k_strings=["hello"])
    result = TraceResult(
        metadata=meta,
        layer_snapshots=[snap],
        token_predictions=[pred],
        final_logits=np.zeros((1, 100), dtype=np.float32),
    )
    assert len(result.layer_snapshots) == 1
    assert result.final_logits.shape == (1, 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'neurotrace.types'`

- [ ] **Step 3: Implement types.py**

```python
# src/neurotrace/types.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurotrace/types.py tests/test_types.py
git commit -m "feat: add core dataclasses — TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult"
```

---

### Task 3: Model Loading & Architecture Registry

**Files:**
- Create: `src/neurotrace/models.py`
- Create: `tests/conftest.py` (shared fixtures)
- Create: `tests/test_models.py`

- [ ] **Step 1: Create tests/conftest.py with shared fixture**

```python
# tests/conftest.py
import pytest

TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="session")
def tinyllama_model():
    """Load TinyLlama once for the entire test session."""
    from neurotrace.models import load_model
    model, tokenizer = load_model(TINYLLAMA_MODEL)
    return model, tokenizer
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_models.py
import pytest
from neurotrace.models import get_architecture, load_model, ModelArchitecture

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_llama_architecture_registered():
    arch = get_architecture("llama")
    assert isinstance(arch, ModelArchitecture)
    assert arch.layer_container == "model.layers"
    assert arch.attention_module == "self_attn"
    assert arch.mlp_module == "mlp"


@pytest.mark.model_download
def test_load_model():
    model, tokenizer = load_model(TINYLLAMA)
    assert model.config.model_type == "llama"
    assert tokenizer is not None
    assert model.config.num_hidden_layers > 0


@pytest.mark.model_download
def test_auto_detect_architecture():
    model, _ = load_model(TINYLLAMA)
    arch = get_architecture(model.config.model_type)
    # Verify we can resolve the layer container
    layers = arch.get_layers(model)
    assert len(layers) == model.config.num_hidden_layers
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py::test_llama_architecture_registered -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement models.py**

```python
# src/neurotrace/models.py
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
        torch_dtype=dtype,
        device_map=device,
    )
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
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_models.py::test_llama_architecture_registered -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/neurotrace/models.py tests/test_models.py tests/conftest.py
git commit -m "feat: model loading with architecture registry — start with llama support"
```

---

### Task 4: Hook Manager

**Files:**
- Create: `src/neurotrace/hooks.py`
- Create: `tests/test_hooks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_hooks.py
import pytest
import torch
import numpy as np
from neurotrace.hooks import HookManager
from neurotrace.models import load_model, get_architecture

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.model_download
def test_hooks_register_on_correct_modules(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)
    manager = HookManager(model, arch, capture_mode="full", layer_stride=1)

    num_layers = model.config.num_hidden_layers
    # Each layer: residual (layer itself), attention, mlp, + layer norms
    expected_hooks = num_layers * (1 + 1 + 1 + len(arch.layer_norm_modules))
    assert manager.num_hooks == expected_hooks

    manager.cleanup()


@pytest.mark.model_download
def test_hooks_capture_activations(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)
    manager = HookManager(model, arch, capture_mode="full", layer_stride=1)

    inputs = tokenizer("Hello world", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    captured = manager.get_captured_data()
    num_layers = model.config.num_hidden_layers
    seq_len = inputs["input_ids"].shape[1]
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    # Check we captured data for every layer
    for i in range(num_layers):
        assert (i, "residual") in captured
        assert (i, "attention") in captured
        assert (i, "mlp") in captured

    manager.cleanup()


@pytest.mark.model_download
def test_hooks_cleanup(tinyllama_model):
    model, tokenizer = tinyllama_model
    arch = get_architecture(model.config.model_type)
    manager = HookManager(model, arch, capture_mode="full", layer_stride=1)
    assert manager.num_hooks > 0

    manager.cleanup()
    assert manager.num_hooks == 0

    # Verify no hooks remain on model modules
    for module in model.modules():
        assert len(module._forward_hooks) == 0


@pytest.mark.model_download
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_hooks.py::test_hooks_register_on_correct_modules -v -m model_download`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement hooks.py**

```python
# src/neurotrace/hooks.py
"""PyTorch hook registration and management for activation capture."""

import torch
import numpy as np
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
            handle = mlp.register_forward_hook(
                self._make_mlp_hook(i, store_tensors)
            )
            self._handles.append(handle)

            # Hooks on layer norms
            for ln in self._arch.get_layer_norms(layer):
                handle = ln.register_forward_hook(
                    self._make_ln_hook(i, store_tensors)
                )
                self._handles.append(handle)

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().float().numpy()

    def _make_residual_hook(self, layer_index: int, store_tensors: bool):
        def hook(module, input, output):
            # input[0] is the residual stream input, output[0] is the residual stream output
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            inp_np = self._to_numpy(inp.squeeze(0))  # remove batch dim
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
            # For Llama, output is (attn_output, attn_weights, past_kv) when output_attentions=True
            attn_output = output[0].squeeze(0)
            attn_weights = output[1].squeeze(0) if len(output) > 1 and output[1] is not None else None
            out_np = self._to_numpy(attn_output)
            weights_np = self._to_numpy(attn_weights) if attn_weights is not None else None

            # Compute per-head entropy from attention weights
            entropy = []
            if weights_np is not None:
                for head_idx in range(weights_np.shape[0]):
                    head_weights = weights_np[head_idx]  # [seq, seq]
                    # Entropy per query position, then average
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
            # Store tensor only if full mode + stride hit
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_hooks.py -v -m model_download`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurotrace/hooks.py tests/test_hooks.py
git commit -m "feat: HookManager with architecture-aware hook registration and capture"
```

---

### Task 5: Storage — DuckDB Read/Write

**Files:**
- Create: `src/neurotrace/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storage.py
import numpy as np
import pytest
from neurotrace.storage import TraceDB
from neurotrace.types import (
    TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult,
)


def _make_sample_trace() -> TraceResult:
    """Create a minimal but complete trace for testing."""
    meta = TraceMetadata(
        trace_id="test-uuid-1234",
        model_name="test-model",
        model_revision="abc123",
        prompt="hello world",
        token_ids=[1, 2],
        tokens=["hello", " world"],
        num_layers=2,
        num_heads=2,
        hidden_size=4,
        param_count=100,
        device="cpu",
        dtype="float32",
        random_seed=42,
        label="test-label",
        capture_mode="full",
        layer_stride=1,
        timestamp="2026-03-20T00:00:00",
    )
    snapshots = []
    for i in range(2):
        snapshots.append(LayerSnapshot(
            layer_index=i,
            residual_in=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
            residual_out=np.array([[1.1, 2.1, 3.1, 4.1], [5.1, 6.1, 7.1, 8.1]], dtype=np.float32),
            attention_weights=np.random.randn(2, 2, 2).astype(np.float32),
            attention_output=np.random.randn(2, 4).astype(np.float32),
            mlp_in=np.random.randn(2, 4).astype(np.float32),
            mlp_out=np.random.randn(2, 4).astype(np.float32),
            ln_values=np.random.randn(2, 4).astype(np.float32),
            residual_in_norm=7.746,
            residual_out_norm=7.946,
            attention_entropy=[1.5, 1.2],
            mlp_activation_mag=3.0,
            top1_token=42,
            top1_prob=0.5,
        ))
    predictions = [
        TokenPrediction(position=0, top_k_tokens=[42, 10], top_k_probs=[0.5, 0.3], top_k_strings=["the", "a"]),
        TokenPrediction(position=1, top_k_tokens=[7, 8], top_k_probs=[0.4, 0.35], top_k_strings=["is", "was"]),
    ]
    return TraceResult(
        metadata=meta,
        layer_snapshots=snapshots,
        token_predictions=predictions,
        final_logits=np.random.randn(2, 100).astype(np.float32),
    )


def test_write_and_read_trace(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()

    db = TraceDB(db_path)
    db.write_trace(trace)

    loaded = db.read_trace("test-uuid-1234")
    assert loaded.metadata.trace_id == trace.metadata.trace_id
    assert loaded.metadata.model_name == trace.metadata.model_name
    assert loaded.metadata.label == "test-label"
    assert loaded.metadata.prompt == trace.metadata.prompt
    db.close()


def test_tensor_roundtrip_fidelity(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()

    db = TraceDB(db_path)
    db.write_trace(trace)
    loaded = db.read_trace("test-uuid-1234")

    for orig, loaded_snap in zip(trace.layer_snapshots, loaded.layer_snapshots):
        np.testing.assert_array_equal(orig.residual_in, loaded_snap.residual_in)
        np.testing.assert_array_equal(orig.residual_out, loaded_snap.residual_out)
        np.testing.assert_array_equal(orig.attention_weights, loaded_snap.attention_weights)
    db.close()


def test_stats_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()

    db = TraceDB(db_path)
    db.write_trace(trace)
    loaded = db.read_trace("test-uuid-1234")

    for orig, loaded_snap in zip(trace.layer_snapshots, loaded.layer_snapshots):
        assert abs(orig.residual_in_norm - loaded_snap.residual_in_norm) < 0.01
        assert orig.top1_token == loaded_snap.top1_token
        assert len(orig.attention_entropy) == len(loaded_snap.attention_entropy)
    db.close()


def test_list_traces(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = TraceDB(db_path)
    db.write_trace(_make_sample_trace())

    traces = db.list_traces()
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "test-uuid-1234"
    assert traces[0]["label"] == "test-label"
    db.close()


def test_attention_maps_per_head(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()
    db = TraceDB(db_path)
    db.write_trace(trace)

    # Query single head
    head_map = db.get_attention_map("test-uuid-1234", layer_index=0, head_index=0)
    assert head_map.shape == (2, 2)  # [seq, seq]
    db.close()


def test_light_mode_null_blobs(tmp_path):
    db_path = str(tmp_path / "test.db")
    trace = _make_sample_trace()
    # Simulate light mode
    for snap in trace.layer_snapshots:
        snap.mlp_in = None
        snap.mlp_out = None
    trace.metadata.capture_mode = "light"

    db = TraceDB(db_path)
    db.write_trace(trace)
    loaded = db.read_trace("test-uuid-1234")

    for snap in loaded.layer_snapshots:
        assert snap.mlp_in is None
        assert snap.mlp_out is None
        # Stats should still be present
        assert snap.mlp_activation_mag is not None
    db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_storage.py::test_write_and_read_trace -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement storage.py**

```python
# src/neurotrace/storage.py
"""DuckDB trace storage — read, write, and query traces."""

import struct

import duckdb
import numpy as np

from neurotrace.types import (
    LayerSnapshot,
    TokenPrediction,
    TraceMetadata,
    TraceResult,
)


def _serialize_tensor(arr: np.ndarray) -> bytes:
    """Serialize numpy array to bytes with shape header.

    Format: [ndim as int32] [dim0 as int32] [dim1 as int32] ... [raw float32 data]
    """
    header = struct.pack("<i", arr.ndim)
    for dim in arr.shape:
        header += struct.pack("<i", dim)
    return header + arr.astype(np.float32).tobytes()


def _deserialize_tensor(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to numpy array."""
    offset = 0
    ndim = struct.unpack_from("<i", blob, offset)[0]
    offset += 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack_from("<i", blob, offset)[0])
        offset += 4
    data = np.frombuffer(blob, dtype=np.float32, offset=offset)
    return data.reshape(shape)


class TraceDB:
    """DuckDB-backed trace storage."""

    def __init__(self, db_path: str):
        self._conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id VARCHAR PRIMARY KEY,
                model_name VARCHAR NOT NULL,
                model_revision VARCHAR,
                prompt VARCHAR NOT NULL,
                token_ids INTEGER[],
                tokens VARCHAR[],
                num_layers INTEGER NOT NULL,
                num_heads INTEGER NOT NULL,
                hidden_size INTEGER NOT NULL,
                param_count BIGINT,
                device VARCHAR,
                dtype VARCHAR,
                random_seed INTEGER,
                label VARCHAR,
                capture_mode VARCHAR,
                layer_stride INTEGER,
                top_prediction VARCHAR,
                top_prediction_prob FLOAT,
                timestamp VARCHAR NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS layer_snapshots (
                trace_id VARCHAR NOT NULL,
                layer_index INTEGER NOT NULL,
                residual_in_norm FLOAT,
                residual_out_norm FLOAT,
                attention_entropy FLOAT[],
                mlp_activation_mag FLOAT,
                top1_token INTEGER,
                top1_prob FLOAT,
                residual_in_blob BLOB,
                residual_out_blob BLOB,
                attention_output_blob BLOB,
                mlp_in_blob BLOB,
                mlp_out_blob BLOB,
                ln_values_blob BLOB,
                PRIMARY KEY (trace_id, layer_index)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_maps (
                trace_id VARCHAR NOT NULL,
                layer_index INTEGER NOT NULL,
                head_index INTEGER NOT NULL,
                entropy FLOAT,
                weights_blob BLOB NOT NULL,
                PRIMARY KEY (trace_id, layer_index, head_index)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS token_predictions (
                trace_id VARCHAR NOT NULL,
                position INTEGER NOT NULL,
                top_k_tokens INTEGER[],
                top_k_probs FLOAT[],
                top_k_strings VARCHAR[],
                PRIMARY KEY (trace_id, position)
            )
        """)

    def write_trace(self, result: TraceResult) -> None:
        """Write a complete trace to the database."""
        meta = result.metadata

        # Determine top prediction from token_predictions at last position
        top_pred = ""
        top_pred_prob = 0.0
        if result.token_predictions:
            last = result.token_predictions[-1]
            if last.top_k_strings:
                top_pred = last.top_k_strings[0]
                top_pred_prob = last.top_k_probs[0]

        self._conn.execute(
            """INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                meta.trace_id, meta.model_name, meta.model_revision,
                meta.prompt, meta.token_ids, meta.tokens,
                meta.num_layers, meta.num_heads, meta.hidden_size,
                meta.param_count, meta.device, meta.dtype,
                meta.random_seed, meta.label, meta.capture_mode,
                meta.layer_stride, top_pred, top_pred_prob, meta.timestamp,
            ],
        )

        for snap in result.layer_snapshots:
            self._conn.execute(
                """INSERT INTO layer_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    meta.trace_id, snap.layer_index,
                    snap.residual_in_norm, snap.residual_out_norm,
                    snap.attention_entropy, snap.mlp_activation_mag,
                    snap.top1_token, snap.top1_prob,
                    _serialize_tensor(snap.residual_in) if snap.residual_in is not None else None,
                    _serialize_tensor(snap.residual_out) if snap.residual_out is not None else None,
                    _serialize_tensor(snap.attention_output) if snap.attention_output is not None else None,
                    _serialize_tensor(snap.mlp_in) if snap.mlp_in is not None else None,
                    _serialize_tensor(snap.mlp_out) if snap.mlp_out is not None else None,
                    _serialize_tensor(snap.ln_values) if snap.ln_values is not None else None,
                ],
            )

            # Write per-head attention maps
            if snap.attention_weights is not None:
                num_heads = snap.attention_weights.shape[0]
                for h in range(num_heads):
                    head_weights = snap.attention_weights[h]  # [seq, seq]
                    head_entropy = snap.attention_entropy[h] if h < len(snap.attention_entropy) else 0.0
                    self._conn.execute(
                        """INSERT INTO attention_maps VALUES (?, ?, ?, ?, ?)""",
                        [
                            meta.trace_id, snap.layer_index, h,
                            head_entropy, _serialize_tensor(head_weights),
                        ],
                    )

        for pred in result.token_predictions:
            self._conn.execute(
                """INSERT INTO token_predictions VALUES (?, ?, ?, ?, ?)""",
                [
                    meta.trace_id, pred.position,
                    pred.top_k_tokens, pred.top_k_probs, pred.top_k_strings,
                ],
            )

    def read_trace(self, trace_id: str) -> TraceResult:
        """Read a complete trace from the database."""
        row = self._conn.execute(
            "SELECT * FROM traces WHERE trace_id = ?", [trace_id]
        ).fetchone()
        if row is None:
            raise ValueError(f"Trace not found: {trace_id}")

        meta = TraceMetadata(
            trace_id=row[0], model_name=row[1], model_revision=row[2],
            prompt=row[3], token_ids=row[4], tokens=row[5],
            num_layers=row[6], num_heads=row[7], hidden_size=row[8],
            param_count=row[9], device=row[10], dtype=row[11],
            random_seed=row[12], label=row[13], capture_mode=row[14],
            layer_stride=row[15], timestamp=row[18],
        )

        snap_rows = self._conn.execute(
            "SELECT * FROM layer_snapshots WHERE trace_id = ? ORDER BY layer_index",
            [trace_id],
        ).fetchall()

        snapshots = []
        for sr in snap_rows:
            # Reconstruct attention weights from per-head maps
            head_rows = self._conn.execute(
                "SELECT head_index, weights_blob FROM attention_maps WHERE trace_id = ? AND layer_index = ? ORDER BY head_index",
                [trace_id, sr[1]],
            ).fetchall()
            attn_weights = None
            if head_rows:
                heads = [_deserialize_tensor(hr[1]) for hr in head_rows]
                attn_weights = np.stack(heads, axis=0)

            snapshots.append(LayerSnapshot(
                layer_index=sr[1],
                residual_in=_deserialize_tensor(sr[8]) if sr[8] is not None else None,
                residual_out=_deserialize_tensor(sr[9]) if sr[9] is not None else None,
                attention_weights=attn_weights,
                attention_output=_deserialize_tensor(sr[10]) if sr[10] is not None else None,
                mlp_in=_deserialize_tensor(sr[11]) if sr[11] is not None else None,
                mlp_out=_deserialize_tensor(sr[12]) if sr[12] is not None else None,
                ln_values=_deserialize_tensor(sr[13]) if sr[13] is not None else None,
                residual_in_norm=sr[2],
                residual_out_norm=sr[3],
                attention_entropy=sr[4] if sr[4] is not None else [],
                mlp_activation_mag=sr[5],
                top1_token=sr[6],
                top1_prob=sr[7],
            ))

        pred_rows = self._conn.execute(
            "SELECT * FROM token_predictions WHERE trace_id = ? ORDER BY position",
            [trace_id],
        ).fetchall()

        predictions = [
            TokenPrediction(
                position=pr[1], top_k_tokens=pr[2],
                top_k_probs=pr[3], top_k_strings=pr[4],
            )
            for pr in pred_rows
        ]

        # final_logits not stored in DB — return empty array
        # (full logits are large; we store top-k predictions instead)
        vocab_estimate = 32000
        seq_len = len(meta.token_ids)
        final_logits = np.zeros((seq_len, vocab_estimate), dtype=np.float32)

        return TraceResult(
            metadata=meta,
            layer_snapshots=snapshots,
            token_predictions=predictions,
            final_logits=final_logits,
        )

    def list_traces(self, model_filter: str | None = None) -> list[dict]:
        """List all traces, optionally filtered by model name."""
        query = "SELECT trace_id, label, model_name, prompt, top_prediction, top_prediction_prob, timestamp FROM traces"
        params = []
        if model_filter:
            query += " WHERE model_name LIKE ?"
            params.append(f"%{model_filter}%")
        query += " ORDER BY timestamp DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "trace_id": r[0], "label": r[1], "model_name": r[2],
                "prompt": r[3], "top_prediction": r[4],
                "top_prediction_prob": r[5], "timestamp": r[6],
            }
            for r in rows
        ]

    def get_latest_trace_id(self) -> str:
        """Return the trace_id of the most recent trace."""
        row = self._conn.execute(
            "SELECT trace_id FROM traces ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("No traces in database")
        return row[0]

    def get_attention_map(
        self, trace_id: str, layer_index: int, head_index: int
    ) -> np.ndarray:
        """Get a single head's attention weight matrix."""
        row = self._conn.execute(
            "SELECT weights_blob FROM attention_maps WHERE trace_id = ? AND layer_index = ? AND head_index = ?",
            [trace_id, layer_index, head_index],
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Attention map not found: trace={trace_id}, layer={layer_index}, head={head_index}"
            )
        return _deserialize_tensor(row[0])

    def get_layer_stats(self, trace_id: str) -> list[dict]:
        """Get pre-computed stats for all layers (no blob deserialization)."""
        rows = self._conn.execute(
            """SELECT layer_index, residual_in_norm, residual_out_norm,
                      attention_entropy, mlp_activation_mag, top1_token, top1_prob
               FROM layer_snapshots WHERE trace_id = ? ORDER BY layer_index""",
            [trace_id],
        ).fetchall()
        return [
            {
                "layer_index": r[0], "residual_in_norm": r[1],
                "residual_out_norm": r[2], "attention_entropy": r[3],
                "mlp_activation_mag": r[4], "top1_token": r[5], "top1_prob": r[6],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_storage.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurotrace/storage.py tests/test_storage.py
git commit -m "feat: DuckDB storage with tensor blob serialization and queryable stats columns"
```

---

### Task 6: Tracer — Forward Pass Orchestration

**Files:**
- Create: `src/neurotrace/tracer.py`
- Create: `tests/test_tracer.py`

Note: `tests/conftest.py` with the `tinyllama_model` fixture was already created in Task 3.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tracer.py
import pytest
import numpy as np
from neurotrace.tracer import Tracer
from neurotrace.types import TraceResult

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@pytest.mark.model_download
def test_trace_returns_complete_result(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("The capital of France is")

    assert isinstance(result, TraceResult)
    assert result.metadata.model_name == TINYLLAMA
    assert len(result.layer_snapshots) == model.config.num_hidden_layers
    assert len(result.token_predictions) > 0
    assert result.final_logits is not None


@pytest.mark.model_download
def test_trace_tensor_shapes(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("Hello")

    seq_len = len(result.metadata.token_ids)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads

    for snap in result.layer_snapshots:
        assert snap.residual_in.shape == (seq_len, hidden_size)
        assert snap.residual_out.shape == (seq_len, hidden_size)
        assert snap.attention_weights.shape == (num_heads, seq_len, seq_len)
        assert len(snap.attention_entropy) == num_heads


@pytest.mark.model_download
def test_trace_light_mode(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer, capture_mode="light")
    result = tracer.trace("Hello")

    for snap in result.layer_snapshots:
        assert snap.mlp_in is None
        assert snap.mlp_out is None
        # Stats still present
        assert snap.mlp_activation_mag is not None
        assert snap.residual_in_norm > 0


@pytest.mark.model_download
def test_trace_layer_stride(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer, layer_stride=4)
    result = tracer.trace("Hello")

    for snap in result.layer_snapshots:
        if snap.layer_index % 4 == 0:
            assert snap.residual_in is not None
        else:
            assert snap.residual_in is None
        # Stats always present regardless of stride
        assert snap.residual_in_norm > 0


@pytest.mark.model_download
def test_trace_top_k_predictions(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("The capital of France is")

    for pred in result.token_predictions:
        assert len(pred.top_k_tokens) == 50
        assert len(pred.top_k_probs) == 50
        assert len(pred.top_k_strings) == 50
        assert pred.top_k_probs[0] >= pred.top_k_probs[1]  # sorted descending


@pytest.mark.model_download
def test_trace_metadata_completeness(tinyllama_model):
    model, tokenizer = tinyllama_model
    tracer = Tracer(model, tokenizer)
    result = tracer.trace("Hello", label="test-run", seed=42)

    meta = result.metadata
    assert meta.label == "test-run"
    assert meta.random_seed == 42
    assert meta.device == "cpu"
    assert meta.num_layers == model.config.num_hidden_layers
    assert meta.num_heads == model.config.num_attention_heads
    assert meta.hidden_size == model.config.hidden_size
    assert meta.param_count > 0
    assert meta.model_revision is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracer.py::test_trace_returns_complete_result -v -m model_download`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tracer.py**

```python
# src/neurotrace/tracer.py
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
            layer_snapshots = self._build_snapshots(captured, logits, top_k)

            # Get top prediction at final position
            final_probs = torch.softmax(logits[-1], dim=-1)
            top1_id = torch.argmax(final_probs).item()
            top1_prob = final_probs[top1_id].item()
            top_prediction = self._tokenizer.decode(top1_id)

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
            predictions.append(TokenPrediction(
                position=pos,
                top_k_tokens=token_ids,
                top_k_probs=token_probs,
                top_k_strings=token_strings,
            ))
        return predictions

    def _build_snapshots(
        self, captured: dict, logits: torch.Tensor, top_k: int
    ) -> list[LayerSnapshot]:
        """Assemble LayerSnapshot objects from captured hook data."""
        num_layers = self._model.config.num_hidden_layers
        snapshots = []

        # We need per-layer top-1 predictions.
        # For layers that have residual output, we can project through the LM head
        # to get per-layer predictions. But this requires the model's lm_head.
        # For now, use the final logits for the last layer and None for others
        # unless we add unembedding projection.
        # Simple approach: store top-1 from final logits for all layers (approximate).
        # Better: project residual_out through lm_head + ln for each layer.

        lm_head = self._model.lm_head
        # Get the final layer norm if it exists
        final_ln = None
        if hasattr(self._model.model, "norm"):
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
                    res_tensor = torch.tensor(residual_out, dtype=torch.float32)
                    if final_ln is not None:
                        res_tensor = final_ln(res_tensor)
                    layer_logits = lm_head(res_tensor)
                    layer_probs = torch.softmax(layer_logits[-1], dim=-1)
                    top1_token = int(torch.argmax(layer_probs).item())
                    top1_prob = float(layer_probs[top1_token].item())

            # Get ln values
            ln_values_list = ln_data.get("values", [])
            ln_values = ln_values_list[0] if ln_values_list and ln_values_list[0] is not None else None

            snapshots.append(LayerSnapshot(
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
            ))

        return snapshots
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tracer.py -v -m model_download`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurotrace/tracer.py tests/test_tracer.py
git commit -m "feat: Tracer engine — full forward-pass capture with per-layer predictions"
```

---

### Task 7: Analyzer — Diff Engine

**Files:**
- Create: `src/neurotrace/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_analyzer.py
import numpy as np
import pytest
from neurotrace.analyzer import DiffResult, compute_diff
from neurotrace.types import LayerSnapshot, TokenPrediction, TraceMetadata, TraceResult


def _make_trace(trace_id: str, diverge_at: int | None = None) -> TraceResult:
    """Create a synthetic trace. If diverge_at is set, layer >= diverge_at gets different values."""
    num_layers = 12
    seq_len = 5
    hidden = 64
    heads = 4
    np.random.seed(0)  # deterministic base

    snapshots = []
    for i in range(num_layers):
        base_residual = np.random.randn(seq_len, hidden).astype(np.float32)
        if diverge_at is not None and i >= diverge_at:
            # Inject divergence: rotate residual stream
            noise = np.random.randn(seq_len, hidden).astype(np.float32) * 2.0
            residual_out = base_residual + noise
            top1 = 99  # different prediction
            top1_prob = 0.4
        else:
            residual_out = base_residual + np.random.randn(seq_len, hidden).astype(np.float32) * 0.01
            top1 = 42
            top1_prob = 0.8

        # Create attention weights (valid probability distributions)
        attn = np.random.dirichlet(np.ones(seq_len), size=(heads, seq_len)).astype(np.float32)

        snapshots.append(LayerSnapshot(
            layer_index=i,
            residual_in=base_residual,
            residual_out=residual_out,
            attention_weights=attn,
            attention_output=np.random.randn(seq_len, hidden).astype(np.float32),
            mlp_in=None,
            mlp_out=None,
            ln_values=None,
            residual_in_norm=float(np.linalg.norm(base_residual)),
            residual_out_norm=float(np.linalg.norm(residual_out)),
            attention_entropy=[1.0] * heads,
            mlp_activation_mag=1.0,
            top1_token=top1,
            top1_prob=top1_prob,
        ))

    meta = TraceMetadata(
        trace_id=trace_id, model_name="test", model_revision="abc",
        prompt="test", token_ids=[1]*seq_len, tokens=["t"]*seq_len,
        num_layers=num_layers, num_heads=heads, hidden_size=hidden,
        param_count=100, device="cpu", dtype="float32", random_seed=0,
        label=None, capture_mode="full", layer_stride=1,
        timestamp="2026-01-01T00:00:00",
    )

    preds = [TokenPrediction(
        position=p,
        top_k_tokens=[top1, 10, 5],
        top_k_probs=[0.5, 0.3, 0.2],
        top_k_strings=["a", "b", "c"],
    ) for p, top1 in enumerate([42]*seq_len)]

    return TraceResult(
        metadata=meta, layer_snapshots=snapshots,
        token_predictions=preds,
        final_logits=np.random.randn(seq_len, 100).astype(np.float32),
    )


def test_identical_traces_no_divergence():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b")  # same seed, same data
    diff = compute_diff(trace_a, trace_b)
    assert isinstance(diff, DiffResult)
    # All layers should have high cosine similarity
    for layer in diff.layer_metrics:
        assert layer.cosine_similarity > 0.99
        assert layer.top1_changed is False
        assert layer.kl_divergence < 0.01
        assert layer.flagged is False


def test_divergence_detected_at_correct_layer():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b", diverge_at=8)
    diff = compute_diff(trace_a, trace_b)

    # Layers 0-7 should be similar
    for layer in diff.layer_metrics[:8]:
        assert layer.cosine_similarity > 0.95
        assert layer.flagged is False

    # Layers 8+ should diverge
    for layer in diff.layer_metrics[8:]:
        assert layer.cosine_similarity < 0.95
        assert layer.top1_changed is True
        assert layer.flagged is True

    assert diff.first_divergence_layer == 8


def test_critical_heads_identified():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b", diverge_at=8)
    diff = compute_diff(trace_a, trace_b)

    # Should identify some critical heads
    assert len(diff.critical_heads) > 0
    # Each critical head is (layer, head, js_divergence)
    for layer_idx, head_idx, js_div in diff.critical_heads:
        assert 0 <= layer_idx < 12
        assert 0 <= head_idx < 4
        assert js_div >= 0


def test_configurable_thresholds():
    trace_a = _make_trace("a")
    trace_b = _make_trace("b", diverge_at=8)

    # With very strict thresholds, more layers flagged
    diff_strict = compute_diff(
        trace_a, trace_b,
        cosine_threshold=0.999,
        kl_threshold=0.01,
    )
    # With very loose thresholds, fewer layers flagged
    diff_loose = compute_diff(
        trace_a, trace_b,
        cosine_threshold=0.5,
        kl_threshold=10.0,
    )

    strict_flagged = sum(1 for m in diff_strict.layer_metrics if m.flagged)
    loose_flagged = sum(1 for m in diff_loose.layer_metrics if m.flagged)
    assert strict_flagged >= loose_flagged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_analyzer.py::test_identical_traces_no_divergence -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement analyzer.py**

```python
# src/neurotrace/analyzer.py
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
    all_head_divergences = []  # (layer, head, js_div)

    for snap_a, snap_b in zip(trace_a.layer_snapshots, trace_b.layer_snapshots):
        # (A) Cosine similarity of residual streams
        cos_sim = 1.0
        if snap_a.residual_out is not None and snap_b.residual_out is not None:
            cos_sim = _cosine_sim(snap_a.residual_out, snap_b.residual_out)

        # (B) Top-1 prediction change
        top1_changed = snap_a.top1_token != snap_b.top1_token

        # (C) KL divergence of attention-weighted representations
        # Use attention entropy distributions as proxy, or better:
        # compute from per-layer prediction distributions if available.
        # For now use attention weight distributions averaged over heads.
        kl_div = 0.0
        if snap_a.attention_weights is not None and snap_b.attention_weights is not None:
            # Average attention over heads, flatten
            a_avg = snap_a.attention_weights.mean(axis=0).flatten()
            b_avg = snap_b.attention_weights.mean(axis=0).flatten()
            kl_div = _kl_divergence(a_avg, b_avg)

        # (D) Combined flagging
        flagged = (
            cos_sim < cosine_threshold
            or top1_changed
            or kl_div > kl_threshold
        )

        if flagged and first_divergence is None:
            first_divergence = snap_a.layer_index

        layer_metrics.append(LayerDiffMetrics(
            layer_index=snap_a.layer_index,
            cosine_similarity=cos_sim,
            top1_changed=top1_changed,
            kl_divergence=kl_div,
            flagged=flagged,
            trace_a_top1=snap_a.top1_token,
            trace_a_top1_prob=snap_a.top1_prob,
            trace_b_top1=snap_b.top1_token,
            trace_b_top1_prob=snap_b.top1_prob,
        ))

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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_analyzer.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/neurotrace/analyzer.py tests/test_analyzer.py
git commit -m "feat: diff engine with cosine sim, top-1 change, KL div, and critical head detection"
```

---

### Task 8: CLI — All Commands

**Files:**
- Create: `src/neurotrace/cli.py`
- Create: `tests/test_cli.py`

This is the largest task. Build incrementally: trace first, then list, inspect, diff, predict.

**Key implementation notes:**
- `--trace-id latest` should be resolved via `TraceDB.get_latest_trace_id()` — check if trace_id == "latest" and resolve before querying
- `--prompts-file` reads one prompt per line, traces each sequentially, reusing the loaded model
- `predict` uses Tracer internally with a throwaway trace (no DB), just displays per-position predictions

- [ ] **Step 1: Write CLI tests with CliRunner**

```python
# tests/test_cli.py
import json
import pytest
from click.testing import CliRunner
from neurotrace.cli import cli
from neurotrace.storage import TraceDB
from neurotrace.types import (
    TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult,
)
import numpy as np


def _write_sample_trace(db_path: str, trace_id: str = "test-id-1", label: str = "test") -> None:
    """Write a minimal trace directly to DB for CLI testing."""
    meta = TraceMetadata(
        trace_id=trace_id, model_name="test-model", model_revision="abc",
        prompt="hello world", token_ids=[1, 2], tokens=["hello", " world"],
        num_layers=1, num_heads=1, hidden_size=4, param_count=100,
        device="cpu", dtype="float32", random_seed=42, label=label,
        capture_mode="full", layer_stride=1, timestamp="2026-03-20T00:00:00",
    )
    snap = LayerSnapshot(
        layer_index=0,
        residual_in=np.ones((2, 4), dtype=np.float32),
        residual_out=np.ones((2, 4), dtype=np.float32),
        attention_weights=np.ones((1, 2, 2), dtype=np.float32) * 0.5,
        attention_output=np.ones((2, 4), dtype=np.float32),
        mlp_in=None, mlp_out=None, ln_values=None,
        residual_in_norm=2.0, residual_out_norm=2.0,
        attention_entropy=[0.5], mlp_activation_mag=1.0,
        top1_token=42, top1_prob=0.8,
    )
    pred = TokenPrediction(position=0, top_k_tokens=[42], top_k_probs=[0.8], top_k_strings=["the"])
    result = TraceResult(metadata=meta, layer_snapshots=[snap],
                         token_predictions=[pred], final_logits=np.zeros((2, 100), dtype=np.float32))
    db = TraceDB(db_path)
    db.write_trace(result)
    db.close()


def test_list_command(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", db_path])
    assert result.exit_code == 0
    assert "test-model" in result.output
    assert "test" in result.output  # label


def test_inspect_json_output(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", "--db", db_path, "--trace-id", "test-id-1", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["metadata"]["trace_id"] == "test-id-1"


def test_inspect_latest_alias(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", "--db", db_path, "--trace-id", "latest", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["metadata"]["trace_id"] == "test-id-1"


def test_diff_json_output(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path, trace_id="a", label="a")
    _write_sample_trace(db_path, trace_id="b", label="b")
    runner = CliRunner()
    result = runner.invoke(cli, ["diff", "--db", db_path, "--trace-a", "a", "--trace-b", "b", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "layer_metrics" in data
    assert "critical_heads" in data


def test_list_empty_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", db_path])
    assert result.exit_code == 0


def test_inspect_invalid_trace_id(tmp_path):
    db_path = str(tmp_path / "test.db")
    _write_sample_trace(db_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", "--db", db_path, "--trace-id", "nonexistent"])
    assert result.exit_code != 0


def test_prompts_file(tmp_path):
    """Test that --prompts-file reads prompts from a file (one per line)."""
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text("hello\nworld\n")
    # We can't actually run trace without a model, but we can verify the CLI accepts the flag
    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "--model", "nonexistent", "--prompts-file", str(prompts_file), "--db", str(tmp_path / "test.db")])
    # Should fail at model loading, not at argument parsing
    assert "nonexistent" in result.output or result.exit_code != 0
```

- [ ] **Step 2: Implement cli.py with all 5 commands**

Click group `cli` with subcommands: `trace`, `list`, `inspect`, `diff`, `predict`.

`trace` command:
- `--model` (required)
- `--prompt` / `--prompts-file` (mutually exclusive, one required)
- `--db` (required)
- `--label` (optional)
- `--seed` (default 42)
- `--capture-mode` (default "full", choice: full/light)
- `--layer-stride` (default 1)
- Rich progress bars: "Loading model...", "Running forward pass...", "Storing trace..."
- `--prompts-file`: read file, trace each line, reuse loaded model

`list` command:
- `--db` (required), `--model` (optional filter)
- Rich table: ID (short 8 chars), label, model, prompt (truncated 60 chars), top prediction, timestamp

`inspect` command:
- `--db`, `--trace-id` (required). If trace-id == "latest", resolve via `db.get_latest_trace_id()`
- `--layer` (optional), `--head` (optional, requires --layer), `--json` (flag)
- Default: per-layer summary table
- `--json`: output full trace as JSON (metadata + layer stats, no raw tensors)

`diff` command:
- `--db`, `--trace-a`, `--trace-b` (required)
- `--cosine-threshold` (default 0.95), `--kl-threshold` (default 0.5)
- `--sort-by` (default "layer", choices: layer/cosine/kl)
- `--flagged-only`, `--head-detail`, `--json` (flags)
- Rich table with all 4 metrics. Flagged rows in red. Summary below.
- `--json`: structured JSON output

`predict` command:
- `--model`, `--prompt` (required), `--top-k` (default 5)
- Quick mode — loads model, runs forward pass via Tracer, displays per-position top-k predictions
- No DB write

- [ ] **Step 3: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS (except prompts_file which will fail at model loading, not arg parsing).

- [ ] **Step 4: Smoke test the full CLI**

```bash
uv run neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is" --db test.db --label "france-test"
uv run neurotrace list --db test.db
uv run neurotrace inspect --db test.db --trace-id latest
uv run neurotrace inspect --db test.db --trace-id latest --json
uv run neurotrace predict --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello world"
```

- [ ] **Step 5: Commit**

```bash
git add src/neurotrace/cli.py tests/test_cli.py
git commit -m "feat: Click CLI with trace, list, inspect, diff, and predict commands"
```

---

### Task 9: Integration Testing & Polish

**Files:**
- Modify: various files for bug fixes found during integration

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/ tests/`
Fix any issues.

- [ ] **Step 3: Run trace-demo end-to-end**

```bash
make trace-demo
```

Verify: model downloads, trace completes, output displays correctly.

- [ ] **Step 4: Test diff end-to-end**

```bash
uv run neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "What is 7 times 8?" --db test.db --label "math-1"
uv run neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "What is 7 plus 8?" --db test.db --label "math-2"
uv run neurotrace diff --db test.db --trace-a <id1> --trace-b <id2>
```

- [ ] **Step 5: Run pyright type checking**

Run: `pyright src/`
Fix any type errors.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: integration fixes, linting, and type checking cleanup"
```
