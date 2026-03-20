# NeuroTrace — Project Overview

Reference document for AI agents writing prompts against this codebase.

## 1. File Tree

```
.
├── docs/superpowers/plans/
│   └── 2026-03-20-neurotrace-build.md
├── src/neurotrace/
│   ├── __init__.py          # version only (__version__ = "0.1.0")
│   ├── types.py             # shared dataclasses
│   ├── models.py            # model loading + architecture registry
│   ├── hooks.py             # activation capture hooks
│   ├── tracer.py            # forward-pass tracing engine
│   ├── storage.py           # DuckDB read/write/query
│   ├── analyzer.py          # trace diff / divergence detection
│   ├── ablate.py            # ablation engine (zero/scale interventions)
│   ├── scan.py              # automated sabotage detection across datasets
│   ├── datasets.py          # built-in evaluation datasets
│   ├── neurons.py           # neuron-level MLP attribution (profile/ablate)
│   ├── finetune.py          # LoRA fine-tuning engine
│   ├── report.py            # self-contained HTML report generation
│   ├── upload.py            # CarbonFiles upload
│   └── cli.py               # Click CLI (all commands)
├── tests/
│   ├── conftest.py          # session-scoped TinyLlama fixture
│   ├── test_types.py
│   ├── test_models.py
│   ├── test_hooks.py
│   ├── test_tracer.py
│   ├── test_storage.py
│   ├── test_analyzer.py
│   ├── test_ablate.py
│   ├── test_report.py
│   ├── test_upload.py
│   ├── test_scan.py
│   ├── test_neurons.py
│   ├── test_finetune.py
│   └── test_cli.py
├── .gitignore
├── Makefile
├── pyproject.toml
├── pyrightconfig.json
├── README.md
└── uv.lock
```

## 2. Module Descriptions

### `types.py` — Shared dataclasses for trace data

```python
@dataclass
class TraceMetadata:
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
    capture_mode: str          # "full" or "light"
    layer_stride: int
    timestamp: str             # ISO 8601

@dataclass
class LayerSnapshot:
    layer_index: int
    residual_in: np.ndarray | None       # [seq_len, hidden_size]
    residual_out: np.ndarray | None      # [seq_len, hidden_size]
    attention_weights: np.ndarray | None # [num_heads, seq_len, seq_len]
    attention_output: np.ndarray | None  # [seq_len, hidden_size]
    mlp_in: np.ndarray | None           # None in light mode
    mlp_out: np.ndarray | None          # None in light mode
    ln_values: np.ndarray | None
    residual_in_norm: float
    residual_out_norm: float
    attention_entropy: list[float]       # per-head
    mlp_activation_mag: float
    top1_token: int
    top1_prob: float

@dataclass
class TokenPrediction:
    position: int
    top_k_tokens: list[int]
    top_k_probs: list[float]
    top_k_strings: list[str]

@dataclass
class TraceResult:
    metadata: TraceMetadata
    layer_snapshots: list[LayerSnapshot]
    token_predictions: list[TokenPrediction]
    final_logits: np.ndarray             # [seq_len, vocab_size]
```

**Key details:**
- All numpy arrays are float32.
- `capture_mode="light"` sets `mlp_in`, `mlp_out`, `ln_values` to None.
- `layer_stride > 1` sets raw tensor fields to None for skipped layers (stats are always present).
- `top1_token` / `top1_prob` are computed by projecting `residual_out` through the model's `lm_head` (with final layer norm applied).

---

### `models.py` — Model loading and architecture registry

```python
@dataclass
class ModelArchitecture:
    layer_container: str              # e.g. "model.layers"
    attention_module: str             # e.g. "self_attn"
    mlp_module: str                   # e.g. "mlp"
    layer_norm_modules: list[str]     # e.g. ["input_layernorm", "post_attention_layernorm"]

    def get_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]
    def get_attention(self, layer: torch.nn.Module) -> torch.nn.Module
    def get_mlp(self, layer: torch.nn.Module) -> torch.nn.Module
    def get_layer_norms(self, layer: torch.nn.Module) -> list[torch.nn.Module]

ARCHITECTURE_REGISTRY: dict[str, ModelArchitecture]
# Currently registered: "llama"

def get_architecture(model_type: str) -> ModelArchitecture
def load_model(model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.nn.Module, AutoTokenizer]
def get_model_revision(model_name: str) -> str
def count_parameters(model: torch.nn.Module) -> int
def get_lm_head_and_norm(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module | None]
def get_final_prediction(token_predictions: list) -> tuple[str, float]
```

**Key details:**
- `load_model` passes `token=False` to avoid HuggingFace auth prompts.
- Only `"llama"` architecture is registered. Layer container is `model.model.layers`.
- Llama attention module: `self_attn`. MLP module: `mlp`. Layer norms: `input_layernorm`, `post_attention_layernorm`.
- Architecture detection uses `model.config.model_type` (e.g. TinyLlama reports `"llama"`).
- `get_lm_head_and_norm()` extracts `model.lm_head` and `model.model.norm` — shared utility used by tracer, ablate, scan, and CLI.
- `get_final_prediction()` gets the last-position top-1 prediction from token predictions — shared by ablate and neurons modules.

---

### `hooks.py` — PyTorch forward hook registration for activation capture

```python
class HookManager:
    def __init__(self, model: torch.nn.Module, architecture: ModelArchitecture, capture_mode: str = "full", layer_stride: int = 1)

    @property
    def num_hooks(self) -> int
    def get_captured_data(self) -> dict[tuple[int, str], dict]
    def cleanup(self) -> None
```

**Key details:**
- Hooks are registered in `__init__` on every layer, immediately active.
- Per layer, registers hooks on: the layer itself (residual), `self_attn` (attention), `mlp` (MLP), and each layer norm.
- Captured data keyed by `(layer_index, component)` where component is `"residual"`, `"attention"`, `"mlp"`, or `"ln"`.
- Residual hook captures: `input_tensor`, `output_tensor`, `input_norm`, `output_norm`.
- Attention hook captures: `output_tensor`, `weights` (from `output[1]`), `entropy` (per-head, computed as mean of `-sum(w * log(w))`).
- MLP hook captures: `input_tensor`, `output_tensor` (both None in light mode), `activation_mag`.
- LN hook captures: `values` (list, None in light mode).
- `layer_stride` controls whether raw tensors are stored (stats always computed).
- `cleanup()` removes all hooks and clears captured data.

---

### `tracer.py` — Core forward-pass tracing engine

```python
class Tracer:
    def __init__(self, model: torch.nn.Module, tokenizer, capture_mode: str = "full", layer_stride: int = 1)
    def trace(self, prompt: str, label: str | None = None, seed: int = 42, top_k: int = 50) -> TraceResult
```

**Key details:**
- `trace()` temporarily sets `model.config.output_attentions = True` and `model.config._attn_implementation = "eager"` (restores original values in `finally`).
- Registers a `HookManager`, runs forward pass with `torch.no_grad()`, then cleans up hooks.
- Generates `trace_id` as `uuid4()`.
- Per-layer `top1_token` / `top1_prob` computed by projecting `residual_out` through `model.model.norm` (final LN) then `model.lm_head`.
- `final_logits` shape: `[seq_len, vocab_size]`.
- `token_predictions` has one entry per token position with top-k tokens.

---

### `storage.py` — DuckDB trace storage

```python
class TraceDB:
    def __init__(self, db_path: str)
    def write_trace(self, result: TraceResult, interventions: str | None = None) -> None
    def read_trace(self, trace_id: str) -> TraceResult
    def list_traces(self, model_filter: str | None = None) -> list[dict]
    def resolve_trace_id(self, identifier: str) -> str
    def get_latest_trace_id(self) -> str
    def get_attention_map(self, trace_id: str, layer_index: int, head_index: int) -> np.ndarray
    def get_layer_stats(self, trace_id: str) -> list[dict]
    def find_existing_trace(self, model_name: str, prompt: str, seed: int, capture_mode: str) -> str | None
    def get_interventions(self, trace_id: str) -> str | None
    def close(self) -> None
```

**Key details:**
- Uses DuckDB, not SQLite. Single `.db` file.
- Tensor serialization: custom binary format with `[ndim:int32][dim0:int32]...[float32 data]`.
- `final_logits` is NOT stored in DB. On `read_trace`, returns a zero array of shape `(seq_len, 32000)`.
- `resolve_trace_id` tries (in order): `"latest"` alias → exact `trace_id` match → label match → prefix match. Raises `ValueError` on ambiguous or not found.
- `write_trace` computes `top_prediction` / `top_prediction_prob` from the last `TokenPrediction` for the traces table.
- `interventions` column was added via migration (`ALTER TABLE ... ADD COLUMN`), so old DBs auto-upgrade.
- `get_layer_stats` returns dicts with keys: `layer_index`, `residual_in_norm`, `residual_out_norm`, `attention_entropy`, `mlp_activation_mag`, `top1_token`, `top1_prob`.
- `list_traces` returns dicts with keys: `trace_id`, `label`, `model_name`, `prompt`, `top_prediction`, `top_prediction_prob`, `timestamp`.

---

### `analyzer.py` — Trace diff / divergence detection

```python
@dataclass
class LayerDiffMetrics:
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
    trace_a_id: str
    trace_b_id: str
    layer_metrics: list[LayerDiffMetrics]
    critical_heads: list[tuple[int, int, float]]  # (layer, head, js_div)
    first_divergence_layer: int | None

def compute_diff(trace_a: TraceResult, trace_b: TraceResult, cosine_threshold: float = 0.95, kl_threshold: float = 0.5) -> DiffResult
```

**Key details:**
- 4 metrics per layer: cosine similarity of residual streams, top-1 prediction change, KL divergence of average attention weights, combined flagging.
- Flagged if: `cos_sim < cosine_threshold OR top1_changed OR kl_div > kl_threshold`.
- Critical heads: top 5 heads by Jensen-Shannon divergence across all layers.
- When traces have different sequence lengths, cosine similarity compares last position only.

---

### `ablate.py` — Ablation engine

```python
@dataclass
class AblationSpec:
    zero_layers: list[int]
    zero_heads: list[tuple[int, int]]      # (layer, head)
    scale_layers: list[tuple[int, float]]   # (layer, factor)
    zero_mlp: list[int]                     # defaults to [] via __post_init__

    def to_json(self) -> str
    def describe(self) -> str

@dataclass
class LayerComparison:
    layer_index: int
    baseline_top1: str
    baseline_top1_prob: float
    ablated_top1: str
    ablated_top1_prob: float
    cosine_similarity: float
    changed: bool

@dataclass
class AblationResult:
    baseline_trace: TraceResult
    ablated_trace: TraceResult
    spec: AblationSpec
    baseline_final_token: str
    baseline_final_prob: float
    ablated_final_token: str
    ablated_final_prob: float
    layer_comparisons: list[LayerComparison]

def parse_zero_layers(value: str) -> list[int]
def parse_zero_heads(value: str) -> list[tuple[int, int]]
def parse_scale_layers(value: str) -> list[tuple[int, float]]

class AblationHookManager:
    def __init__(self, model: torch.nn.Module, architecture: ModelArchitecture, spec: AblationSpec)
    @property
    def num_hooks(self) -> int
    def cleanup(self) -> None

def run_ablation(model: torch.nn.Module, tokenizer, prompt: str, spec: AblationSpec, baseline: TraceResult | None = None, label: str | None = None, seed: int = 42) -> AblationResult
```

**Key details:**
- 4 intervention types: zero entire layer, zero specific attention heads, scale layer contribution, zero MLP sublayer.
- Layer zeroing hook: extracts `contribution = output[0] - input[0]`, returns `input + factor * contribution`. Factor=0.0 for zeroing.
- Head zeroing hook: reshapes attention output to `(batch, seq_len, num_heads, head_dim)`, zeros targeted heads, reshapes back.
- MLP zeroing hook: replaces MLP output with `torch.zeros_like(output)`.
- Hooks registered on: `model.model.layers[L]` (layer), `model.model.layers[L].self_attn` (heads), `model.model.layers[L].mlp` (MLP).
- `run_ablation` optionally runs baseline trace first, then ablated trace with hooks active, then compares per-layer using `_compare_traces`.
- `to_json()` format: `{"zero_layers": [int], "zero_heads": [[layer, head]], "scale_layers": [[layer, factor]], "zero_mlp": [int]}`.
- `describe()` format: `"zero-layers=20,21; zero-heads=20:7; scale-layers=21:0.5; zero-mlp=20"`.

---

### `report.py` — Self-contained HTML report generation

```python
def generate_report(trace: TraceResult, layer_stats: list[dict], layer_predictions: list[dict] | None = None, token_tracks: list[dict] | None = None, full_attention: bool = False, no_attention: bool = False) -> str

def generate_comparison_report(trace_a: TraceResult, trace_b: TraceResult, stats_a: list[dict], stats_b: list[dict], preds_a: list[dict] | None, preds_b: list[dict] | None, tracks_a: list[dict] | None, tracks_b: list[dict] | None, diff_result: DiffResult, token_lookup: dict[int, str], full_attention: bool = False, no_attention: bool = False) -> str
```

**Key details:**
- Returns a complete HTML string with inline CSS and inline SVG charts. No external dependencies.
- Dark theme ("Refined Ocean"). Print-friendly `@media print` styles.
- Charts rendered as SVG: line charts (residual norms, entropy, token tracking) and attention heatmaps.
- `_sample_layers` picks ~6 representative layers for attention heatmaps (first, last, evenly spaced).
- `full_attention=True` shows all layers. `no_attention=True` skips heatmaps entirely.
- `layer_predictions` / `token_tracks` are computed in `cli.py:_compute_layer_predictions()` by projecting residual_out through lm_head. These are optional — report degrades gracefully if None.

---

### `upload.py` — CarbonFiles upload

```python
def upload_report(html_path: str, bucket_id: str | None = None) -> str
```

**Key details:**
- Requires `carbonfiles-client` package (optional dependency group `upload`).
- Requires `CF_URL` and `CF_API_KEY` environment variables.
- If `bucket_id` is None, creates a new bucket named `"neurotrace-report"` with 7-day expiry.
- Returns URL: `{CF_URL}/api/buckets/{bucket_id}/files/{filename}/content`.

---

### `cli.py` — Click CLI entry point

The CLI group is `cli()`, exposed as `neurotrace` console script. Uses Rich for table/progress output. Stderr for progress spinners, stdout for data.

Helper functions:
```python
def _resolve_trace_id(db: TraceDB, trace_id: str) -> str
def _slugify(text: str, max_len: int = 30) -> str
def _decode_tokens(tokenizer, token_ids: list[int]) -> list[dict]
def _compute_layer_predictions(result, top_k: int = 5) -> tuple[list[dict] | None, list[dict] | None]
```

`_compute_layer_predictions` loads the model, projects each layer's `residual_out` through `model.model.norm` + `lm_head`, returns `(layer_predictions, token_tracks)`. Returns `(None, None)` if no residuals or model can't be loaded.

## 3. Data Schema

### DuckDB Tables

```sql
CREATE TABLE traces (
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
    timestamp VARCHAR NOT NULL,
    interventions VARCHAR           -- JSON string or NULL
);

CREATE TABLE layer_snapshots (
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
);

CREATE TABLE attention_maps (
    trace_id VARCHAR NOT NULL,
    layer_index INTEGER NOT NULL,
    head_index INTEGER NOT NULL,
    entropy FLOAT,
    weights_blob BLOB NOT NULL,
    PRIMARY KEY (trace_id, layer_index, head_index)
);

CREATE TABLE token_predictions (
    trace_id VARCHAR NOT NULL,
    position INTEGER NOT NULL,
    top_k_tokens INTEGER[],
    top_k_probs FLOAT[],
    top_k_strings VARCHAR[],
    PRIMARY KEY (trace_id, position)
);
```

### Migration

```sql
ALTER TABLE traces ADD COLUMN interventions VARCHAR;
-- Applied idempotently on every TraceDB.__init__; catches CatalogException if column exists.
```

### Tensor Blob Format

Binary: `[ndim as int32][dim0 as int32][dim1 as int32]...[raw float32 data]`. Little-endian.

### Interventions JSON Format

```json
{
    "zero_layers": [20, 21],
    "zero_heads": [[20, 7], [20, 12]],
    "scale_layers": [[21, 0.5]],
    "zero_mlp": [20]
}
```

Stored as a VARCHAR in the `traces.interventions` column. NULL for non-ablated traces.

## 4. CLI Commands

All commands are under the `neurotrace` entry point.

### `neurotrace trace`

Run a forward-pass trace and store results.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | HuggingFace model name or path |
| `--prompt` | str | None | Prompt text (mutually exclusive with `--prompts-file`) |
| `--prompts-file` | path | None | File with one prompt per line |
| `--db` | str | **required** | DuckDB database file path |
| `--label` | str | None | Optional label for the trace |
| `--seed` | int | 42 | Random seed |
| `--capture-mode` | Choice[full, light] | full | Capture mode |
| `--layer-stride` | int | 1 | Layer stride for tensor capture |

### `neurotrace list`

List all stored traces.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--db` | str | **required** | DuckDB database file path |
| `--model` | str | None | Filter by model name (substring match) |

### `neurotrace inspect`

Inspect a stored trace — shows metadata and layer stats table.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--db` | str | **required** | DuckDB database file path |
| `--trace-id` | str | **required** | Trace ID, label, prefix, or `"latest"` |
| `--layer` | int | None | Show only this layer |
| `--head` | int | None | Show specific head (requires `--layer`) |
| `--json` | flag | False | Output as JSON |

### `neurotrace diff`

Compare two traces and show divergence metrics.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--db` | str | **required** | DuckDB database file path |
| `--trace-a` | str | **required** | First trace ID |
| `--trace-b` | str | **required** | Second trace ID |
| `--cosine-threshold` | float | 0.95 | Cosine sim threshold for flagging |
| `--kl-threshold` | float | 0.5 | KL divergence threshold for flagging |
| `--sort-by` | Choice[layer, cosine, kl] | layer | Sort order |
| `--flagged-only` | flag | False | Show only flagged layers |
| `--head-detail` | flag | False | Show critical head details |
| `--json` | flag | False | Output as JSON |

### `neurotrace predict`

Show top-K token predictions at every layer from a stored trace. Requires model loading to project residual_out through lm_head.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--db` | str | **required** | DuckDB database file path |
| `--trace-id` | str | **required** | Trace ID, label, or `"latest"` |
| `--top-k` | int | 5 | Top predictions per layer |
| `--changes-only` | flag | False | Only show layers where top-1 changed |
| `--layers` | str | None | Comma-separated layer indices |
| `--track` | str | None | Token string to track across layers |
| `--json` | flag | False | Output as JSON |

### `neurotrace decode`

Decode token IDs to human-readable strings.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | HuggingFace model name or path |
| `--tokens` | int (multiple) | None | Token IDs to decode |
| `--from-trace` | str | None | Trace ID — decode all unique top-1 tokens |
| `--db` | str | None | DuckDB file (required with `--from-trace`) |

### `neurotrace compare`

Trace two prompts, diff them, and show decoded results. Reuses existing traces if found (matched by model, prompt, seed, capture_mode).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | str | **required** | HuggingFace model name or path |
| `--prompt-a` | str | **required** | First prompt |
| `--prompt-b` | str | **required** | Second prompt |
| `--db` | str | **required** | DuckDB database file path |
| `--seed` | int | 42 | Random seed |
| `--light` | flag | False | Use light capture mode |
| `--flagged-only` | flag | False | Show only flagged layers |
| `--head-detail` | flag | False | Show critical head details |
| `--json` | flag | False | Output as JSON |

### `neurotrace ablate`

Run inference with targeted components disabled and compare to baseline.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--db` | str | **required** | DuckDB database file path |
| `--model` | str | **required** | HuggingFace model name or path |
| `--prompt` | str | **required** | Prompt text |
| `--zero-layers` | str | None | Zero entire layer outputs (e.g. `"20"` or `"20,21"`) |
| `--zero-heads` | str | None | Zero specific attention heads (e.g. `"20:7,20:12"`) |
| `--scale-layer` | str | None | Scale layer contributions (e.g. `"20:0.5,21:2.0"`) |
| `--zero-mlp` | str | None | Zero MLP sublayer outputs (e.g. `"20"` or `"20,21"`) |
| `--baseline` | str | None | Baseline trace ID/label/prefix. If omitted, runs clean trace. |
| `--label` | str | None | Label for the ablated trace |
| `--seed` | int | 42 | Random seed |

At least one of `--zero-layers`, `--zero-heads`, `--scale-layer`, `--zero-mlp` is required.

### `neurotrace report`

Generate a self-contained HTML report from one or two traces.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--db` | str | **required** | DuckDB database file path |
| `--trace-id` | str | None | Single trace report (mutually exclusive with `--trace-a`/`--trace-b`) |
| `--trace-a` | str | None | Comparison: first trace ID |
| `--trace-b` | str | None | Comparison: second trace ID |
| `-o` / `--output` | str | `"report.html"` | Output file path |
| `--open` | flag | False | Open in browser after generating |
| `--full-attention` | flag | False | Include heatmaps for all layers |
| `--no-attention` | flag | False | Skip attention heatmaps entirely |
| `--upload` | flag | False | Upload to CarbonFiles after generating |
| `--bucket` | str | None | Existing CarbonFiles bucket ID |

## 5. Test Structure

### Organization

Each `src/neurotrace/*.py` module has a corresponding `tests/test_*.py`. Tests use pytest.

### Model Download Tests

Tests that require downloading TinyLlama (~1.1B params) are marked with `@pytest.mark.model_download`. The model is loaded once per session via a `conftest.py` fixture:

```python
# tests/conftest.py
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@pytest.fixture(scope="session")
def tinyllama_model():
    from neurotrace.models import load_model
    model, tokenizer = load_model(TINYLLAMA_MODEL)
    return model, tokenizer
```

### Non-model Tests

Unit tests for parsing, serialization, CLI validation, storage roundtrips, etc. use synthetic data (manually constructed `TraceResult` objects) and `tmp_path` for DuckDB files.

CLI tests use Click's `CliRunner` and mock model loading where needed.

Report and upload tests mock external dependencies.

### Running Tests

```bash
make test          # uv run pytest tests/ -v (all tests including model downloads)
make test-quick    # uv run pytest tests/ -v -m "not model_download" (fast, no downloads)
make lint          # uv run ruff check src/ tests/
```

## 6. Dependencies

### Python Version

`>=3.12`

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.0 | PyTorch for model inference and hooks |
| `transformers` | >=4.30 | HuggingFace model loading (`AutoModelForCausalLM`, `AutoTokenizer`) |
| `huggingface_hub` | >=0.20 | Model revision lookup (`model_info`) |
| `click` | >=8.0 | CLI framework |
| `duckdb` | >=1.0 | Trace storage |
| `rich` | >=13.0 | Terminal tables and progress spinners |
| `numpy` | >=1.24 | Array operations |

### Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `upload` | `carbonfiles-client>=0.5.0` | Upload reports to CarbonFiles |
| `dev` | `pytest>=8.0`, `ruff>=0.4`, `pyright>=1.1` | Testing and linting |

### Build System

Hatchling (`hatchling.build`). Wheel packages `src/neurotrace`.

### Console Script

`neurotrace = neurotrace.cli:cli`

## 7. Known Patterns and Conventions

### Hook Management

Two separate hook systems:

1. **`HookManager`** (hooks.py) — read-only activation capture. Registers forward hooks on layers, attention, MLP, and layer norms. Used by `Tracer.trace()`. Hooks capture data into `_captured` dict, cleaned up in `finally` block.

2. **`AblationHookManager`** (ablate.py) — write hooks for interventions. Registers forward hooks that modify outputs (zero, scale). Used by `_run_ablated_trace()`. Both hook managers coexist during ablated traces (ablation hooks registered first, then tracer hooks).

Both managers have `cleanup()` methods that call `handle.remove()` on all registered hooks.

### Trace ID Resolution

`TraceDB.resolve_trace_id(identifier: str)` resolves in order:
1. `"latest"` → most recent trace by timestamp
2. Exact `trace_id` match
3. Exact `label` match (errors if ambiguous — multiple traces with same label)
4. Prefix match on `trace_id` (errors if ambiguous)
5. Raises `ValueError("Trace not found: ...")` if none match

### Model Loading

- Always passes `token=False` to `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()` to avoid HuggingFace auth prompts.
- Default device: `"cpu"`, default dtype: `torch.float32`.
- `model.eval()` called after loading.
- For tracing, temporarily sets `model.config.output_attentions = True` and `model.config._attn_implementation = "eager"`.

### HTML Reports

- Fully self-contained: all CSS inline, all charts as inline SVG. No JavaScript. No external resources.
- Dark theme with print-friendly media query.
- Attention heatmaps use ~6 sampled layers by default (configurable via `--full-attention` / `--no-attention`).
- Report data comes from: `TraceResult` + `layer_stats` (from DB) + `layer_predictions` / `token_tracks` (computed at report time by loading model).

### CarbonFiles Upload

- Requires `carbonfiles-client` SDK (optional dependency).
- Environment variables: `CF_URL` (base URL), `CF_API_KEY`.
- Creates a bucket with 7-day expiry if no `--bucket` provided.
- Returns a direct content URL.

### Makefile Targets

| Target | Command |
|--------|---------|
| `setup` | `uv sync --all-extras` |
| `test` | `uv run pytest tests/ -v` |
| `test-quick` | `uv run pytest tests/ -v -m "not model_download"` |
| `lint` | `uv run ruff check src/ tests/` |
| `trace-demo` | Trace + list + inspect with TinyLlama |
| `install` | `uv tool install --force --reinstall --from '.[upload]' neurotrace` |
| `uninstall` | `uv tool uninstall neurotrace` |
| `clean` | Remove `__pycache__`, `.pytest_cache`, `*.db` |
