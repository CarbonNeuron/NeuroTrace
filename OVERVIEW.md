# NeuroTrace — Project Overview

Reference document for AI agents writing code against this codebase.

## 1. File Tree

```
.
├── src/neurotrace/
│   ├── __init__.py          # version (__version__ = "0.1.0")
│   ├── types.py             # TraceMetadata, LayerSnapshot, TokenPrediction, TraceResult
│   ├── models.py            # load_model, get_architecture, get_lm_head_and_norm, get_final_prediction
│   ├── hooks.py             # HookManager — read-only activation capture
│   ├── tracer.py            # Tracer.trace() — forward-pass tracing engine
│   ├── storage.py           # TraceDB — DuckDB read/write/query
│   ├── analyzer.py          # compute_diff — trace comparison
│   ├── ablate.py            # AblationSpec, AblationHookManager, run_ablation
│   ├── scan.py              # run_scan — automated sabotage detection
│   ├── datasets.py          # get_builtin_dataset, format_prompt, load_dataset
│   ├── neurons.py           # profile_neurons, ablate_neurons
│   ├── finetune.py          # run_finetune — LoRA training
│   ├── probe.py             # run_probe — per-domain linear probe
│   ├── probe_universal.py   # train_universal_probe — cross-domain probe
│   ├── circuit.py           # run_circuit — MLP circuit analysis
│   ├── heatmap.py           # run_heatmap_local — full layer×prompt ablation matrix
│   ├── commitment.py        # run_commitment_local — margin tracking
│   ├── contrast.py          # run_contrast_local — cross-domain MLP geometry
│   ├── attribute.py         # gradient + ablation attribution
│   ├── token_trace.py       # run_token_trace_local — position×layer MLP projection
│   ├── attention_trace.py   # run_attention_trace_local — per-head contribution decomposition
│   ├── decompose.py         # decompose_logits — Logit Prism per-layer contribution
│   ├── diagnose.py          # run_diagnosis — combined diagnostic pipeline
│   ├── repair.py            # run_repair_local — ROME rank-one edits
│   ├── fingerprint.py       # compute_fingerprint_local, check_regressions_fast
│   ├── bench.py             # run_bench_local, run_bench_remote
│   ├── heal.py              # run_heal_local — autonomous scan→repair→verify pipeline
│   ├── remote.py            # RemoteWorker — HTTP/SSE client for GPU worker
│   ├── report.py            # generate_report, generate_comparison_report (HTML)
│   ├── upload.py            # upload_report to CarbonFiles
│   └── cli/
│       ├── __init__.py      # Click group + command registration
│       ├── trace.py         # trace, list, inspect, diff, predict, decode, compare
│       ├── analysis.py      # ablate, sweep, scan, finetune, neurons, probe, circuit, probe_universal
│       ├── domain.py        # experiment, heatmap, commitment, contrast, attribute
│       ├── advanced.py      # token_trace, attention_trace, diagnose, decompose
│       ├── repair.py        # repair, fingerprint
│       ├── bench.py         # bench
│       ├── heal.py          # heal
│       ├── report.py        # report
│       └── worker.py        # worker (version, update, reload)
├── tools/
│   └── gpu-worker.py        # FastAPI + SSE GPU inference server (runs on Windows w/ torch-directml)
├── tests/
│   ├── conftest.py          # session-scoped model fixture, model_available() helper
│   ├── test_ablate.py
│   ├── test_analyzer.py
│   ├── test_attention_trace.py
│   ├── test_attribute.py
│   ├── test_bench.py
│   ├── test_circuit.py
│   ├── test_cli.py
│   ├── test_commitment.py
│   ├── test_contrast.py
│   ├── test_decompose.py
│   ├── test_diagnose.py
│   ├── test_experiment.py
│   ├── test_finetune.py
│   ├── test_fingerprint.py
│   ├── test_heal.py
│   ├── test_heatmap.py
│   ├── test_hooks.py
│   ├── test_models.py
│   ├── test_neurons.py
│   ├── test_probe.py
│   ├── test_probe_universal.py
│   ├── test_remote.py
│   ├── test_repair.py
│   ├── test_report.py
│   ├── test_scan.py
│   ├── test_storage.py
│   ├── test_token_trace.py
│   ├── test_tracer.py
│   ├── test_types.py
│   ├── test_upload.py
│   └── test_worker.py
├── adapters/
│   └── tinyllama-capitals-fix/  # LoRA adapter (~1.5MB)
├── .gitignore
├── Makefile
├── pyproject.toml
├── pyrightconfig.json
├── README.md
├── OVERVIEW.md
└── uv.lock
```

## 2. Architecture & Patterns

### CLI Framework: Click
- Entry point: `neurotrace.cli:cli` (Click group)
- All commands registered in `cli/__init__.py` via `cli.add_command()`
- Top-level `--model` option available to all commands via `ctx.obj["model"]`
- Output: Rich tables/progress to stderr, data to stdout
- `--json` flag on most commands for machine-readable output
- `--remote URL` flag routes inference to GPU worker

### Two Hook Systems
1. **HookManager** (hooks.py) — read-only capture for tracing
2. **AblationHookManager** (ablate.py) — write hooks for interventions
Both coexist during ablated traces. Both have `cleanup()`.

### Remote Worker Pattern
- `RemoteWorker` (remote.py) wraps all GPU worker HTTP endpoints
- SSE streaming for long operations (batch-ablate, extract-activations, repair, etc.)
- Worker endpoints mirror local function signatures
- Most CLI commands check `--remote` flag and dispatch to either local or remote path

### HTML Report Pattern
- Every major module has `generate_*_html()` functions
- All reports: self-contained HTML, inline CSS + SVG, dark theme ("Refined Ocean")
- Common helpers: `_esc()`, `_*_css()`, `_value_to_color()`, `_*_color()`
- Uploadable to CarbonFiles via `--upload` flag

### Dataset Pattern
- Built-in datasets in `datasets.py`: capitals (50), math_simple (50), history_dates (40), science_symbols (40)
- `get_builtin_dataset(name)` → `list[dict]` with `{"prompt": str, "answer": str}`
- Custom datasets: JSON files with same format
- `format_prompt(raw_prompt, tokenizer)` applies chat template

### Test Pattern
- `@pytest.mark.slow` for tests requiring real model download
- `model_available()` helper for conditional skipping
- `conftest.py` has session-scoped TinyLlama fixture
- Non-slow tests use synthetic data + mocks

## 3. Key Module Signatures

### scan.py
```python
class ScanResult:
    model_name: str; dataset_name: str; prompt_results: list[PromptResult]
    # Properties: correct_count, sabotaged_count, weak_count, wrong_count

def run_scan(model, tokenizer, dataset: list[dict], dataset_name: str,
             seed: int = 42, ...) -> ScanResult
```

### diagnose.py
```python
class DiagnosisResult:
    prompt: str; answer: str; circuit: CircuitAnalysis
    suppression: SuppressionAnalysis; verdict: Verdict
    repair: RepairPrescription | None

def run_diagnosis(attn_result, token_result) -> DiagnosisResult
```

### repair.py
```python
class RepairResult:
    prompt: str; answer: str; status: str  # "repaired"|"skipped"|"regression"
    before: RepairBefore; after: RepairAfter; edit: RepairEdit
    regressions: list[RegressionResult]

def run_repair_local(model, tokenizer, prompt, answer, ...) -> RepairResult
def apply_rank_one_edit(model, layer: int, k_star, delta)
def undo_rank_one_edit(model, layer: int, k_star, delta)
```

### fingerprint.py
```python
class Fingerprint:
    prompt: str; answer: str; competitor: str
    margin: float; key_vectors: np.ndarray; p_answer: np.ndarray; p_competitor: np.ndarray

def compute_fingerprint_local(model, tokenizer, prompt, answer, seed=42) -> Fingerprint
def check_regressions_fast(delta, k_star, layer, fingerprints, threshold=0.5) -> list[AnalyticalRegression]
```

### bench.py
```python
class BenchRun:
    run_id: str; model_name: str; domain_results: list[DomainBenchResult]
    baseline_ppl: float; combined_ppl: float

def run_bench_local(model, tokenizer, dataset_names, ...) -> BenchRun
def run_bench_remote(remote_url, dataset_names, ...) -> BenchRun
def compute_perplexity_local(model, tokenizer, ...) -> dict
```

### heal.py
```python
class HealResult:
    model_name: str; dataset_name: str
    baseline_total: int; baseline_correct: int; baseline_wrong: int
    baseline_sabotaged: int; baseline_weak: int
    healed_total: int; healed_correct: int; healed_wrong: int
    healed_sabotaged: int; healed_weak: int
    edits_attempted: int; edits_applied: int; edits_rolled_back: int; edits_skipped: int
    regressions_checked: int; regressions_found: int
    ppl_before: float | None; ppl_after: float | None; ppl_delta: float | None
    output_path: str | None; duration_seconds: float
    prompt_results: list[PromptHealResult]; catastrophic_undo: bool

def run_heal_local(model, tokenizer, dataset, dataset_name, db_path="demo.db",
                   max_edits=500, regression_threshold=0.05, ppl_threshold=0.1,
                   dry_run=False, seed=42, output_path=None,
                   progress_callback=None) -> HealResult
def heal_result_to_dict(result: HealResult) -> dict
def generate_heal_html(result: HealResult) -> str
```

### decompose.py
```python
class DecomposeResult:
    prompt: str; answer: str; margin: float
    component_margins: list[dict]  # [{layer, attn_margin, mlp_margin}]

def run_decompose_local(model, tokenizer, prompt, answer, competitors, seed=42) -> list[DecomposeResult]
def run_decompose_remote(worker, prompt, tokens, seed=42) -> DecomposeResult
```

### remote.py — RemoteWorker endpoints
```python
class RemoteWorker:
    def __init__(self, base_url: str, timeout: float = 300.0)
    def health(self) -> dict
    def get_model_config(self) -> dict
    def format_prompt(self, messages: list[dict]) -> dict
    def trace(self, prompt, seed=42, top_k=5) -> dict
    def batch_ablate_stream(self, prompt, num_layers, seed=42) -> Generator[dict]
    def extract_activations_stream(self, prompts, layer_start, layer_end) -> Generator[dict]
    def forward_states_stream(self, prompts, seed=42) -> Generator[dict]
    def forward_mlp_deltas_stream(self, prompts, layers=None) -> Generator[dict]
    def forward_mlp_deltas_all_positions_stream(self, prompt, layers=None) -> Generator[dict]
    def attribute_gradients_stream(self, prompts, layer, target_token_ids) -> Generator[dict]
    def attention_contributions_stream(self, prompt, layers=None) -> Generator[dict]
    def decompose_stream(self, prompt, tokens, seed=42) -> Generator[dict]
    def fingerprint_stream(self, prompts, seed=42) -> Generator[dict]
    def repair_stream(self, prompt, answer, ...) -> Generator[dict]
    def perplexity_stream(self, max_samples=100) -> Generator[dict]
    def repair_and_measure_stream(self, prompts, target_margin=0.0) -> Generator[dict]
    def repair_undo(self) -> dict
    def repair_save(self, path: str) -> dict
    def worker_version(self) -> dict
    def reload_stream(self, model=None, dtype=None) -> Generator[dict]
    def worker_update_stream(self) -> Generator[dict]
    def finetune_stream(self, config: dict) -> Generator[dict]
    def download_adapter(self, adapter_id, output_path) -> None
```

## 4. CLI Commands (30 total)

| Command | Module | Description |
|---------|--------|-------------|
| `trace` | trace.py | Forward-pass trace, store to DB |
| `list` | trace.py | List stored traces |
| `inspect` | trace.py | Inspect trace metadata + layer stats |
| `diff` | trace.py | Compare two traces |
| `predict` | trace.py | Top-K predictions at every layer |
| `decode` | trace.py | Decode token IDs to strings |
| `compare` | trace.py | Trace + diff two prompts |
| `ablate` | analysis.py | Run with components disabled |
| `sweep` | analysis.py | Parameter sweep (MLP scale factors) |
| `scan` | analysis.py | Batch sabotage detection |
| `finetune` | analysis.py | LoRA fine-tuning |
| `neurons` | analysis.py | Neuron-level MLP attribution |
| `probe` | analysis.py | Per-domain linear probe |
| `circuit` | analysis.py | MLP circuit analysis |
| `probe-universal` | analysis.py | Cross-domain probe |
| `experiment` | domain.py | Run dataset through trace pipeline |
| `heatmap` | domain.py | Full layer×prompt ablation matrix |
| `commitment` | domain.py | Margin tracking across layers |
| `contrast` | domain.py | Cross-domain MLP geometry |
| `attribute` | domain.py | Gradient/ablation input attribution |
| `token-trace` | advanced.py | Position×layer MLP projection |
| `attention-trace` | advanced.py | Per-head contribution decomposition |
| `diagnose` | advanced.py | Combined diagnostic pipeline |
| `decompose` | advanced.py | Logit Prism per-layer decomposition |
| `repair` | repair.py | ROME rank-one weight edits |
| `fingerprint` | repair.py | Fast regression checking |
| `bench` | bench.py | Benchmark with repair + perplexity |
| `heal` | heal.py | Autonomous self-repair loop (scan→repair→verify) |
| `report` | report.py | HTML report generation |
| `worker` | worker.py | GPU worker management (version/update/reload) |

### Common CLI Flags
| Flag | Description |
|------|-------------|
| `--model` | HuggingFace model ID (top-level group option) |
| `--db` | DuckDB database path |
| `--remote URL` | GPU worker URL |
| `--json` | Machine-readable JSON output |
| `--report PATH` | Generate HTML report |
| `--upload` | Upload report to CarbonFiles |
| `--adapter PATH` | LoRA adapter directory |
| `--seed INT` | Random seed (default: 42) |
| `--device` | Compute device (cpu/cuda/auto) |
| `--dtype` | Data type (auto/float16/float32/bfloat16) |

## 5. Data Schema (DuckDB)

4 tables: `traces`, `layer_snapshots`, `attention_maps`, `token_predictions`.
Tensor blob format: `[ndim:i32][dims:i32...][float32 data]`, little-endian.

## 6. Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = ["torch>=2.0", "transformers>=4.30", "huggingface_hub>=0.20",
                "click>=8.0", "duckdb>=1.0", "rich>=13.0", "numpy>=1.24"]

[project.optional-dependencies]
upload = ["carbonfiles-client>=0.5.0"]
finetune = ["peft>=0.14.0", "datasets>=3.0.0", "accelerate>=1.0.0"]
probe = ["scikit-learn>=1.5.0"]
remote = ["httpx>=0.27.0"]
dev = ["pytest>=8.0", "ruff>=0.4", "pyright>=1.1"]
```

Build: hatchling. Console script: `neurotrace = neurotrace.cli:cli`

## 7. Makefile

| Target | Command |
|--------|---------|
| `setup` | `uv sync --all-extras` |
| `test` | `uv run pytest tests/ -v` |
| `test-quick` | `uv run pytest tests/ -v -m "not slow"` |
| `lint` | `uv run ruff check src/ tests/` |
| `install` | `uv tool install --force --reinstall --from '.[upload,finetune,probe]' neurotrace` |
| `clean` | Remove __pycache__, .pytest_cache, *.db |

## 8. GPU Worker (tools/gpu-worker.py)

FastAPI server running on Windows with torch-directml.

**Key endpoints:**
- `GET /health` — model info + device
- `GET /model/config` — architecture details
- `GET /version` — git commit, branch, uptime, device
- `POST /format` — apply chat template
- `POST /trace` — single forward-pass trace
- `POST /batch-ablate` — SSE stream of ablation results
- `POST /extract-activations` — SSE stream of hidden states
- `POST /forward-states` — SSE stream of all-layer hidden states
- `POST /forward-mlp-deltas` — SSE stream of MLP input/output
- `POST /forward-mlp-deltas-all-positions` — SSE all positions×layers
- `POST /attribute-gradients` — SSE gradient attribution
- `POST /attention-contributions` — SSE per-head o_proj decomposition
- `POST /decompose` — SSE logit decomposition
- `POST /fingerprint` — SSE MLP key vector caching
- `POST /repair` — SSE ROME edit with before/after/regression
- `POST /repair/undo` — undo last edit
- `POST /repair/save` — save edited weights
- `POST /bench/perplexity` — SSE perplexity computation
- `POST /bench/repair-and-measure` — SSE batch repair + measure
- `POST /reload` — SSE model reload (switch model/dtype)
- `POST /update` — SSE git pull + restart
- `POST /finetune` — SSE LoRA training
- `GET /finetune/{id}/download` — download adapter tarball

**DirectML notes:**
- bfloat16 unsupported, auto-selects float16
- REPO_DIR = Path(__file__).resolve().parent.parent
- torch-directml 0.2.5.dev240914 requires torch ~2.4
