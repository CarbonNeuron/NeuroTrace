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
│   ├── discover.py          # discover knowledge gaps via structured fact extraction
│   ├── chain.py             # chain trace — multi-hop reasoning through residual streams
│   ├── remote.py            # WorkerClient — HTTP/SSE client for GPU worker
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
│       ├── discover.py      # discover
│       ├── chain.py         # chain
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
│   ├── test_chain.py
│   ├── test_circuit.py
│   ├── test_cli.py
│   ├── test_commitment.py
│   ├── test_contrast.py
│   ├── test_decompose.py
│   ├── test_diagnose.py
│   ├── test_discover.py
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
│   ├── test_raw_inference.py
│   ├── test_remote.py
│   ├── test_repair.py
│   ├── test_report.py
│   ├── test_scan.py
│   ├── test_storage.py
│   ├── test_token_trace.py
│   ├── test_tracer.py
│   ├── test_types.py
│   ├── test_upload.py
│   ├── test_worker.py
│   └── test_worker_v2.py
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
- `WorkerClient` (remote.py) wraps all GPU worker HTTP endpoints
- `RemoteWorker` kept as backward-compat alias for `WorkerClient`
- SSE streaming for long operations (batch-ablate, extract-activations, repair, etc.)
- v2 primitives: `forward()`, `hooked()`, `generate()`, `batch_forward()`, `decompose()`, `rome_edit()`
- Legacy streaming methods for backward compatibility with existing worker endpoints
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
- `@pytest.mark.model_download` for tests requiring real model download
- `model_available()` helper for conditional skipping
- `conftest.py` has session-scoped TinyLlama fixture
- Non-slow tests use synthetic data + mocks
- **Test counts: 649 total, 564 quick (non-model_download)**

## 3. CLI Commands (30 total)

| Command | Module | Remote | Description |
|---------|--------|--------|-------------|
| `trace` | trace.py | yes | Forward-pass trace, store to DB |
| `list` | trace.py | no | List stored traces |
| `inspect` | trace.py | no | Inspect trace metadata + layer stats |
| `diff` | trace.py | no | Compare two traces |
| `predict` | trace.py | yes | Top-K predictions at every layer |
| `decode` | trace.py | yes | Decode token IDs to strings |
| `compare` | trace.py | yes | Trace + diff two prompts |
| `ablate` | analysis.py | yes | Run with components disabled |
| `sweep` | analysis.py | yes | Parameter sweep (MLP scale factors) |
| `scan` | analysis.py | yes | Batch sabotage detection |
| `finetune` | analysis.py | yes | LoRA fine-tuning |
| `neurons` | analysis.py | yes | Neuron-level MLP attribution |
| `probe` | analysis.py | yes | Per-domain linear probe |
| `circuit` | analysis.py | yes | MLP circuit analysis |
| `probe-universal` | analysis.py | yes | Cross-domain probe |
| `experiment` | domain.py | yes | Run dataset through trace pipeline |
| `heatmap` | domain.py | yes | Full layer×prompt ablation matrix |
| `commitment` | domain.py | yes | Margin tracking across layers |
| `contrast` | domain.py | yes | Cross-domain MLP geometry |
| `attribute` | domain.py | yes | Gradient/ablation input attribution |
| `token-trace` | advanced.py | yes | Position×layer MLP projection |
| `attention-trace` | advanced.py | yes | Per-head contribution decomposition |
| `diagnose` | advanced.py | yes | Combined diagnostic pipeline |
| `decompose` | advanced.py | yes | Logit Prism per-layer decomposition |
| `repair` | repair.py | yes | ROME rank-one weight edits |
| `fingerprint` | repair.py | yes | Fast regression checking |
| `bench` | bench.py | yes | Benchmark with repair + perplexity |
| `heal` | heal.py | yes | Autonomous self-repair loop (scan→repair→verify) |
| `discover` | discover.py | yes | Knowledge gap discovery via fact extraction |
| `chain` | chain.py | yes | Multi-hop reasoning trace through residual streams |
| `report` | report.py | no | HTML report generation |
| `worker` | worker.py | — | GPU worker management (version/update/reload) |

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

## 4. WorkerClient → Worker Endpoint Mapping

| WorkerClient Method | Endpoint | Type |
|---------------------|----------|------|
| `forward()` | `POST /inference/forward` | v2 |
| `hooked()` | `POST /inference/hooked` | v2 |
| `generate()` | `POST /inference/generate` | v2 |
| `batch_forward()` | `POST /inference/batch` | v2 SSE |
| `batch_hooked()` | (sequential `hooked()` calls) | v2 |
| `edit()` | `POST /model/edit` | v2 |
| `edit_undo()` | `POST /model/edit/undo` | v2 |
| `edit_clear()` | `POST /model/edit/clear` | v2 |
| `edit_stack()` | `GET /model/edit/stack` | v2 |
| `decompose()` | `POST /decompose` | v2 SSE |
| `rome_edit()` | `POST /model/rome-edit` | v2 |
| `batch_edit()` | (sequential `rome_edit()` calls) | v2 |
| `health()` | `GET /health` | mgmt |
| `get_model_config()` | `GET /model/config` | mgmt |
| `format_prompt()` | `POST /format` | mgmt |
| `worker_version()` | `GET /version` | mgmt |
| `reload_stream()` | `POST /reload` | mgmt SSE |
| `worker_update_stream()` | `POST /update` | mgmt SSE |
| `trace()` | `POST /trace` | legacy |
| `batch_ablate_stream()` | `POST /batch-ablate` | legacy SSE |
| `extract_activations_stream()` | `POST /extract-activations` | legacy SSE |
| `forward_states_stream()` | `POST /forward-states` | legacy SSE |
| `forward_mlp_deltas_stream()` | `POST /forward-mlp-deltas` | legacy SSE |
| `forward_mlp_deltas_all_positions_stream()` | `POST /forward-mlp-deltas-all-positions` | legacy SSE |
| `attribute_gradients_stream()` | `POST /attribute-gradients` | legacy SSE |
| `attention_contributions_stream()` | `POST /attention-contributions` | legacy SSE |
| `finetune_stream()` | `POST /finetune` | legacy SSE |
| `download_adapter()` | `GET /finetune/{id}/download` | legacy |
| `decompose_stream()` | `POST /decompose` | legacy SSE |
| `fingerprint_stream()` | `POST /fingerprint` | legacy SSE |
| `repair_stream()` | `POST /repair` | legacy SSE |
| `perplexity_stream()` | `POST /bench/perplexity` | legacy SSE |
| `repair_and_measure_stream()` | `POST /bench/repair-and-measure` | legacy SSE |
| `repair_undo()` | `POST /repair/undo` | legacy |
| `repair_save()` | `POST /repair/save` | legacy |

## 5. Worker Endpoints (tools/gpu-worker.py)

| Endpoint | Method | Format |
|----------|--------|--------|
| `/health` | GET | JSON |
| `/model/config` | GET | JSON |
| `/version` | GET | JSON |
| `/format` | POST | JSON |
| `/inference/forward` | POST | JSON |
| `/inference/hooked` | POST | JSON |
| `/inference/generate` | POST | JSON |
| `/inference/batch` | POST | SSE stream |
| `/model/edit` | POST | JSON |
| `/model/rome-edit` | POST | JSON |
| `/model/edit/undo` | POST | JSON |
| `/model/edit/clear` | POST | JSON |
| `/model/edit/stack` | GET | JSON |
| `/trace` | POST | JSON |
| `/batch-ablate` | POST | SSE stream |
| `/forward-states` | POST | SSE stream |
| `/extract-activations` | POST | SSE stream |
| `/forward-mlp-deltas` | POST | SSE stream |
| `/forward-mlp-deltas-all-positions` | POST | SSE stream |
| `/attention-contributions` | POST | SSE stream |
| `/attribute-gradients` | POST | SSE stream |
| `/finetune` | POST | SSE stream |
| `/finetune/{id}/download` | GET | tar.gz stream |
| `/fingerprint` | POST | SSE stream |
| `/decompose` | POST | SSE stream |
| `/repair` | POST | SSE stream |
| `/repair/undo` | POST | JSON |
| `/repair/save` | POST | JSON |
| `/bench/perplexity` | POST | SSE stream |
| `/bench/repair-and-measure` | POST | SSE stream |
| `/reload` | POST | SSE stream |
| `/update` | POST | SSE stream |

## 6. Data Schema (DuckDB)

4 tables: `traces`, `layer_snapshots`, `attention_maps`, `token_predictions`.
Tensor blob format: `[ndim:i32][dims:i32...][float32 data]`, little-endian.

## 7. Dependencies

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

## 8. Makefile

| Target | Command |
|--------|---------|
| `setup` | `uv sync --all-extras` |
| `test` | `uv run pytest tests/ -v` |
| `test-quick` | `uv run pytest tests/ -v -m "not model_download"` |
| `lint` | `uv run ruff check src/ tests/` |
| `install` | `uv tool install --force --reinstall --from '.[upload,finetune,probe]' neurotrace` |
| `clean` | Remove __pycache__, .pytest_cache, *.db |

## 9. Cleanup Audit (2026-03-21)

**Before:** 695 tests (564 quick)
**After:** 649 tests (564 quick)

**Dead code removed:**
- `run_chain_batch()` from chain.py (never called)
- `run_commitment_remote()` from commitment.py (never called)
- `WorkerClient.fingerprint()` — called `/inference/fingerprint` which doesn't exist in worker; CLI uses `fingerprint_stream()` instead
- `WorkerClient.attention()` — called `/inference/attention` which doesn't exist in worker; CLI uses `attention_contributions_stream()` instead
- `WorkerClient.contrast()` — called `/inference/contrast` which doesn't exist in worker; never called by CLI
- `FingerprintVector`, `FingerprintResult`, `HeadContribution`, `AttentionResult`, `ContrastLayerResult`, `ContrastResult` dataclasses from remote.py (only used by deleted methods)
- Unused import `run_chain_batch` from cli/chain.py

**Theatrical tests deleted (131 tests):**
- Dataclass instantiation tests (test_types.py, test_attention_trace.py)
- CLI registration tests (test_attribute.py, test_chain.py, test_commitment.py, test_contrast.py)
- Mock-the-function-under-test tests (test_remote.py, test_worker.py, test_upload.py)
- Import/callable tests (test_bench.py)
- Tests for deleted methods (test_worker_v2.py: test_attention, test_dataclass_imports)

**No behavioral changes.** All remaining tests pass.
