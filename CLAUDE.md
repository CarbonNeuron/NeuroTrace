# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make test              # full test suite (downloads TinyLlama ~1.1B on first run)
make test-quick        # fast: skips tests that need model downloads
make lint              # ruff check src/ tests/
uv run pytest tests/test_ablate.py::test_ablation_spec_to_json -v   # single test
uv run pytest tests/test_ablate.py -v -m "not model_download"       # one file, no downloads
make install           # install neurotrace CLI to ~/.local/bin via uv tool
```

**Always use `make install` to install the CLI.** Do not run `uv tool install` directly — the Makefile has the correct flags (`--force`, `--reinstall`, extras, etc.).

## Architecture

NeuroTrace captures activations from transformer forward passes, stores them in DuckDB, and provides CLI commands for analysis, ablation, probing, and repair.

### Data flow

`Tracer` → `HookManager` hooks capture activations → `TraceResult` (dataclasses) → `TraceDB` writes to DuckDB → CLI commands read back for analysis.

### Two hook systems

1. **`HookManager`** (hooks.py): Read-only capture hooks on layers, attention, MLP, and layer norms. Used by `Tracer.trace()`. Captures data into a dict keyed by `(layer_index, component_name)`.

2. **`AblationHookManager`** (ablate.py): Write hooks that modify outputs (zero layers, zero heads, scale layers, zero MLP). Both coexist during ablated traces — ablation hooks register first, then tracer hooks capture the modified activations.

### Model architecture registry

`models.py` has a registry mapping `model.config.model_type` → `ModelArchitecture` dataclass that describes where to find layers, attention, MLP, and layer norms. Currently only `"llama"` is registered. Adding a new architecture means adding an entry to `ARCHITECTURE_REGISTRY`.

For Llama: layers at `model.model.layers`, attention at `.self_attn`, MLP at `.mlp`, layer norms at `.input_layernorm` and `.post_attention_layernorm`.

### Storage

DuckDB (not SQLite). Single `.db` file. 4 tables: `traces`, `layer_snapshots`, `attention_maps`, `token_predictions`. Tensors stored as BLOBs with a custom binary format (ndim + shape + float32 data). `final_logits` is NOT persisted — reconstructed as zeros on read.

### Trace ID resolution

`TraceDB.resolve_trace_id()` resolves in order: `"latest"` → exact UUID → label match → prefix match. Raises `ValueError` on ambiguous or not found.

### Per-layer predictions

`top1_token`/`top1_prob` in `LayerSnapshot` are computed by projecting `residual_out` through `model.model.norm` (final layer norm) then `model.lm_head`. This happens during tracing and again in `predict`/`report` commands.

### Analysis pipeline

The full interpretability pipeline (available via `neurotrace experiment`) runs: scan → ablate → finetune → verify. Each stage can also be run independently.

- **scan.py**: Tests a model against a JSONL dataset to find anomalous predictions.
- **probe.py**: Trains a linear probe on activations at a target layer to find a "sabotage direction" separating correct from incorrect predictions. Requires `scikit-learn`.
- **circuit.py**: Projects the probe direction through MLP weight matrices to identify which neurons contribute most to the direction.
- **neurons.py**: Profiles individual neuron attributions in an MLP layer, or ablates top-N neurons to measure their causal effect.
- **finetune.py**: Trains a LoRA adapter (via PEFT) on specified MLP layers to repair identified failures. Requires `peft`, `datasets`, `accelerate`.
- **datasets.py**: Loads JSONL dataset files for scan/probe/finetune commands.

### CLI modules

- **cli.py**: All Click commands. Progress/spinners go to stderr (`err_console`), data to stdout.
- **report.py**: Generates self-contained HTML reports (inline CSS, inline SVG, no JS, no external resources).
- **upload.py**: Optional upload to CarbonFiles (requires `carbonfiles-client`).

## Source files

| File | Purpose |
|---|---|
| `types.py` | Core dataclasses: `TraceResult`, `LayerSnapshot`, `AttentionMap`, etc. |
| `models.py` | Model loading, architecture registry (`ARCHITECTURE_REGISTRY`) |
| `hooks.py` | `HookManager` — read-only activation capture |
| `tracer.py` | `Tracer` — orchestrates forward pass with hooks |
| `storage.py` | `TraceDB` — DuckDB persistence with tensor blob serialization |
| `analyzer.py` | Diff engine: cosine sim, KL divergence, top-1 change, critical heads |
| `ablate.py` | `AblationSpec`, `AblationHookManager` — component zeroing/scaling |
| `scan.py` | Dataset scanning for anomalous predictions |
| `probe.py` | Linear probe training on activation geometry |
| `circuit.py` | Circuit tracing from probe direction through MLP weights |
| `neurons.py` | Per-neuron MLP attribution profiling and ablation |
| `finetune.py` | LoRA adapter training for targeted repair |
| `datasets.py` | JSONL dataset loading |
| `report.py` | Self-contained HTML report generation |
| `cli.py` | Click CLI (17 commands) |
| `upload.py` | Optional CarbonFiles upload |

## Conventions

- Python 3.12+. Use `uv` for dependency management.
- All model loading uses `token=False` to avoid HuggingFace auth prompts.
- Tests requiring TinyLlama use `@pytest.mark.model_download` and the session-scoped `tinyllama_model` fixture from `conftest.py`.
- CLI uses Click. Progress/spinners go to stderr (`err_console`), data to stdout.
- HTML reports are fully self-contained (inline CSS, inline SVG, no JS, no external resources).
- `AblationSpec.zero_mlp` defaults to `[]` via `__post_init__` for backward compatibility.
- The `interventions` column in the `traces` table is nullable JSON (VARCHAR). Added via idempotent `ALTER TABLE` migration on every `TraceDB.__init__`.
- Optional features use extras: `finetune` (peft, datasets, accelerate), `probe` (scikit-learn), `upload` (carbonfiles-client).
