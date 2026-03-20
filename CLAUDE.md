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

## Architecture

NeuroTrace captures activations from transformer forward passes, stores them in DuckDB, and provides CLI commands for analysis and ablation.

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

## Conventions

- Python 3.12+. Use `uv` for dependency management.
- All model loading uses `token=False` to avoid HuggingFace auth prompts.
- Tests requiring TinyLlama use `@pytest.mark.model_download` and the session-scoped `tinyllama_model` fixture from `conftest.py`.
- CLI uses Click. Progress/spinners go to stderr (`err_console`), data to stdout.
- HTML reports are fully self-contained (inline CSS, inline SVG, no JS, no external resources).
- `AblationSpec.zero_mlp` defaults to `[]` via `__post_init__` for backward compatibility.
- The `interventions` column in the `traces` table is nullable JSON (VARCHAR). Added via idempotent `ALTER TABLE` migration on every `TraceDB.__init__`.
