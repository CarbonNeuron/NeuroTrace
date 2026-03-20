# NeuroTrace

Interpretability toolkit for tracing inference through open-source transformer models.

Captures attention patterns, MLP activations, and residual streams during a forward pass, stores them in DuckDB, and exposes a CLI for trace, inspect, diff, and predict workflows.

## Install

```bash
# Development (run via uv)
uv sync

# Global PATH install (~/.local/bin/neurotrace)
make install
```

After `make install`, use `neurotrace` directly. For development, prefix with `uv run`.

## Quickstart

```bash
# Run a trace
neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "The capital of France is" --db traces.db

# List stored traces
neurotrace list --db traces.db

# Inspect a trace
neurotrace inspect --db traces.db --trace-id latest

# Diff two traces
neurotrace diff --db traces.db --trace-id-a <id-a> --trace-id-b <id-b>

# Predict next token
neurotrace predict --db traces.db --trace-id latest
```

## Dev Commands

```bash
make setup       # install all dependencies
make test        # run full test suite
make test-quick  # run tests excluding model downloads
make lint        # run ruff linter
make clean       # remove caches and temp db files
```
