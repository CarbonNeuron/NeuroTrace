# NeuroTrace

**Interpretability toolkit for tracing, analyzing, and repairing transformer inference.**

NeuroTrace hooks into a model's forward pass to capture residual streams, attention patterns, and MLP activations layer by layer. Everything is stored in a local DuckDB database. A single CLI lets you trace, diff, ablate, scan for anomalies, probe activation geometry, profile individual neurons, train targeted LoRA repairs, and generate self-contained HTML reports — no notebooks, no cloud, no dependencies beyond PyTorch and HuggingFace.

## Install

```bash
# Development (run commands via `uv run neurotrace`)
uv sync

# Global install to ~/.local/bin/neurotrace
make install
```

## Workflow

### 1. Trace a forward pass

Capture every layer's residual stream, attention maps, MLP outputs, and per-layer token predictions:

```bash
neurotrace trace --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --db traces.db
```

### 2. Inspect and compare

```bash
neurotrace inspect --db traces.db --trace-id latest
neurotrace predict --db traces.db --trace-id latest --top-k 5

# Diff two traces — cosine similarity, KL divergence, top-1 changes
neurotrace diff --db traces.db --trace-id-a <id-a> --trace-id-b <id-b>

# Or trace-and-diff in one shot
neurotrace compare --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt-a "2+2=" --prompt-b "2+3=" --db traces.db
```

### 3. Ablate components

Disable layers, attention heads, or MLPs and observe the effect:

```bash
neurotrace ablate --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --db traces.db \
  --zero-layers 15,16 --zero-heads 10:0,10:1

# Sweep a parameter range in a single model load
neurotrace sweep --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --db traces.db \
  --sweep zero-layers --range 0-21
```

### 4. Scan for anomalies

Test a model against a dataset to find systematic failures:

```bash
neurotrace scan --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset datasets/capitals.jsonl --db traces.db
```

### 5. Probe activation geometry

Train a linear probe to find directions in activation space that separate correct from sabotaged predictions:

```bash
neurotrace probe --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset datasets/capitals.jsonl --db traces.db --layer 16
```

### 6. Trace circuits

Follow a probe's learned direction back through MLP layers to identify which neurons contribute most:

```bash
neurotrace circuit --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset datasets/capitals.jsonl --db traces.db --layer 16
```

### 7. Profile and ablate neurons

Identify the highest-attribution neurons in a layer's MLP, then ablate them individually:

```bash
neurotrace neurons --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --db traces.db \
  --layer 16 --mode profile

neurotrace neurons --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "The capital of France is" --db traces.db \
  --layer 16 --mode ablate --top-n 10
```

### 8. Repair with LoRA

Train a targeted LoRA adapter on specific MLP layers to fix identified failures:

```bash
neurotrace finetune --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset datasets/capitals.jsonl --target-layers 15,16 \
  --output adapters/fix-capitals
```

### 9. Run the full pipeline

Scan, ablate, finetune, and verify in one command:

```bash
neurotrace experiment --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset datasets/capitals.jsonl --db traces.db
```

### 10. Generate reports

Produce a self-contained HTML report (inline CSS, inline SVG, no JS, no external resources):

```bash
neurotrace report --db traces.db --trace-id latest -o report.html
```

## CLI Reference

| Command | Description |
|---|---|
| `trace` | Run a forward-pass trace and store results |
| `list` | List all stored traces |
| `inspect` | Inspect a stored trace |
| `predict` | Show top-K token predictions at every layer |
| `diff` | Compare two traces with divergence metrics |
| `compare` | Trace two prompts and diff in one step |
| `decode` | Decode token IDs to human-readable strings |
| `ablate` | Disable components and compare to baseline |
| `sweep` | Sweep a parameter range across multiple ablations |
| `scan` | Scan a dataset for anomalous predictions |
| `probe` | Find directions in activation space via linear probing |
| `circuit` | Trace a probe direction back through MLP layers |
| `neurons` | Profile or ablate individual neurons in an MLP |
| `finetune` | Train a LoRA adapter to fix targeted failures |
| `experiment` | Run the full diagnostic pipeline end-to-end |
| `report` | Generate a self-contained HTML report |

## Architecture

```
Tracer -> HookManager -> TraceResult -> TraceDB (DuckDB) -> CLI analysis
                |                                               |
        AblationHookManager                              HTML reports
        (modify activations)                             Probe / Circuit
                                                         LoRA finetune
```

- **HookManager** — read-only hooks that capture activations during `Tracer.trace()`
- **AblationHookManager** — write hooks that zero/scale components before capture
- **TraceDB** — DuckDB storage with tensor BLOBs and queryable stat columns
- **Model registry** — architecture-aware hook placement (currently Llama)

## Development

```bash
make test              # full suite (downloads TinyLlama ~1.1B on first run)
make test-quick        # skip model-download tests
make lint              # ruff check
make clean             # remove caches and temp files
```

Run a single test:

```bash
uv run pytest tests/test_ablate.py::test_ablation_spec_to_json -v
```

## Stack

Python 3.12+ / PyTorch / HuggingFace Transformers / DuckDB / Click / Rich
