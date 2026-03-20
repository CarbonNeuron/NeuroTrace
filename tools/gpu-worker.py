#!/usr/bin/env python3
"""
NeuroTrace GPU Inference Server

A standalone FastAPI server for GPU-accelerated transformer inference,
ablation sweeps, and LoRA fine-tuning. Self-contained -- does NOT import
from neurotrace.

Dependencies: torch, transformers, peft, fastapi, uvicorn
Optional:     torch_directml (for AMD/Intel GPUs on Windows)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gpu-worker")

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _list_devices() -> list[dict[str, str]]:
    """Return a list of all available GPU devices with index and name."""
    devices: list[dict[str, str]] = []
    try:
        import torch_directml  # type: ignore[import-untyped]

        count = torch_directml.device_count()
        for i in range(count):
            devices.append(
                {
                    "backend": "directml",
                    "index": str(i),
                    "name": torch_directml.device_name(i),
                }
            )
    except ImportError:
        pass
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(
                {
                    "backend": "cuda",
                    "index": str(i),
                    "name": torch.cuda.get_device_name(i),
                }
            )
    devices.append({"backend": "cpu", "index": "0", "name": "CPU"})
    return devices


def _resolve_device_index_by_name(
    backend: str, name_query: str
) -> int:
    """Find a device index by substring match on device name. Errors on ambiguity."""
    query_lower = name_query.lower()
    matches: list[tuple[int, str]] = []

    if backend == "directml":
        try:
            import torch_directml  # type: ignore[import-untyped]

            for i in range(torch_directml.device_count()):
                dname = torch_directml.device_name(i)
                if query_lower in dname.lower():
                    matches.append((i, dname))
        except ImportError:
            pass
    elif backend == "cuda":
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                dname = torch.cuda.get_device_name(i)
                if query_lower in dname.lower():
                    matches.append((i, dname))
    else:
        raise ValueError(f"--device-name is not supported with --device {backend}")

    if len(matches) == 0:
        raise ValueError(
            f"No {backend} device matching '{name_query}' found"
        )
    if len(matches) > 1:
        match_list = ", ".join(f"[{i}] {n}" for i, n in matches)
        raise ValueError(
            f"Multiple {backend} devices match '{name_query}': {match_list}"
        )
    return matches[0][0]


def _auto_detect_device() -> torch.device:
    """Pick the best available device: DirectML > CUDA > CPU."""
    try:
        import torch_directml  # type: ignore[import-untyped]

        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_device(
    choice: str | None,
    device_index: int = 0,
    device_name: str | None = None,
) -> torch.device:
    """Resolve a user-specified device string with optional index or name."""
    if device_name is not None and choice is None:
        raise ValueError("--device-name requires --device to be specified")

    if device_name is not None:
        assert choice is not None
        device_index = _resolve_device_index_by_name(choice, device_name)

    if choice is None:
        return _auto_detect_device()
    choice_lower = choice.lower()
    if choice_lower == "directml":
        import torch_directml  # type: ignore[import-untyped]

        return torch_directml.device(device_index)
    if choice_lower == "cuda":
        return torch.device(f"cuda:{device_index}")
    if choice_lower == "cpu":
        return torch.device("cpu")
    # Allow raw torch device strings like "cuda:1"
    return torch.device(choice)


def _device_display_name(device: torch.device) -> str:
    """Human-readable device name for /health."""
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        return torch.cuda.get_device_name(idx)
    if device.type == "privateuseone":
        try:
            import torch_directml  # type: ignore[import-untyped]

            idx = device.index or 0
            return torch_directml.device_name(idx)
        except ImportError:
            pass
        return "DirectML"
    return str(device)


# ---------------------------------------------------------------------------
# Safe tensor operations (DirectML workarounds)
# ---------------------------------------------------------------------------


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    """softmax that falls back to CPU when the current device chokes."""
    try:
        return torch.softmax(logits.float().cpu(), dim=-1)
    except Exception:
        log.warning("softmax fallback to CPU")
        return torch.softmax(logits.float().cpu(), dim=-1)


def _safe_topk(probs: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """topk that falls back to CPU when the current device chokes."""
    try:
        return torch.topk(probs, k)
    except Exception:
        log.warning("topk fallback to CPU")
        cpu_probs = probs.float().cpu()
        return torch.topk(cpu_probs, k)


# ---------------------------------------------------------------------------
# Global state filled at startup
# ---------------------------------------------------------------------------

app = FastAPI(title="NeuroTrace GPU Worker", version="1.0.0")

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None
_device: torch.device | None = None
_model_name: str = ""
_num_layers: int = 0
# adapter_id -> path on disk
_adapters: dict[str, Path] = {}


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class TraceRequest(BaseModel):
    prompt: str
    seed: int = 42
    top_k: int = 5


class TopToken(BaseModel):
    token: str
    prob: float


class LayerTrace(BaseModel):
    layer: int
    top_tokens: list[TopToken]


class TraceResponse(BaseModel):
    prompt: str
    num_layers: int
    final_token: str
    final_prob: float
    layers: list[LayerTrace]


class AblationItem(BaseModel):
    zero_mlp_layers: list[int] = Field(default_factory=list)


class BatchAblateRequest(BaseModel):
    prompt: str
    ablations: list[AblationItem]
    seed: int = 42
    top_k: int = 1


class ExtractActivationsRequest(BaseModel):
    prompts: list[str]
    layer_start: int
    layer_end: int
    seed: int = 42


class DatasetEntry(BaseModel):
    prompt: str
    answer: str


class FinetuneRequest(BaseModel):
    dataset: list[DatasetEntry]
    target_layers: list[int]
    target_modules: list[str] = Field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    epochs: int = 3
    lr: float = 2e-4
    rank: int = 8
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_transformer_layers() -> torch.nn.ModuleList:
    """Return the list of transformer blocks from the loaded model."""
    assert _model is not None
    return _model.model.layers  # type: ignore[union-attr]


def _project_to_vocab(hidden: torch.Tensor) -> torch.Tensor:
    """Project a hidden state through final norm + lm_head to get logits."""
    assert _model is not None
    normed = _model.model.norm(hidden)  # type: ignore[union-attr]
    return _model.lm_head(normed)  # type: ignore[union-attr]


def _decode_topk(
    logits: torch.Tensor, k: int
) -> list[dict[str, Any]]:
    """Decode top-k tokens from a logits vector (last position)."""
    assert _tokenizer is not None
    probs = _safe_softmax(logits)
    top_probs, top_ids = _safe_topk(probs, min(k, probs.shape[-1]))
    results: list[dict[str, Any]] = []
    for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
        results.append(
            {"token": _tokenizer.decode([tid]), "prob": round(prob, 6)}
        )
    return results


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse_line(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    assert _device is not None
    return {
        "status": "ok",
        "model": _model_name,
        "device": str(_device),
        "device_name": _device_display_name(_device),
        "num_layers": _num_layers,
    }


@app.post("/trace", response_model=TraceResponse)
def trace(req: TraceRequest) -> TraceResponse:
    assert _model is not None and _tokenizer is not None and _device is not None

    torch.manual_seed(req.seed)
    inputs = _tokenizer(req.prompt, return_tensors="pt").to(_device)

    layers = _get_transformer_layers()
    layer_results: list[LayerTrace] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(layer_idx: int):
        def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
            # output is a tuple; first element is the hidden state
            hidden = output[0] if isinstance(output, tuple) else output
            # Project the last token's hidden state to vocab
            last_hidden = hidden[:, -1, :]
            logits = _project_to_vocab(last_hidden.float())
            top_tokens = _decode_topk(logits[0], req.top_k)
            layer_results.append(
                LayerTrace(
                    layer=layer_idx,
                    top_tokens=[TopToken(**t) for t in top_tokens],
                )
            )

        return hook_fn

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(_make_hook(i)))

        with torch.no_grad():
            outputs = _model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    # Final prediction from actual model output
    final_logits = outputs.logits[:, -1, :].float().cpu()
    final_probs = torch.softmax(final_logits, dim=-1)
    top_prob, top_id = torch.topk(final_probs[0], 1)
    final_token = _tokenizer.decode([top_id.item()])
    final_prob = round(top_prob.item(), 6)

    # Sort layer results by layer index (hooks may fire in order, but be safe)
    layer_results.sort(key=lambda lr: lr.layer)

    return TraceResponse(
        prompt=req.prompt,
        num_layers=_num_layers,
        final_token=final_token,
        final_prob=final_prob,
        layers=layer_results,
    )


@app.post("/batch-ablate")
def batch_ablate(req: BatchAblateRequest) -> StreamingResponse:
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        total = len(req.ablations)
        inputs = _tokenizer(req.prompt, return_tensors="pt").to(_device)

        for idx, ablation in enumerate(req.ablations):
            description = (
                "baseline"
                if not ablation.zero_mlp_layers
                else f"zero_mlp:{ablation.zero_mlp_layers}"
            )
            yield _sse_line(
                {
                    "type": "progress",
                    "index": idx,
                    "total": total,
                    "description": description,
                }
            )

            torch.manual_seed(req.seed)
            hooks: list[torch.utils.hooks.RemovableHandle] = []

            def _zero_mlp_hook(
                module: torch.nn.Module, input: Any, output: Any
            ) -> torch.Tensor:
                return torch.zeros_like(output)

            try:
                layers = _get_transformer_layers()
                for layer_idx in ablation.zero_mlp_layers:
                    if 0 <= layer_idx < len(layers):
                        handle = layers[layer_idx].mlp.register_forward_hook(
                            _zero_mlp_hook
                        )
                        hooks.append(handle)

                with torch.no_grad():
                    outputs = _model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            final_logits = outputs.logits[:, -1, :].float().cpu()
            final_probs = torch.softmax(final_logits, dim=-1)
            top_prob, top_id = _safe_topk(final_probs[0], 1)
            final_token = _tokenizer.decode([top_id.item()])

            yield _sse_line(
                {
                    "type": "result",
                    "index": idx,
                    "zero_mlp_layers": ablation.zero_mlp_layers,
                    "final_token": final_token,
                    "final_prob": round(top_prob.item(), 6),
                }
            )

            # Let the event loop breathe between ablations
            await asyncio.sleep(0)

        yield _sse_line({"type": "done", "total_results": total})

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/extract-activations")
def extract_activations(req: ExtractActivationsRequest) -> StreamingResponse:
    """Extract layer output activations for multiple prompts via SSE."""
    assert _model is not None and _tokenizer is not None and _device is not None

    import base64

    import numpy as np

    async def _generate():
        layers = _get_transformer_layers()
        total = len(req.prompts)

        for idx, prompt in enumerate(req.prompts):
            yield _sse_line({
                "type": "progress",
                "index": idx,
                "total": total,
            })

            torch.manual_seed(req.seed)
            inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

            captured: dict[int, torch.Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHandle] = []

            for layer_idx in range(req.layer_start, req.layer_end + 1):
                def _make_hook(li: int):
                    def hook_fn(
                        module: torch.nn.Module,
                        input: Any,
                        output: Any,
                    ) -> None:
                        out = output[0] if isinstance(output, tuple) else output
                        captured[li] = out.detach()
                    return hook_fn

                if layer_idx < len(layers):
                    h = layers[layer_idx].register_forward_hook(
                        _make_hook(layer_idx)
                    )
                    hooks.append(h)

            try:
                with torch.no_grad():
                    _model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            # Concatenate last-token activations from each layer
            layer_acts = []
            for li in range(req.layer_start, req.layer_end + 1):
                if li in captured:
                    act = captured[li][0, -1, :].cpu().float().numpy()
                    layer_acts.append(act)

            concat = np.concatenate(layer_acts).astype(np.float32)
            encoded = base64.b64encode(concat.tobytes()).decode("ascii")

            yield _sse_line({
                "type": "activations",
                "index": idx,
                "shape": list(concat.shape),
                "dtype": "float32",
                "data": encoded,
            })

            await asyncio.sleep(0)

        yield _sse_line({"type": "done", "total": total})

    return StreamingResponse(
        _generate(), media_type="text/event-stream"
    )


@app.post("/finetune")
def finetune(req: FinetuneRequest) -> StreamingResponse:
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
        except ImportError:
            yield _sse_line(
                {
                    "type": "error",
                    "message": "peft is not installed. Install with: pip install peft",
                }
            )
            return

        # Build target module list scoped to requested layers
        target_modules: list[str] = []
        for layer_idx in req.target_layers:
            for module in req.target_modules:
                target_modules.append(f"model.layers.{layer_idx}.mlp.{module}")

        if not target_modules:
            yield _sse_line(
                {"type": "error", "message": "No target modules resolved."}
            )
            return

        lora_config = LoraConfig(
            r=req.rank,
            lora_alpha=req.rank * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Wrap model with LoRA
        _model.train()
        peft_model = get_peft_model(_model, lora_config)
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )

        # Tokenize dataset
        encodings: list[dict[str, torch.Tensor]] = []
        for entry in req.dataset:
            text = f"{entry.prompt} {entry.answer}"
            enc = _tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
            enc["labels"] = enc["input_ids"].clone()
            encodings.append({k: v.to(_device) for k, v in enc.items()})

        total_samples = len(encodings)
        steps_per_epoch = max(1, (total_samples + req.batch_size - 1) // req.batch_size)
        total_steps = steps_per_epoch * req.epochs

        yield _sse_line(
            {
                "type": "config",
                "total_steps": total_steps,
                "trainable_params": trainable_params,
                "target_modules": target_modules,
                "epochs": req.epochs,
                "lr": req.lr,
                "rank": req.rank,
                "batch_size": req.batch_size,
                "dataset_size": total_samples,
            }
        )

        optimizer = torch.optim.AdamW(
            peft_model.parameters(), lr=req.lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        start_time = time.monotonic()
        global_step = 0
        final_loss = 0.0

        for epoch in range(1, req.epochs + 1):
            epoch_loss = 0.0
            epoch_steps = 0

            # Simple batching by accumulation
            optimizer.zero_grad()
            accum_loss = 0.0

            for sample_idx, enc in enumerate(encodings):
                outputs = peft_model(**enc)
                loss = outputs.loss / min(req.batch_size, total_samples)
                loss.backward()
                accum_loss += loss.item()

                # Step when batch is full or at end of epoch
                if (sample_idx + 1) % req.batch_size == 0 or (
                    sample_idx + 1
                ) == total_samples:
                    torch.nn.utils.clip_grad_norm_(
                        peft_model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    step_loss = accum_loss
                    epoch_loss += step_loss
                    epoch_steps += 1
                    final_loss = step_loss

                    yield _sse_line(
                        {
                            "type": "progress",
                            "epoch": epoch,
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": round(step_loss, 4),
                            "lr": round(scheduler.get_last_lr()[0], 8),
                        }
                    )
                    accum_loss = 0.0

                    await asyncio.sleep(0)

            avg_loss = epoch_loss / max(epoch_steps, 1)
            yield _sse_line(
                {
                    "type": "epoch_end",
                    "epoch": epoch,
                    "avg_loss": round(avg_loss, 4),
                }
            )

        # Save adapter
        adapter_id = f"ft-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        adapter_dir = Path(tempfile.gettempdir()) / "neurotrace_adapters" / adapter_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(adapter_dir))
        _adapters[adapter_id] = adapter_dir

        elapsed = time.monotonic() - start_time

        yield _sse_line(
            {
                "type": "complete",
                "final_loss": round(final_loss, 4),
                "total_time_seconds": round(elapsed, 1),
                "adapter_id": adapter_id,
            }
        )

        # Restore model to eval mode and unload LoRA
        peft_model.eval()
        try:
            peft_model.merge_and_unload()
        except Exception:
            log.warning("Could not merge and unload LoRA; model remains wrapped.")

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get("/finetune/{adapter_id}/download")
def download_adapter(adapter_id: str) -> StreamingResponse:
    if adapter_id not in _adapters:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_id}' not found")

    adapter_dir = _adapters[adapter_id]
    if not adapter_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Adapter directory missing on disk"
        )

    def _tar_stream():
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for fpath in adapter_dir.iterdir():
                if fpath.is_file():
                    tar.add(str(fpath), arcname=fpath.name)
        buf.seek(0)
        yield buf.read()

    return StreamingResponse(
        _tar_stream(),
        media_type="application/gzip",
        headers={
            "Content-Disposition": f"attachment; filename={adapter_id}.tar.gz"
        },
    )


# ---------------------------------------------------------------------------
# Startup & CLI
# ---------------------------------------------------------------------------


def _load_model(model_name: str, device: torch.device, cache_dir: str | None) -> None:
    global _model, _tokenizer, _device, _model_name, _num_layers

    log.info("Loading tokenizer: %s", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=False, cache_dir=cache_dir
    )

    log.info("Loading model: %s", model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, token=False, cache_dir=cache_dir
    )
    _model = _model.to(device)
    _model.eval()

    _device = device
    _model_name = model_name
    _num_layers = len(_model.model.layers)

    log.info(
        "Model ready: %s | %d layers | device: %s (%s)",
        model_name,
        _num_layers,
        device,
        _device_display_name(device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NeuroTrace GPU inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--port", type=int, default=8877, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--device",
        default=None,
        choices=["directml", "cuda", "cpu"],
        help="Force device (default: auto-detect)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="GPU ordinal index (default: 0)",
    )
    parser.add_argument(
        "--device-name",
        default=None,
        help="Select GPU by substring match on device name (case-insensitive)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print all available devices and exit",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Model cache directory"
    )
    args = parser.parse_args()

    if args.list_devices:
        devices = _list_devices()
        for d in devices:
            print(f"[{d['backend']}:{d['index']}] {d['name']}")
        return

    if args.device_index != 0 and args.device_name is not None:
        parser.error("--device-index and --device-name are mutually exclusive")

    device = _resolve_device(args.device, args.device_index, args.device_name)
    _load_model(args.model, device, args.cache_dir)

    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
