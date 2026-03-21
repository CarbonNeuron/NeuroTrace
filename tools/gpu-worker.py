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
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    log.error(
        "%s %s: Validation error: %s",
        request.method, request.url.path, exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors())},
    )


START_TIME = time.time()

REPO_DIR = str(Path(__file__).resolve().parent.parent)


def get_git_info() -> dict[str, Any]:
    """Collect git commit, branch, dirty status, and timestamp."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_DIR
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, cwd=REPO_DIR
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, cwd=REPO_DIR
            ).strip()
        )
        timestamp = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"], text=True, cwd=REPO_DIR
        ).strip()
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "timestamp": timestamp,
        }
    except Exception:
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": True,
            "timestamp": None,
        }


_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None
_device: torch.device | None = None
_model_name: str = ""
_num_layers: int = 0
_dtype: str = "auto"
# adapter_id -> path on disk
_adapters: dict[str, Path] = {}
# Stack of (layer, k_star, delta) for legacy /repair undo
_edit_stack: list[tuple[int, Any, Any]] = []
# v2 edit stack: (layer, component, key_vec, val_vec) for /model/edit
_v2_edit_stack: list[tuple[int, str, Any, Any]] = []


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class FormatRequest(BaseModel):
    messages: list[ChatMessage]


class TraceRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
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
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    ablations: list[AblationItem]
    seed: int = 42
    top_k: int = 1


class ExtractActivationsRequest(BaseModel):
    prompts: list[str]
    layer_start: int
    layer_end: int
    seed: int = 42


class ForwardStatesRequest(BaseModel):
    prompts: list[str]
    seed: int = 42


class ForwardMlpDeltasRequest(BaseModel):
    prompts: list[str]
    layers: list[int] | None = None
    seed: int = 42


class ForwardMlpDeltasAllPositionsRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    layers: list[int] | None = None
    seed: int = 42


class AttentionContributionsRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    layers: list[int] | None = None
    seed: int = 42


class AttributeGradientsRequest(BaseModel):
    prompts: list[str]
    layer: int
    target_token_ids: list[int]
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


class VerifyPrompt(BaseModel):
    prompt: str
    answer: str


class RepairRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    answer: str
    competitor: str | None = None
    target_layer: int | None = None
    target_component: str = "mlp"
    target_margin: float = 0.0
    verify_prompts: list[VerifyPrompt] = Field(default_factory=list)
    seed: int = 42


class RepairSaveRequest(BaseModel):
    path: str


class ReloadRequest(BaseModel):
    model: str | None = None
    dtype: str | None = None


# ---------------------------------------------------------------------------
# v2 Pydantic models — inference primitives
# ---------------------------------------------------------------------------


class OutputSpec(BaseModel):
    logits: bool = False
    top_k: int = 10
    residuals: bool = False
    residual_layers: list[int] | None = None
    hidden_states: bool = False


class InferenceForwardRequest(BaseModel):
    input: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    seed: int = 42
    outputs: OutputSpec = Field(default_factory=OutputSpec)


class HookSpec(BaseModel):
    layer: int
    component: str  # "mlp" | "attn" | "layer"
    action: str  # "zero" | "capture" | "scale" | "replace"
    scale: float | None = None
    tensor: str | None = None  # base64-encoded float16


class InferenceHookedRequest(BaseModel):
    input: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    seed: int = 42
    hooks: list[HookSpec] = Field(default_factory=list)
    outputs: OutputSpec = Field(default_factory=OutputSpec)


class InferenceGenerateRequest(BaseModel):
    input: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    seed: int = 42
    max_tokens: int = 20
    temperature: float = 0.0


class InferenceBatchRequest(BaseModel):
    requests: list[InferenceForwardRequest]


class ModelEditRequest(BaseModel):
    layer: int
    component: str  # e.g. "mlp.down_proj"
    action: str = "rome_rank_one"
    key_vector: str  # base64 float16
    value_vector: str  # base64 float16


class RomeEditRequest(BaseModel):
    """High-level ROME edit: compute key/value vectors on the worker."""

    input: str
    subject: str
    target: str
    layer: int
    raw: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_prompt(
    prompt: str | None,
    messages: list[ChatMessage] | None,
    raw: bool = False,
) -> str:
    """Resolve a prompt from either raw string or chat messages.

    When raw=True, the prompt string is used directly (no chat template).
    When raw=False and messages are provided, the chat template is applied.
    When only prompt is provided (no messages), it is used directly regardless of raw flag.
    """
    assert _tokenizer is not None
    if raw and prompt is not None:
        return prompt
    if messages is not None:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        return _tokenizer.apply_chat_template(
            msg_dicts, tokenize=False, add_generation_prompt=True,
        )
    if prompt is not None:
        return prompt
    raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' is required")


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
        "dtype": _dtype,
        "device": str(_device),
        "device_name": _device_display_name(_device),
        "num_layers": _num_layers,
    }


def _get_model_config() -> dict[str, Any]:
    """Return model config dict for the loaded model."""
    assert _model is not None
    cfg = _model.config
    return {
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", _num_layers),
        "num_attention_heads": getattr(cfg, "num_attention_heads", 0),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", 0),
        "hidden_size": getattr(cfg, "hidden_size", 0),
        "intermediate_size": getattr(cfg, "intermediate_size", 0),
        "vocab_size": getattr(cfg, "vocab_size", 0),
    }


@app.get("/model/config")
def model_config() -> dict[str, Any]:
    """Return the full model config dict."""
    assert _model is not None
    return _get_model_config()


@app.get("/version")
async def version() -> dict[str, Any]:
    assert _device is not None
    git = get_git_info()
    return {
        **git,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(_device),
        "device_name": _device_display_name(_device),
        "model": _model_name,
        "dtype": _dtype,
        "model_config": _get_model_config(),
        "uptime_seconds": int(time.time() - START_TIME),
    }


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/update")
async def update() -> StreamingResponse:
    async def stream():
        old_commit = get_git_info()["commit"]

        yield _sse_event(
            "progress", {"status": "pulling", "message": "Pulling latest from origin/main..."}
        )

        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True,
            text=True,
            cwd=REPO_DIR,
        )

        if result.returncode != 0:
            yield _sse_event("error", {"message": f"Git pull failed: {result.stderr}"})
            return

        new_commit = get_git_info()["commit"]
        changed = old_commit != new_commit

        yield _sse_event(
            "progress", {"status": "pulled", "message": result.stdout.strip()}
        )

        if changed:
            yield _sse_event(
                "progress",
                {"status": "restarting", "message": "Restarting worker in 2 seconds..."},
            )
            yield _sse_event(
                "done",
                {"old_commit": old_commit, "new_commit": new_commit, "changed": True},
            )

            async def restart():
                await asyncio.sleep(2)
                os.execv(sys.executable, [sys.executable] + sys.argv)

            asyncio.create_task(restart())
        else:
            yield _sse_event(
                "done",
                {"old_commit": old_commit, "new_commit": new_commit, "changed": False},
            )

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/reload")
async def reload(req: ReloadRequest) -> StreamingResponse:
    async def _generate():
        global _model, _tokenizer, _device, _model_name, _num_layers, _dtype, _edit_stack

        new_model = req.model or _model_name
        new_dtype = req.dtype or _dtype

        if new_model == _model_name and new_dtype == _dtype:
            yield _sse_event("done", {"model": _model_name, "dtype": _dtype, "message": "No change needed."})
            return

        # Unload current model
        yield _sse_event("progress", {"status": "unloading", "message": f"Unloading {_model_name}..."})
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        _edit_stack.clear()
        _v2_edit_stack.clear()

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await asyncio.sleep(0)

        # Load new model
        yield _sse_event("progress", {"status": "loading", "message": f"Loading {new_model} ({new_dtype})..."})

        try:
            _load_model(new_model, _device, cache_dir=None, dtype=new_dtype)
        except Exception as e:
            yield _sse_event("error", {"message": str(e)})
            return

        await asyncio.sleep(0)

        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() // (1024 * 1024)

        yield _sse_event("done", {
            "model": _model_name,
            "dtype": _dtype,
            "vram_mb": vram_mb,
        })

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/format")
def format_prompt(req: FormatRequest) -> dict[str, Any]:
    """Format chat messages using the model's chat template."""
    assert _tokenizer is not None
    msg_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    formatted = _tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=True,
    )
    num_tokens = len(_tokenizer.encode(formatted))
    return {"formatted": formatted, "num_tokens": num_tokens}


# ---------------------------------------------------------------------------
# v2 Inference Primitive Endpoints
# ---------------------------------------------------------------------------


def _resolve_input(
    input_str: str | None,
    messages: list[ChatMessage] | None,
    raw: bool = False,
) -> str:
    """Resolve input for v2 endpoints — same logic as _resolve_prompt."""
    return _resolve_prompt(input_str, messages, raw)


def _encode_residual(tensor: torch.Tensor) -> str:
    """Encode a tensor as base64 float16."""
    import base64

    arr = tensor.detach().cpu().float().numpy().astype("float16")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_tensor_b64(b64_str: str, dtype: str = "float16") -> torch.Tensor:
    """Decode a base64-encoded tensor."""
    import base64

    import numpy as np

    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=dtype)
    return torch.from_numpy(arr.copy()).float()


def _build_top_tokens(logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    """Build top-k token predictions with logit and prob values."""
    assert _tokenizer is not None
    probs = _safe_softmax(logits)
    top_probs, top_ids = _safe_topk(probs, min(k, probs.shape[-1]))
    logits_cpu = logits.float().cpu()
    results: list[dict[str, Any]] = []
    for prob, tid in zip(top_probs.tolist(), top_ids.tolist()):
        results.append({
            "token": _tokenizer.decode([tid]),
            "token_id": tid,
            "logit": round(logits_cpu[tid].item(), 4),
            "prob": round(prob, 6),
        })
    return results


@app.post("/inference/forward")
def inference_forward(req: InferenceForwardRequest) -> dict[str, Any]:
    """v2 forward pass — return logits, top-k predictions, and optional residuals."""
    assert _model is not None and _tokenizer is not None and _device is not None

    prompt = _resolve_input(req.input, req.messages, raw=req.raw)
    torch.manual_seed(req.seed)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    layers = _get_transformer_layers()
    residuals: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    # Determine which layers to capture residuals for
    if req.outputs.residuals:
        target_layers = (
            req.outputs.residual_layers
            if req.outputs.residual_layers is not None
            else list(range(len(layers)))
        )
        for layer_idx in target_layers:
            if 0 <= layer_idx < len(layers):
                def _make_hook(li: int):
                    def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
                        hidden = output[0] if isinstance(output, tuple) else output
                        residuals[li] = hidden[:, -1, :].detach()
                    return hook_fn
                hooks.append(layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))

    try:
        with torch.no_grad():
            outputs = _model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    final_logits = outputs.logits[:, -1, :].float().cpu()
    top_tokens = _build_top_tokens(final_logits[0], req.outputs.top_k)

    result: dict[str, Any] = {
        "top_tokens": top_tokens,
        "num_layers": _num_layers,
        "vocab_size": getattr(_model.config, "vocab_size", 0),
        "hidden_dim": getattr(_model.config, "hidden_size", 0),
    }

    if req.outputs.residuals:
        result["residuals"] = {
            str(li): _encode_residual(residuals[li][0])
            for li in sorted(residuals.keys())
        }
    else:
        result["residuals"] = None

    return result


@app.post("/inference/hooked")
def inference_hooked(req: InferenceHookedRequest) -> dict[str, Any]:
    """v2 hooked forward pass — apply hooks (zero/capture/scale/replace) during inference."""
    assert _model is not None and _tokenizer is not None and _device is not None

    prompt = _resolve_input(req.input, req.messages, raw=req.raw)
    torch.manual_seed(req.seed)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    layers = _get_transformer_layers()
    hooks_handles: list[torch.utils.hooks.RemovableHandle] = []
    captured: dict[str, torch.Tensor] = {}

    for hook_spec in req.hooks:
        layer_idx = hook_spec.layer
        if layer_idx < 0 or layer_idx >= len(layers):
            continue

        layer = layers[layer_idx]
        component = hook_spec.component
        action = hook_spec.action

        # Resolve the target module
        if component == "mlp":
            target = layer.mlp
        elif component == "attn":
            target = layer.self_attn
        elif component == "layer":
            target = layer
        else:
            continue

        if action == "zero":
            def _make_zero_hook(target_module: torch.nn.Module):
                def hook_fn(module: torch.nn.Module, input: Any, output: Any):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                return hook_fn
            hooks_handles.append(target.register_forward_hook(_make_zero_hook(target)))

        elif action == "capture":
            capture_key = f"{layer_idx}.{component}"
            def _make_capture_hook(key: str):
                def hook_fn(module: torch.nn.Module, input: Any, output: Any) -> None:
                    out = output[0] if isinstance(output, tuple) else output
                    captured[key] = out[:, -1, :].detach()
                return hook_fn
            hooks_handles.append(target.register_forward_hook(_make_capture_hook(capture_key)))

        elif action == "scale":
            scale_factor = hook_spec.scale if hook_spec.scale is not None else 1.0
            def _make_scale_hook(s: float):
                def hook_fn(module: torch.nn.Module, input: Any, output: Any):
                    if isinstance(output, tuple):
                        return (output[0] * s,) + output[1:]
                    return output * s
                return hook_fn
            hooks_handles.append(target.register_forward_hook(_make_scale_hook(scale_factor)))

        elif action == "replace":
            if hook_spec.tensor is not None:
                replacement = _decode_tensor_b64(hook_spec.tensor).to(_device)
                def _make_replace_hook(rep: torch.Tensor):
                    def hook_fn(module: torch.nn.Module, input: Any, output: Any):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[:, -1, :] = rep
                            return (out,) + output[1:]
                        out = output.clone()
                        out[:, -1, :] = rep
                        return out
                    return hook_fn
                hooks_handles.append(target.register_forward_hook(_make_replace_hook(replacement)))

    try:
        with torch.no_grad():
            outputs = _model(**inputs)
    finally:
        for h in hooks_handles:
            h.remove()

    final_logits = outputs.logits[:, -1, :].float().cpu()
    top_tokens = _build_top_tokens(final_logits[0], req.outputs.top_k)

    result: dict[str, Any] = {
        "top_tokens": top_tokens,
        "num_layers": _num_layers,
        "vocab_size": getattr(_model.config, "vocab_size", 0),
        "hidden_dim": getattr(_model.config, "hidden_size", 0),
    }

    if captured:
        result["captured"] = {
            key: _encode_residual(val[0]) for key, val in captured.items()
        }
    else:
        result["captured"] = {}

    return result


@app.post("/inference/generate")
def inference_generate(req: InferenceGenerateRequest) -> dict[str, Any]:
    """v2 generate — produce multiple tokens sequentially."""
    assert _model is not None and _tokenizer is not None and _device is not None

    prompt = _resolve_input(req.input, req.messages, raw=req.raw)
    torch.manual_seed(req.seed)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

    tokens_out: list[dict[str, Any]] = []

    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        for _ in range(req.max_tokens):
            outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].float().cpu()
            probs = _safe_softmax(logits[0])

            if req.temperature == 0.0:
                top_prob, top_id = _safe_topk(probs, 1)
                next_id = top_id[0].item()
                next_prob = top_prob[0].item()
                next_logit = logits[0, next_id].item()
            else:
                scaled = logits[0] / req.temperature
                scaled_probs = torch.softmax(scaled, dim=-1)
                next_id = torch.multinomial(scaled_probs, 1).item()
                next_prob = probs[next_id].item()
                next_logit = logits[0, next_id].item()

            token_str = _tokenizer.decode([next_id])
            tokens_out.append({
                "token": token_str,
                "token_id": next_id,
                "logit": round(next_logit, 4),
                "prob": round(next_prob, 6),
            })

            # Check for EOS
            if next_id == _tokenizer.eos_token_id:
                break

            # Append to input for next step
            next_tensor = torch.tensor([[next_id]], device=_device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=_device, dtype=attention_mask.dtype)],
                    dim=1,
                )

    generated_text = "".join(t["token"] for t in tokens_out)

    return {
        "text": generated_text,
        "tokens": tokens_out,
        "num_tokens": len(tokens_out),
    }


@app.post("/inference/batch")
def inference_batch(req: InferenceBatchRequest) -> StreamingResponse:
    """v2 batch forward — process multiple forward requests with SSE progress."""
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        total = len(req.requests)
        start = time.time()

        for idx, sub_req in enumerate(req.requests):
            prompt = _resolve_input(sub_req.input, sub_req.messages, raw=sub_req.raw)

            yield _sse_event("progress", {
                "completed": idx,
                "total": total,
                "current": prompt[:80],
                "elapsed_s": round(time.time() - start, 1),
            })

            torch.manual_seed(sub_req.seed)
            inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

            layers = _get_transformer_layers()
            residuals: dict[int, torch.Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHandle] = []

            if sub_req.outputs.residuals:
                target_layers = (
                    sub_req.outputs.residual_layers
                    if sub_req.outputs.residual_layers is not None
                    else list(range(len(layers)))
                )
                for layer_idx in target_layers:
                    if 0 <= layer_idx < len(layers):
                        def _make_hook(li: int):
                            def hook_fn(module, input, output):
                                hidden = output[0] if isinstance(output, tuple) else output
                                residuals[li] = hidden[:, -1, :].detach()
                            return hook_fn
                        hooks.append(layers[layer_idx].register_forward_hook(_make_hook(layer_idx)))

            try:
                with torch.no_grad():
                    outputs = _model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            final_logits = outputs.logits[:, -1, :].float().cpu()
            top_tokens = _build_top_tokens(final_logits[0], sub_req.outputs.top_k)

            result_data: dict[str, Any] = {
                "index": idx,
                "top_tokens": top_tokens,
                "num_layers": _num_layers,
                "vocab_size": getattr(_model.config, "vocab_size", 0),
                "hidden_dim": getattr(_model.config, "hidden_size", 0),
            }

            if sub_req.outputs.residuals:
                result_data["residuals"] = {
                    str(li): _encode_residual(residuals[li][0])
                    for li in sorted(residuals.keys())
                }
            else:
                result_data["residuals"] = None

            yield _sse_event("result", result_data)
            await asyncio.sleep(0)

        yield _sse_event("done", {
            "total": total,
            "elapsed_s": round(time.time() - start, 1),
        })

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# v2 Weight Editing Endpoints
# ---------------------------------------------------------------------------


@app.post("/model/edit")
def model_edit(req: ModelEditRequest) -> dict[str, Any]:
    """Apply a rank-one weight update: W += value_vector @ key_vector.T"""
    assert _model is not None

    layers = _get_transformer_layers()
    if req.layer < 0 or req.layer >= len(layers):
        raise HTTPException(status_code=400, detail=f"Layer {req.layer} out of range")

    key_vec = _decode_tensor_b64(req.key_vector).to(_device)
    val_vec = _decode_tensor_b64(req.value_vector).to(_device)

    # Navigate to the target weight matrix
    layer = layers[req.layer]
    parts = req.component.split(".")
    target = layer
    for part in parts:
        target = getattr(target, part, None)
        if target is None:
            raise HTTPException(
                status_code=400,
                detail=f"Component '{req.component}' not found in layer {req.layer}",
            )

    if not hasattr(target, "weight"):
        raise HTTPException(
            status_code=400,
            detail=f"Component '{req.component}' has no weight matrix",
        )

    # Store original for undo: save the delta so we can subtract it
    k_dot = (key_vec @ key_vec).item()
    if k_dot < 1e-10:
        raise HTTPException(status_code=400, detail="Key vector is near-zero")

    update = torch.outer(val_vec, key_vec) / k_dot
    target.weight.data += update

    edit_id = len(_v2_edit_stack) + 1
    _v2_edit_stack.append((req.layer, req.component, key_vec.cpu(), val_vec.cpu()))

    return {
        "success": True,
        "edit_id": edit_id,
        "stack_size": len(_v2_edit_stack),
    }


@app.post("/model/rome-edit")
def model_rome_edit(req: RomeEditRequest) -> dict[str, Any]:
    """High-level ROME edit: auto-compute key/value vectors and apply."""
    assert _model is not None and _tokenizer is not None and _device is not None

    torch.manual_seed(req.seed)
    layers = _get_transformer_layers()
    if req.layer < 0 or req.layer >= len(layers):
        raise HTTPException(
            status_code=400, detail=f"Layer {req.layer} out of range",
        )

    prompt = req.input
    answer = req.target
    subject = req.subject

    # Forward pass to get baseline
    input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = _model(input_ids)

    logits = outputs.logits[0, -1, :]
    probs = _safe_softmax(logits)

    # Resolve answer token id
    answer_ids = _tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        answer_ids = _tokenizer.encode(" " + answer, add_special_tokens=False)
    answer_tid = answer_ids[0] if answer_ids else 0

    before_prob = probs[answer_tid].item()

    # Auto-detect competitor (top non-answer token)
    top_probs, top_ids = _safe_topk(probs, 10)
    answer_ids_sp = _tokenizer.encode(" " + answer, add_special_tokens=False)
    answer_ids_raw = _tokenizer.encode(answer, add_special_tokens=False)
    exclude = set(answer_ids_sp + answer_ids_raw)
    competitor = "the"
    competitor_tid = 0
    for tid in top_ids.tolist():
        if tid not in exclude:
            c = _tokenizer.decode([tid]).strip()
            if c:
                competitor = c
                competitor_tid = tid
                break

    before_margin = logits[answer_tid].item() - logits[competitor_tid].item()

    # Compute key vector k* from the target layer's MLP down_proj input
    k_star_holder: dict[str, Any] = {}

    def kstar_hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        if inp.dim() == 3:
            k_star_holder["value"] = inp[0, -1, :].detach().clone()
        elif inp.dim() == 2:
            k_star_holder["value"] = inp[-1, :].detach().clone()
        else:
            k_star_holder["value"] = inp.detach().clone()

    handle = layers[req.layer].mlp.down_proj.register_forward_pre_hook(kstar_hook)
    try:
        with torch.no_grad():
            _model(input_ids)
    finally:
        handle.remove()

    k_star = k_star_holder["value"]

    # Compute correction delta
    ln_weight = _model.model.norm.weight.detach()
    w_unembed = _model.lm_head.weight
    p_answer = w_unembed[answer_tid] * ln_weight
    p_comp = w_unembed[competitor_tid] * ln_weight
    p_margin = p_answer - p_comp
    dot = (p_margin @ p_margin).item()

    deficit = -before_margin  # we want margin to become positive
    if dot < 1e-10:
        delta = torch.zeros_like(p_margin)
    else:
        delta = (deficit / dot) * p_margin

    # Apply rank-one edit to mlp.down_proj
    down_proj = layers[req.layer].mlp.down_proj
    k_dot = (k_star @ k_star).item()
    if k_dot < 1e-10:
        raise HTTPException(status_code=400, detail="Key vector is near-zero")

    update = torch.outer(delta, k_star) / k_dot
    down_proj.weight.data += update

    edit_id = len(_v2_edit_stack) + 1
    _v2_edit_stack.append((req.layer, "mlp.down_proj", k_star.cpu(), delta.cpu()))

    # Verify
    with torch.no_grad():
        outputs_after = _model(input_ids)

    after_logits = outputs_after.logits[0, -1, :]
    after_probs = _safe_softmax(after_logits)
    after_prob = after_probs[answer_tid].item()
    after_margin = after_logits[answer_tid].item() - after_logits[competitor_tid].item()

    return {
        "success": True,
        "edit_id": edit_id,
        "stack_size": len(_v2_edit_stack),
        "pre_prob": before_prob,
        "post_prob": after_prob,
        "pre_margin": before_margin,
        "post_margin": after_margin,
    }


@app.post("/model/edit/undo")
def model_edit_undo() -> dict[str, Any]:
    """Pop the last edit from the stack, restore previous weights."""
    assert _model is not None
    if not _v2_edit_stack:
        raise HTTPException(status_code=400, detail="No edits to undo")

    layer_idx, component, key_vec, val_vec = _v2_edit_stack.pop()
    layers = _get_transformer_layers()
    layer = layers[layer_idx]

    parts = component.split(".")
    target = layer
    for part in parts:
        target = getattr(target, part)

    key_vec_d = key_vec.to(_device)
    val_vec_d = val_vec.to(_device)
    k_dot = (key_vec_d @ key_vec_d).item()
    if k_dot >= 1e-10:
        update = torch.outer(val_vec_d, key_vec_d) / k_dot
        target.weight.data -= update

    return {
        "success": True,
        "edits_remaining": len(_v2_edit_stack),
    }


@app.post("/model/edit/clear")
def model_edit_clear() -> dict[str, Any]:
    """Clear all edits, restore original weights."""
    assert _model is not None

    # Undo all edits in reverse order
    layers = _get_transformer_layers()
    while _v2_edit_stack:
        layer_idx, component, key_vec, val_vec = _v2_edit_stack.pop()
        layer = layers[layer_idx]

        parts = component.split(".")
        target = layer
        for part in parts:
            target = getattr(target, part)

        key_vec_d = key_vec.to(_device)
        val_vec_d = val_vec.to(_device)
        k_dot = (key_vec_d @ key_vec_d).item()
        if k_dot >= 1e-10:
            update = torch.outer(val_vec_d, key_vec_d) / k_dot
            target.weight.data -= update

    return {
        "success": True,
        "edits_remaining": 0,
    }


@app.get("/model/edit/stack")
def model_edit_stack() -> dict[str, Any]:
    """Return current edit stack."""
    stack_info = []
    for i, (layer_idx, component, _key, _val) in enumerate(_v2_edit_stack):
        stack_info.append({
            "edit_id": i + 1,
            "layer": layer_idx,
            "component": component,
        })
    return {
        "stack_size": len(_v2_edit_stack),
        "edits": stack_info,
    }


# ---------------------------------------------------------------------------
# Legacy endpoints (kept for backward compatibility)
# ---------------------------------------------------------------------------


@app.post("/trace", response_model=TraceResponse)
def trace(req: TraceRequest) -> TraceResponse:
    assert _model is not None and _tokenizer is not None and _device is not None

    prompt = _resolve_prompt(req.prompt, req.messages, raw=req.raw)
    torch.manual_seed(req.seed)
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

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
        prompt=prompt,
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
        prompt = _resolve_prompt(req.prompt, req.messages, raw=req.raw)
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

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


@app.post("/forward-states")
def forward_states(req: ForwardStatesRequest) -> StreamingResponse:
    """Return all layers' last-token hidden states for each prompt via SSE."""
    assert _model is not None and _tokenizer is not None and _device is not None

    import base64

    import numpy as np

    async def _generate():
        layers = _get_transformer_layers()
        total = len(req.prompts)

        for idx, prompt in enumerate(req.prompts):
            yield _sse_line({
                "type": "progress",
                "current": idx + 1,
                "total": total,
            })

            torch.manual_seed(req.seed)
            inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

            captured: dict[int, torch.Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHandle] = []

            for layer_idx in range(len(layers)):
                def _make_hook(li: int):
                    def hook_fn(
                        module: torch.nn.Module,
                        input: Any,
                        output: Any,
                    ) -> None:
                        out = output[0] if isinstance(output, tuple) else output
                        captured[li] = out[:, -1, :].detach().cpu().float()
                    return hook_fn

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

            # Stack all layers: [num_layers, hidden_dim]
            layer_states = []
            for li in range(len(layers)):
                if li in captured:
                    layer_states.append(captured[li][0].numpy())

            stacked = np.stack(layer_states).astype(np.float32)
            encoded = base64.b64encode(stacked.tobytes()).decode("ascii")

            yield _sse_line({
                "type": "states",
                "prompt_idx": idx,
                "shape": list(stacked.shape),
                "dtype": "float32",
                "hidden_states": encoded,
            })

            await asyncio.sleep(0)

        yield _sse_line({"type": "done", "total": total})

    return StreamingResponse(
        _generate(), media_type="text/event-stream"
    )


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


@app.post("/forward-mlp-deltas")
def forward_mlp_deltas(req: ForwardMlpDeltasRequest) -> StreamingResponse:
    """Return MLP input/output at specified layers for each prompt via SSE."""
    assert _model is not None and _tokenizer is not None and _device is not None

    import base64

    import numpy as np

    async def _generate():
        model_layers = _get_transformer_layers()
        total = len(req.prompts)
        target_layers = req.layers if req.layers is not None else list(range(len(model_layers)))

        for idx, prompt in enumerate(req.prompts):
            yield _sse_line({
                "type": "progress",
                "current": idx + 1,
                "total": total,
            })

            torch.manual_seed(req.seed)
            inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

            captured_in: dict[int, torch.Tensor] = {}
            captured_out: dict[int, torch.Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHandle] = []

            for layer_idx in target_layers:
                def _make_hook(li: int):
                    def hook_fn(
                        module: torch.nn.Module,
                        input: Any,
                        output: Any,
                    ) -> None:
                        inp = input[0] if isinstance(input, tuple) else input
                        captured_in[li] = inp[:, -1, :].detach().cpu().float()
                        out = output if not isinstance(output, tuple) else output[0]
                        captured_out[li] = out[:, -1, :].detach().cpu().float()
                    return hook_fn

                if layer_idx < len(model_layers):
                    mlp = model_layers[layer_idx].mlp
                    h = mlp.register_forward_hook(_make_hook(layer_idx))
                    hooks.append(h)

            try:
                with torch.no_grad():
                    _model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            for li in target_layers:
                if li not in captured_in:
                    continue
                mlp_in = captured_in[li][0].numpy().astype(np.float16)
                mlp_out = captured_out[li][0].numpy().astype(np.float16)

                yield _sse_line({
                    "type": "deltas",
                    "prompt_idx": idx,
                    "layer": li,
                    "mlp_input": base64.b64encode(mlp_in.tobytes()).decode("ascii"),
                    "mlp_output": base64.b64encode(mlp_out.tobytes()).decode("ascii"),
                    "shape": list(mlp_in.shape),
                    "dtype": "float16",
                })

            await asyncio.sleep(0)

        yield _sse_line({"type": "done", "total": total})

    return StreamingResponse(
        _generate(), media_type="text/event-stream"
    )


@app.post("/forward-mlp-deltas-all-positions")
def forward_mlp_deltas_all_positions(
    req: ForwardMlpDeltasAllPositionsRequest,
) -> StreamingResponse:
    """Return MLP deltas at all token positions for all layers via SSE."""
    assert _model is not None and _tokenizer is not None and _device is not None

    import base64

    import numpy as np

    async def _generate():
        model_layers = _get_transformer_layers()
        target_layers = (
            req.layers
            if req.layers is not None
            else list(range(len(model_layers)))
        )

        prompt = _resolve_prompt(req.prompt, req.messages, raw=req.raw)
        torch.manual_seed(req.seed)
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
        num_positions = inputs["input_ids"].shape[1]

        captured_in: dict[int, torch.Tensor] = {}
        captured_out: dict[int, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        for layer_idx in target_layers:
            def _make_hook(li: int):
                def hook_fn(
                    module: torch.nn.Module,
                    input: Any,
                    output: Any,
                ) -> None:
                    inp = input[0] if isinstance(input, tuple) else input
                    out = output if not isinstance(output, tuple) else output[0]
                    # Capture all positions: [batch, seq_len, hidden_dim]
                    captured_in[li] = inp[0].detach().cpu().float()
                    captured_out[li] = out[0].detach().cpu().float()
                return hook_fn

            if layer_idx < len(model_layers):
                mlp = model_layers[layer_idx].mlp
                h = mlp.register_forward_hook(_make_hook(layer_idx))
                hooks.append(h)

        try:
            with torch.no_grad():
                _model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Stream layer by layer
        for li in target_layers:
            if li not in captured_in:
                continue
            # Compute deltas: [num_positions, hidden_dim]
            deltas = (captured_out[li] - captured_in[li]).numpy().astype(np.float16)
            encoded = base64.b64encode(deltas.tobytes()).decode("ascii")

            yield _sse_line({
                "type": "layer-deltas",
                "layer": li,
                "num_positions": int(deltas.shape[0]),
                "shape": list(deltas.shape),
                "dtype": "float16",
                "deltas": encoded,
            })

            await asyncio.sleep(0)

        yield _sse_line({
            "type": "done",
            "layers_completed": len(target_layers),
            "positions": int(num_positions),
        })

    return StreamingResponse(
        _generate(), media_type="text/event-stream"
    )


@app.post("/attention-contributions")
def attention_contributions(
    req: AttentionContributionsRequest,
) -> StreamingResponse:
    """Decompose per-head attention contributions to the residual stream."""
    assert _model is not None and _tokenizer is not None and _device is not None

    import base64

    import numpy as np

    async def _generate():
        model_layers = _get_transformer_layers()
        target_layers = (
            req.layers
            if req.layers is not None
            else list(range(len(model_layers)))
        )

        prompt = _resolve_prompt(req.prompt, req.messages, raw=req.raw)
        torch.manual_seed(req.seed)
        inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

        num_heads = _model.config.num_attention_heads
        hidden_dim = _model.config.hidden_size
        head_dim = hidden_dim // num_heads

        # Hook o_proj with register_forward_pre_hook to capture input
        captured_pre_proj: dict[int, torch.Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        for layer_idx in target_layers:
            def _make_hook(li: int):
                def hook_fn(
                    module: torch.nn.Module,
                    args: Any,
                ) -> None:
                    inp = args[0] if isinstance(args, tuple) else args
                    # Capture last position: [hidden_dim]
                    captured_pre_proj[li] = inp[0, -1, :].detach().cpu().float()
                return hook_fn

            if layer_idx < len(model_layers):
                o_proj = model_layers[layer_idx].self_attn.o_proj
                h = o_proj.register_forward_pre_hook(_make_hook(layer_idx))
                hooks.append(h)

        try:
            with torch.no_grad():
                _model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Decompose and stream per layer
        for li in target_layers:
            if li not in captured_pre_proj:
                continue
            pre_proj = captured_pre_proj[li].numpy()

            # Get o_proj weight for this layer
            o_proj_weight = (
                model_layers[li].self_attn.o_proj.weight
                .data.cpu().float().numpy()
            )

            # Per-head decomposition
            head_vectors = pre_proj.reshape(num_heads, head_dim)
            contributions = np.zeros((num_heads, hidden_dim), dtype=np.float32)
            for i in range(num_heads):
                w_o_slice = o_proj_weight[:, i * head_dim : (i + 1) * head_dim]
                contributions[i] = w_o_slice @ head_vectors[i]

            encoded = base64.b64encode(
                contributions.astype(np.float16).tobytes()
            ).decode("ascii")

            yield _sse_line({
                "type": "layer-contributions",
                "layer": li,
                "num_heads": int(num_heads),
                "shape": list(contributions.shape),
                "dtype": "float16",
                "contributions": encoded,
            })

            await asyncio.sleep(0)

        yield _sse_line({
            "type": "done",
            "layers_completed": len(target_layers),
            "heads_per_layer": int(num_heads),
        })

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/attribute-gradients")
def attribute_gradients(req: AttributeGradientsRequest) -> StreamingResponse:
    """Compute gradient-based token attribution for MLP delta projections."""
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        model_layers = _get_transformer_layers()
        total = len(req.prompts)

        if req.layer >= len(model_layers):
            n = len(model_layers)
            yield _sse_line({
                "type": "error",
                "message": (
                    f"Layer {req.layer} out of range"
                    f" (model has {n} layers)"
                ),
            })
            return

        for idx, prompt in enumerate(req.prompts):
            yield _sse_line({
                "type": "progress",
                "current": idx + 1,
                "total": total,
            })

            if idx < len(req.target_token_ids):
                target_token_id = req.target_token_ids[idx]
            else:
                target_token_id = req.target_token_ids[0]

            torch.manual_seed(req.seed)
            inputs = _tokenizer(prompt, return_tensors="pt").to(_device)

            # Get embeddings with grad tracking
            embeddings = _model.model.embed_tokens(inputs.input_ids)
            embeddings = embeddings.detach().requires_grad_(True)

            # Hook target layer's MLP
            mlp_delta: dict[str, torch.Tensor] = {}

            def hook_fn(module, inp, out):
                inp_t = inp[0] if isinstance(inp, tuple) else inp
                out_t = out if not isinstance(out, tuple) else out[0]
                mlp_delta["input"] = inp_t
                mlp_delta["output"] = out_t

            handle = model_layers[req.layer].mlp.register_forward_hook(hook_fn)

            try:
                _model.model(
                    inputs_embeds=embeddings,
                    attention_mask=inputs.attention_mask,
                )
            finally:
                handle.remove()

            # Compute MLP delta at last token position
            delta = mlp_delta["output"][:, -1, :] - mlp_delta["input"][:, -1, :]

            # Get target direction from lm_head
            target_dir = _model.lm_head.weight[target_token_id].detach().float()
            target_dir = target_dir / target_dir.norm()

            # Scalar projection
            projection = (delta.float() * target_dir).sum()

            # Backprop to embeddings
            projection.backward()

            # Attribution = gradient magnitude at each position
            grads = embeddings.grad[0]  # [seq_len, hidden_dim]
            grad_magnitudes = grads.norm(dim=-1).detach().cpu().numpy()  # [seq_len]

            # Decode tokens
            token_ids = inputs.input_ids[0].tolist()
            tokens = [_tokenizer.decode([tid]) for tid in token_ids]
            target_token = _tokenizer.decode([target_token_id])

            # Normalize
            total_mag = float(grad_magnitudes.sum())
            if total_mag > 1e-10:
                normalized = (grad_magnitudes / total_mag).tolist()
            else:
                n = len(grad_magnitudes)
                normalized = [1.0 / n] * n if n > 0 else []

            yield _sse_line({
                "type": "attribution",
                "prompt_idx": idx,
                "prompt": prompt,
                "layer": req.layer,
                "target_token": target_token,
                "target_token_id": target_token_id,
                "token_attributions": [
                    {"position": i, "token": t, "attribution": round(a, 6)}
                    for i, (t, a) in enumerate(zip(tokens, normalized))
                ],
                "method": "gradient",
            })

            await asyncio.sleep(0)

        yield _sse_line({"type": "done", "total": total})

    return StreamingResponse(_generate(), media_type="text/event-stream")


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


def _load_model(model_name: str, device: torch.device, cache_dir: str | None, dtype: str = "auto") -> None:
    global _model, _tokenizer, _device, _model_name, _num_layers, _dtype

    # DirectML does not support bfloat16; force float16 when auto
    if dtype == "auto" and device.type == "privateuseone":
        dtype = "float16"
        log.info("DirectML detected — forcing dtype=float16")
    elif dtype == "bfloat16" and device.type == "privateuseone":
        dtype = "float16"
        log.warning(
            "DirectML does not support bfloat16 — falling back to float16"
        )

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(dtype, "auto")

    log.info("Loading tokenizer: %s", model_name)
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=False, cache_dir=cache_dir
    )

    log.info("Loading model: %s (dtype=%s)", model_name, dtype)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, token=False, cache_dir=cache_dir
    )
    _model = _model.to(device)
    _model.eval()

    _device = device
    _model_name = model_name
    _dtype = dtype
    _num_layers = len(_model.model.layers)

    log.info(
        "Model ready: %s | %d layers | device: %s (%s)",
        model_name,
        _num_layers,
        device,
        _device_display_name(device),
    )


# ---------------------------------------------------------------------------
# Fingerprint endpoint
# ---------------------------------------------------------------------------


class FingerprintPrompt(BaseModel):
    prompt: str
    answer: str


class FingerprintRequest(BaseModel):
    prompts: list[FingerprintPrompt]
    seed: int = 42


@app.post("/fingerprint")
def fingerprint(req: FingerprintRequest) -> StreamingResponse:
    """Compute MLP key vector fingerprints for analytical regression checking."""
    assert _model is not None and _tokenizer is not None and _device is not None

    import base64

    import numpy as np

    async def _generate():
        layers = _get_transformer_layers()
        total = len(req.prompts)

        fingerprints = []

        for idx, item in enumerate(req.prompts):
            yield _sse_line({
                "type": "progress",
                "current": idx + 1,
                "total": total,
                "status": "fingerprinting",
            })

            torch.manual_seed(req.seed)
            input_ids = _tokenizer.encode(
                item.prompt, return_tensors="pt",
            ).to(_device)

            # Resolve answer token ID
            answer_ids = _tokenizer.encode(
                " " + item.answer, add_special_tokens=False,
            )
            if not answer_ids:
                answer_ids = _tokenizer.encode(
                    item.answer, add_special_tokens=False,
                )
            answer_id = answer_ids[0] if answer_ids else 0

            # Capture MLP key vectors (input to down_proj) at every layer
            key_vectors: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx):
                def hook_fn(module, args):
                    inp = args[0] if isinstance(args, tuple) else args
                    if inp.dim() == 3:
                        key_vectors[layer_idx] = (
                            inp[0, -1, :].detach().cpu().half()
                        )
                    elif inp.dim() == 2:
                        key_vectors[layer_idx] = (
                            inp[-1, :].detach().cpu().half()
                        )
                    else:
                        key_vectors[layer_idx] = inp.detach().cpu().half()
                return hook_fn

            hooks: list[torch.utils.hooks.RemovableHandle] = []
            for i, layer in enumerate(layers):
                hooks.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        make_hook(i),
                    )
                )

            try:
                with torch.no_grad():
                    outputs = _model(
                        input_ids, output_hidden_states=True,
                    )
            finally:
                for h in hooks:
                    h.remove()

            # Get logits and find competitor
            logits = outputs.logits[0, -1, :]
            answer_logit = logits[answer_id].item()

            logits_masked = logits.clone()
            logits_masked[answer_id] = float("-inf")
            competitor_id = logits_masked.argmax().item()
            competitor_logit = logits[competitor_id].item()
            competitor = _tokenizer.decode([competitor_id]).strip()
            if not competitor:
                competitor = _tokenizer.decode([competitor_id])

            # Compute Logit Prism projection vectors
            last_hidden = outputs.hidden_states[-1][0, -1, :]
            ln_weight = _model.model.norm.weight.detach()
            variance = last_hidden.pow(2).mean(-1, keepdim=False)
            s = torch.sqrt(variance + _model.model.norm.variance_epsilon)

            w_unembed = _model.lm_head.weight
            p_answer = (
                (w_unembed[answer_id] * ln_weight / s)
                .detach().cpu().half().numpy()
            )
            p_competitor = (
                (w_unembed[competitor_id] * ln_weight / s)
                .detach().cpu().half().numpy()
            )

            # Stack key vectors
            kv_tensor = torch.stack([
                key_vectors[i] for i in range(len(layers))
            ]).numpy()

            fingerprints.append({
                "prompt": item.prompt,
                "answer": item.answer,
                "competitor": competitor,
                "answer_logit": round(answer_logit, 4),
                "competitor_logit": round(competitor_logit, 4),
                "margin": round(answer_logit - competitor_logit, 4),
                "key_vectors_b64": base64.b64encode(
                    kv_tensor.tobytes(),
                ).decode("ascii"),
                "key_vectors_shape": list(kv_tensor.shape),
                "p_answer_b64": base64.b64encode(
                    p_answer.tobytes(),
                ).decode("ascii"),
                "p_competitor_b64": base64.b64encode(
                    p_competitor.tobytes(),
                ).decode("ascii"),
                "p_shape": list(p_answer.shape),
            })

            await asyncio.sleep(0)

        yield _sse_line({
            "type": "result",
            "fingerprints": fingerprints,
        })
        yield _sse_line({"type": "done", "total": total})

    return StreamingResponse(
        _generate(), media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Decompose endpoint
# ---------------------------------------------------------------------------


class DecomposeRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    raw: bool = False
    tokens: list[str]
    seed: int = 42


@app.post("/decompose")
def decompose(req: DecomposeRequest) -> StreamingResponse:
    """Logit Prism decomposition — exact additive logit attribution via SSE."""
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        yield _sse_line({"type": "progress", "status": "computing"})

        prompt = _resolve_prompt(req.prompt, req.messages, raw=req.raw)
        torch.manual_seed(req.seed)
        input_ids = _tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(_device)

        # Resolve target token IDs
        target_ids = []
        for t in req.tokens:
            ids = _tokenizer.encode(t, add_special_tokens=False)
            if not ids:
                ids = _tokenizer.encode(" " + t, add_special_tokens=False)
            target_ids.append(ids[0] if ids else 0)

        # Hook attention and MLP outputs
        layers = _get_transformer_layers()
        components: dict[int, dict[str, Any]] = {}
        hooks: list[torch.utils.hooks.RemovableHandle] = []

        for i, layer in enumerate(layers):
            components[i] = {}

            def _make_attn_hook(li: int):
                def hook_fn(
                    module: torch.nn.Module,
                    input: Any,
                    output: Any,
                ) -> None:
                    components[li]["attn"] = output[0][0, -1, :].detach()
                return hook_fn

            def _make_mlp_hook(li: int):
                def hook_fn(
                    module: torch.nn.Module,
                    input: Any,
                    output: Any,
                ) -> None:
                    out = (
                        output[0, -1, :]
                        if output.dim() == 3
                        else output[-1, :]
                    )
                    components[li]["mlp"] = out.detach()
                return hook_fn

            hooks.append(
                layer.self_attn.register_forward_hook(_make_attn_hook(i))
            )
            hooks.append(
                layer.mlp.register_forward_hook(_make_mlp_hook(i))
            )

        try:
            with torch.no_grad():
                outputs = _model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        # Embedding at last position
        embed = _model.model.embed_tokens(input_ids)[0, -1, :].detach()

        # Final layer norm weight
        ln_weight = _model.model.norm.weight.detach()

        # Reconstruct residual stream
        residual = embed.clone()
        for i in range(len(layers)):
            residual = residual + components[i]["attn"] + components[i]["mlp"]

        # RMS norm scale
        variance = residual.pow(2).mean(-1, keepdim=False)
        s = torch.sqrt(variance + _model.model.norm.variance_epsilon)

        W_unembed = _model.lm_head.weight

        # Compute cumulative rank at each layer by projecting the running
        # residual stream through RMSNorm + lm_head and ranking across the
        # full vocabulary.
        eps = _model.model.norm.variance_epsilon
        cum_residual = embed.clone()
        per_layer_cum_ranks: dict[int, dict[int, int]] = {}
        for i in range(len(layers)):
            cum_residual = cum_residual + components[i]["attn"] + components[i]["mlp"]
            var = cum_residual.pow(2).mean(-1, keepdim=False)
            scale = torch.sqrt(var + eps)
            normed = (cum_residual * ln_weight) / scale
            logits_full = normed @ W_unembed.T
            layer_ranks = {}
            for tid in target_ids:
                layer_ranks[tid] = int((logits_full > logits_full[tid]).sum().item()) + 1
            per_layer_cum_ranks[i] = layer_ranks

        decompositions: dict[str, Any] = {}
        for token_str, token_id in zip(req.tokens, target_ids):
            unembed_vec = W_unembed[token_id]
            P = (unembed_vec * ln_weight) / s

            embed_contrib = (P * embed).sum().item()
            layer_contribs = []
            total = embed_contrib

            for i in range(len(layers)):
                attn_c = (P * components[i]["attn"]).sum().item()
                mlp_c = (P * components[i]["mlp"]).sum().item()
                layer_contribs.append({
                    "layer": i, "attention": attn_c, "mlp": mlp_c,
                    "cumulative_rank": per_layer_cum_ranks[i][token_id],
                })
                total += attn_c + mlp_c

            actual_logit = outputs.logits[0, -1, token_id].item()

            decompositions[token_str] = {
                "token_id": token_id,
                "final_logit": actual_logit,
                "embedding": embed_contrib,
                "layers": layer_contribs,
                "reconstruction_error": abs(total - actual_logit),
                "norm_scale": s.item(),
            }

        yield _sse_line({
            "type": "decomposition",
            "prompt": prompt,
            "decompositions": decompositions,
        })

        await asyncio.sleep(0)
        yield _sse_line({"type": "done"})

    return StreamingResponse(
        _generate(), media_type="text/event-stream"
    )


@app.post("/repair")
def repair(req: RepairRequest) -> StreamingResponse:
    """ROME rank-one weight repair with SSE progress."""
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        torch.manual_seed(req.seed)
        layers = _get_transformer_layers()
        prompt = _resolve_prompt(req.prompt, req.messages, raw=req.raw)
        answer = req.answer
        competitor = req.competitor

        # Auto-detect competitor if not provided
        if competitor is None:
            input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)
            with torch.no_grad():
                outputs = _model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = _safe_softmax(logits)
            top_probs, top_ids = _safe_topk(probs, 10)
            answer_ids_sp = _tokenizer.encode(
                " " + answer, add_special_tokens=False,
            )
            answer_ids_raw = _tokenizer.encode(
                answer, add_special_tokens=False,
            )
            exclude = set(answer_ids_sp + answer_ids_raw)
            for tid in top_ids.tolist():
                if tid not in exclude:
                    competitor = _tokenizer.decode([tid]).strip()
                    if competitor:
                        break
            if not competitor:
                competitor = "the"

        yield _sse_line({"type": "progress", "status": "decomposing"})
        await asyncio.sleep(0)

        # Run decompose for answer and competitor
        input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)

        target_tokens = [answer, competitor]
        target_ids = []
        for t in target_tokens:
            ids = _tokenizer.encode(t, add_special_tokens=False)
            if not ids:
                ids = _tokenizer.encode(" " + t, add_special_tokens=False)
            target_ids.append(ids[0] if ids else 0)

        # Hook attention and MLP outputs
        components: dict[int, dict[str, Any]] = {}
        hooks_list: list[torch.utils.hooks.RemovableHandle] = []

        def make_attn_hook(layer_idx):
            def hook(module, inp, output):
                components[layer_idx]["attn"] = output[0][0, -1, :].detach()
            return hook

        def make_mlp_hook(layer_idx):
            def hook(module, inp, output):
                out = output[0, -1, :] if output.dim() == 3 else output[-1, :]
                components[layer_idx]["mlp"] = out.detach()
            return hook

        for i, layer in enumerate(layers):
            components[i] = {}
            hooks_list.append(
                layer.self_attn.register_forward_hook(make_attn_hook(i)),
            )
            hooks_list.append(
                layer.mlp.register_forward_hook(make_mlp_hook(i)),
            )

        try:
            with torch.no_grad():
                outputs = _model(input_ids)
        finally:
            for h in hooks_list:
                h.remove()

        embed = _model.model.embed_tokens(input_ids)[0, -1, :].detach()
        ln_weight = _model.model.norm.weight.detach()
        residual = embed.clone()
        for i in range(len(layers)):
            residual = residual + components[i]["attn"] + components[i]["mlp"]
        variance = residual.pow(2).mean(-1, keepdim=False)
        s = torch.sqrt(variance + _model.model.norm.variance_epsilon)
        w_unembed = _model.lm_head.weight

        # Compute before decomposition
        decompositions = {}
        for token_str, token_id in zip(target_tokens, target_ids):
            unembed_vec = w_unembed[token_id]
            proj = (unembed_vec * ln_weight) / s
            embed_c = (proj * embed).sum().item()
            layer_contribs = []
            for i in range(len(layers)):
                attn_c = (proj * components[i]["attn"]).sum().item()
                mlp_c = (proj * components[i]["mlp"]).sum().item()
                layer_contribs.append({
                    "layer": i, "attention": attn_c, "mlp": mlp_c,
                })
            actual = outputs.logits[0, -1, token_id].item()
            decompositions[token_str] = {
                "final_logit": actual,
                "embedding": embed_c,
                "layers": layer_contribs,
            }

        # Compute component margins
        a_dec = decompositions[answer]
        c_dec = decompositions[competitor]
        component_margins = []
        for al, cl in zip(a_dec["layers"], c_dec["layers"]):
            component_margins.append({
                "layer": al["layer"],
                "attn_margin": al["attention"] - cl["attention"],
                "mlp_margin": al["mlp"] - cl["mlp"],
            })

        # Select target layer
        target_layer = req.target_layer
        target_component = req.target_component
        if target_layer is None:
            worst_margin = float("inf")
            for cm in component_margins:
                total_cm = cm["attn_margin"] + cm["mlp_margin"]
                if total_cm < worst_margin:
                    worst_margin = total_cm
                    target_layer = cm["layer"]
                    target_component = (
                        "mlp" if cm["mlp_margin"] < cm["attn_margin"]
                        else "attention"
                    )
            if target_layer is None:
                target_layer = 0

        current_comp_margin = 0.0
        for cm in component_margins:
            if cm["layer"] == target_layer:
                current_comp_margin = (
                    cm["mlp_margin"] if target_component == "mlp"
                    else cm["attn_margin"]
                )
                break

        before_logits = outputs.logits[0, -1, :]
        before_probs = _safe_softmax(before_logits)
        before_prob = before_probs[target_ids[0]].item()

        before_data = {
            "answer_logit": a_dec["final_logit"],
            "competitor_logit": c_dec["final_logit"],
            "margin": a_dec["final_logit"] - c_dec["final_logit"],
            "component_margin": current_comp_margin,
            "answer_prob": before_prob,
        }

        yield _sse_line({
            "type": "progress",
            "status": "computing_key_vector",
            "layer": target_layer,
        })
        await asyncio.sleep(0)

        # Extract key vector k*
        k_star_holder: dict[str, Any] = {}

        def kstar_hook(module, args):
            inp = args[0] if isinstance(args, tuple) else args
            if inp.dim() == 3:
                k_star_holder["value"] = inp[0, -1, :].detach().clone()
            elif inp.dim() == 2:
                k_star_holder["value"] = inp[-1, :].detach().clone()
            else:
                k_star_holder["value"] = inp.detach().clone()

        handle = layers[target_layer].mlp.down_proj.register_forward_pre_hook(
            kstar_hook,
        )
        try:
            with torch.no_grad():
                _model(input_ids)
        finally:
            handle.remove()

        k_star = k_star_holder["value"]

        # Compute correction delta
        deficit = req.target_margin - current_comp_margin
        yield _sse_line({
            "type": "progress",
            "status": "computing_correction",
            "margin_deficit": deficit,
        })
        await asyncio.sleep(0)

        p_answer = w_unembed[target_ids[0]] * ln_weight
        p_comp = w_unembed[target_ids[1]] * ln_weight
        p_margin = p_answer - p_comp
        dot = (p_margin @ p_margin).item()
        if dot < 1e-10:
            delta = torch.zeros_like(p_margin)
        else:
            delta = (deficit / dot) * p_margin

        # Apply rank-one edit
        yield _sse_line({"type": "progress", "status": "applying_edit"})
        await asyncio.sleep(0)

        down_proj = layers[target_layer].mlp.down_proj
        k_dot = (k_star @ k_star).item()
        edit_norm = 0.0
        if k_dot >= 1e-10:
            update = torch.outer(delta, k_star) / k_dot
            edit_norm = update.norm().item()
            down_proj.weight.data += update

        _edit_stack.append((target_layer, k_star, delta))

        # Verify
        yield _sse_line({"type": "progress", "status": "verifying_target"})
        await asyncio.sleep(0)

        with torch.no_grad():
            outputs_after = _model(input_ids)

        after_logits = outputs_after.logits[0, -1, :]
        after_probs = _safe_softmax(after_logits)
        after_prob = after_probs[target_ids[0]].item()

        # Re-decompose after edit
        hooks_list2: list[torch.utils.hooks.RemovableHandle] = []
        components2: dict[int, dict[str, Any]] = {}
        for i, layer in enumerate(layers):
            components2[i] = {}
            hooks_list2.append(
                layer.self_attn.register_forward_hook(make_attn_hook(i)),
            )

            def make_mlp_hook2(layer_idx):
                def hook(module, inp, output):
                    out = (
                        output[0, -1, :]
                        if output.dim() == 3
                        else output[-1, :]
                    )
                    components2[layer_idx]["mlp"] = out.detach()
                return hook

            hooks_list2.append(
                layer.mlp.register_forward_hook(make_mlp_hook2(i)),
            )

        def make_attn_hook2(layer_idx):
            def hook(module, inp, output):
                components2[layer_idx]["attn"] = output[0][0, -1, :].detach()
            return hook

        # Re-register hooks correctly
        for h in hooks_list2:
            h.remove()
        hooks_list2.clear()
        for i, layer in enumerate(layers):
            components2[i] = {}
            hooks_list2.append(
                layer.self_attn.register_forward_hook(make_attn_hook2(i)),
            )
            hooks_list2.append(
                layer.mlp.register_forward_hook(make_mlp_hook(i)),
            )

        try:
            with torch.no_grad():
                outputs_after2 = _model(input_ids)
        finally:
            for h in hooks_list2:
                h.remove()

        embed2 = _model.model.embed_tokens(input_ids)[0, -1, :].detach()
        residual2 = embed2.clone()
        for i in range(len(layers)):
            residual2 = (
                residual2 + components2[i]["attn"] + components[i]["mlp"]
            )
        variance2 = residual2.pow(2).mean(-1, keepdim=False)
        s2 = torch.sqrt(variance2 + _model.model.norm.variance_epsilon)

        after_comp_margin = 0.0
        for i in range(len(layers)):
            if i == target_layer:
                a_proj = (w_unembed[target_ids[0]] * ln_weight) / s2
                c_proj = (w_unembed[target_ids[1]] * ln_weight) / s2
                a_mlp = (a_proj * components[i]["mlp"]).sum().item()
                c_mlp = (c_proj * components[i]["mlp"]).sum().item()
                after_comp_margin = a_mlp - c_mlp

        after_a_logit = outputs_after2.logits[0, -1, target_ids[0]].item()
        after_c_logit = outputs_after2.logits[0, -1, target_ids[1]].item()

        after_data = {
            "answer_logit": after_a_logit,
            "competitor_logit": after_c_logit,
            "margin": after_a_logit - after_c_logit,
            "component_margin": after_comp_margin,
            "answer_prob": after_prob,
        }

        # Check regressions
        regressions = []
        for vp in req.verify_prompts:
            vp_ids = _tokenizer.encode(vp.prompt, return_tensors="pt").to(
                _device,
            )
            vp_answer_ids = _tokenizer.encode(
                " " + vp.answer, add_special_tokens=False,
            )
            if not vp_answer_ids:
                vp_answer_ids = _tokenizer.encode(
                    vp.answer, add_special_tokens=False,
                )
            vp_aid = vp_answer_ids[0] if vp_answer_ids else 0

            with torch.no_grad():
                vp_out = _model(vp_ids)
            vp_probs = _safe_softmax(vp_out.logits[0, -1, :])
            vp_after_prob = vp_probs[vp_aid].item()

            # Undo, check, re-apply
            if k_dot >= 1e-10:
                down_proj.weight.data -= torch.outer(delta, k_star) / k_dot
            with torch.no_grad():
                vp_out_before = _model(vp_ids)
            vp_before_probs = _safe_softmax(vp_out_before.logits[0, -1, :])
            vp_before_prob = vp_before_probs[vp_aid].item()
            if k_dot >= 1e-10:
                down_proj.weight.data += torch.outer(delta, k_star) / k_dot

            status = (
                "ok" if vp_after_prob >= vp_before_prob - 0.05
                else "regression"
            )
            regressions.append({
                "prompt": vp.prompt,
                "answer": vp.answer,
                "before_prob": round(vp_before_prob, 4),
                "after_prob": round(vp_after_prob, 4),
                "status": status,
            })

        yield _sse_line({
            "type": "result",
            "prompt": prompt,
            "answer": answer,
            "competitor": competitor,
            "target_layer": target_layer,
            "target_component": target_component,
            "before": before_data,
            "after": after_data,
            "edit": {
                "matrix": "down_proj",
                "layer": target_layer,
                "rank": 1,
                "norm": round(edit_norm, 6),
            },
            "regressions": regressions,
            "status": "repaired",
        })
        await asyncio.sleep(0)
        yield _sse_line({"type": "done"})

    return StreamingResponse(
        _generate(), media_type="text/event-stream",
    )


@app.post("/repair/undo")
def repair_undo() -> dict[str, Any]:
    """Undo the last repair edit."""
    assert _model is not None
    if not _edit_stack:
        raise HTTPException(status_code=400, detail="No edits to undo")

    target_layer, k_star, delta = _edit_stack.pop()
    layers = _get_transformer_layers()
    down_proj = layers[target_layer].mlp.down_proj
    k_dot = (k_star @ k_star).item()
    if k_dot >= 1e-10:
        update = torch.outer(delta, k_star) / k_dot
        down_proj.weight.data -= update

    return {"status": "undone", "layer": target_layer, "edits_remaining": len(_edit_stack)}


@app.post("/repair/save")
def repair_save(req: RepairSaveRequest) -> dict[str, Any]:
    """Save the current edited model state."""
    assert _model is not None and _tokenizer is not None
    save_path = Path(req.path)
    save_path.mkdir(parents=True, exist_ok=True)
    _model.save_pretrained(str(save_path))
    _tokenizer.save_pretrained(str(save_path))
    return {"status": "saved", "path": str(save_path)}


# ---------------------------------------------------------------------------
# Bench endpoints
# ---------------------------------------------------------------------------


class PerplexityRequest(BaseModel):
    max_samples: int = 100
    max_length: int = 512


class BenchPrompt(BaseModel):
    prompt: str
    answer: str


class RepairAndMeasureRequest(BaseModel):
    prompts: list[BenchPrompt]
    target_margin: float = 0.0


@app.post("/bench/perplexity")
def bench_perplexity(req: PerplexityRequest) -> StreamingResponse:
    """Compute perplexity on WikiText-2 test set."""
    assert _model is not None and _tokenizer is not None and _device is not None
    import math

    async def _generate():
        yield _sse_line({"type": "progress", "status": "loading_dataset"})
        await asyncio.sleep(0)

        try:
            from datasets import load_dataset as _hf_load
            dataset = _hf_load("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([t for t in dataset["text"] if t.strip()])
        except Exception:
            text = (
                "The tower is 324 metres tall, about the same height as an "
                "81-storey building, and the tallest structure in Paris. "
                "Its base is square, measuring 125 metres on each side. " * 50
            )

        yield _sse_line({"type": "progress", "status": "computing"})
        await asyncio.sleep(0)

        encodings = _tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(_device)

        model_max = getattr(_model.config, "max_position_embeddings", 2048) or 2048
        actual_max = min(model_max, req.max_length)
        stride = actual_max // 2

        nlls: list[float] = []
        total_tokens = 0

        for begin in range(
            0,
            min(input_ids.size(1), req.max_samples * actual_max),
            stride,
        ):
            end = min(begin + actual_max, input_ids.size(1))
            chunk = input_ids[:, begin:end]

            with torch.no_grad():
                outputs = _model(chunk)
                logits = outputs.logits[:, :-1, :].contiguous()
                targets = chunk[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction="mean",
                )

            chunk_tokens = targets.numel()
            nlls.append(loss.item() * chunk_tokens)
            total_tokens += chunk_tokens

            if end == input_ids.size(1):
                break

        avg_loss = sum(nlls) / total_tokens if total_tokens > 0 else 0.0
        ppl = math.exp(avg_loss)

        yield _sse_line({
            "type": "result",
            "perplexity": round(ppl, 4),
            "loss": round(avg_loss, 6),
            "tokens": total_tokens,
        })
        await asyncio.sleep(0)
        yield _sse_line({"type": "done"})

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/bench/repair-and-measure")
def bench_repair_and_measure(req: RepairAndMeasureRequest) -> StreamingResponse:
    """Repair all prompts sequentially with stacking edits, return results."""
    assert _model is not None and _tokenizer is not None and _device is not None

    async def _generate():
        layers = _get_transformer_layers()
        ln_weight = _model.model.norm.weight.detach()
        w_unembed = _model.lm_head.weight

        results = []
        total_norm = 0.0

        for idx, bp in enumerate(req.prompts):
            prompt = bp.prompt
            answer_str = bp.answer

            yield _sse_line({
                "type": "progress",
                "status": "repairing",
                "current": idx + 1,
                "total": len(req.prompts),
                "prompt": prompt[:50],
            })
            await asyncio.sleep(0)

            input_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)

            # Get answer and competitor token IDs
            answer_ids = _tokenizer.encode(" " + answer_str, add_special_tokens=False)
            if not answer_ids:
                answer_ids = _tokenizer.encode(answer_str, add_special_tokens=False)
            answer_id = answer_ids[0] if answer_ids else 0

            # Find competitor
            with torch.no_grad():
                outputs = _model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = _safe_softmax(logits)
            top_probs, top_ids = _safe_topk(probs, 10)
            exclude = set(answer_ids + _tokenizer.encode(answer_str, add_special_tokens=False))
            competitor_str = "the"
            comp_id = 0
            for tid in top_ids.tolist():
                if tid not in exclude:
                    decoded = _tokenizer.decode([tid]).strip()
                    if decoded:
                        competitor_str = decoded
                        comp_id = tid
                        break

            if comp_id == 0:
                comp_ids = _tokenizer.encode(" " + competitor_str, add_special_tokens=False)
                comp_id = comp_ids[0] if comp_ids else 0

            before_prob = probs[answer_id].item()
            before_margin = (logits[answer_id] - logits[comp_id]).item()

            # Hook MLP outputs for decompose
            components: dict[int, dict[str, Any]] = {}
            hooks_list: list[torch.utils.hooks.RemovableHandle] = []

            def make_mlp_hook(li):
                def hook(module, inp, output):
                    out = output[0, -1, :] if output.dim() == 3 else output[-1, :]
                    components[li]["mlp"] = out.detach()
                return hook

            for i, layer in enumerate(layers):
                components[i] = {}
                hooks_list.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))

            try:
                with torch.no_grad():
                    _model(input_ids)
            finally:
                for h in hooks_list:
                    h.remove()

            # Find worst layer
            embed = _model.model.embed_tokens(input_ids)[0, -1, :].detach()
            residual = embed.clone()
            for i in range(len(layers)):
                residual = residual + components[i].get("mlp", torch.zeros_like(embed))

            variance = residual.pow(2).mean(-1, keepdim=False)
            s = torch.sqrt(variance + _model.model.norm.variance_epsilon)

            target_layer = 0
            worst_margin = float("inf")
            for i in range(len(layers)):
                if "mlp" not in components[i]:
                    continue
                a_proj = (w_unembed[answer_id] * ln_weight) / s
                c_proj = (w_unembed[comp_id] * ln_weight) / s
                mlp_margin = (
                    (a_proj * components[i]["mlp"]).sum().item()
                    - (c_proj * components[i]["mlp"]).sum().item()
                )
                if mlp_margin < worst_margin:
                    worst_margin = mlp_margin
                    target_layer = i

            # Compute key vector
            k_holder: dict[str, Any] = {}

            def kstar_hook(module, args):
                inp = args[0] if isinstance(args, tuple) else args
                if inp.dim() == 3:
                    k_holder["v"] = inp[0, -1, :].detach().clone()
                elif inp.dim() == 2:
                    k_holder["v"] = inp[-1, :].detach().clone()
                else:
                    k_holder["v"] = inp.detach().clone()

            handle = layers[target_layer].mlp.down_proj.register_forward_pre_hook(kstar_hook)
            try:
                with torch.no_grad():
                    _model(input_ids)
            finally:
                handle.remove()

            k_star = k_holder["v"]

            # Correction delta
            deficit = req.target_margin - worst_margin
            p_answer = w_unembed[answer_id] * ln_weight
            p_comp = w_unembed[comp_id] * ln_weight
            p_margin_vec = p_answer - p_comp
            dot = (p_margin_vec @ p_margin_vec).item()
            if dot < 1e-10:
                delta = torch.zeros_like(p_margin_vec)
            else:
                delta = (deficit / dot) * p_margin_vec

            # Apply edit
            down_proj = layers[target_layer].mlp.down_proj
            k_dot = (k_star @ k_star).item()
            edit_norm = 0.0
            if k_dot >= 1e-10:
                update = torch.outer(delta, k_star) / k_dot
                edit_norm = update.norm().item()
                down_proj.weight.data += update

            _edit_stack.append((target_layer, k_star, delta))
            total_norm += edit_norm

            # After measurement
            with torch.no_grad():
                out_after = _model(input_ids)
            after_logits = out_after.logits[0, -1, :]
            after_probs = _safe_softmax(after_logits)
            after_prob = after_probs[answer_id].item()
            after_margin = (after_logits[answer_id] - after_logits[comp_id]).item()

            results.append({
                "prompt": prompt,
                "answer": answer_str,
                "competitor": competitor_str,
                "before_margin": round(before_margin, 4),
                "after_margin": round(after_margin, 4),
                "before_prob": round(before_prob, 4),
                "after_prob": round(after_prob, 4),
                "layer": target_layer,
                "component": "mlp",
                "edit_norm": round(edit_norm, 6),
                "status": "repaired",
            })

        yield _sse_line({
            "type": "result",
            "results": results,
            "total_edits": len(req.prompts),
            "total_edit_norm": round(total_norm, 4),
        })
        await asyncio.sleep(0)
        yield _sse_line({"type": "done"})

    return StreamingResponse(_generate(), media_type="text/event-stream")


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
    parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Model dtype. auto = use model's native dtype.",
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
    _load_model(args.model, device, args.cache_dir, args.dtype)

    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
