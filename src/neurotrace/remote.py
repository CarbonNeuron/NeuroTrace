"""Remote GPU worker client for NeuroTrace.

WorkerClient is the typed client for the v2 inference primitives.
RemoteWorker is kept as an alias for backward compatibility.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Typed dataclasses for v2 return types
# ---------------------------------------------------------------------------


@dataclass
class TokenPrediction:
    token: str
    token_id: int
    logit: float
    prob: float


@dataclass
class ForwardResult:
    top_tokens: list[TokenPrediction]
    residuals: dict[int, np.ndarray] | None
    num_layers: int
    vocab_size: int
    hidden_dim: int


@dataclass
class HookedResult:
    top_tokens: list[TokenPrediction]
    captured: dict[str, np.ndarray]
    num_layers: int
    vocab_size: int
    hidden_dim: int


@dataclass
class GenerateResult:
    text: str
    tokens: list[TokenPrediction]
    num_tokens: int


@dataclass
class Hook:
    layer: int
    component: str  # "mlp" | "attn" | "layer"
    action: str  # "zero" | "capture" | "scale" | "replace"
    scale: float | None = None
    tensor: np.ndarray | None = None


@dataclass
class EditResult:
    success: bool
    edit_id: int
    stack_size: int


@dataclass
class EditInfo:
    edit_id: int
    layer: int
    component: str


@dataclass
class HealthInfo:
    status: str
    model: str
    dtype: str
    device: str
    device_name: str
    num_layers: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionInfo:
    commit: str
    branch: str
    dirty: bool
    model: str
    device: str
    device_name: str
    uptime_seconds: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_residual(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded float16 array."""
    raw = base64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.float16).copy()


def _encode_tensor(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 float16."""
    return base64.b64encode(arr.astype(np.float16).tobytes()).decode("ascii")


def _parse_top_tokens(data: list[dict]) -> list[TokenPrediction]:
    """Parse top_tokens from JSON response."""
    return [
        TokenPrediction(
            token=t["token"],
            token_id=t.get("token_id", 0),
            logit=t.get("logit", 0.0),
            prob=t.get("prob", 0.0),
        )
        for t in data
    ]


# ---------------------------------------------------------------------------
# WorkerClient — typed v2 client
# ---------------------------------------------------------------------------


class WorkerClient:
    """Typed client for the GPU inference worker (v2 primitives)."""

    def __init__(self, base_url: str, timeout: float = 300.0):
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for remote worker support. "
                "Install with: pip install neurotrace[remote]"
            )
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    # -------------------------------------------------------------------
    # v2 Inference Primitives
    # -------------------------------------------------------------------

    def forward(
        self,
        prompt: str,
        *,
        raw: bool = True,
        top_k: int = 10,
        residual_layers: list[int] | None = None,
        seed: int = 42,
    ) -> ForwardResult:
        """Run a forward pass via /inference/forward."""
        payload: dict[str, Any] = {
            "input": prompt,
            "raw": raw,
            "seed": seed,
            "outputs": {
                "top_k": top_k,
                "residuals": residual_layers is not None,
            },
        }
        if residual_layers is not None:
            payload["outputs"]["residual_layers"] = residual_layers

        r = self.client.post(f"{self.base_url}/inference/forward", json=payload)
        r.raise_for_status()
        data = r.json()

        residuals = None
        if data.get("residuals"):
            residuals = {
                int(k): _decode_residual(v) for k, v in data["residuals"].items()
            }

        return ForwardResult(
            top_tokens=_parse_top_tokens(data["top_tokens"]),
            residuals=residuals,
            num_layers=data["num_layers"],
            vocab_size=data["vocab_size"],
            hidden_dim=data["hidden_dim"],
        )

    def hooked(
        self,
        prompt: str,
        hooks: list[Hook],
        *,
        raw: bool = True,
        top_k: int = 10,
        seed: int = 42,
    ) -> HookedResult:
        """Run a hooked forward pass via /inference/hooked."""
        hook_specs = []
        for h in hooks:
            spec: dict[str, Any] = {
                "layer": h.layer,
                "component": h.component,
                "action": h.action,
            }
            if h.scale is not None:
                spec["scale"] = h.scale
            if h.tensor is not None:
                spec["tensor"] = _encode_tensor(h.tensor)
            hook_specs.append(spec)

        payload: dict[str, Any] = {
            "input": prompt,
            "raw": raw,
            "seed": seed,
            "hooks": hook_specs,
            "outputs": {"top_k": top_k},
        }

        r = self.client.post(f"{self.base_url}/inference/hooked", json=payload)
        r.raise_for_status()
        data = r.json()

        captured = {}
        for key, b64 in data.get("captured", {}).items():
            captured[key] = _decode_residual(b64)

        return HookedResult(
            top_tokens=_parse_top_tokens(data["top_tokens"]),
            captured=captured,
            num_layers=data["num_layers"],
            vocab_size=data["vocab_size"],
            hidden_dim=data["hidden_dim"],
        )

    def generate(
        self,
        prompt: str,
        *,
        raw: bool = True,
        max_tokens: int = 20,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> GenerateResult:
        """Generate multiple tokens via /inference/generate."""
        payload: dict[str, Any] = {
            "input": prompt,
            "raw": raw,
            "seed": seed,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        r = self.client.post(f"{self.base_url}/inference/generate", json=payload)
        r.raise_for_status()
        data = r.json()

        return GenerateResult(
            text=data["text"],
            tokens=_parse_top_tokens(data["tokens"]),
            num_tokens=data["num_tokens"],
        )

    # -------------------------------------------------------------------
    # v2 Batch (SSE streaming)
    # -------------------------------------------------------------------

    def batch_forward(
        self,
        prompts: list[str],
        *,
        raw: bool = True,
        top_k: int = 10,
        seed: int = 42,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ForwardResult]:
        """Batch forward pass via /inference/batch with SSE streaming."""
        requests = [
            {
                "input": p,
                "raw": raw,
                "seed": seed,
                "outputs": {"top_k": top_k},
            }
            for p in prompts
        ]

        results: list[ForwardResult] = []
        with self.client.stream(
            "POST",
            f"{self.base_url}/inference/batch",
            json={"requests": requests},
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    # Skip event: lines
                    if line.startswith("event: "):
                        continue
                    continue
                data = json.loads(line[6:])

                if "completed" in data and on_progress:
                    on_progress(data["completed"], data.get("total", len(prompts)))

                if "top_tokens" in data:
                    residuals = None
                    if data.get("residuals"):
                        residuals = {
                            int(k): _decode_residual(v)
                            for k, v in data["residuals"].items()
                        }
                    results.append(ForwardResult(
                        top_tokens=_parse_top_tokens(data["top_tokens"]),
                        residuals=residuals,
                        num_layers=data.get("num_layers", 0),
                        vocab_size=data.get("vocab_size", 0),
                        hidden_dim=data.get("hidden_dim", 0),
                    ))

        return results

    def batch_hooked(
        self,
        prompts: list[str],
        hooks: list[Hook],
        *,
        raw: bool = True,
        top_k: int = 10,
        seed: int = 42,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[HookedResult]:
        """Batch hooked forward pass — runs individual hooked calls."""
        results: list[HookedResult] = []
        for i, prompt in enumerate(prompts):
            if on_progress:
                on_progress(i, len(prompts))
            result = self.hooked(prompt, hooks, raw=raw, top_k=top_k, seed=seed)
            results.append(result)
        return results

    # -------------------------------------------------------------------
    # v2 Weight Editing (ROME)
    # -------------------------------------------------------------------

    def edit(
        self,
        layer: int,
        component: str,
        key_vector: np.ndarray,
        value_vector: np.ndarray,
    ) -> EditResult:
        """Apply a rank-one weight edit via /model/edit."""
        payload = {
            "layer": layer,
            "component": component,
            "action": "rome_rank_one",
            "key_vector": _encode_tensor(key_vector),
            "value_vector": _encode_tensor(value_vector),
        }
        r = self.client.post(f"{self.base_url}/model/edit", json=payload)
        r.raise_for_status()
        data = r.json()
        return EditResult(
            success=data["success"],
            edit_id=data["edit_id"],
            stack_size=data["stack_size"],
        )

    def edit_undo(self) -> None:
        """Undo the last edit via /model/edit/undo."""
        r = self.client.post(f"{self.base_url}/model/edit/undo")
        r.raise_for_status()

    def edit_clear(self) -> None:
        """Clear all edits via /model/edit/clear."""
        r = self.client.post(f"{self.base_url}/model/edit/clear")
        r.raise_for_status()

    def edit_stack(self) -> list[EditInfo]:
        """Get current edit stack via /model/edit/stack."""
        r = self.client.get(f"{self.base_url}/model/edit/stack")
        r.raise_for_status()
        data = r.json()
        return [
            EditInfo(
                edit_id=e["edit_id"],
                layer=e["layer"],
                component=e["component"],
            )
            for e in data["edits"]
        ]

    # -------------------------------------------------------------------
    # Management
    # -------------------------------------------------------------------

    def health(self) -> dict:
        """Check worker health and get model info."""
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def get_model_config(self) -> dict:
        """Fetch model config from GPU worker."""
        r = self.client.get(f"{self.base_url}/model/config")
        r.raise_for_status()
        return r.json()

    def format_prompt(self, messages: list[dict]) -> dict:
        """Format chat messages using the model's chat template."""
        r = self.client.post(
            f"{self.base_url}/format", json={"messages": messages},
        )
        r.raise_for_status()
        return r.json()

    def worker_version(self) -> dict:
        """Get worker version and status info."""
        r = self.client.get(f"{self.base_url}/version")
        r.raise_for_status()
        return r.json()

    def reload_stream(
        self,
        model: str | None = None,
        dtype: str | None = None,
    ) -> Generator[dict, None, None]:
        """Stream model reload progress via SSE."""
        payload: dict = {}
        if model is not None:
            payload["model"] = model
        if dtype is not None:
            payload["dtype"] = dtype

        with self.client.stream(
            "POST",
            f"{self.base_url}/reload",
            json=payload,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def worker_update_stream(self) -> Generator[dict, None, None]:
        """Stream worker update progress via SSE."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/update",
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    # -------------------------------------------------------------------
    # Legacy methods — kept for backward compatibility with existing callers
    # -------------------------------------------------------------------

    def trace(
        self, prompt: str, seed: int = 42, top_k: int = 5, raw: bool = True,
    ) -> dict:
        """Run a single trace on the remote worker (legacy)."""
        payload: dict = {"seed": seed, "top_k": top_k}
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        r = self.client.post(f"{self.base_url}/trace", json=payload)
        r.raise_for_status()
        return r.json()

    def batch_ablate_stream(
        self, prompt: str, num_layers: int, seed: int = 42, top_k: int = 1,
        raw: bool = True,
    ) -> Generator[dict, None, None]:
        """Stream batch-ablate results via SSE (legacy)."""
        ablations = [{"zero_mlp_layers": []}]
        for layer in range(num_layers):
            ablations.append({"zero_mlp_layers": [layer]})

        payload: dict = {
            "ablations": ablations,
            "seed": seed,
            "top_k": top_k,
        }
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]

        with self.client.stream(
            "POST",
            f"{self.base_url}/batch-ablate",
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def extract_activations_stream(
        self,
        prompts: list[str],
        layer_start: int,
        layer_end: int,
        seed: int = 42,
    ) -> Generator[dict, None, None]:
        """Stream activation extraction results via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/extract-activations",
            json={
                "prompts": prompts,
                "layer_start": layer_start,
                "layer_end": layer_end,
                "seed": seed,
            },
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def forward_states_stream(
        self, prompts: list[str], seed: int = 42
    ) -> Generator[dict, None, None]:
        """Stream forward-pass hidden states via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/forward-states",
            json={"prompts": prompts, "seed": seed},
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def forward_mlp_deltas_stream(
        self,
        prompts: list[str],
        layers: list[int] | None = None,
        seed: int = 42,
    ) -> Generator[dict, None, None]:
        """Stream MLP input/output activations via SSE (legacy)."""
        payload: dict = {"prompts": prompts, "seed": seed}
        if layers is not None:
            payload["layers"] = layers

        with self.client.stream(
            "POST",
            f"{self.base_url}/forward-mlp-deltas",
            json=payload,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def attribute_gradients_stream(
        self,
        prompts: list[str],
        layer: int,
        target_token_ids: list[int],
        seed: int = 42,
    ) -> Generator[dict, None, None]:
        """Stream gradient attribution results via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/attribute-gradients",
            json={
                "prompts": prompts,
                "layer": layer,
                "target_token_ids": target_token_ids,
                "seed": seed,
            },
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def attention_contributions_stream(
        self,
        prompt: str,
        layers: list[int] | None = None,
        seed: int = 42,
        raw: bool = True,
    ) -> Generator[dict, None, None]:
        """Stream per-head attention contributions via SSE (legacy)."""
        payload: dict = {"seed": seed}
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        if layers is not None:
            payload["layers"] = layers

        with self.client.stream(
            "POST",
            f"{self.base_url}/attention-contributions",
            json=payload,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def forward_mlp_deltas_all_positions_stream(
        self,
        prompt: str,
        layers: list[int] | None = None,
        seed: int = 42,
        raw: bool = True,
    ) -> Generator[dict, None, None]:
        """Stream MLP deltas at all token positions via SSE (legacy)."""
        payload: dict = {"seed": seed}
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        if layers is not None:
            payload["layers"] = layers

        with self.client.stream(
            "POST",
            f"{self.base_url}/forward-mlp-deltas-all-positions",
            json=payload,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def finetune_stream(self, config: dict) -> Generator[dict, None, None]:
        """Stream finetune progress via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/finetune",
            json=config,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def decompose_stream(
        self,
        prompt: str,
        tokens: list[str],
        seed: int = 42,
        raw: bool = True,
    ) -> Generator[dict, None, None]:
        """Stream logit decomposition results via SSE (legacy)."""
        payload: dict = {"tokens": tokens, "seed": seed}
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]

        with self.client.stream(
            "POST",
            f"{self.base_url}/decompose",
            json=payload,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def fingerprint_stream(
        self,
        prompts: list[dict],
        seed: int = 42,
    ) -> Generator[dict, None, None]:
        """Stream fingerprint results via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/fingerprint",
            json={"prompts": prompts, "seed": seed},
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def repair_stream(
        self,
        prompt: str,
        answer: str,
        competitor: str | None = None,
        target_layer: int | None = None,
        target_component: str = "mlp",
        target_margin: float = 0.0,
        verify_prompts: list[dict] | None = None,
        seed: int = 42,
        raw: bool = True,
    ) -> Generator[dict, None, None]:
        """Stream repair results via SSE (legacy)."""
        payload: dict = {
            "answer": answer,
            "target_margin": target_margin,
            "seed": seed,
        }
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        if competitor is not None:
            payload["competitor"] = competitor
        if target_layer is not None:
            payload["target_layer"] = target_layer
            payload["target_component"] = target_component
        if verify_prompts:
            payload["verify_prompts"] = verify_prompts

        with self.client.stream(
            "POST",
            f"{self.base_url}/repair",
            json=payload,
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def perplexity_stream(
        self,
        max_samples: int = 100,
        max_length: int = 512,
    ) -> Generator[dict, None, None]:
        """Stream perplexity computation results via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/bench/perplexity",
            json={"max_samples": max_samples, "max_length": max_length},
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def repair_and_measure_stream(
        self,
        prompts: list[dict],
        target_margin: float = 0.0,
    ) -> Generator[dict, None, None]:
        """Stream repair-and-measure results via SSE (legacy)."""
        with self.client.stream(
            "POST",
            f"{self.base_url}/bench/repair-and-measure",
            json={"prompts": prompts, "target_margin": target_margin},
            timeout=600.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def repair_undo(self) -> dict:
        """Undo the last repair edit on the worker (legacy)."""
        r = self.client.post(f"{self.base_url}/repair/undo")
        r.raise_for_status()
        return r.json()

    def repair_save(self, path: str) -> dict:
        """Save the edited model on the worker (legacy)."""
        r = self.client.post(
            f"{self.base_url}/repair/save", json={"path": path},
        )
        r.raise_for_status()
        return r.json()

    def download_adapter(self, adapter_id: str, output_path: str) -> None:
        """Download trained adapter weights to local path (legacy)."""
        import io
        import tarfile

        with self.client.stream(
            "GET", f"{self.base_url}/finetune/{adapter_id}/download"
        ) as response:
            response.raise_for_status()
            buf = io.BytesIO()
            for chunk in response.iter_bytes():
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                tar.extractall(path=output_path, filter="data")


# Backward compatibility alias
RemoteWorker = WorkerClient
