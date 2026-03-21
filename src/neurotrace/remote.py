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
class LayerPrediction:
    layer: int
    top_tokens: list[TokenPrediction]


@dataclass
class ForwardResult:
    top_tokens: list[TokenPrediction]
    residuals: dict[int, np.ndarray] | None
    num_layers: int
    vocab_size: int
    hidden_dim: int
    layer_predictions: list[LayerPrediction] | None = None


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
# Phase 2 types
# ---------------------------------------------------------------------------


@dataclass
class DecomposeLayerResult:
    layer: int
    attn_logit: float
    mlp_logit: float
    cumulative: float
    cumulative_rank: int | None = None


@dataclass
class DecomposeCompetitor:
    token: str
    total_logit: float
    margin: float


@dataclass
class DecomposeResult:
    answer_token_id: int
    total_logit: float
    layers: list[DecomposeLayerResult]
    competitors: list[DecomposeCompetitor]
    reconstruction_error: float


@dataclass
class RomeEditResult:
    success: bool
    edit_id: int
    stack_size: int
    pre_prob: float
    post_prob: float
    pre_margin: float
    post_margin: float


@dataclass
class FingerprintVector:
    key: np.ndarray
    projection: np.ndarray


@dataclass
class FingerprintResult:
    vectors: dict[int, FingerprintVector]  # layer -> vectors


@dataclass
class HeadContribution:
    layer: int
    head: int
    logit_contribution: float


@dataclass
class AttentionResult:
    heads: list[HeadContribution]


@dataclass
class ContrastLayerResult:
    layer: int
    cosine_similarity: float
    prompt_a_norm: float
    prompt_b_norm: float


@dataclass
class ContrastResult:
    layers: list[ContrastLayerResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class WorkerError(Exception):
    """Raised when the GPU worker returns a non-200 response."""

    def __init__(self, status_code: int, detail: str, endpoint: str):
        self.status_code = status_code
        self.detail = detail
        self.endpoint = endpoint
        super().__init__(f"Worker {endpoint} returned {status_code}: {detail}")


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
        layer_predictions: bool = False,
        layer_predictions_top_k: int = 5,
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
        if layer_predictions:
            payload["outputs"]["layer_predictions"] = True
            payload["outputs"]["layer_predictions_top_k"] = layer_predictions_top_k

        r = self.client.post(f"{self.base_url}/inference/forward", json=payload)
        r.raise_for_status()
        data = r.json()

        residuals = None
        if data.get("residuals"):
            residuals = {
                int(k): _decode_residual(v) for k, v in data["residuals"].items()
            }

        lp = None
        if data.get("layer_predictions"):
            lp = [
                LayerPrediction(
                    layer=lp_data["layer"],
                    top_tokens=_parse_top_tokens(lp_data["top_tokens"]),
                )
                for lp_data in data["layer_predictions"]
            ]

        return ForwardResult(
            top_tokens=_parse_top_tokens(data["top_tokens"]),
            residuals=residuals,
            num_layers=data["num_layers"],
            vocab_size=data["vocab_size"],
            hidden_dim=data["hidden_dim"],
            layer_predictions=lp,
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
        layer_predictions: bool = False,
        layer_predictions_top_k: int = 5,
        seed: int = 42,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[ForwardResult]:
        """Batch forward pass via /inference/batch with SSE streaming."""
        outputs: dict[str, Any] = {"top_k": top_k}
        if layer_predictions:
            outputs["layer_predictions"] = True
            outputs["layer_predictions_top_k"] = layer_predictions_top_k
        requests = [
            {
                "input": p,
                "raw": raw,
                "seed": seed,
                "outputs": outputs,
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
                    lp = None
                    if data.get("layer_predictions"):
                        lp = [
                            LayerPrediction(
                                layer=lp_data["layer"],
                                top_tokens=_parse_top_tokens(
                                    lp_data["top_tokens"],
                                ),
                            )
                            for lp_data in data["layer_predictions"]
                        ]
                    results.append(ForwardResult(
                        top_tokens=_parse_top_tokens(data["top_tokens"]),
                        residuals=residuals,
                        num_layers=data.get("num_layers", 0),
                        vocab_size=data.get("vocab_size", 0),
                        hidden_dim=data.get("hidden_dim", 0),
                        layer_predictions=lp,
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
    # v2 Phase 2: High-level inference primitives
    # -------------------------------------------------------------------

    def decompose(
        self,
        prompt: str,
        answer: str,
        *,
        raw: bool = True,
        seed: int = 42,
    ) -> DecomposeResult:
        """Logit Prism decomposition via /decompose SSE endpoint."""
        payload: dict[str, Any] = {
            "tokens": [answer],
            "seed": seed,
        }
        if raw:
            payload["prompt"] = prompt
            payload["raw"] = True
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]

        # Consume SSE stream to find the decomposition event
        data = None
        with self.client.stream(
            "POST",
            f"{self.base_url}/decompose",
            json=payload,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event.get("type") == "decomposition":
                        data = event
                        break

        if data is None:
            raise WorkerError("No decomposition event received from worker")

        decomp = data["decompositions"][answer]
        token_id = decomp["token_id"]
        total_logit = decomp["final_logit"]
        embedding = decomp["embedding"]

        # Build per-layer results with cumulative logits
        layers = []
        cumulative = embedding
        for lc in decomp["layers"]:
            cumulative += lc["attention"] + lc["mlp"]
            layers.append(DecomposeLayerResult(
                layer=lc["layer"],
                attn_logit=lc["attention"],
                mlp_logit=lc["mlp"],
                cumulative=cumulative,
                cumulative_rank=lc.get("cumulative_rank"),
            ))

        return DecomposeResult(
            answer_token_id=token_id,
            total_logit=total_logit,
            layers=layers,
            competitors=[],
            reconstruction_error=decomp.get("reconstruction_error", 0.0),
        )

    def rome_edit(
        self,
        prompt: str,
        subject: str,
        target: str,
        layer: int,
        *,
        raw: bool = True,
        seed: int = 42,
    ) -> RomeEditResult:
        """ROME rank-one weight edit via /model/rome-edit."""
        payload: dict[str, Any] = {
            "input": prompt,
            "subject": subject,
            "target": target,
            "layer": layer,
            "raw": raw,
            "seed": seed,
        }
        endpoint = "/model/rome-edit"
        r = self.client.post(f"{self.base_url}{endpoint}", json=payload)
        if r.status_code != 200:
            ct = r.headers.get("content-type", "")
            if ct.startswith("application/json"):
                detail = r.json().get("detail", r.text)
            else:
                detail = r.text
            raise WorkerError(r.status_code, str(detail), endpoint)
        data = r.json()
        return RomeEditResult(
            success=data["success"],
            edit_id=data["edit_id"],
            stack_size=data["stack_size"],
            pre_prob=data.get("pre_prob", 0.0),
            post_prob=data.get("post_prob", 0.0),
            pre_margin=data.get("pre_margin", 0.0),
            post_margin=data.get("post_margin", 0.0),
        )

    def fingerprint(
        self,
        prompt: str,
        layers: list[int],
        *,
        raw: bool = True,
        seed: int = 42,
    ) -> FingerprintResult:
        """Capture key/projection vectors via /inference/fingerprint."""
        payload: dict[str, Any] = {
            "input": prompt,
            "raw": raw,
            "seed": seed,
            "layers": layers,
        }
        r = self.client.post(
            f"{self.base_url}/inference/fingerprint", json=payload,
        )
        r.raise_for_status()
        data = r.json()
        vectors: dict[int, FingerprintVector] = {}
        for layer_str, vecs in data.get("vectors", {}).items():
            vectors[int(layer_str)] = FingerprintVector(
                key=_decode_residual(vecs["key"]),
                projection=_decode_residual(vecs["projection"]),
            )
        return FingerprintResult(vectors=vectors)

    def attention(
        self,
        prompt: str,
        answer: str,
        *,
        raw: bool = True,
        method: str = "o_proj",
        seed: int = 42,
    ) -> AttentionResult:
        """Per-head attention contributions via /inference/attention."""
        payload: dict[str, Any] = {
            "input": prompt,
            "answer": answer,
            "raw": raw,
            "method": method,
            "seed": seed,
        }
        r = self.client.post(
            f"{self.base_url}/inference/attention", json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return AttentionResult(
            heads=[
                HeadContribution(
                    layer=h["layer"],
                    head=h["head"],
                    logit_contribution=h["logit_contribution"],
                )
                for h in data.get("heads", [])
            ],
        )

    def contrast(
        self,
        prompt_a: str,
        prompt_b: str,
        *,
        raw: bool = True,
        seed: int = 42,
    ) -> ContrastResult:
        """Per-layer residual stream contrast via /inference/contrast."""
        payload: dict[str, Any] = {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "raw": raw,
            "seed": seed,
        }
        r = self.client.post(
            f"{self.base_url}/inference/contrast", json=payload,
        )
        r.raise_for_status()
        data = r.json()
        return ContrastResult(
            layers=[
                ContrastLayerResult(
                    layer=ld["layer"],
                    cosine_similarity=ld["cosine_similarity"],
                    prompt_a_norm=ld.get("prompt_a_norm", 0.0),
                    prompt_b_norm=ld.get("prompt_b_norm", 0.0),
                )
                for ld in data.get("layers", [])
            ],
        )

    def batch_edit(
        self,
        edits: list[dict[str, Any]],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[RomeEditResult]:
        """Batch ROME edits — sequential calls to /model/edit."""
        results: list[RomeEditResult] = []
        for i, edit in enumerate(edits):
            if on_progress:
                on_progress(i, len(edits))
            result = self.rome_edit(
                prompt=edit["prompt"],
                subject=edit.get("subject", ""),
                target=edit["target"],
                layer=edit["layer"],
                raw=edit.get("raw", True),
                seed=edit.get("seed", 42),
            )
            results.append(result)
        return results

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
