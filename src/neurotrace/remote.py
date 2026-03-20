"""Remote GPU worker client for NeuroTrace."""

import json
from typing import Generator


class RemoteWorker:
    """HTTP client for the NeuroTrace GPU worker."""

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

    def health(self) -> dict:
        """Check worker health and get model info."""
        r = self.client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def trace(self, prompt: str, seed: int = 42, top_k: int = 5) -> dict:
        """Run a single trace on the remote worker."""
        r = self.client.post(f"{self.base_url}/trace", json={
            "prompt": prompt, "seed": seed, "top_k": top_k
        })
        r.raise_for_status()
        return r.json()

    def batch_ablate_stream(
        self, prompt: str, num_layers: int, seed: int = 42, top_k: int = 1
    ) -> Generator[dict, None, None]:
        """Stream batch-ablate results via SSE. Yields parsed event dicts."""
        ablations = [{"zero_mlp_layers": []}]  # baseline first
        for layer in range(num_layers):
            ablations.append({"zero_mlp_layers": [layer]})

        with self.client.stream(
            "POST",
            f"{self.base_url}/batch-ablate",
            json={
                "prompt": prompt,
                "ablations": ablations,
                "seed": seed,
                "top_k": top_k,
            },
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
        """Stream activation extraction results via SSE.

        Yields parsed event dicts with types: progress, activations, done.
        """
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
        """Stream forward-pass hidden states via SSE.

        Yields parsed event dicts with types: progress, states, done.
        The 'states' event contains base64-encoded float32 data for
        all layers' last-token hidden states: shape [num_layers, hidden_dim].
        """
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
        """Stream MLP input/output activations via SSE.

        Yields parsed event dicts with types: progress, deltas, done.
        """
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
        """Stream gradient attribution results via SSE.

        Yields parsed event dicts with types: progress, attribution, done.
        """
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
    ) -> Generator[dict, None, None]:
        """Stream per-head attention contributions via SSE.

        Yields parsed event dicts with types: layer-contributions, done.
        """
        payload: dict = {"prompt": prompt, "seed": seed}
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
    ) -> Generator[dict, None, None]:
        """Stream MLP deltas at all token positions via SSE.

        Yields parsed event dicts with types: layer-deltas, done.
        """
        payload: dict = {"prompt": prompt, "seed": seed}
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
        """Stream finetune progress via SSE. Yields parsed event dicts."""
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
    ) -> Generator[dict, None, None]:
        """Stream logit decomposition results via SSE.

        Yields parsed event dicts with types: progress, decomposition, done.
        """
        with self.client.stream(
            "POST",
            f"{self.base_url}/decompose",
            json={
                "prompt": prompt,
                "tokens": tokens,
                "seed": seed,
            },
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
        """Stream fingerprint results via SSE.

        prompts: list of {"prompt": str, "answer": str}
        Yields parsed event dicts with types: progress, result, done.
        """
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
    ) -> Generator[dict, None, None]:
        """Stream repair results via SSE."""
        payload: dict = {
            "prompt": prompt,
            "answer": answer,
            "target_margin": target_margin,
            "seed": seed,
        }
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
        """Stream perplexity computation results via SSE."""
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
        """Stream repair-and-measure results via SSE."""
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
        """Undo the last repair edit on the worker."""
        r = self.client.post(f"{self.base_url}/repair/undo")
        r.raise_for_status()
        return r.json()

    def repair_save(self, path: str) -> dict:
        """Save the edited model on the worker."""
        r = self.client.post(
            f"{self.base_url}/repair/save", json={"path": path},
        )
        r.raise_for_status()
        return r.json()

    def worker_version(self) -> dict:
        """Get worker version and status info."""
        r = self.client.get(f"{self.base_url}/version")
        r.raise_for_status()
        return r.json()

    def worker_update_stream(self) -> Generator[dict, None, None]:
        """Stream worker update progress via SSE.

        Yields parsed event dicts with types: progress, done, error.
        """
        with self.client.stream(
            "POST",
            f"{self.base_url}/update",
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    def download_adapter(self, adapter_id: str, output_path: str) -> None:
        """Download trained adapter weights to local path."""
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
