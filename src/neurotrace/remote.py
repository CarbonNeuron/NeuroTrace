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
