"""Trace-related CLI commands: trace, list, inspect, diff, predict, decode, compare."""

import json

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.cli import (
    _maybe_load_adapter,
    _resolve_device,
    _resolve_trace_id,
    console,
    err_console,
)
from neurotrace.storage import TraceDB


def _slugify(text: str, max_len: int = 30) -> str:
    """Slugify text for use as a trace label."""
    import re

    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")[:max_len].rstrip("-")
    return slug


def _decode_tokens(tokenizer, token_ids: list[int]) -> list[dict]:
    """Decode token IDs to strings, hex bytes, and special flag."""
    results = []
    special_ids = set(tokenizer.all_special_ids)
    for tid in sorted(token_ids):
        decoded = tokenizer.decode(tid)
        raw_token = tokenizer.convert_ids_to_tokens(tid)
        token_bytes = decoded.encode("utf-8")
        hex_bytes = " ".join(f"{b:02x}" for b in token_bytes)
        results.append(
            {
                "token_id": tid,
                "string": decoded,
                "raw_token": raw_token,
                "hex_bytes": hex_bytes,
                "is_special": tid in special_ids,
            }
        )
    return results


@click.command()
@click.option("--model", default=None, help="HuggingFace model name or path.")
@click.option("--prompt", default=None, help="Prompt text to trace.")
@click.option(
    "--prompts-file",
    default=None,
    type=click.Path(exists=True),
    help="File with one prompt per line.",
)
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--label", default=None, help="Optional label for the trace.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--capture-mode",
    default="full",
    type=click.Choice(["full", "light"]),
    help="Capture mode.",
)
@click.option("--layer-stride", default=1, type=int, help="Layer stride for capture.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--raw", "use_raw", is_flag=True, default=None,
              help="Raw inference (no chat template). Default when --remote is used.")
@click.option("--chat", "use_chat", is_flag=True, default=False,
              help="Force chat template mode (override raw default for --remote).")
def trace(
    model,
    prompt,
    prompts_file,
    db,
    label,
    seed,
    capture_mode,
    layer_stride,
    adapter,
    remote,
    device,
    use_raw,
    use_chat,
):
    """Run a forward-pass trace and store results."""
    if use_raw and use_chat:
        raise click.UsageError("Cannot use both --raw and --chat.")
    if prompt is None and prompts_file is None:
        raise click.UsageError("Must provide either --prompt or --prompts-file.")
    if prompt is not None and prompts_file is not None:
        raise click.UsageError("Cannot provide both --prompt and --prompts-file.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Resolve raw mode: default True for --remote, False for local
    raw = use_raw if use_raw is not None else (remote is not None and not use_chat)

    # Collect prompts
    if prompts_file is not None:
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [prompt]

    if remote is not None:
        import uuid
        from datetime import datetime, timezone

        from neurotrace.remote import WorkerClient
        from neurotrace.types import LayerSnapshot, TraceMetadata, TraceResult

        worker = WorkerClient(remote)
        health = worker.health()
        model_name = health["model"]
        num_layers = health["num_layers"]
        err_console.print(
            f"GPU: {health.get('device_name', 'unknown')} via {remote}"
        )

        db_conn = TraceDB(db)
        try:
            for i, p in enumerate(prompts):
                err_console.print(
                    f"Tracing prompt {i + 1}/{len(prompts)}..."
                )
                all_layers = list(range(num_layers))
                fwd = worker.forward(
                    p, raw=raw, top_k=5,
                    residual_layers=all_layers, seed=seed,
                )

                # Build layer snapshots from residuals
                layer_snapshots = []
                for layer_idx in range(fwd.num_layers):
                    import numpy as np

                    residual = None
                    res_norm = 0.0
                    if fwd.residuals and layer_idx in fwd.residuals:
                        residual = fwd.residuals[layer_idx].astype(
                            np.float32,
                        )
                        res_norm = float(np.linalg.norm(residual))

                    snap = LayerSnapshot(
                        layer_index=layer_idx,
                        residual_in=None,
                        residual_out=residual,
                        attention_weights=None,
                        attention_output=None,
                        mlp_in=None,
                        mlp_out=None,
                        ln_values=None,
                        residual_in_norm=0.0,
                        residual_out_norm=res_norm,
                        attention_entropy=[],
                        mlp_activation_mag=0.0,
                        top1_token=(
                            fwd.top_tokens[0].token_id
                            if fwd.top_tokens else 0
                        ),
                        top1_prob=(
                            fwd.top_tokens[0].prob
                            if fwd.top_tokens else 0.0
                        ),
                    )
                    layer_snapshots.append(snap)

                # Build metadata
                meta = TraceMetadata(
                    trace_id=str(uuid.uuid4()),
                    model_name=model_name,
                    model_revision="remote",
                    prompt=p,
                    token_ids=[],
                    tokens=[],
                    num_layers=fwd.num_layers,
                    num_heads=0,
                    hidden_size=fwd.hidden_dim,
                    param_count=0,
                    device="remote",
                    dtype="remote",
                    random_seed=seed,
                    label=label,
                    capture_mode="remote",
                    layer_stride=1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

                result = TraceResult(
                    metadata=meta,
                    layer_snapshots=layer_snapshots,
                    token_predictions=[],
                    final_logits=None,
                )

                db_conn.write_trace(result)
                console.print(
                    f"[green]Stored trace {result.metadata.trace_id}[/green]"
                )
        finally:
            db_conn.close()
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model
        from neurotrace.tracer import Tracer

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        tracer = Tracer(
            model_obj, tokenizer, capture_mode=capture_mode, layer_stride=layer_stride
        )
        db_conn = TraceDB(db)

        try:
            for i, p in enumerate(prompts):
                desc = f"Tracing prompt {i + 1}/{len(prompts)}..."
                progress.update(task, description=desc)
                result = tracer.trace(p, label=label, seed=seed)

                progress.update(task, description="Storing trace...")
                db_conn.write_trace(result)
                console.print(f"[green]Stored trace {result.metadata.trace_id}[/green]")
        finally:
            db_conn.close()


@click.command("list")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="Filter by model name.")
def list_traces(db, model):
    """List all stored traces."""
    db_conn = TraceDB(db)
    try:
        rows = db_conn.list_traces(model_filter=model)
    finally:
        db_conn.close()

    if not rows:
        console.print("[dim]No traces found.[/dim]")
        return

    table = Table(title="Traces")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Label")
    table.add_column("Model")
    table.add_column("Prompt")
    table.add_column("Top Prediction")
    table.add_column("Timestamp")

    for r in rows:
        prompt_display = r["prompt"]
        if len(prompt_display) > 60:
            prompt_display = prompt_display[:57] + "..."
        top_pred = r["top_prediction"] or ""
        if r["top_prediction_prob"]:
            top_pred += f" ({r['top_prediction_prob']:.2f})"
        table.add_row(
            r["trace_id"][:8],
            r["label"] or "",
            r["model_name"],
            prompt_display,
            top_pred,
            r["timestamp"],
        )

    console.print(table)


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--trace-id", required=True, help="Trace ID or 'latest'.")
@click.option("--layer", default=None, type=int, help="Show only this layer.")
@click.option(
    "--head", default=None, type=int, help="Show specific head (requires --layer)."
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def inspect(db, trace_id, layer, head, output_json):
    """Inspect a stored trace."""
    if head is not None and layer is None:
        raise click.UsageError("--head requires --layer.")

    db_conn = TraceDB(db)
    try:
        trace_id = _resolve_trace_id(db_conn, trace_id)
        result = db_conn.read_trace(trace_id)
        layer_stats = db_conn.get_layer_stats(trace_id)
    except ValueError as e:
        db_conn.close()
        raise click.ClickException(str(e))
    finally:
        db_conn.close()

    if output_json:
        meta = result.metadata
        meta_dict = {
            "trace_id": meta.trace_id,
            "model_name": meta.model_name,
            "model_revision": meta.model_revision,
            "prompt": meta.prompt,
            "token_ids": meta.token_ids,
            "tokens": meta.tokens,
            "num_layers": meta.num_layers,
            "num_heads": meta.num_heads,
            "hidden_size": meta.hidden_size,
            "param_count": meta.param_count,
            "device": meta.device,
            "dtype": meta.dtype,
            "random_seed": meta.random_seed,
            "label": meta.label,
            "capture_mode": meta.capture_mode,
            "layer_stride": meta.layer_stride,
            "timestamp": meta.timestamp,
        }
        predictions = [
            {
                "position": p.position,
                "top_k_tokens": p.top_k_tokens,
                "top_k_probs": p.top_k_probs,
                "top_k_strings": p.top_k_strings,
            }
            for p in result.token_predictions
        ]

        # Filter by layer if requested
        stats = layer_stats
        if layer is not None:
            stats = [s for s in stats if s["layer_index"] == layer]

        output = {
            "metadata": meta_dict,
            "layer_stats": stats,
            "token_predictions": predictions,
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    meta = result.metadata
    console.print(f"\n[bold]Trace:[/bold] {meta.trace_id}")
    console.print(f"[bold]Model:[/bold] {meta.model_name} (rev {meta.model_revision})")
    console.print(f"[bold]Prompt:[/bold] {meta.prompt}")
    console.print(f"[bold]Tokens:[/bold] {len(meta.token_ids)}")
    console.print(
        f"[bold]Layers:[/bold] {meta.num_layers}  [bold]Heads:[/bold] {meta.num_heads}"
    )
    console.print(
        f"[bold]Capture:[/bold] {meta.capture_mode}"
        f"  [bold]Stride:[/bold] {meta.layer_stride}"
    )
    console.print()

    # Filter stats
    stats = layer_stats
    if layer is not None:
        stats = [s for s in stats if s["layer_index"] == layer]

    table = Table(title="Layer Stats")
    table.add_column("Layer", justify="right")
    table.add_column("Res-In Norm", justify="right")
    table.add_column("Res-Out Norm", justify="right")
    table.add_column("Attn Entropy", justify="right")
    table.add_column("MLP Mag", justify="right")
    table.add_column("Top-1 Token", justify="right")
    table.add_column("Top-1 Prob", justify="right")

    for s in stats:
        entropy_str = ""
        if s["attention_entropy"]:
            avg_ent = sum(s["attention_entropy"]) / len(s["attention_entropy"])
            entropy_str = f"{avg_ent:.3f}"
        table.add_row(
            str(s["layer_index"]),
            f"{s['residual_in_norm']:.3f}",
            f"{s['residual_out_norm']:.3f}",
            entropy_str,
            f"{s['mlp_activation_mag']:.3f}",
            str(s["top1_token"]),
            f"{s['top1_prob']:.3f}",
        )

    console.print(table)


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--trace-a", required=True, help="First trace ID.")
@click.option("--trace-b", required=True, help="Second trace ID.")
@click.option(
    "--cosine-threshold",
    default=0.95,
    type=float,
    help="Cosine similarity threshold for flagging.",
)
@click.option(
    "--kl-threshold",
    default=0.5,
    type=float,
    help="KL divergence threshold for flagging.",
)
@click.option(
    "--sort-by",
    default="layer",
    type=click.Choice(["layer", "cosine", "kl"]),
    help="Sort order for diff table.",
)
@click.option("--flagged-only", is_flag=True, help="Show only flagged layers.")
@click.option("--head-detail", is_flag=True, help="Show critical head details.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def diff(
    db,
    trace_a,
    trace_b,
    cosine_threshold,
    kl_threshold,
    sort_by,
    flagged_only,
    head_detail,
    output_json,
):
    """Compare two traces and show divergence metrics."""
    from neurotrace.analyzer import compute_diff

    db_conn = TraceDB(db)
    try:
        trace_a_id = _resolve_trace_id(db_conn, trace_a)
        trace_b_id = _resolve_trace_id(db_conn, trace_b)
        result_a = db_conn.read_trace(trace_a_id)
        result_b = db_conn.read_trace(trace_b_id)
    except ValueError as e:
        db_conn.close()
        raise click.ClickException(str(e))
    finally:
        db_conn.close()

    diff_result = compute_diff(
        result_a, result_b, cosine_threshold=cosine_threshold, kl_threshold=kl_threshold
    )

    if output_json:
        output = {
            "trace_a_id": diff_result.trace_a_id,
            "trace_b_id": diff_result.trace_b_id,
            "first_divergence_layer": diff_result.first_divergence_layer,
            "layer_metrics": [
                {
                    "layer_index": m.layer_index,
                    "cosine_similarity": m.cosine_similarity,
                    "top1_changed": m.top1_changed,
                    "kl_divergence": m.kl_divergence,
                    "flagged": m.flagged,
                    "trace_a_top1": m.trace_a_top1,
                    "trace_a_top1_prob": m.trace_a_top1_prob,
                    "trace_b_top1": m.trace_b_top1,
                    "trace_b_top1_prob": m.trace_b_top1_prob,
                }
                for m in diff_result.layer_metrics
            ],
            "critical_heads": [
                {"layer": h[0], "head": h[1], "js_divergence": h[2]}
                for h in diff_result.critical_heads
            ],
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    metrics = list(diff_result.layer_metrics)
    if flagged_only:
        metrics = [m for m in metrics if m.flagged]

    if sort_by == "cosine":
        metrics.sort(key=lambda m: m.cosine_similarity)
    elif sort_by == "kl":
        metrics.sort(key=lambda m: m.kl_divergence, reverse=True)

    table = Table(
        title=f"Diff: {diff_result.trace_a_id[:8]} vs {diff_result.trace_b_id[:8]}"
    )
    table.add_column("Layer", justify="right")
    table.add_column("Cosine Sim", justify="right")
    table.add_column("Top-1 Changed")
    table.add_column("KL Div", justify="right")
    table.add_column("Flagged")

    for m in metrics:
        style = "red" if m.flagged else None
        table.add_row(
            str(m.layer_index),
            f"{m.cosine_similarity:.4f}",
            "YES" if m.top1_changed else "no",
            f"{m.kl_divergence:.4f}",
            "FLAGGED" if m.flagged else "",
            style=style,
        )

    console.print(table)

    # Summary
    flagged_count = sum(1 for m in diff_result.layer_metrics if m.flagged)
    console.print(
        f"\n[bold]Flagged layers:[/bold] "
        f"{flagged_count}/{len(diff_result.layer_metrics)}"
    )
    if diff_result.first_divergence_layer is not None:
        console.print(
            f"[bold]First divergence:[/bold] layer {diff_result.first_divergence_layer}"
        )

    if head_detail and diff_result.critical_heads:
        console.print("\n[bold]Critical Heads (top JS divergence):[/bold]")
        head_table = Table()
        head_table.add_column("Layer", justify="right")
        head_table.add_column("Head", justify="right")
        head_table.add_column("JS Divergence", justify="right")
        for layer_idx, head_idx, js_div in diff_result.critical_heads:
            head_table.add_row(str(layer_idx), str(head_idx), f"{js_div:.6f}")
        console.print(head_table)


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--trace-id", required=True, help="Trace ID, label, or 'latest'.")
@click.option("--top-k", default=5, type=int, help="Top predictions per layer.")
@click.option(
    "--changes-only", is_flag=True, help="Only show layers where top-1 changed."
)
@click.option("--layers", default=None, help="Comma-separated layer indices to show.")
@click.option("--track", default=None, help="Token string to track across all layers.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def predict(
    db, trace_id, top_k, changes_only, layers, track,
    output_json, adapter, remote, device,
):
    """Show top-K token predictions at every layer from a stored trace."""
    import torch

    db_conn = TraceDB(db)
    try:
        trace_id = _resolve_trace_id(db_conn, trace_id)
        result = db_conn.read_trace(trace_id)
    except ValueError as e:
        db_conn.close()
        raise click.ClickException(str(e))
    finally:
        db_conn.close()

    # Check that residual_out vectors are available
    has_residuals = any(
        snap.residual_out is not None for snap in result.layer_snapshots
    )
    if not has_residuals:
        raise click.ClickException(
            "Residuals not stored in this trace. "
            "Re-trace with current version (layer_stride=1) to enable predict."
        )

    # Parse --layers filter
    layer_filter = None
    if layers is not None:
        layer_filter = set(int(x.strip()) for x in layers.split(","))

    # Load model for lm_head projection and tokenizer
    if remote is not None:
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
        health = worker.health()
        model_name = health["model"]
    else:
        model_name = result.metadata.model_name
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model_name, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Projecting layers...")

    from neurotrace.models import get_lm_head_and_norm

    lm_head, final_ln = get_lm_head_and_norm(model_obj)

    # Resolve --track token to ID
    track_token_id = None
    if track is not None:
        vocab = tokenizer.get_vocab()
        # Try exact match first, then with leading space (common BPE pattern)
        candidates = [track, f"\u2581{track}", f" {track}"]
        for candidate in candidates:
            if candidate in vocab:
                track_token_id = vocab[candidate]
                break
        if track_token_id is None:
            # Encode the string and take the first token
            encoded = tokenizer.encode(track, add_special_tokens=False)
            if encoded:
                track_token_id = encoded[0]
            else:
                raise click.ClickException(f"Could not resolve token: {track!r}")

    # Project each layer's residual_out through lm_head
    layer_predictions = []
    prev_ranking = {}  # token_id -> rank in previous layer

    for snap in result.layer_snapshots:
        if snap.residual_out is None:
            continue
        if layer_filter is not None and snap.layer_index not in layer_filter:
            continue

        with torch.no_grad():
            res_tensor = torch.tensor(snap.residual_out, dtype=torch.float32).unsqueeze(
                0
            )
            if final_ln is not None:
                res_tensor = final_ln(res_tensor)
            layer_logits = lm_head(res_tensor.squeeze(0))
            # Use last token position
            layer_probs = torch.softmax(layer_logits[-1], dim=-1)

            topk_result = torch.topk(layer_probs, k=top_k)
            topk_ids = topk_result.indices.tolist()
            topk_probs = topk_result.values.tolist()
            topk_strings = [tokenizer.decode(tid) for tid in topk_ids]

        # Build current ranking map
        current_ranking = {tid: rank for rank, tid in enumerate(topk_ids)}

        # Compute rank change annotations
        annotations = []
        for rank, tid in enumerate(topk_ids):
            if not prev_ranking:
                annotations.append("")
            elif tid not in prev_ranking:
                annotations.append("NEW")
            elif prev_ranking[tid] == rank:
                annotations.append("--")
            elif prev_ranking[tid] > rank:
                annotations.append(f"^ from #{prev_ranking[tid] + 1}")
            else:
                annotations.append(f"v from #{prev_ranking[tid] + 1}")

        # Track specific token
        track_info = None
        if track_token_id is not None:
            track_prob = float(layer_probs[track_token_id].item())
            # Find rank (search full sorted probs)
            track_rank = int((layer_probs >= track_prob).sum().item())
            track_info = {
                "token_id": track_token_id,
                "token_str": tokenizer.decode(track_token_id),
                "rank": track_rank,
                "prob": track_prob,
            }

        entry = {
            "layer_index": snap.layer_index,
            "top_k_ids": topk_ids,
            "top_k_probs": topk_probs,
            "top_k_strings": topk_strings,
            "annotations": annotations,
            "track": track_info,
        }

        # For --changes-only: skip if top-1 didn't change
        if changes_only and prev_ranking:
            prev_top1 = next((tid for tid, r in prev_ranking.items() if r == 0), None)
            if prev_top1 == topk_ids[0]:
                # Update prev_ranking but skip display
                prev_ranking = current_ranking
                continue

        layer_predictions.append(entry)
        prev_ranking = current_ranking

    if output_json:
        click.echo(json.dumps(layer_predictions, indent=2, default=str))
        return

    # Rich output
    if not layer_predictions:
        console.print("[dim]No layers to display.[/dim]")
        return

    for entry in layer_predictions:
        console.print(f"\n[bold]Layer {entry['layer_index']}[/bold] " + "-" * 50)
        for rank, (tid, prob, s, ann) in enumerate(
            zip(
                entry["top_k_ids"],
                entry["top_k_probs"],
                entry["top_k_strings"],
                entry["annotations"],
            )
        ):
            ann_str = f"  [dim]{ann}[/dim]" if ann else ""
            console.print(f"  {rank + 1}. {repr(s):20s} ({prob:.3f}){ann_str}")
        if entry["track"] is not None:
            t = entry["track"]
            console.print(
                f"  [yellow]>> {repr(t['token_str'])}: "
                f"rank #{t['rank']}, prob {t['prob']:.4f}[/yellow]"
            )


@click.command()
@click.option("--model", default=None, help="HuggingFace model name or path.")
@click.option(
    "--tokens",
    default=None,
    multiple=True,
    type=int,
    help="Token IDs to decode.",
)
@click.option(
    "--from-trace",
    default=None,
    help="Trace ID — decode all unique top-1 tokens.",
)
@click.option("--db", default=None, help="Path to DuckDB database file.")
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
def decode(model, tokens, from_trace, db, remote):
    """Decode token IDs to human-readable strings."""
    if not tokens and from_trace is None:
        raise click.UsageError("Must provide --tokens or --from-trace.")
    if from_trace is not None and db is None:
        raise click.UsageError("--from-trace requires --db.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if remote is not None:
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
        health = worker.health()
        model = health["model"]

    token_ids = list(tokens)

    if from_trace is not None:
        db_conn = TraceDB(db)
        try:
            trace_id = _resolve_trace_id(db_conn, from_trace)
            stats = db_conn.get_layer_stats(trace_id)
        finally:
            db_conn.close()
        unique_ids = {s["top1_token"] for s in stats}
        token_ids = sorted(unique_ids)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading tokenizer...", total=None)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, token=False)
        progress.update(task, description="Done.")

    decoded = _decode_tokens(tokenizer, token_ids)

    table = Table(title="Token Decode")
    table.add_column("Token ID", justify="right", style="cyan")
    table.add_column("String")
    table.add_column("Raw Token")
    table.add_column("Bytes")
    table.add_column("Special")

    for d in decoded:
        table.add_row(
            str(d["token_id"]),
            repr(d["string"]),
            str(d["raw_token"]),
            d["hex_bytes"],
            "YES" if d["is_special"] else "",
        )

    console.print(table)


@click.command()
@click.option("--model", default=None, help="HuggingFace model name or path.")
@click.option("--prompt-a", required=True, help="First prompt.")
@click.option("--prompt-b", required=True, help="Second prompt.")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--light", is_flag=True, help="Use light capture mode.")
@click.option("--flagged-only", is_flag=True, help="Show only flagged layers.")
@click.option("--head-detail", is_flag=True, help="Show critical head details.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def compare(
    model,
    prompt_a,
    prompt_b,
    db,
    seed,
    light,
    flagged_only,
    head_detail,
    output_json,
    remote,
    device,
):
    """Trace two prompts, diff them, and show decoded results."""
    from neurotrace.analyzer import compute_diff
    from neurotrace.models import load_model
    from neurotrace.tracer import Tracer

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if remote is not None:
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
        health = worker.health()
        model = health["model"]
        err_console.print(
            f"GPU: {health.get('device_name', 'unknown')} via {remote}"
        )

    capture_mode = "light" if light else "full"
    db_conn = TraceDB(db)

    status_console = err_console if output_json else console

    def find_or_trace(prompt, tracer_obj, tokenizer_obj):
        """Return trace_id, reusing existing trace if possible."""
        existing = db_conn.find_existing_trace(
            model,
            prompt,
            seed,
            capture_mode,
        )
        if existing is not None:
            status_console.print(f"[dim]Reusing existing trace {existing[:8]}[/dim]")
            return existing

        label = _slugify(prompt)
        result = tracer_obj.trace(prompt, label=label, seed=seed)
        db_conn.write_trace(result)
        status_console.print(
            f"[green]Traced: {result.metadata.trace_id[:8]}[/green]"
            f" ({repr(prompt[:40])})"
        )
        return result.metadata.trace_id

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        progress.update(task, description="Model loaded.")

        tracer = Tracer(
            model_obj,
            tokenizer,
            capture_mode=capture_mode,
        )

        progress.update(task, description="Tracing prompt A...")
        trace_a_id = find_or_trace(prompt_a, tracer, tokenizer)

        progress.update(task, description="Tracing prompt B...")
        trace_b_id = find_or_trace(prompt_b, tracer, tokenizer)

        progress.update(task, description="Done.")

    # Load traces and diff
    result_a = db_conn.read_trace(trace_a_id)
    result_b = db_conn.read_trace(trace_b_id)
    db_conn.close()

    diff_result = compute_diff(result_a, result_b)

    # Build token decode lookup from both traces
    all_top1_ids = set()
    for m in diff_result.layer_metrics:
        all_top1_ids.add(m.trace_a_top1)
        all_top1_ids.add(m.trace_b_top1)
    token_lookup = {}
    for tid in all_top1_ids:
        token_lookup[tid] = tokenizer.decode(tid)

    if output_json:
        output = {
            "trace_a_id": diff_result.trace_a_id,
            "trace_b_id": diff_result.trace_b_id,
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "first_divergence_layer": diff_result.first_divergence_layer,
            "token_legend": {str(k): v for k, v in token_lookup.items()},
            "layer_metrics": [
                {
                    "layer_index": m.layer_index,
                    "cosine_similarity": m.cosine_similarity,
                    "top1_changed": m.top1_changed,
                    "kl_divergence": m.kl_divergence,
                    "flagged": m.flagged,
                    "trace_a_top1": m.trace_a_top1,
                    "trace_a_top1_str": token_lookup.get(m.trace_a_top1, "?"),
                    "trace_a_top1_prob": m.trace_a_top1_prob,
                    "trace_b_top1": m.trace_b_top1,
                    "trace_b_top1_str": token_lookup.get(m.trace_b_top1, "?"),
                    "trace_b_top1_prob": m.trace_b_top1_prob,
                }
                for m in diff_result.layer_metrics
            ],
            "critical_heads": [
                {"layer": h[0], "head": h[1], "js_divergence": h[2]}
                for h in diff_result.critical_heads
            ],
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    metrics = list(diff_result.layer_metrics)
    if flagged_only:
        metrics = [m for m in metrics if m.flagged]

    table = Table(title=f"Compare: {trace_a_id[:8]} vs {trace_b_id[:8]}")
    table.add_column("Layer", justify="right")
    table.add_column("Cos Sim", justify="right")
    table.add_column("Top-1 Changed")
    table.add_column("KL Div", justify="right")
    table.add_column("Flagged")
    table.add_column("Trace A Top-1")
    table.add_column("Trace B Top-1")

    for m in metrics:
        style = "red" if m.flagged else None
        a_str = token_lookup.get(m.trace_a_top1, "?")
        b_str = token_lookup.get(m.trace_b_top1, "?")
        table.add_row(
            str(m.layer_index),
            f"{m.cosine_similarity:.4f}",
            "YES" if m.top1_changed else "no",
            f"{m.kl_divergence:.4f}",
            "FLAGGED" if m.flagged else "",
            f"{repr(a_str)} ({m.trace_a_top1_prob:.2f})",
            f"{repr(b_str)} ({m.trace_b_top1_prob:.2f})",
            style=style,
        )

    console.print(table)

    # Summary
    flagged_count = sum(1 for m in diff_result.layer_metrics if m.flagged)
    console.print(
        f"\n[bold]Flagged layers:[/bold] "
        f"{flagged_count}/{len(diff_result.layer_metrics)}"
    )
    if diff_result.first_divergence_layer is not None:
        console.print(
            f"[bold]First divergence:[/bold] layer {diff_result.first_divergence_layer}"
        )

    if head_detail and diff_result.critical_heads:
        console.print("\n[bold]Critical Heads (top JS divergence):[/bold]")
        head_table = Table()
        head_table.add_column("Layer", justify="right")
        head_table.add_column("Head", justify="right")
        head_table.add_column("JS Divergence", justify="right")
        for layer_idx, head_idx, js_div in diff_result.critical_heads:
            head_table.add_row(str(layer_idx), str(head_idx), f"{js_div:.6f}")
        console.print(head_table)

    # Token legend
    console.print("\n[bold]Token Legend:[/bold]")
    legend_table = Table()
    legend_table.add_column("ID", justify="right", style="cyan")
    legend_table.add_column("String")
    for tid in sorted(token_lookup.keys()):
        legend_table.add_row(str(tid), repr(token_lookup[tid]))
    console.print(legend_table)
