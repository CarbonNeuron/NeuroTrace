"""Click CLI for NeuroTrace — trace, list, inspect, diff, predict commands."""

import json

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.storage import TraceDB

console = Console()
err_console = Console(stderr=True)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to best available, pass others through."""
    if device != "auto":
        return device
    try:
        import torch_directml  # noqa: F401

        return "privateuseone:0"
    except ImportError:
        pass
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_trace_id(db: TraceDB, trace_id: str) -> str:
    """Resolve label, prefix, 'latest', or full trace_id."""
    return db.resolve_trace_id(trace_id)


def _maybe_load_adapter(model, adapter_path: str | None):
    """Load and merge a LoRA adapter if path is provided."""
    if adapter_path is None:
        return model
    from neurotrace.finetune import load_adapter

    return load_adapter(model, adapter_path)


@click.group()
def cli() -> None:
    """NeuroTrace: interpretability toolkit for tracing transformer inference."""


@cli.command()
@click.option("--model", required=True, help="HuggingFace model name or path.")
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
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
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
    device,
):
    """Run a forward-pass trace and store results."""
    if prompt is None and prompts_file is None:
        raise click.UsageError("Must provide either --prompt or --prompts-file.")
    if prompt is not None and prompts_file is not None:
        raise click.UsageError("Cannot provide both --prompt and --prompts-file.")

    # Collect prompts
    if prompts_file is not None:
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [prompt]

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


@cli.command("list")
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


@cli.command()
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


@cli.command()
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


@cli.command()
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
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def predict(
    db, trace_id, top_k, changes_only, layers, track, output_json, adapter, device
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


@cli.command()
@click.option("--model", required=True, help="HuggingFace model name or path.")
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
def decode(model, tokens, from_trace, db):
    """Decode token IDs to human-readable strings."""
    if not tokens and from_trace is None:
        raise click.UsageError("Must provide --tokens or --from-trace.")
    if from_trace is not None and db is None:
        raise click.UsageError("--from-trace requires --db.")

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


@cli.command()
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt-a", required=True, help="First prompt.")
@click.option("--prompt-b", required=True, help="Second prompt.")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--light", is_flag=True, help="Use light capture mode.")
@click.option("--flagged-only", is_flag=True, help="Show only flagged layers.")
@click.option("--head-detail", is_flag=True, help="Show critical head details.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
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
    device,
):
    """Trace two prompts, diff them, and show decoded results."""
    from neurotrace.analyzer import compute_diff
    from neurotrace.models import load_model
    from neurotrace.tracer import Tracer

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


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt", required=True, help="Prompt text to trace.")
@click.option(
    "--zero-layers",
    default=None,
    help="Zero entire layer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--zero-heads",
    default=None,
    help="Zero specific attention heads. E.g. '20:7,20:12'.",
)
@click.option(
    "--scale-layer",
    default=None,
    help="Scale layer contributions. E.g. '20:0.5,21:2.0'.",
)
@click.option(
    "--zero-mlp",
    default=None,
    help="Zero MLP sublayer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--scale-mlp",
    default=None,
    help="Scale MLP sublayer outputs. E.g. '20:0.5' or '20:0.3,21:0.7'.",
)
@click.option(
    "--baseline",
    default=None,
    help="Baseline trace ID, label, or prefix. If omitted, runs a clean trace.",
)
@click.option("--label", default=None, help="Label for the ablated trace.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def ablate(
    db,
    model,
    prompt,
    zero_layers,
    zero_heads,
    scale_layer,
    zero_mlp,
    scale_mlp,
    baseline,
    label,
    seed,
    adapter,
    device,
):
    """Run inference with targeted components disabled and compare to baseline."""
    from neurotrace.ablate import (
        AblationSpec,
        parse_scale_layers,
        parse_zero_heads,
        parse_zero_layers,
        run_ablation,
    )

    # Parse interventions
    zl = parse_zero_layers(zero_layers) if zero_layers else []
    zh = parse_zero_heads(zero_heads) if zero_heads else []
    sl = parse_scale_layers(scale_layer) if scale_layer else []
    zm = parse_zero_layers(zero_mlp) if zero_mlp else []
    sm = parse_scale_layers(scale_mlp) if scale_mlp else []

    if not zl and not zh and not sl and not zm and not sm:
        raise click.UsageError(
            "At least one intervention required: "
            "--zero-layers, --zero-heads, --scale-layer, --zero-mlp, or --scale-mlp."
        )

    # Conflict check: same layer in both --zero-mlp and --scale-mlp
    zm_set = set(zm)
    sm_set = {ly for ly, _ in sm}
    conflict = zm_set & sm_set
    if conflict:
        layers_str = ", ".join(str(x) for x in sorted(conflict))
        raise click.UsageError(
            f"Layer {layers_str} appears in both --zero-mlp and --scale-mlp"
        )

    spec = AblationSpec(
        zero_layers=zl,
        zero_heads=zh,
        scale_layers=sl,
        zero_mlp=zm,
        scale_mlp=sm,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        # Load baseline if provided
        baseline_trace = None
        db_conn = TraceDB(db)
        if baseline is not None:
            try:
                baseline_id = _resolve_trace_id(db_conn, baseline)
                baseline_trace = db_conn.read_trace(baseline_id)
                progress.update(
                    task,
                    description=f"Loaded baseline {baseline_id[:8]}.",
                )
            except ValueError as e:
                db_conn.close()
                raise click.ClickException(str(e))

        progress.update(task, description="Running ablation...")
        result = run_ablation(
            model_obj,
            tokenizer,
            prompt,
            spec,
            baseline=baseline_trace,
            label=label,
            seed=seed,
        )
        progress.update(task, description="Storing traces...")

        # Store baseline if we created it
        if baseline is None:
            db_conn.write_trace(result.baseline_trace)
            console.print(
                f"[green]Baseline stored:"
                f" {result.baseline_trace.metadata.trace_id[:8]}[/green]"
            )

        # Store ablated trace with interventions metadata
        db_conn.write_trace(result.ablated_trace, interventions=spec.to_json())
        console.print(
            f"[green]Ablated stored:"
            f" {result.ablated_trace.metadata.trace_id[:8]}[/green]"
        )
        db_conn.close()

    # Print summary
    console.print(
        f"\n[bold]Baseline:[/bold] {result.baseline_final_token!r}"
        f" (p={result.baseline_final_prob:.2f})"
    )
    console.print(
        f"[bold]Ablated:[/bold]  {result.ablated_final_token!r}"
        f" (p={result.ablated_final_prob:.2f})"
    )
    console.print(f"[dim]Interventions: {spec.describe()}[/dim]\n")

    # Layer comparison table
    table = Table(title="Layer Comparison")
    table.add_column("Layer", justify="right")
    table.add_column("Baseline Top-1")
    table.add_column("Ablated Top-1")
    table.add_column("Cos Sim", justify="right")
    table.add_column("Changed?")

    for lc in result.layer_comparisons:
        style = "red" if lc.changed else None
        marker = "\u2713" if lc.changed else ""
        # Mark intervened layers
        is_intervened = (
            lc.layer_index in spec.zero_layers
            or any(ly == lc.layer_index for ly, _ in spec.zero_heads)
            or any(ly == lc.layer_index for ly, _ in spec.scale_layers)
            or lc.layer_index in spec.zero_mlp
            or any(ly == lc.layer_index for ly, _ in spec.scale_mlp)
        )
        if is_intervened and lc.changed:
            marker += " \u2190"
        table.add_row(
            str(lc.layer_index),
            f"{lc.baseline_top1} ({lc.baseline_top1_prob:.2f})",
            f"{lc.ablated_top1} ({lc.ablated_top1_prob:.2f})",
            f"{lc.cosine_similarity:.4f}",
            marker,
            style=style,
        )

    console.print(table)

    # Summary line
    changed_count = sum(1 for lc in result.layer_comparisons if lc.changed)
    console.print(
        f"\n[bold]Summary:[/bold] Ablating ({spec.describe()}) changed prediction"
        f" from {result.baseline_final_token!r} (p={result.baseline_final_prob:.2f})"
        f" to {result.ablated_final_token!r} (p={result.ablated_final_prob:.2f})."
        f" {changed_count}/{len(result.layer_comparisons)} layers changed top-1."
    )


def _compute_layer_predictions(
    result,
    top_k: int = 5,
):
    """Project residual_out at each layer through lm_head to get top-k predictions.

    Returns (layer_predictions, token_tracks) or (None, None) if residuals missing.
    """
    import torch

    has_residuals = any(s.residual_out is not None for s in result.layer_snapshots)
    if not has_residuals:
        return None, None

    from neurotrace.models import load_model

    try:
        model_obj, tokenizer = load_model(result.metadata.model_name)
    except (OSError, ValueError):
        return None, None
    from neurotrace.models import get_lm_head_and_norm

    lm_head, final_ln = get_lm_head_and_norm(model_obj)

    layer_preds = []
    prev_ranking: dict[int, int] = {}
    # Store full prob vectors to build token tracks later
    prob_vectors: dict[int, torch.Tensor] = {}

    for snap in result.layer_snapshots:
        if snap.residual_out is None:
            continue

        with torch.no_grad():
            res = torch.tensor(snap.residual_out, dtype=torch.float32).unsqueeze(0)
            if final_ln is not None:
                res = final_ln(res)
            logits = lm_head(res.squeeze(0))
            probs = torch.softmax(logits[-1], dim=-1)

            topk = torch.topk(probs, k=top_k)
            topk_ids = topk.indices.tolist()
            topk_probs = topk.values.tolist()
            topk_strings = [tokenizer.decode(tid) for tid in topk_ids]

        current_ranking = {tid: rank for rank, tid in enumerate(topk_ids)}
        annotations = []
        for rank, tid in enumerate(topk_ids):
            if not prev_ranking:
                annotations.append("")
            elif tid not in prev_ranking:
                annotations.append("NEW")
            elif prev_ranking[tid] == rank:
                annotations.append("--")
            elif prev_ranking[tid] > rank:
                annotations.append(f"^ #{prev_ranking[tid] + 1}")
            else:
                annotations.append(f"v #{prev_ranking[tid] + 1}")

        layer_preds.append(
            {
                "layer_index": snap.layer_index,
                "top_k_ids": topk_ids,
                "top_k_probs": topk_probs,
                "top_k_strings": topk_strings,
                "annotations": annotations,
            }
        )

        prob_vectors[snap.layer_index] = probs
        prev_ranking = current_ranking

    # Token tracking: top-3 final tokens
    token_tracks = []
    if layer_preds:
        final = layer_preds[-1]
        for i in range(min(3, len(final["top_k_ids"]))):
            tid = final["top_k_ids"][i]
            tok_str = final["top_k_strings"][i]
            track_probs = []
            for lp in layer_preds:
                pv = prob_vectors.get(lp["layer_index"])
                if pv is not None:
                    track_probs.append(float(pv[tid].item()))
                else:
                    track_probs.append(0.0)
            token_tracks.append(
                {
                    "token": tok_str,
                    "token_id": tid,
                    "probs": track_probs,
                    "layers": [lp["layer_index"] for lp in layer_preds],
                }
            )

    return layer_preds, token_tracks


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--trace-id", default=None, help="Single trace report: trace ID or label."
)
@click.option("--trace-a", default=None, help="Comparison report: first trace ID.")
@click.option("--trace-b", default=None, help="Comparison report: second trace ID.")
@click.option("-o", "--output", default="report.html", help="Output file path.")
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open report in browser after generating.",
)
@click.option(
    "--full-attention",
    is_flag=True,
    help="Include attention heatmaps for all layers.",
)
@click.option(
    "--no-attention",
    is_flag=True,
    help="Skip attention heatmaps entirely.",
)
@click.option(
    "--upload",
    is_flag=True,
    help="Upload report to CarbonFiles after generating.",
)
@click.option(
    "--bucket",
    default=None,
    help="Existing CarbonFiles bucket ID for upload.",
)
def report(
    db,
    trace_id,
    trace_a,
    trace_b,
    output,
    open_browser,
    full_attention,
    no_attention,
    upload,
    bucket,
):
    """Generate a self-contained HTML report from one or two traces."""
    import os
    import webbrowser

    from neurotrace.report import generate_comparison_report, generate_report

    # Validate options
    if trace_id and (trace_a or trace_b):
        raise click.UsageError("Cannot use --trace-id with --trace-a/--trace-b.")
    if not trace_id and not (trace_a and trace_b):
        raise click.UsageError("Provide --trace-id or both --trace-a and --trace-b.")
    if (trace_a and not trace_b) or (trace_b and not trace_a):
        raise click.UsageError("Must provide both --trace-a and --trace-b.")

    db_conn = TraceDB(db)

    try:
        if trace_id:
            # Single trace report
            resolved_id = _resolve_trace_id(db_conn, trace_id)
            result = db_conn.read_trace(resolved_id)
            layer_stats = db_conn.get_layer_stats(resolved_id)
            db_conn.close()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=err_console,
            ) as progress:
                task = progress.add_task("Computing predictions...", total=None)
                layer_preds, token_tracks = _compute_layer_predictions(result)
                progress.update(task, description="Generating report...")

                html = generate_report(
                    result,
                    layer_stats,
                    layer_preds,
                    token_tracks,
                    full_attention,
                    no_attention,
                )
        else:
            # Comparison report
            from neurotrace.analyzer import compute_diff

            id_a = _resolve_trace_id(db_conn, trace_a)
            id_b = _resolve_trace_id(db_conn, trace_b)
            result_a = db_conn.read_trace(id_a)
            result_b = db_conn.read_trace(id_b)
            stats_a = db_conn.get_layer_stats(id_a)
            stats_b = db_conn.get_layer_stats(id_b)
            db_conn.close()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=err_console,
            ) as progress:
                task = progress.add_task(
                    "Computing predictions for trace A...", total=None
                )
                preds_a, tracks_a = _compute_layer_predictions(result_a)
                progress.update(
                    task,
                    description="Computing predictions for trace B...",
                )
                preds_b, tracks_b = _compute_layer_predictions(result_b)
                progress.update(task, description="Computing diff...")

                diff_result = compute_diff(result_a, result_b)

                # Build token lookup for diff table
                token_lookup: dict[int, str] = {}
                if preds_a:
                    for lp in preds_a:
                        for tid, s in zip(lp["top_k_ids"], lp["top_k_strings"]):
                            token_lookup[tid] = s
                if preds_b:
                    for lp in preds_b:
                        for tid, s in zip(lp["top_k_ids"], lp["top_k_strings"]):
                            token_lookup[tid] = s
                # Also add from diff metrics
                for m in diff_result.layer_metrics:
                    if m.trace_a_top1 not in token_lookup:
                        token_lookup[m.trace_a_top1] = str(m.trace_a_top1)
                    if m.trace_b_top1 not in token_lookup:
                        token_lookup[m.trace_b_top1] = str(m.trace_b_top1)

                progress.update(task, description="Generating report...")
                html = generate_comparison_report(
                    result_a,
                    result_b,
                    stats_a,
                    stats_b,
                    preds_a,
                    preds_b,
                    tracks_a,
                    tracks_b,
                    diff_result,
                    token_lookup,
                    full_attention,
                    no_attention,
                )
    except ValueError as e:
        db_conn.close()
        raise click.ClickException(str(e))

    with open(output, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(output) / 1024
    console.print(f"[green]Report written to {output} ({size_kb:.0f} KB)[/green]")

    if upload:
        from neurotrace.upload import upload_report

        url = upload_report(output, bucket_id=bucket)
        console.print(f"[green]\u2713[/green] Uploaded: {url}")

    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(output)}")


def _parse_sweep_range(value: str) -> tuple[int, list[float]]:
    """Parse 'L:start:end:step' into (layer, [values]).

    Returns the layer index and a list of float values from start to end (inclusive)
    stepping by step.
    """
    parts = value.split(":")
    if len(parts) != 4:
        raise click.BadParameter(f"Expected format L:start:end:step, got {value!r}")
    layer = int(parts[0])
    start = float(parts[1])
    end = float(parts[2])
    step = float(parts[3])
    values = []
    v = start
    while v <= end + step * 0.001:  # small epsilon for float rounding
        values.append(round(v, 6))
        v += step
    return layer, values


def _parse_sweep_zero_heads(value: str) -> tuple[int, list[int]]:
    """Parse 'L:start-end' into (layer, [head_indices])."""
    parts = value.split(":")
    if len(parts) != 2:
        raise click.BadParameter(f"Expected format L:start-end, got {value!r}")
    layer = int(parts[0])
    head_range = parts[1].split("-")
    if len(head_range) != 2:
        raise click.BadParameter(f"Expected format L:start-end, got {value!r}")
    start = int(head_range[0])
    end = int(head_range[1])
    return layer, list(range(start, end + 1))


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt", required=True, help="Prompt text to trace.")
@click.option(
    "--baseline",
    default=None,
    help="Baseline trace ID, label, or prefix. If omitted, runs a clean trace.",
)
@click.option("--label", default=None, help="Base label for sweep traces.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--sweep-scale-mlp",
    default=None,
    help="Sweep MLP scale factor. Format: L:start:end:step (e.g. '20:0.1:0.9:0.1').",
)
@click.option(
    "--sweep-zero-heads",
    "sweep_zero_heads_opt",
    default=None,
    help="Sweep zero-heads one at a time. Format: L:start-end (e.g. '20:0-31').",
)
@click.option(
    "--sweep-zero-mlp",
    default=None,
    help="Sweep zero-mlp one layer at a time. Format: L1,L2,L3 (e.g. '18,19,20').",
)
@click.option(
    "--sweep-scale-layer",
    default=None,
    help="Sweep layer scale factor. Format: L:start:end:step (e.g. '20:0.1:0.9:0.1').",
)
@click.option(
    "--zero-layers",
    default=None,
    help="Fixed intervention: zero entire layer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--zero-heads",
    default=None,
    help="Fixed intervention: zero specific attention heads. E.g. '20:7,20:12'.",
)
@click.option(
    "--scale-layer",
    default=None,
    help="Fixed intervention: scale layer contributions. E.g. '20:0.5,21:2.0'.",
)
@click.option(
    "--zero-mlp",
    default=None,
    help="Fixed intervention: zero MLP sublayer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--scale-mlp",
    default=None,
    help="Fixed intervention: scale MLP sublayer outputs. E.g. '20:0.5'.",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def sweep(
    db,
    model,
    prompt,
    baseline,
    label,
    seed,
    sweep_scale_mlp,
    sweep_zero_heads_opt,
    sweep_zero_mlp,
    sweep_scale_layer,
    zero_layers,
    zero_heads,
    scale_layer,
    zero_mlp,
    scale_mlp,
    output_json,
    adapter,
    device,
):
    """Run multiple ablations in a single model load, sweeping a parameter range."""
    from neurotrace.ablate import (
        AblationSpec,
        parse_scale_layers,
        parse_zero_heads,
        parse_zero_layers,
        run_ablation,
    )

    # Count sweep flags
    sweep_flags = [
        sweep_scale_mlp,
        sweep_zero_heads_opt,
        sweep_zero_mlp,
        sweep_scale_layer,
    ]
    active_sweeps = [f for f in sweep_flags if f is not None]
    if len(active_sweeps) == 0:
        raise click.UsageError("Exactly one --sweep-* flag is required.")
    if len(active_sweeps) > 1:
        raise click.UsageError("Only one --sweep-* flag per invocation.")

    # Parse fixed interventions
    fixed_zl = parse_zero_layers(zero_layers) if zero_layers else []
    fixed_zh = parse_zero_heads(zero_heads) if zero_heads else []
    fixed_sl = parse_scale_layers(scale_layer) if scale_layer else []
    fixed_zm = parse_zero_layers(zero_mlp) if zero_mlp else []
    fixed_sm = parse_scale_layers(scale_mlp) if scale_mlp else []

    # Build sweep specs: list of (description_value, AblationSpec)
    sweep_specs: list[tuple[str, AblationSpec]] = []
    sweep_description = ""
    sweep_target_layer = -1

    if sweep_scale_mlp:
        target_layer, values = _parse_sweep_range(sweep_scale_mlp)
        sweep_target_layer = target_layer
        step = values[1] - values[0] if len(values) > 1 else 0
        sweep_description = (
            f"scale-mlp layer {target_layer} ({values[0]} → {values[-1]}, step {step})"
        )
        for v in values:
            sm = fixed_sm + [(target_layer, v)]
            spec = AblationSpec(
                zero_layers=fixed_zl,
                zero_heads=fixed_zh,
                scale_layers=fixed_sl,
                zero_mlp=fixed_zm,
                scale_mlp=sm,
            )
            sweep_specs.append((str(v), spec))

    elif sweep_scale_layer:
        target_layer, values = _parse_sweep_range(sweep_scale_layer)
        sweep_target_layer = target_layer
        step = values[1] - values[0] if len(values) > 1 else 0
        sweep_description = (
            f"scale-layer layer {target_layer}"
            f" ({values[0]} → {values[-1]}, step {step})"
        )
        for v in values:
            sl = fixed_sl + [(target_layer, v)]
            spec = AblationSpec(
                zero_layers=fixed_zl,
                zero_heads=fixed_zh,
                scale_layers=sl,
                zero_mlp=fixed_zm,
                scale_mlp=fixed_sm,
            )
            sweep_specs.append((str(v), spec))

    elif sweep_zero_heads_opt:
        target_layer, heads = _parse_sweep_zero_heads(sweep_zero_heads_opt)
        sweep_target_layer = target_layer
        sweep_description = (
            f"zero-heads layer {target_layer} (head {heads[0]} → {heads[-1]})"
        )
        for h in heads:
            zh = fixed_zh + [(target_layer, h)]
            spec = AblationSpec(
                zero_layers=fixed_zl,
                zero_heads=zh,
                scale_layers=fixed_sl,
                zero_mlp=fixed_zm,
                scale_mlp=fixed_sm,
            )
            sweep_specs.append((f"h{h}", spec))

    elif sweep_zero_mlp:
        layers = parse_zero_layers(sweep_zero_mlp)
        sweep_target_layer = layers[0] if layers else -1
        sweep_description = f"zero-mlp layers {','.join(map(str, layers))}"
        for l_idx in layers:
            zm = fixed_zm + [l_idx]
            spec = AblationSpec(
                zero_layers=fixed_zl,
                zero_heads=fixed_zh,
                scale_layers=fixed_sl,
                zero_mlp=zm,
                scale_mlp=fixed_sm,
            )
            sweep_specs.append((f"L{l_idx}", spec))

    if not sweep_specs:
        raise click.UsageError("Sweep produced no configurations.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        db_conn = TraceDB(db)

        # Resolve or run baseline
        baseline_trace = None
        if baseline is not None:
            try:
                baseline_id = _resolve_trace_id(db_conn, baseline)
                baseline_trace = db_conn.read_trace(baseline_id)
                progress.update(
                    task,
                    description=f"Loaded baseline {baseline_id[:8]}.",
                )
            except ValueError as e:
                db_conn.close()
                raise click.ClickException(str(e))
        else:
            progress.update(task, description="Running baseline trace...")
            from neurotrace.tracer import Tracer

            tracer = Tracer(model_obj, tokenizer)
            baseline_label = f"{label}-baseline" if label else None
            baseline_trace = tracer.trace(prompt, label=baseline_label, seed=seed)
            db_conn.write_trace(baseline_trace)
            if not output_json:
                console.print(
                    f"[green]Baseline stored:"
                    f" {baseline_trace.metadata.trace_id[:8]}[/green]"
                )

        # Run sweep
        results: list[dict] = []
        for i, (sweep_val, spec) in enumerate(sweep_specs):
            desc = f"Sweep {i + 1}/{len(sweep_specs)}: {sweep_val}..."
            progress.update(task, description=desc)

            trace_label = f"{label}-{sweep_val}" if label else f"sweep-{sweep_val}"
            ablation_result = run_ablation(
                model_obj,
                tokenizer,
                prompt,
                spec,
                baseline=baseline_trace,
                label=trace_label,
                seed=seed,
            )

            # Store ablated trace
            db_conn.write_trace(
                ablation_result.ablated_trace,
                interventions=spec.to_json(),
            )

            # Collect summary data
            trace_id = ablation_result.ablated_trace.metadata.trace_id

            # Get predictions at layers surrounding the intervention
            layer_preds = {}
            for lc in ablation_result.layer_comparisons:
                layer_preds[lc.layer_index] = {
                    "token": lc.ablated_top1,
                    "prob": lc.ablated_top1_prob,
                }

            results.append(
                {
                    "sweep_value": sweep_val,
                    "trace_id": trace_id,
                    "final_token": ablation_result.ablated_final_token,
                    "final_prob": ablation_result.ablated_final_prob,
                    "layer_preds": layer_preds,
                    "spec": spec.describe(),
                }
            )

        progress.update(task, description="Done.")
        db_conn.close()

    # Determine display layers: target ± 1 and final
    all_layers = set()
    for r in results:
        all_layers.update(r["layer_preds"].keys())
    all_layers_sorted = sorted(all_layers)

    # Pick layers to show: target-1, target, target+1, and last layer
    display_layers = set()
    if sweep_target_layer >= 0:
        for offset in [-1, 0, 1]:
            candidate = sweep_target_layer + offset
            if candidate in all_layers:
                display_layers.add(candidate)
    if all_layers_sorted:
        display_layers.add(all_layers_sorted[-1])
    display_layers_sorted = sorted(display_layers)

    if output_json:
        json_output = {
            "sweep_description": sweep_description,
            "baseline_trace_id": baseline_trace.metadata.trace_id,
            "results": [
                {
                    "sweep_value": r["sweep_value"],
                    "trace_id": r["trace_id"],
                    "final_token": r["final_token"],
                    "final_prob": r["final_prob"],
                    "interventions": r["spec"],
                    "layer_predictions": {
                        str(li): r["layer_preds"].get(li, {})
                        for li in display_layers_sorted
                    },
                }
                for r in results
            ],
        }
        click.echo(json.dumps(json_output, indent=2, default=str))
        return

    # Rich table output
    console.print(f"\n[bold]Sweep: {sweep_description}[/bold]")
    console.print(f"[dim]Baseline: {baseline_trace.metadata.trace_id[:8]}[/dim]\n")

    table = Table()
    table.add_column("Value", justify="right")
    table.add_column("Trace ID", style="cyan", no_wrap=True)
    for l_idx in display_layers_sorted:
        col_name = f"Layer {l_idx}"
        if l_idx == all_layers_sorted[-1]:
            col_name = "Final"
        table.add_column(col_name)

    for r in results:
        row = [r["sweep_value"], r["trace_id"][:8]]
        for l_idx in display_layers_sorted:
            pred = r["layer_preds"].get(l_idx)
            if pred:
                row.append(f"{pred['token']} ({pred['prob']:.2f})")
            else:
                row.append("")
        table.add_row(*row)

    console.print(table)


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    help="Path to JSON dataset file.",
)
@click.option(
    "--dataset-builtin",
    default=None,
    help="Use a built-in dataset (e.g. 'capitals').",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--sabotage-threshold",
    default=0.5,
    type=float,
    help="Fraction of peak prob drop to flag as sabotage.",
)
@click.option(
    "--final-threshold",
    default=0.3,
    type=float,
    help="Minimum final probability to avoid weak flag.",
)
@click.option(
    "--save-traces",
    is_flag=True,
    help="Store all traces to database.",
)
@click.option(
    "--save-flagged",
    is_flag=True,
    help="Store only flagged traces to database.",
)
@click.option("--details", is_flag=True, help="Show layer details for flagged prompts.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def scan(
    db,
    model,
    dataset_path,
    dataset_builtin,
    seed,
    sabotage_threshold,
    final_threshold,
    save_traces,
    save_flagged,
    details,
    output_json,
    adapter,
    device,
):
    """Scan a dataset for sabotaged predictions."""
    from neurotrace.datasets import get_builtin_dataset, load_dataset
    from neurotrace.scan import run_scan

    if dataset_path is None and dataset_builtin is None:
        raise click.UsageError("Must provide either --dataset or --dataset-builtin.")
    if dataset_path is not None and dataset_builtin is not None:
        raise click.UsageError("Cannot provide both --dataset and --dataset-builtin.")

    # Load dataset
    if dataset_builtin is not None:
        dataset = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    else:
        dataset = load_dataset(dataset_path)
        dataset_name = dataset_path

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        db_conn = None
        if save_traces or save_flagged:
            db_conn = TraceDB(db)

        def progress_cb(i, total, prompt):
            desc = f"Scanning {i + 1}/{total}: {prompt[:40]}..."
            progress.update(task, description=desc)

        scan_result = run_scan(
            model_obj,
            tokenizer,
            dataset,
            dataset_name,
            seed=seed,
            sabotage_threshold=sabotage_threshold,
            final_threshold=final_threshold,
            save_traces=save_traces,
            save_flagged=save_flagged,
            db=db_conn,
            progress_callback=progress_cb,
        )

        progress.update(task, description="Done.")
        if db_conn:
            db_conn.close()

    if output_json:
        output = {
            "model": scan_result.model_name,
            "dataset": scan_result.dataset_name,
            "total": len(scan_result.prompt_results),
            "correct": scan_result.correct_count,
            "sabotaged": scan_result.sabotaged_count,
            "weak": scan_result.weak_count,
            "wrong": scan_result.wrong_count,
            "results": [
                {
                    "prompt": r.prompt,
                    "answer": r.answer,
                    "final_token": r.final_token,
                    "final_prob": r.final_prob,
                    "final_rank": r.final_rank,
                    "peak_prob": r.peak_prob,
                    "peak_layer": r.peak_layer,
                    "commitment_layer": r.commitment_layer,
                    "sabotage_layers": r.sabotage_layers,
                    "flags": r.flags,
                    "status": r.status,
                }
                for r in scan_result.prompt_results
            ],
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    console.print(
        f"\n[bold]Scan:[/bold] {len(scan_result.prompt_results)} prompts, "
        f"model {scan_result.model_name}\n"
    )

    table = Table()
    table.add_column("Status", justify="center")
    table.add_column("Prompt")
    table.add_column("Answer")
    table.add_column("Final (prob)")
    table.add_column("Peak", justify="right")
    table.add_column("Flags")

    for r in scan_result.prompt_results:
        if r.status == "correct":
            status_str = "[green]\u2713[/green]"
            style = None
        elif r.status == "sabotaged":
            status_str = "[red]\u26a0 SABO[/red]"
            style = "red"
        elif r.status == "weak":
            status_str = "[yellow]\u26a0 WEAK[/yellow]"
            style = "yellow"
        else:
            status_str = "[red]\u2717 WRONG[/red]"
            style = "red"

        prompt_display = r.prompt
        if len(prompt_display) > 30:
            prompt_display = "..." + prompt_display[-27:]

        final_display = f"{r.final_token} ({r.final_prob:.2f})"
        peak_display = f"{r.peak_prob:.2f}" if r.peak_prob > 0 else ""
        flags_display = ", ".join(r.flags)

        table.add_row(
            status_str,
            prompt_display,
            r.answer,
            final_display,
            peak_display,
            flags_display,
            style=style,
        )

    console.print(table)

    # Summary
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"{scan_result.correct_count} \u2713 correct | "
        f"{scan_result.sabotaged_count} \u26a0 sabotaged | "
        f"{scan_result.weak_count} \u26a0 weak | "
        f"{scan_result.wrong_count} \u2717 wrong"
    )

    # Details for flagged prompts
    if details:
        flagged = [r for r in scan_result.prompt_results if r.flags]
        if flagged:
            console.print("\n[bold]Flagged Details:[/bold]\n")
            for r in flagged:
                console.print(f'[bold]\u26a0 "{r.prompt}" \u2192 {r.answer}[/bold]')
                if r.ranks and r.probs:
                    # Show compact layer-by-layer
                    parts = []
                    peak_idx = (
                        r.probs.index(r.peak_prob) if r.peak_prob in r.probs else None
                    )
                    for i, (rank, prob) in enumerate(zip(r.ranks, r.probs)):
                        entry = f"L{i}: rank #{rank} ({prob:.3f})"
                        if i == peak_idx:
                            entry += " \u2190 peak"
                        if i in r.sabotage_layers:
                            entry += " \u2190 SABOTAGE"
                        parts.append(entry)
                    # Show last few layers around the action
                    if len(parts) > 10:
                        start = max(0, (r.commitment_layer or 0) - 2)
                        parts = parts[start:]
                    console.print("  " + " | ".join(parts))
                if r.sabotage_layers:
                    severity = (
                        (r.peak_prob - min(r.probs[s] for s in r.sabotage_layers))
                        / r.peak_prob
                        if r.peak_prob > 0
                        else 0
                    )
                    console.print(
                        f"  Sabotage layers: {r.sabotage_layers} | "
                        f"Severity: {severity:.2f} (prob drop from peak)"
                    )
                console.print()


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option(
    "--dataset-builtin",
    default=None,
    help="Use built-in dataset (e.g., 'capitals').",
)
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    help="Path to JSONL training data.",
)
@click.option(
    "--from-scan",
    "from_scan",
    default=None,
    help="Use scan results — trace ID or 'latest'."
    " Auto-generates weighted training data.",
)
@click.option(
    "--target-layers",
    default="20,21",
    help="MLP layers to apply LoRA to (comma-separated).",
)
@click.option("--rank", default=8, type=int, help="LoRA rank.")
@click.option("--alpha", default=16, type=int, help="LoRA alpha.")
@click.option("--epochs", default=10, type=int, help="Training epochs.")
@click.option("--lr", default=1e-4, type=float, help="Learning rate.")
@click.option(
    "--output",
    "output_dir",
    default=None,
    help="Directory to save adapter weights.",
)
@click.option("--eval-before", is_flag=True, help="Run scan before training.")
@click.option("--eval-after", is_flag=True, help="Run scan after training.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877).")
def finetune(
    db,
    model,
    dataset_builtin,
    dataset_path,
    from_scan,
    target_layers,
    rank,
    alpha,
    epochs,
    lr,
    output_dir,
    eval_before,
    eval_after,
    seed,
    output_json,
    device,
    remote,
):
    """Train a LoRA adapter on specified MLP layers to fix sabotaged predictions."""
    from neurotrace.finetune import (
        FinetuneConfig,
        generate_training_data_from_builtin,
        generate_training_data_from_jsonl,
        generate_training_data_from_scan,
        run_finetune,
    )

    # Remote GPU finetune
    if remote is not None:
        _finetune_remote(
            remote=remote,
            dataset_builtin=dataset_builtin,
            dataset_path=dataset_path,
            target_layers=target_layers,
            rank=rank,
            alpha=alpha,
            epochs=epochs,
            lr=lr,
            output_dir=output_dir,
            seed=seed,
            output_json=output_json,
            db=db,
        )
        return

    # Validate data source
    sources = [dataset_builtin, dataset_path, from_scan]
    source_count = sum(1 for s in sources if s is not None)
    if source_count == 0:
        raise click.UsageError(
            "Must provide --dataset-builtin, --dataset, or --from-scan."
        )
    if source_count > 1:
        raise click.UsageError(
            "Only one data source allowed:"
            " --dataset-builtin, --dataset, or --from-scan."
        )

    # Parse target layers
    layers = [int(x.strip()) for x in target_layers.split(",") if x.strip()]

    # Determine output directory
    if output_dir is None:
        import re
        from datetime import datetime

        model_slug = re.sub(r"[^a-zA-Z0-9]", "-", model).strip("-")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"adapters/{model_slug}-{timestamp}"

    # Generate training data
    dataset_name = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Preparing training data...", total=None)

        if dataset_builtin is not None:
            from neurotrace.datasets import get_builtin_dataset

            dataset = get_builtin_dataset(dataset_builtin)
            dataset_name = dataset_builtin

            # Optionally get sabotaged prompts from scan
            sabotaged = None
            if eval_before:
                progress.update(task, description="Running pre-training scan...")
                sabotaged = _run_eval_scan(
                    model,
                    dataset_builtin,
                    seed,
                    progress,
                    task,
                    device=device,
                )

            examples = generate_training_data_from_builtin(
                dataset,
                sabotaged_prompts=sabotaged.get("sabotaged_prompts")
                if sabotaged
                else None,
            )
            scan_before = sabotaged.get("summary") if sabotaged else None

        elif dataset_path is not None:
            examples = generate_training_data_from_jsonl(dataset_path)
            dataset_name = dataset_path
            scan_before = None

        else:
            # from_scan: run a fresh scan to get sabotage results for weighting
            progress.update(task, description="Running scan for training data...")
            from neurotrace.datasets import get_builtin_dataset
            from neurotrace.models import load_model
            from neurotrace.scan import run_scan

            device = _resolve_device(device)
            model_obj, tokenizer = load_model(model, device=device)
            dataset = get_builtin_dataset("capitals")
            scan_result = run_scan(model_obj, tokenizer, dataset, "capitals", seed=seed)
            examples = generate_training_data_from_scan(scan_result)
            dataset_name = "capitals (from scan)"
            scan_before = {
                "correct": scan_result.correct_count,
                "sabotaged": scan_result.sabotaged_count,
                "weak": scan_result.weak_count,
                "wrong": scan_result.wrong_count,
            }
            del model_obj, tokenizer

        progress.update(
            task,
            description=f"Training data ready: {len(examples)} examples.",
        )

        # Build config
        config = FinetuneConfig(
            target_layers=layers,
            rank=rank,
            alpha=alpha,
            epochs=epochs,
            learning_rate=lr,
            seed=seed,
        )

        # Run training
        def training_progress(desc):
            progress.update(task, description=desc)

        result = run_finetune(
            model_name=model,
            examples=examples,
            config=config,
            output_dir=output_dir,
            progress_callback=training_progress,
        )
        result.dataset_name = dataset_name
        result.scan_before = scan_before if eval_before else None

        # Post-training eval
        scan_after = None
        if eval_after and dataset_builtin is not None:
            progress.update(task, description="Running post-training scan...")
            scan_after = _run_eval_scan(
                model,
                dataset_builtin,
                seed,
                progress,
                task,
                adapter_path=output_dir,
                device=device,
            )
            result.scan_after = scan_after.get("summary") if scan_after else None

        # Save run to DB
        progress.update(task, description="Saving run metadata...")
        db_conn = TraceDB(db)
        db_conn.save_finetune_run(result)
        db_conn.close()

        progress.update(task, description="Done.")

    if output_json:
        output = {
            "run_id": result.run_id,
            "model": result.model_name,
            "adapter_path": result.adapter_path,
            "target_layers": result.target_layers,
            "lora_rank": result.lora_rank,
            "lora_alpha": result.lora_alpha,
            "dataset": result.dataset_name,
            "dataset_size": result.dataset_size,
            "epochs": result.epochs,
            "learning_rate": result.learning_rate,
            "train_loss_start": result.train_loss_start,
            "train_loss_end": result.train_loss_end,
            "scan_before": result.scan_before,
            "scan_after": result.scan_after,
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich output
    console.print("\n[bold]Fine-tuning complete[/bold]")
    console.print(f"[bold]Run ID:[/bold] {result.run_id[:8]}")
    console.print(f"[bold]Model:[/bold] {result.model_name}")
    console.print(f"[bold]Target layers:[/bold] {result.target_layers}")
    console.print(
        f"[bold]LoRA rank:[/bold] {result.lora_rank}, alpha: {result.lora_alpha}"
    )
    console.print(
        f"[bold]Dataset:[/bold] {result.dataset_name} ({result.dataset_size} examples)"
    )
    console.print(f"[bold]Epochs:[/bold] {result.epochs}")
    if result.train_loss_start is not None and result.train_loss_end is not None:
        console.print(
            f"[bold]Loss:[/bold]"
            f" {result.train_loss_start:.4f} → {result.train_loss_end:.4f}"
        )
    console.print(f"[green]Adapter saved to {result.adapter_path}[/green]")

    if result.scan_before:
        console.print(f"\n[bold]Pre-training scan:[/bold] {result.scan_before}")
    if result.scan_after:
        console.print(f"[bold]Post-training scan:[/bold] {result.scan_after}")


def _run_eval_scan(
    model_name,
    dataset_builtin,
    seed,
    progress,
    task,
    adapter_path=None,
    device="cpu",
):
    """Run a scan for evaluation, returning summary dict."""
    from neurotrace.datasets import get_builtin_dataset
    from neurotrace.models import load_model
    from neurotrace.scan import run_scan

    dataset = get_builtin_dataset(dataset_builtin)

    device = _resolve_device(device)
    model_obj, tokenizer = load_model(model_name, device=device)
    model_obj = _maybe_load_adapter(model_obj, adapter_path)

    def progress_cb(i, total, _prompt):
        progress.update(task, description=f"Eval scan {i + 1}/{total}...")

    scan_result = run_scan(
        model_obj,
        tokenizer,
        dataset,
        dataset_builtin,
        seed=seed,
        progress_callback=progress_cb,
    )

    sabotaged_prompts = {
        r.prompt for r in scan_result.prompt_results if r.status == "sabotaged"
    }

    return {
        "summary": {
            "correct": scan_result.correct_count,
            "sabotaged": scan_result.sabotaged_count,
            "weak": scan_result.weak_count,
            "wrong": scan_result.wrong_count,
        },
        "sabotaged_prompts": sabotaged_prompts,
    }


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option(
    "--dataset-builtin",
    required=True,
    help="Built-in dataset name (e.g. capitals, math_simple, all).",
)
@click.option(
    "--target-layers",
    default=None,
    help="MLP layers to ablate/finetune (comma-separated). Auto-detected if omitted.",
)
@click.option(
    "--output",
    "output_dir",
    default=None,
    help="Directory for experiment results.",
)
@click.option("--skip-ablate", is_flag=True, help="Skip the ablation step.")
@click.option("--skip-finetune", is_flag=True, help="Skip the finetune step.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def experiment(
    db,
    model,
    dataset_builtin,
    target_layers,
    output_dir,
    skip_ablate,
    skip_finetune,
    seed,
    output_json,
    device,
):
    """Run a full diagnostic pipeline: scan, ablate, finetune, verify."""
    import os
    import time
    import uuid
    from datetime import datetime

    from neurotrace.datasets import get_builtin_dataset

    start_time = time.time()
    experiment_id = str(uuid.uuid4())

    # Determine output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"experiments/{dataset_builtin}-{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    dataset = get_builtin_dataset(dataset_builtin)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        from neurotrace.models import load_model
        from neurotrace.scan import run_scan

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        progress.update(task, description="Model loaded.")

        # --- Step 1: Baseline scan ---
        progress.update(task, description="Step 1/4: Running baseline scan...")

        def scan_progress(i, total, prompt):
            progress.update(
                task, description=f"Scanning {i + 1}/{total}: {prompt[:40]}..."
            )

        scan_result = run_scan(
            model_obj,
            tokenizer,
            dataset,
            dataset_builtin,
            seed=seed,
            progress_callback=scan_progress,
        )

        baseline_summary = {
            "model": scan_result.model_name,
            "dataset": scan_result.dataset_name,
            "total": len(scan_result.prompt_results),
            "correct": scan_result.correct_count,
            "sabotaged": scan_result.sabotaged_count,
            "weak": scan_result.weak_count,
            "wrong": scan_result.wrong_count,
            "results": [
                {
                    "prompt": r.prompt,
                    "answer": r.answer,
                    "final_token": r.final_token,
                    "final_prob": r.final_prob,
                    "status": r.status,
                    "flags": r.flags,
                    "sabotage_layers": r.sabotage_layers,
                }
                for r in scan_result.prompt_results
            ],
        }
        scan_baseline_path = os.path.join(output_dir, "scan_baseline.json")
        with open(scan_baseline_path, "w") as f:
            json.dump(baseline_summary, f, indent=2, default=str)

        err_console.print(
            f"  Baseline: {scan_result.correct_count} correct, "
            f"{scan_result.sabotaged_count} sabotaged, "
            f"{scan_result.weak_count} weak, "
            f"{scan_result.wrong_count} wrong"
        )

        # --- Step 2: Ablation analysis ---
        ablation_summary = {"target_layers": [], "layer_impacts": {}}
        detected_layers = None

        if not skip_ablate:
            progress.update(task, description="Step 2/4: Running ablation analysis...")
            from neurotrace.ablate import AblationSpec, run_ablation

            # Get prompts that were sabotaged or wrong
            problem_prompts = [
                r
                for r in scan_result.prompt_results
                if r.status in ("sabotaged", "wrong")
            ]

            if problem_prompts:
                # For each problem prompt, ablate MLP at each layer
                num_layers = len(scan_result.prompt_results[0].ranks)
                layer_improvement_counts: dict[int, int] = {}

                for pi, pr in enumerate(problem_prompts):
                    progress.update(
                        task,
                        description=(
                            f"Ablating {pi + 1}/{len(problem_prompts)}: "
                            f"{pr.prompt[:30]}..."
                        ),
                    )
                    for layer_idx in range(num_layers):
                        spec = AblationSpec(
                            zero_layers=[],
                            zero_heads=[],
                            scale_layers=[],
                            zero_mlp=[layer_idx],
                        )
                        try:
                            abl_result = run_ablation(
                                model_obj,
                                tokenizer,
                                pr.prompt,
                                spec,
                                seed=seed,
                            )
                            # Check if ablating this layer improved the prediction
                            ablated_token = abl_result.ablated_final_token.strip()
                            answer_lower = pr.answer.strip().lower()
                            ablated_lower = ablated_token.lstrip("\u2581").lower()
                            if ablated_lower and answer_lower.startswith(ablated_lower):
                                layer_improvement_counts[layer_idx] = (
                                    layer_improvement_counts.get(layer_idx, 0) + 1
                                )
                        except Exception:
                            continue

                ablation_summary["layer_impacts"] = {
                    str(k): v
                    for k, v in sorted(
                        layer_improvement_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                }

                # Pick top 2 layers
                if layer_improvement_counts:
                    sorted_layers = sorted(
                        layer_improvement_counts.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    detected_layers = [ly for ly, _ in sorted_layers[:2]]
                    ablation_summary["target_layers"] = detected_layers
            else:
                err_console.print("  No problem prompts found, skipping ablation.")
        else:
            err_console.print("  Skipping ablation step.")

        ablation_path = os.path.join(output_dir, "ablation_summary.json")
        with open(ablation_path, "w") as f:
            json.dump(ablation_summary, f, indent=2)

        # Resolve target layers
        if target_layers is not None:
            layers = [int(x.strip()) for x in target_layers.split(",") if x.strip()]
        elif detected_layers:
            layers = detected_layers
        else:
            layers = [20, 21]
            err_console.print(
                "  No impactful layers detected, using defaults: [20, 21]"
            )

        err_console.print(f"  Target layers: {layers}")

        # --- Step 3: Finetune ---
        finetune_run_id = None
        ft_result = None
        verify_result = None
        adapter_path = os.path.join(output_dir, "adapter")

        if not skip_finetune:
            progress.update(task, description="Step 3/4: Fine-tuning LoRA adapter...")
            from neurotrace.finetune import (
                FinetuneConfig,
                generate_training_data_from_scan,
                run_finetune,
            )

            examples = generate_training_data_from_scan(scan_result, sabotage_weight=3)
            config = FinetuneConfig(
                target_layers=layers,
                seed=seed,
            )

            def ft_progress(desc):
                progress.update(task, description=f"  {desc}")

            # Need to delete model to free memory before loading for training
            del model_obj, tokenizer

            ft_result = run_finetune(
                model_name=model,
                examples=examples,
                config=config,
                output_dir=adapter_path,
                progress_callback=ft_progress,
            )
            ft_result.dataset_name = dataset_builtin
            finetune_run_id = ft_result.run_id

            # Save finetune run to DB
            db_conn = TraceDB(db)
            db_conn.save_finetune_run(ft_result)
            db_conn.close()

            err_console.print(
                f"  Training: loss {ft_result.train_loss_start:.4f} -> "
                f"{ft_result.train_loss_end:.4f}"
            )
        else:
            err_console.print("  Skipping finetune step.")

        # --- Step 4: Verify ---
        adapted_summary = None
        if not skip_finetune:
            progress.update(task, description="Step 4/4: Verifying with adapter...")

            device = _resolve_device(device)
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter_path)

            def verify_progress(i, total, prompt):
                progress.update(
                    task,
                    description=f"Verifying {i + 1}/{total}: {prompt[:40]}...",
                )

            verify_result = run_scan(
                model_obj,
                tokenizer,
                dataset,
                dataset_builtin,
                seed=seed,
                progress_callback=verify_progress,
            )

            adapted_summary = {
                "model": verify_result.model_name,
                "dataset": verify_result.dataset_name,
                "total": len(verify_result.prompt_results),
                "correct": verify_result.correct_count,
                "sabotaged": verify_result.sabotaged_count,
                "weak": verify_result.weak_count,
                "wrong": verify_result.wrong_count,
                "results": [
                    {
                        "prompt": r.prompt,
                        "answer": r.answer,
                        "final_token": r.final_token,
                        "final_prob": r.final_prob,
                        "status": r.status,
                        "flags": r.flags,
                    }
                    for r in verify_result.prompt_results
                ],
            }
            scan_adapted_path = os.path.join(output_dir, "scan_adapted.json")
            with open(scan_adapted_path, "w") as f:
                json.dump(adapted_summary, f, indent=2, default=str)

            err_console.print(
                f"  Adapted: {verify_result.correct_count} correct, "
                f"{verify_result.sabotaged_count} sabotaged, "
                f"{verify_result.weak_count} weak, "
                f"{verify_result.wrong_count} wrong"
            )
            del model_obj, tokenizer
        else:
            err_console.print("  Skipping verification (no finetune).")

        progress.update(task, description="Saving experiment...")

    duration = time.time() - start_time

    # --- Generate report.md ---
    report_lines = [
        f"# Experiment Report: {dataset_builtin}",
        "",
        f"**Experiment ID:** {experiment_id[:8]}",
        f"**Model:** {model}",
        f"**Dataset:** {dataset_builtin} ({len(dataset)} examples)",
        f"**Duration:** {duration:.1f}s",
        "",
        "## Baseline Accuracy",
        "",
        f"- Correct: {baseline_summary['correct']}",
        f"- Sabotaged: {baseline_summary['sabotaged']}",
        f"- Weak: {baseline_summary['weak']}",
        f"- Wrong: {baseline_summary['wrong']}",
        "",
    ]

    if not skip_ablate and ablation_summary.get("layer_impacts"):
        report_lines.extend([
            "## Ablation Analysis",
            "",
            "Top impactful layers (MLP-zero improvement count):",
            "",
        ])
        for layer_str, count in list(ablation_summary["layer_impacts"].items())[:5]:
            report_lines.append(f"- Layer {layer_str}: {count} improvements")
        report_lines.extend(["", f"**Selected target layers:** {layers}", ""])

    if not skip_finetune and finetune_run_id and ft_result is not None:
        report_lines.extend([
            "## Fine-tuning",
            "",
            f"- Target layers: {layers}",
            "- LoRA rank: 8, alpha: 16",
            "- Epochs: 10",
            f"- Loss: {ft_result.train_loss_start:.4f}"
            f" -> {ft_result.train_loss_end:.4f}",
            f"- Adapter: {adapter_path}",
            "",
        ])

    if adapted_summary:
        report_lines.extend([
            "## Post-Fix Accuracy",
            "",
            f"- Correct: {adapted_summary['correct']}",
            f"- Sabotaged: {adapted_summary['sabotaged']}",
            f"- Weak: {adapted_summary['weak']}",
            f"- Wrong: {adapted_summary['wrong']}",
            "",
            "## Delta",
            "",
            f"- Correct: {baseline_summary['correct']} -> {adapted_summary['correct']}"
            f" ({adapted_summary['correct'] - baseline_summary['correct']:+d})",
            f"- Sabotaged: {baseline_summary['sabotaged']}"
            f" -> {adapted_summary['sabotaged']}"
            f" ({adapted_summary['sabotaged'] - baseline_summary['sabotaged']:+d})",
            f"- Wrong: {baseline_summary['wrong']} -> {adapted_summary['wrong']}"
            f" ({adapted_summary['wrong'] - baseline_summary['wrong']:+d})",
            "",
        ])

        # Remaining failures
        remaining = [
            r for r in (verify_result.prompt_results if verify_result else [])
            if r.status != "correct"
        ]
        if remaining:
            report_lines.extend(["## Remaining Failures", ""])
            for r in remaining:
                report_lines.append(
                    f"- [{r.status}] {r.prompt} -> {r.final_token} "
                    f"(expected {r.answer})"
                )
            report_lines.append("")

    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # --- Save experiment to DB ---
    db_conn = TraceDB(db)
    db_conn.save_experiment({
        "id": experiment_id,
        "dataset_name": dataset_builtin,
        "model": model,
        "baseline_correct": baseline_summary["correct"],
        "baseline_sabotaged": baseline_summary["sabotaged"],
        "baseline_weak": baseline_summary["weak"],
        "baseline_wrong": baseline_summary["wrong"],
        "target_layers": json.dumps(layers),
        "finetune_run_id": finetune_run_id,
        "adapter_path": adapter_path if not skip_finetune else None,
        "result_correct": adapted_summary["correct"] if adapted_summary else None,
        "result_sabotaged": adapted_summary["sabotaged"] if adapted_summary else None,
        "result_weak": adapted_summary["weak"] if adapted_summary else None,
        "result_wrong": adapted_summary["wrong"] if adapted_summary else None,
        "created_at": datetime.now().isoformat(),
        "duration_seconds": duration,
    })
    db_conn.close()

    # --- Output ---
    if output_json:
        output = {
            "experiment_id": experiment_id,
            "model": model,
            "dataset": dataset_builtin,
            "output_dir": output_dir,
            "target_layers": layers,
            "baseline": {
                "correct": baseline_summary["correct"],
                "sabotaged": baseline_summary["sabotaged"],
                "weak": baseline_summary["weak"],
                "wrong": baseline_summary["wrong"],
            },
            "ablation": ablation_summary,
            "result": {
                "correct": adapted_summary["correct"],
                "sabotaged": adapted_summary["sabotaged"],
                "weak": adapted_summary["weak"],
                "wrong": adapted_summary["wrong"],
            }
            if adapted_summary
            else None,
            "duration_seconds": round(duration, 1),
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    console.print("\n[bold]Experiment complete[/bold]")
    console.print(f"[bold]ID:[/bold] {experiment_id[:8]}")
    console.print(f"[bold]Dataset:[/bold] {dataset_builtin} ({len(dataset)} examples)")
    console.print(f"[bold]Target layers:[/bold] {layers}")
    console.print(
        f"\n[bold]Baseline:[/bold] "
        f"{baseline_summary['correct']} correct, "
        f"{baseline_summary['sabotaged']} sabotaged, "
        f"{baseline_summary['weak']} weak, "
        f"{baseline_summary['wrong']} wrong"
    )
    if adapted_summary:
        console.print(
            f"[bold]Adapted:[/bold]  "
            f"{adapted_summary['correct']} correct, "
            f"{adapted_summary['sabotaged']} sabotaged, "
            f"{adapted_summary['weak']} weak, "
            f"{adapted_summary['wrong']} wrong"
        )
        delta_correct = adapted_summary["correct"] - baseline_summary["correct"]
        console.print(
            f"[bold]Delta:[/bold]    {delta_correct:+d} correct"
        )
    console.print(f"\n[green]Results saved to {output_dir}/[/green]")
    console.print(f"Duration: {duration:.1f}s")


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt", required=True, help="Target prompt.")
@click.option("--layer", required=True, type=int, help="MLP layer to analyze.")
@click.option(
    "--contrast",
    default=None,
    help="Contrast prompt (model gets this right).",
)
@click.option("--top-n", default=20, type=int, help="Top N neurons.")
@click.option(
    "--save-profile",
    default=None,
    help="Label to save profile.",
)
@click.option(
    "--ablate",
    "ablate_mode",
    is_flag=True,
    help="Enable ablation mode.",
)
@click.option(
    "--neurons",
    "neuron_str",
    default=None,
    help="Neuron indices: '0,1,2' or '100-200'.",
)
@click.option(
    "--from-profile",
    default=None,
    help="Use top-N neurons from a saved profile.",
)
@click.option(
    "--group-size",
    default=1,
    type=int,
    help="Ablate in groups of N (default: 1).",
)
@click.option("--baseline", default=None, help="Baseline trace ID for comparison.")
@click.option("--label", default=None, help="Label prefix for ablation traces.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def neurons(
    db,
    model,
    prompt,
    layer,
    contrast,
    top_n,
    save_profile,
    ablate_mode,
    neuron_str,
    from_profile,
    group_size,
    baseline,
    label,
    seed,
    output_json,
    adapter,
    device,
):
    """Neuron-level MLP attribution: profile or ablate individual neurons."""
    if ablate_mode:
        _neurons_ablate(
            db,
            model,
            prompt,
            layer,
            top_n,
            neuron_str,
            from_profile,
            group_size,
            baseline,
            label,
            seed,
            output_json,
            adapter,
            device=device,
        )
    else:
        _neurons_profile(
            db,
            model,
            prompt,
            layer,
            contrast,
            top_n,
            save_profile,
            seed,
            output_json,
            adapter,
            device=device,
        )


def _neurons_profile(
    db,
    model,
    prompt,
    layer,
    contrast,
    top_n,
    save_profile,
    seed,
    output_json,
    adapter=None,
    device="cpu",
):
    """Profile mode: capture and rank MLP intermediate neuron activations."""
    from neurotrace.neurons import profile_neurons

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        disable=output_json,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Profiling neurons...")

        profile = profile_neurons(
            model_obj,
            tokenizer,
            prompt,
            layer,
            top_n=top_n,
            contrast_prompt=contrast,
            seed=seed,
            label=save_profile,
        )

        if save_profile:
            progress.update(task, description="Saving profile...")
            db_conn = TraceDB(db)
            db_conn.save_neuron_profile(profile)
            db_conn.close()

        progress.update(task, description="Done.")

    if output_json:
        output = {
            "profile_id": profile.profile_id,
            "layer": profile.layer,
            "prompt": profile.prompt,
            "contrast_prompt": profile.contrast_prompt,
            "model": profile.model_name,
            "neurons": [
                {
                    "index": idx,
                    "target_activation": t_act,
                    "contrast_activation": c_act,
                    "diff_activation": d_act,
                    "abs_diff": abs(d_act) if d_act is not None else abs(t_act),
                }
                for idx, t_act, c_act, d_act in zip(
                    profile.neuron_indices,
                    profile.target_activations,
                    profile.contrast_activations
                    or [None] * len(profile.neuron_indices),
                    profile.diff_activations or [None] * len(profile.neuron_indices),
                )
            ],
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    console.print(f"\n[bold]Neuron Profile:[/bold] Layer {profile.layer} MLP")
    console.print(f"[bold]Prompt:[/bold] {profile.prompt}")
    if profile.contrast_prompt:
        console.print(f"[bold]Contrast:[/bold] {profile.contrast_prompt}")
    console.print()

    table = Table(title=f"Top {top_n} Neurons (Layer {layer} MLP)")
    table.add_column("Neuron", justify="right", style="cyan")
    table.add_column("Target Act", justify="right")
    if profile.contrast_activations:
        table.add_column("Contrast Act", justify="right")
    if profile.diff_activations:
        table.add_column("Diff", justify="right")
        table.add_column("Abs Diff", justify="right")

    for i, idx in enumerate(profile.neuron_indices):
        row = [str(idx), f"{profile.target_activations[i]:.4f}"]
        if profile.contrast_activations:
            row.append(f"{profile.contrast_activations[i]:.4f}")
        if profile.diff_activations:
            d = profile.diff_activations[i]
            row.append(f"{d:+.4f}")
            row.append(f"{abs(d):.4f}")
        table.add_row(*row)

    console.print(table)

    if save_profile:
        console.print(
            f"\n[green]Profile saved as {save_profile!r}"
            f" (id: {profile.profile_id[:8]})[/green]"
        )


def _neurons_ablate(
    db,
    model,
    prompt,
    layer,
    top_n,
    neuron_str,
    from_profile,
    group_size,
    baseline,
    label,
    seed,
    output_json,
    adapter=None,
    device="cpu",
):
    """Ablate mode: zero specific neurons and measure prediction impact."""
    from neurotrace.neurons import ablate_neurons, parse_neurons

    # Determine which neurons to ablate
    if neuron_str is None and from_profile is None:
        raise click.UsageError("Ablation mode requires --neurons or --from-profile.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        disable=output_json,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        db_conn = TraceDB(db)

        # Resolve neuron indices
        if from_profile:
            progress.update(task, description="Loading profile...")
            profile = db_conn.load_neuron_profile(from_profile)
            if profile is None:
                db_conn.close()
                raise click.ClickException(f"Profile not found: {from_profile}")
            neuron_indices = profile.neuron_indices[:top_n]
        else:
            neuron_indices = parse_neurons(neuron_str)

        # Build groups
        if group_size <= 0:
            group_size = 1
        neuron_groups = []
        for i in range(0, len(neuron_indices), group_size):
            neuron_groups.append(neuron_indices[i : i + group_size])

        # Resolve baseline
        baseline_trace = None
        if baseline is not None:
            try:
                baseline_id = _resolve_trace_id(db_conn, baseline)
                baseline_trace = db_conn.read_trace(baseline_id)
                progress.update(task, description=f"Loaded baseline {baseline_id[:8]}.")
            except ValueError as e:
                db_conn.close()
                raise click.ClickException(str(e))

        # Run ablations
        progress.update(
            task,
            description=f"Ablating {len(neuron_groups)} group(s)...",
        )
        baseline_created, results = ablate_neurons(
            model_obj,
            tokenizer,
            prompt,
            layer,
            neuron_groups=neuron_groups,
            baseline=baseline_trace,
            label_prefix=label,
            seed=seed,
        )

        # Store traces
        progress.update(task, description="Storing traces...")
        if baseline_created is not None:
            db_conn.write_trace(baseline_created)

        for r in results:
            interventions = json.dumps(
                {"zero_neurons": {"layer": layer, "neurons": r.neurons}}
            )
            db_conn.write_trace(r.trace, interventions=interventions)

        db_conn.close()
        progress.update(task, description="Done.")

    if output_json:
        output = {
            "layer": layer,
            "prompt": prompt,
            "results": [
                {
                    "neurons": r.neurons,
                    "baseline_top1": r.baseline_top1,
                    "baseline_top1_prob": r.baseline_top1_prob,
                    "ablated_top1": r.ablated_top1,
                    "ablated_top1_prob": r.ablated_top1_prob,
                    "changed": r.changed,
                    "trace_id": r.trace.metadata.trace_id,
                }
                for r in results
            ],
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    console.print(f"\n[bold]Neuron Ablation:[/bold] Layer {layer} MLP")
    console.print(f"[bold]Prompt:[/bold] {prompt}\n")

    table = Table(title="Neuron Ablation Results")
    table.add_column("Neuron(s)", justify="right", style="cyan")
    table.add_column("Baseline Top-1")
    table.add_column("Ablated Top-1")
    table.add_column("Changed?")

    for r in results:
        neurons_str = ",".join(map(str, r.neurons))
        if len(neurons_str) > 20:
            neurons_str = neurons_str[:17] + "..."
        style = "green" if r.changed else None
        marker = "\u2713 YES" if r.changed else ""
        table.add_row(
            neurons_str,
            f"{r.baseline_top1} ({r.baseline_top1_prob:.3f})",
            f"{r.ablated_top1} ({r.ablated_top1_prob:.3f})",
            marker,
            style=style,
        )

    console.print(table)

    changed_count = sum(1 for r in results if r.changed)
    console.print(
        f"\n[bold]Summary:[/bold] {changed_count}/{len(results)}"
        f" ablation(s) changed the top-1 prediction."
    )


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option(
    "--dataset-builtin",
    default=None,
    help="Built-in dataset name.",
)
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    type=click.Path(exists=True),
    help="Custom dataset JSON file.",
)
@click.option(
    "--layer", default=None, type=int,
    help="Layer to extract (default: auto).",
)
@click.option(
    "--extraction-point",
    default="pre_mlp",
    type=click.Choice(["pre_mlp", "post_attn", "post_mlp"]),
    help="Activation extraction point.",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--output", default=None, help="Output directory.")
@click.option(
    "--cross-dataset",
    default=None,
    help="Cross-domain dataset (builtin name or path).",
)
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def probe(
    db,
    model,
    dataset_builtin,
    dataset_path,
    layer,
    extraction_point,
    seed,
    output,
    cross_dataset,
    adapter,
    output_json,
    device,
):
    """Find the sabotage direction in activation space via linear probing."""
    from neurotrace.datasets import get_builtin_dataset, load_dataset
    from neurotrace.probe import (
        auto_detect_layer,
        run_probe,
        save_probe_outputs,
    )
    from neurotrace.scan import run_scan

    if dataset_path is None and dataset_builtin is None:
        raise click.UsageError("Must provide either --dataset or --dataset-builtin.")
    if dataset_path is not None and dataset_builtin is not None:
        raise click.UsageError("Cannot provide both --dataset and --dataset-builtin.")

    # Load dataset
    if dataset_builtin is not None:
        dataset = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    else:
        dataset = load_dataset(dataset_path)
        dataset_name = dataset_path

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        # Run scan to get labels
        progress.update(task, description="Scanning dataset for sabotage labels...")

        def scan_progress(i, total, prompt):
            progress.update(
                task,
                description=f"Scanning {i + 1}/{total}: {prompt[:40]}...",
            )

        scan_result = run_scan(
            model_obj,
            tokenizer,
            dataset,
            dataset_name,
            seed=seed,
            progress_callback=scan_progress,
        )

        # Auto-detect layer if not specified
        if layer is None:
            layer = auto_detect_layer(scan_result)
            if layer is None:
                raise click.ClickException(
                    "No sabotage layers found in scan. Specify --layer manually."
                )
            err_console.print(f"Auto-detected sabotage layer: {layer}")

        # Cross-dataset scan
        cross_scan = None
        cross_name = None
        if cross_dataset is not None:
            progress.update(
                task,
                description=f"Cross-scan: {cross_dataset}...",
            )
            try:
                cross_ds = get_builtin_dataset(cross_dataset)
                cross_name = cross_dataset
            except ValueError:
                cross_ds = load_dataset(cross_dataset)
                cross_name = cross_dataset

            def cross_progress(i, total, prompt):
                progress.update(
                    task,
                    description=f"Cross-scan {i + 1}/{total}: {prompt[:40]}...",
                )

            cross_scan = run_scan(
                model_obj,
                tokenizer,
                cross_ds,
                cross_name,
                seed=seed,
                progress_callback=cross_progress,
            )

        # Run probe
        def probe_progress(desc):
            progress.update(task, description=desc)

        result = run_probe(
            model_obj,
            tokenizer,
            scan_result,
            layer=layer,
            extraction_point=extraction_point,
            seed=seed,
            cross_scan_result=cross_scan,
            cross_dataset_name=cross_name,
            progress_callback=probe_progress,
        )

        # Determine output directory
        if output is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = f"probes/{dataset_name}-{timestamp}"

        # Save outputs
        progress.update(task, description="Saving outputs...")
        save_probe_outputs(result, output)

        # Save to DB
        import uuid

        probe_id = str(uuid.uuid4())
        db_conn = TraceDB(db)
        db_conn.save_probe(probe_id, result)
        db_conn.close()

        progress.update(task, description="Done.")

    if output_json:
        json_output = {
            "probe_id": probe_id,
            "dataset": result.dataset_name,
            "model": result.model_name,
            "layer": result.layer,
            "extraction_point": result.extraction_point,
            "num_clean": result.num_clean,
            "num_sabotaged": result.num_sabotaged,
            "cohens_d": result.cohens_d,
            "auc_roc": result.auc_roc,
            "probe_accuracy": result.probe_accuracy,
            "probe_correct": result.probe_correct,
            "probe_total": result.probe_total,
            "direction_alignment": result.direction_alignment,
            "pca_explained_variance": result.pca_explained_variance.tolist(),
            "output_dir": output,
        }
        if result.cross_dataset:
            json_output["cross_dataset"] = result.cross_dataset
            json_output["cross_auc_roc"] = result.cross_auc_roc
        click.echo(json.dumps(json_output, indent=2, default=str))
        return

    # Rich output
    console.print(f"\n[bold]Probe Analysis:[/bold] {result.dataset_name}")
    console.print(
        f"Layer {result.layer} | {result.extraction_point} | "
        f"{result.num_clean} clean, {result.num_sabotaged} sabotaged"
    )

    if result.num_sabotaged < 5:
        console.print(
            f"\n[yellow]Warning: Only {result.num_sabotaged} sabotaged examples."
            f" Results may be unreliable.[/yellow]"
        )

    console.print("\n[bold]Mean Difference Direction:[/bold]")
    clean_scores = result.projection_scores[~result.labels]
    sabo_scores = result.projection_scores[result.labels]
    c_m, c_s = clean_scores.mean(), clean_scores.std()
    s_m, s_s = sabo_scores.mean(), sabo_scores.std()
    console.print(f"  Clean mean:  {c_m:.4f} +/- {c_s:.4f}")
    console.print(f"  Sabo mean:   {s_m:.4f} +/- {s_s:.4f}")
    console.print(f"  Cohen's d:   {result.cohens_d:.4f}")
    console.print(f"  AUC-ROC:     {result.auc_roc:.4f}")

    console.print("\n[bold]Linear Probe (LOO):[/bold]")
    acc = f"{result.probe_correct}/{result.probe_total}"
    console.print(f"  Accuracy: {acc} ({result.probe_accuracy:.1%})")
    console.print(f"  Direction alignment: {result.direction_alignment:.4f}")

    # Per-prompt table
    table = Table(title="Per-Prompt Projections")
    table.add_column("Prompt")
    table.add_column("Label", justify="center")
    table.add_column("Score", justify="right")

    for i, prompt in enumerate(result.prompts):
        label = "SABO" if result.labels[i] else "clean"
        score = result.projection_scores[i]
        style = "red" if result.labels[i] else None
        prompt_display = prompt if len(prompt) <= 40 else "..." + prompt[-37:]
        table.add_row(prompt_display, label, f"{score:.4f}", style=style)

    console.print(table)

    pca_var = result.pca_explained_variance
    console.print(
        f"\n[bold]PCA:[/bold] PC1={pca_var[0]:.1%},"
        f" PC2={pca_var[1]:.1%}, PC3={pca_var[2]:.1%}"
    )

    if result.cross_dataset:
        console.print(
            f"\n[bold]Cross-Domain ({result.cross_dataset}):[/bold] "
            f"AUC-ROC = {result.cross_auc_roc:.4f}"
        )

    console.print(f"\n[green]Results saved to {output}/[/green]")


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option(
    "--probe-dir", required=True,
    help="Path to probe output directory (contains mean_direction.npy).",
)
@click.option(
    "--layer", default=None, type=int,
    help="MLP layer to decompose (default: auto from probe).",
)
@click.option(
    "--top-k", default=30, type=int,
    help="Number of top tokens to show.",
)
@click.option(
    "--reverse-tokens", default=None,
    help="Comma-separated tokens to trace backwards.",
)
@click.option(
    "--compare-prompt", multiple=True,
    help="Compare MLP behavior on a prompt vs direction.",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--output", default=None, help="Output directory.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
def circuit(
    db,
    model,
    probe_dir,
    layer,
    top_k,
    reverse_tokens,
    compare_prompt,
    seed,
    output,
    adapter,
    output_json,
    device,
):
    """Trace computational circuit from probe direction through MLP to token outputs."""
    from neurotrace.circuit import (
        detect_layer_from_probe,
        run_circuit,
        save_circuit_outputs,
    )

    # Auto-detect layer if not specified
    if layer is None:
        layer = detect_layer_from_probe(probe_dir)
        if layer is None:
            raise click.ClickException(
                "Could not auto-detect layer from probe. Specify --layer manually."
            )
        err_console.print(f"Auto-detected layer: {layer}")

    # Parse reverse tokens
    reverse_list = None
    if reverse_tokens:
        reverse_list = [t.strip() for t in reverse_tokens.split(",") if t.strip()]

    # Parse compare prompts
    compare_list = list(compare_prompt) if compare_prompt else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
        disable=output_json,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        device = _resolve_device(device)
        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        progress.update(task, description="Model loaded.")

        progress.update(task, description="Running circuit analysis...")
        result = run_circuit(
            model_obj,
            tokenizer,
            probe_dir,
            layer=layer,
            top_k=top_k,
            reverse_tokens=reverse_list,
            compare_prompts=compare_list,
            seed=seed,
        )

        # Determine output directory
        if output is None:
            import os
            from datetime import datetime

            probe_name = os.path.basename(probe_dir.rstrip("/"))
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = f"circuits/{probe_name}-{timestamp}"

        progress.update(task, description="Saving outputs...")
        save_circuit_outputs(result, output)

        # Save to DB
        import uuid

        circuit_id = str(uuid.uuid4())
        db_conn = TraceDB(db)
        db_conn.save_circuit(circuit_id, result)
        db_conn.close()

        progress.update(task, description="Done.")

    if output_json:
        forward_data = {
            "circuit_id": circuit_id,
            "model": result.model_name,
            "layer": result.layer,
            "top_boosted": result.forward.top_boosted,
            "top_suppressed": result.forward.top_suppressed,
            "pre_mlp_top": result.forward.pre_mlp_top,
            "output_dir": output,
        }
        if result.reverse:
            forward_data["reverse"] = [
                {"token": r.token, "cosine_sim": r.cosine_sim_with_probe}
                for r in result.reverse
            ]
        if result.comparisons:
            forward_data["comparisons"] = [
                {
                    "prompt": c.prompt,
                    "activation_norm": c.activation_norm,
                    "cosine_sim_with_direction": c.cosine_sim_with_direction,
                    "top_boosted": c.top_boosted[:10],
                    "top_suppressed": c.top_suppressed[:10],
                }
                for c in result.comparisons
            ]
        if result.real_comparison:
            rc = result.real_comparison
            forward_data["real_comparison"] = {
                "prompts": [
                    {
                        "text": p.prompt,
                        "mlp_output_norm": p.mlp_output_norm,
                        "sabotage_projection": p.sabotage_projection,
                        "top_boosted": p.top_boosted[:10],
                        "top_suppressed": p.top_suppressed[:10],
                    }
                    for p in rc.prompts
                ],
                "pairwise": [
                    {
                        "prompt_a": pw.prompt_a,
                        "prompt_b": pw.prompt_b,
                        "cosine_similarity": pw.cosine_similarity,
                        "norm_ratio": pw.norm_ratio,
                        "diff_boosted": pw.diff_boosted[:10],
                        "diff_suppressed": pw.diff_suppressed[:10],
                    }
                    for pw in rc.pairwise
                ],
            }
        click.echo(json.dumps(forward_data, indent=2))
        return

    # Rich output
    console.print(f"\n[bold]Circuit Analysis:[/bold] Layer {result.layer} MLP")
    console.print(f"Probe: {probe_dir}")

    # Forward analysis
    console.print("\n[bold]Tokens BOOSTED by MLP:[/bold]")
    boost_table = Table()
    boost_table.add_column("Rank", justify="right")
    boost_table.add_column("Token")
    boost_table.add_column("Logit", justify="right")
    for i, t in enumerate(result.forward.top_boosted[:15], 1):
        boost_table.add_row(str(i), t["token"], f"{t['logit']:+.4f}")
    console.print(boost_table)

    console.print("\n[bold]Tokens SUPPRESSED by MLP:[/bold]")
    suppress_table = Table()
    suppress_table.add_column("Rank", justify="right")
    suppress_table.add_column("Token")
    suppress_table.add_column("Logit", justify="right")
    for i, t in enumerate(result.forward.top_suppressed[:15], 1):
        suppress_table.add_row(str(i), t["token"], f"{t['logit']:+.4f}")
    console.print(suppress_table)

    console.print("\n[bold]Pre-MLP Prediction (logit lens):[/bold]")
    pre_table = Table()
    pre_table.add_column("Rank", justify="right")
    pre_table.add_column("Token")
    pre_table.add_column("Logit", justify="right")
    for i, t in enumerate(result.forward.pre_mlp_top[:15], 1):
        pre_table.add_row(str(i), t["token"], f"{t['logit']:+.4f}")
    console.print(pre_table)

    # Reverse analysis
    if result.reverse:
        console.print("\n[bold]Reverse Circuit (Token -> Input Direction):[/bold]")
        rev_table = Table()
        rev_table.add_column("Token")
        rev_table.add_column("Cosine Sim", justify="right")
        for r in result.reverse:
            style = "green" if abs(r.cosine_sim_with_probe) > 0.3 else None
            rev_table.add_row(r.token, f"{r.cosine_sim_with_probe:.4f}", style=style)
        console.print(rev_table)

    # Prompt comparisons
    if result.comparisons:
        for comp in result.comparisons:
            console.print(f"\n[bold]Prompt:[/bold] \"{comp.prompt}\"")
            console.print(
                f"  Activation norm: {comp.activation_norm:.4f} | "
                f"Cosine sim with direction: {comp.cosine_sim_with_direction:.4f}"
            )
            top_tokens = ", ".join(
                f"{t['token']}({t['logit']:+.2f})" for t in comp.top_boosted[:5]
            )
            console.print(f"  Top boosted: {top_tokens}")

    # Real MLP comparison
    if result.real_comparison:
        rc = result.real_comparison
        console.print("\n[bold cyan]Real MLP Comparison:[/bold cyan]")
        for p in rc.prompts:
            console.print(f"\n[bold]Prompt:[/bold] \"{p.prompt}\"")
            console.print(
                f"  MLP output norm: {p.mlp_output_norm:.2f} | "
                f"Sabotage projection: {p.sabotage_projection:.4f}"
            )
            real_top = ", ".join(
                f"{t['token']}({t['logit']:+.2f})" for t in p.top_boosted[:5]
            )
            console.print(f"  Top MLP-boosted: {real_top}")

        for pw in rc.pairwise:
            console.print(
                f'\n[bold]Diff:[/bold] "{pw.prompt_a}"'
                f' vs "{pw.prompt_b}"'
            )
            console.print(
                f"  Cosine sim: {pw.cosine_similarity:.4f} | "
                f"Norm ratio: {pw.norm_ratio:.4f}"
            )
            diff_tokens = ", ".join(
                f"{t['token']}({t['logit']:+.2f})" for t in pw.diff_boosted[:5]
            )
            console.print(f"  More boosted for first: {diff_tokens}")

    console.print(f"\n[green]Results saved to {output}/[/green]")


def _finetune_remote(
    remote,
    dataset_builtin,
    dataset_path,
    target_layers,
    rank,
    alpha,
    epochs,
    lr,
    output_dir,
    seed,
    output_json,
    db,
):
    """Run LoRA fine-tuning on a remote GPU worker."""
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote, timeout=600.0)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))

    # Parse target layers
    layers = [int(x.strip()) for x in target_layers.split(",") if x.strip()]

    # Load dataset
    dataset_items = []
    if dataset_builtin is not None:
        from neurotrace.datasets import get_builtin_dataset

        raw = get_builtin_dataset(dataset_builtin)
        dataset_items = [{"prompt": d["prompt"], "answer": d["answer"]} for d in raw]
    elif dataset_path is not None:
        from neurotrace.datasets import load_dataset

        raw = load_dataset(dataset_path)
        dataset_items = [{"prompt": d["prompt"], "answer": d["answer"]} for d in raw]

    # Determine output directory
    if output_dir is None:
        import re
        from datetime import datetime

        model_name = health.get("model", "unknown")
        model_slug = re.sub(r"[^a-zA-Z0-9]", "-", model_name).strip("-")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = f"adapters/{model_slug}-{timestamp}"

    config = {
        "dataset": dataset_items,
        "target_layers": layers,
        "target_modules": ["gate_proj", "up_proj", "down_proj"],
        "epochs": epochs,
        "lr": lr,
        "rank": rank,
        "batch_size": 4,
    }

    total_time = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            f"Training LoRA on {device_name}...", total=None
        )

        adapter_id = None
        final_loss = None
        for event in worker.finetune_stream(config):
            etype = event.get("type")
            if etype == "config":
                total_steps = event.get("total_steps", 0)
                progress.update(task, total=total_steps)
            elif etype == "progress":
                step = event.get("step", 0)
                loss = event.get("loss", 0)
                epoch = event.get("epoch", 0)
                progress.update(
                    task,
                    completed=step,
                    description=(
                        f"Epoch {epoch} step {step} loss: {loss:.4f}"
                    ),
                )
            elif etype == "epoch_end":
                avg_loss = event.get("avg_loss", 0)
                epoch = event.get("epoch", 0)
                err_console.print(
                    f"  Epoch {epoch} avg loss: {avg_loss:.4f}"
                )
            elif etype == "complete":
                adapter_id = event.get("adapter_id")
                final_loss = event.get("final_loss")
                total_time = event.get("total_time_seconds", 0)
                progress.update(
                    task,
                    description=f"Training complete! loss={final_loss:.4f}",
                )

        if adapter_id:
            progress.update(
                task, description=f"Downloading adapter {adapter_id}..."
            )
            import os

            os.makedirs(output_dir, exist_ok=True)
            worker.download_adapter(adapter_id, output_dir)
            progress.update(task, description="Done.")

    console.print("\n[bold]Remote LoRA Training Complete[/bold]")
    console.print(f"Device: {device_name}")
    if final_loss is not None:
        console.print(f"Final loss: {final_loss:.4f}")
    if total_time:
        console.print(f"Training time: {total_time:.1f}s")
    console.print(f"Adapter saved to: {output_dir}")

    if output_json:
        click.echo(
            json.dumps(
                {
                    "adapter_id": adapter_id,
                    "output_dir": output_dir,
                    "final_loss": final_loss,
                    "total_time_seconds": total_time,
                    "device": device_name,
                },
                indent=2,
            )
        )


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--dataset-builtin",
    default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    type=click.Path(exists=True),
    help="Custom JSONL dataset path.",
)
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
@click.option(
    "--device", default="cpu", help="Device: cpu, cuda, directml, auto."
)
@click.option("--layers", default=None, help="Layer range (default: all).")
@click.option("--html", "html_path", default=None, help="HTML heatmap output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def heatmap(
    model,
    db,
    dataset_builtin,
    dataset_path,
    remote,
    device,
    layers,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Generate MLP ablation heatmap: zero each layer, measure changes."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.datasets import get_builtin_dataset, load_dataset
    from neurotrace.heatmap import (
        HeatmapResult,
        build_layer_summaries,
        generate_heatmap_html,
        heatmap_result_to_dict,
    )

    # Validate inputs
    if dataset_path is None and dataset_builtin is None:
        raise click.UsageError("Must provide either --dataset or --dataset-builtin.")
    if dataset_path is not None and dataset_builtin is not None:
        raise click.UsageError("Cannot provide both --dataset and --dataset-builtin.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Load dataset
    if dataset_builtin is not None:
        dataset = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    else:
        dataset = load_dataset(dataset_path)
        dataset_name = dataset_path

    prompts = [{"prompt": d["prompt"], "answer": d["answer"]} for d in dataset]

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    if remote is not None:
        cells = _heatmap_remote(remote, prompts, seed)
        # Get model name and num_layers from worker
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        health = worker.health()
        model_name = health["model"]
        num_layers = health["num_layers"]
    else:
        from neurotrace.heatmap import run_heatmap_local
        from neurotrace.models import get_architecture, load_model

        device = _resolve_device(device)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=err_console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter)
            arch = get_architecture(model_obj.config.model_type)
            num_layers = len(arch.get_layers(model_obj))
            model_name = model

            progress.update(
                task,
                description=(
                    f"Running heatmap: {len(prompts)} prompts x {num_layers} layers"
                ),
                total=len(prompts) * num_layers,
            )

            def progress_cb(prompt_idx, layer_idx, n_prompts, n_layers):
                completed = prompt_idx * n_layers + layer_idx + 1
                progress.update(
                    task,
                    completed=completed,
                    description=(
                        f"Prompt {prompt_idx + 1}/{n_prompts} "
                        f"Layer {layer_idx}/{n_layers}"
                    ),
                )

            cells = run_heatmap_local(
                model_obj,
                tokenizer,
                arch,
                prompts,
                seed=seed,
                progress_callback=progress_cb,
            )
            progress.update(task, description="Done.")

    # Build summaries
    layer_summaries = build_layer_summaries(cells, num_layers, len(prompts))

    result = HeatmapResult(
        run_id=run_id,
        dataset_name=dataset_name,
        model_name=model_name,
        num_layers=num_layers,
        num_prompts=len(prompts),
        cells=cells,
        layer_summaries=layer_summaries,
        created_at=created_at,
    )

    # Save to DB
    result_dict = heatmap_result_to_dict(result)
    db_conn = TraceDB(db)
    db_conn.write_heatmap_run(
        run_id=run_id,
        dataset_name=dataset_name,
        model_name=model_name,
        num_layers=num_layers,
        num_prompts=len(prompts),
        cells_json=json.dumps(result_dict["cells"]),
        summaries_json=json.dumps(result_dict["layer_summaries"]),
        adapter_path=adapter,
    )
    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        html = generate_heatmap_html(result)
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Heatmap saved to {html_path}[/green]")

    # JSON output
    if output_json:
        click.echo(json.dumps(result_dict, indent=2))
        return

    # Terminal table
    console.print(
        f"\n[bold]Heatmap:[/bold] {dataset_name} "
        f"({len(prompts)} prompts x {num_layers} layers)"
    )
    console.print(f"Model: {model_name}\n")

    table = Table()
    table.add_column("Layer", justify="right")
    table.add_column("Fixes", justify="right")
    table.add_column("Breaks", justify="right")
    table.add_column("Changes", justify="right")
    table.add_column("Avg DProb", justify="right")
    table.add_column("Impact", justify="right")

    for s in layer_summaries:
        style = None
        if s.impact_score > 0.05:
            style = "green"
        elif s.impact_score < -0.05:
            style = "red"
        table.add_row(
            str(s.layer),
            str(s.fixes),
            str(s.breaks),
            str(s.changes),
            f"{s.avg_delta_correct_prob:+.3f}",
            f"{s.impact_score:+.3f}",
            style=style,
        )

    console.print(table)


def _heatmap_remote(remote_url, prompts, seed):
    """Run heatmap via remote GPU worker with SSE streaming."""
    from neurotrace.heatmap import HeatmapCell, check_correct
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    num_layers = health["num_layers"]
    device_name = health.get("device_name", health.get("device", "unknown"))

    err_console.print(
        f"GPU: {device_name} via {remote_url}"
    )

    cells = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        outer_task = progress.add_task(
            "Heatmap", total=len(prompts)
        )
        inner_task = progress.add_task(
            "Layer", total=num_layers + 1
        )

        for prompt_idx, item in enumerate(prompts):
            prompt = item["prompt"]
            answer = item["answer"]
            progress.update(
                outer_task,
                description=f"Prompt {prompt_idx + 1}/{len(prompts)}",
            )
            progress.update(inner_task, completed=0)

            baseline_token = None
            baseline_prob = 0.0
            baseline_correct = False

            for event in worker.batch_ablate_stream(
                prompt, num_layers, seed=seed
            ):
                etype = event.get("type")
                if etype == "progress":
                    idx = event.get("index", 0)
                    progress.update(inner_task, completed=idx)
                elif etype == "result":
                    idx = event.get("index", 0)
                    token = event.get("final_token", "")
                    prob = event.get("final_prob", 0.0)
                    zero_layers = event.get("zero_mlp_layers", [])

                    if idx == 0:
                        # Baseline
                        baseline_token = token
                        baseline_prob = prob
                        baseline_correct = check_correct(token, answer)
                    else:
                        # Ablated — zero_layers has one element
                        layer = zero_layers[0] if zero_layers else idx - 1
                        ablated_correct = check_correct(token, answer)

                        if not baseline_correct and ablated_correct:
                            flip_dir = "fixed"
                        elif baseline_correct and not ablated_correct:
                            flip_dir = "broke"
                        elif token != baseline_token:
                            flip_dir = "changed"
                        else:
                            flip_dir = "none"

                        cells.append(
                            HeatmapCell(
                                prompt_index=prompt_idx,
                                prompt=prompt,
                                expected_answer=answer,
                                layer=layer,
                                baseline_token=baseline_token,
                                baseline_prob=baseline_prob,
                                baseline_correct=baseline_correct,
                                ablated_token=token,
                                ablated_prob=prob,
                                ablated_correct=ablated_correct,
                                delta_correct_prob=prob - baseline_prob,
                                flipped=token != baseline_token,
                                flip_direction=flip_dir,
                            )
                        )

            progress.update(outer_task, advance=1)

    return cells


@cli.command("probe-universal")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--heatmap-run",
    default="latest",
    help="Heatmap run ID, or 'latest' (default), or 'all'.",
)
@click.option(
    "--layer-range",
    default="14-21",
    help="Layer range for activation extraction (e.g. '14-21').",
)
@click.option("--output", default=None, help="Output directory for probe files.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
@click.option(
    "--model", default=None,
    help="HuggingFace model name (auto from heatmap if omitted).",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option(
    "--device", default="cpu",
    help="Device: cpu, cuda, directml, auto.",
)
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def probe_universal(
    db,
    heatmap_run,
    layer_range,
    output,
    output_json,
    model,
    device,
    adapter,
    remote,
    seed,
):
    """Train a universal vulnerability probe using heatmap data."""
    import uuid
    from datetime import datetime

    from neurotrace.probe_universal import (
        build_labels_from_heatmap_runs,
        label_from_heatmap,
        save_universal_probe,
        train_universal_probe,
    )

    # Parse layer range
    try:
        parts = layer_range.split("-")
        layer_start, layer_end = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        raise click.UsageError(
            f"Invalid --layer-range: {layer_range!r}. "
            "Use 'START-END'."
        )

    if remote is None and model is None:
        # Will auto-detect from heatmap run below
        pass

    db_conn = TraceDB(db)

    # Resolve heatmap runs
    if heatmap_run == "all":
        runs = db_conn.get_all_heatmap_runs()
        if not runs:
            raise click.ClickException(
                "No heatmap runs found in database."
            )
    elif heatmap_run == "latest":
        run_id = db_conn.get_latest_heatmap_run_id()
        runs = [db_conn.read_heatmap_run(run_id)]
    else:
        runs = [db_conn.read_heatmap_run(heatmap_run)]

    run_ids = [r["run_id"] for r in runs]
    model_name = model or runs[0]["model_name"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            "Building labels from heatmap data...", total=None,
        )

        # Build labels (filter breaks to layer range)
        lr = (layer_start, layer_end)
        prompts, labels, domains = build_labels_from_heatmap_runs(
            runs, layer_range=lr,
        )
        n_excluded = sum(
            1 for r in runs
            for info in label_from_heatmap(
                r["cells"], layer_range=lr,
            ).values()
            if not info["baseline_correct"]
        )

        n_vuln = int(labels.sum())
        n_robust = int((~labels).sum())
        err_console.print(
            f"  Samples: {len(prompts)} "
            f"({n_vuln} vulnerable, {n_robust} robust, "
            f"{n_excluded} excluded)"
        )

        if n_vuln < 2 or n_robust < 2:
            raise click.ClickException(
                f"Need at least 2 vulnerable and 2 robust "
                f"samples. Got {n_vuln} vulnerable, "
                f"{n_robust} robust."
            )

        # Extract activations — remote or local
        if remote is not None:
            activations = _probe_universal_remote(
                remote, prompts, layer_start, layer_end,
                seed, progress, task,
            )
        else:
            from neurotrace.probe_universal import (
                extract_multilayer_activations,
            )

            progress.update(
                task, description="Loading model...",
            )
            from neurotrace.models import load_model

            device = _resolve_device(device)
            model_obj, tokenizer = load_model(
                model_name, device=device,
            )
            model_obj = _maybe_load_adapter(model_obj, adapter)

            progress.update(
                task,
                description=(
                    f"Extracting activations "
                    f"(layers {layer_start}-{layer_end}, "
                    f"{len(prompts)} prompts)..."
                ),
            )
            activations = extract_multilayer_activations(
                model_obj, tokenizer, prompts,
                layer_start, layer_end, seed=seed,
            )

        # Train probe
        progress.update(
            task, description="Training universal probe (LOO CV)...",
        )
        result = train_universal_probe(activations, labels, domains)

        # Fill in metadata
        result.heatmap_run_ids = run_ids
        result.model_name = model_name
        result.layer_range = (layer_start, layer_end)
        result.n_excluded = n_excluded
        result.prompts = prompts

        # Save outputs
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output = f"probes/universal-{timestamp}"

        progress.update(task, description="Saving outputs...")
        save_universal_probe(result, output)

        # Save to DB
        probe_id = str(uuid.uuid4())
        db_conn.save_universal_probe(probe_id, result, output)
        db_conn.close()

        progress.update(task, description="Done.")

    if output_json:
        json_output = {
            "probe_id": probe_id,
            "heatmap_run_ids": run_ids,
            "model": model_name,
            "layer_range": [layer_start, layer_end],
            "n_samples": result.n_samples,
            "n_vulnerable": result.n_vulnerable,
            "n_robust": result.n_robust,
            "n_excluded": result.n_excluded,
            "auc_roc": result.auc_roc,
            "cohens_d": result.cohens_d,
            "accuracy": result.accuracy,
            "per_domain_auc": result.per_domain_auc,
            "confusion_matrix": result.confusion_matrix,
            "top_features": result.top_features[:10],
            "output_dir": output,
        }
        click.echo(json.dumps(json_output, indent=2, default=str))
        return

    # Rich output
    console.print("\n[bold]Universal Vulnerability Probe[/bold]")
    console.print(
        f"Layers {layer_start}-{layer_end} | "
        f"{result.n_vulnerable} vulnerable, "
        f"{result.n_robust} robust, "
        f"{result.n_excluded} excluded"
    )
    console.print(
        f"Heatmap runs: {len(run_ids)} | "
        f"Domains: {len(set(domains))}\n"
    )

    console.print("[bold]Overall Metrics:[/bold]")
    console.print(f"  AUC-ROC:   {result.auc_roc:.4f}")
    console.print(f"  Cohen's d: {result.cohens_d:.4f}")
    console.print(f"  Accuracy:  {result.accuracy:.1%}")

    cm = result.confusion_matrix
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(f"  TP={cm['tp']}  FP={cm['fp']}")
    console.print(f"  FN={cm['fn']}  TN={cm['tn']}")

    if result.per_domain_auc:
        console.print("\n[bold]Per-Domain AUC:[/bold]")
        table = Table()
        table.add_column("Domain")
        table.add_column("AUC-ROC", justify="right")
        for domain, auc in sorted(result.per_domain_auc.items()):
            style = (
                "green" if auc >= 0.8
                else ("yellow" if auc >= 0.6 else "red")
            )
            table.add_row(domain, f"{auc:.4f}", style=style)
        console.print(table)

    console.print("\n[bold]Top Features:[/bold]")
    feat_table = Table()
    feat_table.add_column("Feature Index", justify="right")
    feat_table.add_column("Importance", justify="right")
    for feat in result.top_features[:10]:
        feat_table.add_row(
            str(feat["feature_index"]),
            f"{feat['importance']:.4f}",
        )
    console.print(feat_table)

    console.print(f"\n[green]Probe saved to {output}/[/green]")


def _probe_universal_remote(
    remote_url, prompts, layer_start, layer_end,
    seed, progress, task,
):
    """Extract activations via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get(
        "device_name", health.get("device", "unknown"),
    )
    err_console.print(f"GPU: {device_name} via {remote_url}")

    progress.update(
        task,
        description=(
            f"Extracting activations remotely "
            f"(layers {layer_start}-{layer_end}, "
            f"{len(prompts)} prompts)..."
        ),
    )

    rows = []
    for event in worker.extract_activations_stream(
        prompts, layer_start, layer_end, seed=seed,
    ):
        etype = event.get("type")
        if etype == "progress":
            idx = event["index"]
            total = event["total"]
            progress.update(
                task,
                description=(
                    f"Extracting {idx + 1}/{total}..."
                ),
            )
        elif etype == "activations":
            raw = base64.b64decode(event["data"])
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            rows.append(arr)

    return np.stack(rows)


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--dataset-builtin",
    default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    type=click.Path(exists=True),
    help="Custom JSONL dataset path.",
)
@click.option("--prompt", default=None, help="Single prompt to analyze.")
@click.option("--answer", default=None, help="Expected answer for --prompt.")
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
@click.option(
    "--device", default="cpu", help="Device: cpu, cuda, directml, auto."
)
@click.option(
    "--threshold", default=0.1, type=float,
    help="Min margin threshold for vulnerability (default: 0.1).",
)
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def commitment(
    model,
    db,
    dataset_builtin,
    dataset_path,
    prompt,
    answer,
    remote,
    device,
    threshold,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Measure how strongly the model commits to the correct answer at each layer."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.commitment import (
        CommitmentRun,
        commitment_run_to_dict,
        generate_commitment_html,
        validate_against_heatmap,
    )

    # Validate inputs
    has_dataset = dataset_path is not None or dataset_builtin is not None
    has_single = prompt is not None
    if not has_dataset and not has_single:
        raise click.UsageError(
            "Must provide --dataset/--dataset-builtin or --prompt/--answer."
        )
    if has_dataset and has_single:
        raise click.UsageError(
            "Cannot provide both dataset and --prompt/--answer."
        )
    if has_single and answer is None:
        raise click.UsageError("--answer is required with --prompt.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Build prompt list
    if has_single:
        prompts = [{"prompt": prompt, "answer": answer}]
        dataset_name = "single"
    else:
        from neurotrace.datasets import get_builtin_dataset, load_dataset

        if dataset_builtin is not None:
            dataset = get_builtin_dataset(dataset_builtin)
            dataset_name = dataset_builtin
        else:
            dataset = load_dataset(dataset_path)
            dataset_name = dataset_path
        prompts = [{"prompt": d["prompt"], "answer": d["answer"]} for d in dataset]

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    if remote is not None:
        results = _commitment_remote(
            remote, prompts, seed, threshold, model,
        )
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        health = worker.health()
        model_name = health["model"]
        num_layers = health["num_layers"]
    else:
        from neurotrace.commitment import run_commitment_local
        from neurotrace.models import get_architecture, load_model

        device = _resolve_device(device)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=err_console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter)
            arch = get_architecture(model_obj.config.model_type)
            num_layers = len(arch.get_layers(model_obj))
            model_name = model

            progress.update(
                task,
                description=(
                    f"Commitment scan: {len(prompts)} prompts x {num_layers} layers"
                ),
                total=len(prompts),
            )

            def progress_cb(prompt_idx, n_prompts):
                progress.update(
                    task,
                    completed=prompt_idx,
                    description=(
                        f"Prompt {prompt_idx + 1}/{n_prompts}"
                    ),
                )

            results = run_commitment_local(
                model_obj,
                tokenizer,
                arch,
                prompts,
                seed=seed,
                threshold=threshold,
                progress_callback=progress_cb,
            )
            progress.update(task, description="Done.", completed=len(prompts))

    # Build run object
    run = CommitmentRun(
        run_id=run_id,
        dataset_name=dataset_name,
        model_name=model_name,
        num_layers=num_layers,
        num_prompts=len(prompts),
        threshold=threshold,
        results=results,
        created_at=created_at,
    )

    # Cross-validate against heatmap if available
    db_conn = TraceDB(db)
    try:
        heatmap_runs = db_conn.get_all_heatmap_runs()
        matching = [
            r for r in heatmap_runs
            if r["dataset_name"] == dataset_name
        ]
        if matching:
            run.validation = validate_against_heatmap(
                results, matching[0]["cells"]
            )
    except Exception:
        pass  # No heatmap data, skip validation

    # Save to DB
    run_dict = commitment_run_to_dict(run)
    db_conn.write_commitment_run(
        run_id=run_id,
        dataset_name=dataset_name,
        model_name=model_name,
        n_prompts=len(prompts),
        n_vulnerable=run_dict["n_vulnerable"],
        n_robust=run_dict["n_robust"],
        threshold=threshold,
        avg_commitment_score=run_dict["avg_commitment_score"],
    )
    for r in results:
        db_conn.write_commitment_result(
            run_id=run_id,
            prompt=r.prompt,
            answer=r.answer,
            peak_prob=r.peak_prob,
            peak_layer=r.peak_layer,
            final_prob=r.final_prob,
            min_margin=r.min_margin,
            margin_at_final=r.margin_at_final,
            competitor_token=r.competitor_token,
            competitor_peak=r.competitor_peak,
            crossover_layer=r.crossover_layer,
            vulnerable=r.vulnerable,
            trajectory=json.dumps(r.trajectory),
            margin_trajectory=json.dumps(r.margin_trajectory),
            competitor_trajectory=json.dumps(
                r.competitor_trajectory
            ),
        )
    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        html = generate_commitment_html(run)
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    if output_json:
        click.echo(json.dumps(run_dict, indent=2))
        return

    # Terminal output
    n_vulnerable = run_dict["n_vulnerable"]
    n_robust = run_dict["n_robust"]
    console.print(
        f"\n[bold]Commitment Analysis:[/bold] {dataset_name} "
        f"({len(prompts)} prompts)"
    )
    console.print(
        f"Model: {model_name} | Threshold: {threshold}"
    )
    console.print(
        f"[green]{n_robust} robust[/green] | "
        f"[red]{n_vulnerable} vulnerable[/red] | "
        f"Avg commitment: {run_dict['avg_commitment_score']:.3f}\n"
    )

    table = Table()
    table.add_column("Prompt", max_width=50)
    table.add_column("Answer")
    table.add_column("Peak", justify="right")
    table.add_column("MinMargin", justify="right")
    table.add_column("Competitor")
    table.add_column("CompPeak", justify="right")
    table.add_column("Crossover", justify="right")
    table.add_column("Status")

    for r in results:
        style = "red" if r.vulnerable else "green"
        status = "VULN" if r.vulnerable else "OK"
        xover = (
            f"L{r.crossover_layer}"
            if r.crossover_layer is not None
            else "-"
        )
        table.add_row(
            r.prompt[:50],
            r.answer,
            f"{r.peak_prob:.3f}",
            f"{r.min_margin:+.3f}",
            r.competitor_token,
            f"{r.competitor_peak:.3f}",
            xover,
            status,
            style=style,
        )
    console.print(table)

    if run.validation:
        v = run.validation
        console.print(
            f"\n[bold]Heatmap Validation:[/bold] "
            f"TP={v['tp']} FP={v['fp']} FN={v['fn']} TN={v['tn']} "
            f"Accuracy={v['accuracy']:.1%}"
        )
        if v.get("auc_roc") is not None:
            console.print(f"AUC-ROC: {v['auc_roc']:.4f}")


def _commitment_remote(remote_url, prompts, seed, threshold, model_name_hint):
    """Run commitment analysis via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.commitment import run_commitment_remote
    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    model_name = health["model"]

    err_console.print(f"GPU: {device_name} via {remote_url}")

    # Fetch hidden states from remote
    prompt_texts = [p["prompt"] for p in prompts]
    hidden_states_list = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            "Fetching hidden states...", total=len(prompts),
        )

        for event in worker.forward_states_stream(prompt_texts, seed=seed):
            etype = event.get("type")
            if etype == "progress":
                progress.update(
                    task,
                    completed=event.get("current", 0),
                    description=(
                        f"Fetching states {event.get('current', 0)}"
                        f"/{event.get('total', len(prompts))}..."
                    ),
                )
            elif etype == "states":
                raw = base64.b64decode(event["hidden_states"])
                shape = event["shape"]
                arr = np.frombuffer(raw, dtype=np.float32).copy().reshape(shape)
                hidden_states_list.append(arr)

        progress.update(task, description="Loading lm_head locally...")

        # Load just model for lm_head + norm (from HF cache)
        model_obj, tokenizer = load_model(model_name, device="cpu")
        lm_head, final_ln = get_lm_head_and_norm(model_obj)

        progress.update(task, description="Computing commitment scores...")

        lm_head_weight = lm_head.weight.data

        results = run_commitment_remote(
            hidden_states_list,
            lm_head_weight,
            final_ln,
            tokenizer,
            prompts,
            threshold=threshold,
        )
        progress.update(task, description="Done.")

    return results


# ---------------------------------------------------------------------------
# contrast command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--domains", required=True,
    help="Comma-separated built-in dataset names (e.g. 'capitals,math_simple').",
)
@click.option(
    "--layers", "layer_spec", required=True,
    help="Comma-separated layer indices (e.g. '0,2,10,20').",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def contrast(
    model,
    db,
    domains,
    layer_spec,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Compare MLP activation geometry across domains at specific layers."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.contrast import (
        ContrastResult,
        analyze_vulnerable_vs_robust,
        build_domain_summaries,
        compute_domain_centroid,
        contrast_result_to_dict,
        generate_contrast_html,
    )

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Parse domains and layers
    domain_names = [d.strip() for d in domains.split(",")]
    layers = [int(x.strip()) for x in layer_spec.split(",")]

    # Load datasets for each domain
    from neurotrace.datasets import get_builtin_dataset

    domain_prompts: dict[str, list[dict]] = {}
    for name in domain_names:
        dataset = get_builtin_dataset(name)
        domain_prompts[name] = [
            {"prompt": d["prompt"], "answer": d["answer"]}
            for d in dataset
        ]

    # Load commitment data if available (for competitor projections)
    db_conn = TraceDB(db)
    commitment_data: dict[str, dict] = {}
    try:
        for run_info in db_conn.list_commitment_runs():
            results = db_conn.read_commitment_results(run_info["run_id"])
            for r in results:
                commitment_data[r["prompt"]] = {
                    "competitor_token": r["competitor_token"] or "",
                }
    except Exception:
        pass

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    if remote is not None:
        cells, domain_deltas = _contrast_remote(
            remote, domain_prompts, layers, commitment_data, seed, model,
        )
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        health = worker.health()
        model_name = health["model"]
    else:
        from neurotrace.contrast import run_contrast_local
        from neurotrace.models import get_architecture, load_model

        device = _resolve_device(device)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=err_console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter)
            arch = get_architecture(model_obj.config.model_type)
            model_name = model

            total_prompts = sum(len(ps) for ps in domain_prompts.values())
            progress.update(
                task,
                description=(
                    f"Contrast scan: {total_prompts} prompts x "
                    f"{len(layers)} layers"
                ),
                total=total_prompts,
            )

            def progress_cb(current, total):
                progress.update(
                    task, completed=current,
                    description=f"Processing {current + 1}/{total}...",
                )

            cells, domain_deltas = run_contrast_local(
                model_obj, tokenizer, arch,
                domain_prompts, layers,
                commitment_data=commitment_data,
                seed=seed,
                progress_callback=progress_cb,
            )
            progress.update(task, description="Done.", completed=total_prompts)

    # Build domain centroids for cosine similarity
    centroids: dict[tuple[str, int], any] = {}
    for domain_name in domain_names:
        for layer in layers:
            deltas = [
                domain_deltas[k]
                for k in domain_deltas
                if k[0] == domain_name and k[1] == layer
            ]
            if deltas:
                centroids[(domain_name, layer)] = compute_domain_centroid(deltas)

    summaries = build_domain_summaries(cells, domain_names, layers, centroids)

    # Analysis 5: vulnerable vs robust within first domain (with heatmap data)
    vuln_robust = None
    try:
        heatmap_runs = db_conn.get_all_heatmap_runs()
        for domain_name in domain_names:
            matching = [
                r for r in heatmap_runs
                if r["dataset_name"] == domain_name
            ]
            if matching:
                vuln_robust = analyze_vulnerable_vs_robust(
                    cells, matching[0]["cells"],
                    domain_name, layers, domain_deltas,
                )
                break
    except Exception:
        pass

    result = ContrastResult(
        run_id=run_id,
        domains=domain_names,
        layers=layers,
        model_name=model_name,
        cells=cells,
        summaries=summaries,
        vulnerable_vs_robust=vuln_robust,
        created_at=created_at,
    )

    # Save to DB
    db_conn.write_contrast_run(
        run_id=run_id,
        domains=json.dumps(domain_names),
        layers=json.dumps(layers),
        model_name=model_name,
    )
    for c in cells:
        db_conn.write_contrast_result(
            run_id=run_id,
            domain=c.domain,
            layer=c.layer,
            prompt=c.prompt,
            answer=c.answer,
            mlp_delta_norm=c.mlp_delta_norm,
            answer_projection=c.answer_projection,
            competitor_projection=c.competitor_projection,
            competitor_token=c.competitor_token,
        )
    for s in summaries:
        db_conn.write_contrast_summary(
            run_id=run_id,
            domain=s.domain,
            layer=s.layer,
            mean_delta_norm=s.mean_delta_norm,
            std_delta_norm=s.std_delta_norm,
            mean_answer_proj=s.mean_answer_proj,
            mean_competitor_proj=s.mean_competitor_proj,
            cosine_similarities=json.dumps(s.cosine_similarities),
        )
    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        html = generate_contrast_html(result)
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    result_dict = contrast_result_to_dict(result)
    if output_json:
        click.echo(json.dumps(result_dict, indent=2))
        return

    # Terminal output
    console.print(
        f"\n[bold]Domain Geometry Contrast:[/bold] "
        f"{', '.join(domain_names)} | Layers {layers}"
    )
    console.print(f"Model: {model_name}\n")

    for layer in layers:
        layer_sums = [s for s in summaries if s.layer == layer]
        if not layer_sums:
            continue

        console.print(f"[bold]Layer {layer} MLP Delta Analysis:[/bold]")
        table = Table()
        table.add_column("Domain")
        table.add_column("Mean|D|", justify="right")
        table.add_column("Std|D|", justify="right")
        table.add_column("->Answer", justify="right")
        table.add_column("->Competitor", justify="right")
        for d in domain_names:
            table.add_column(f"cos({d[:6]})", justify="right")

        for s in layer_sums:
            cos_vals = [
                f"{s.cosine_similarities.get(d, 0.0):.3f}"
                for d in domain_names
            ]
            style = ""
            if s.mean_answer_proj < -0.1:
                style = "red"
            elif s.mean_answer_proj > 0.1:
                style = "green"

            table.add_row(
                s.domain,
                f"{s.mean_delta_norm:.2f}",
                f"{s.std_delta_norm:.2f}",
                f"{s.mean_answer_proj:+.3f}",
                f"{s.mean_competitor_proj:+.3f}",
                *cos_vals,
                style=style,
            )
        console.print(table)
        console.print()

    # Print natural language summary for first two domains at first layer
    if len(summaries) >= 2:
        s0 = summaries[0]
        s1 = summaries[1]
        if s0.mean_delta_norm > 0 and s1.mean_delta_norm > 0:
            ratio = s0.mean_delta_norm / s1.mean_delta_norm
            console.print(
                f"[dim]Layer {s0.layer}: MLP pushes "
                f"{ratio:.1f}x harder on {s0.domain} "
                f"(|D|={s0.mean_delta_norm:.2f}) than "
                f"{s1.domain} (|D|={s1.mean_delta_norm:.2f}).[/dim]"
            )

    if vuln_robust and vuln_robust.get("status") == "ok":
        for layer_str, info in vuln_robust.get("layers", {}).items():
            cos_sim = info["cosine_similarity"]
            interp = (
                "confirming weak immunity hypothesis"
                if cos_sim > 0.8
                else "suggesting targeted interference"
            )
            console.print(
                f"[dim]L{layer_str} within-domain: vulnerable and robust "
                f"receive similar MLP push (cos={cos_sim:.2f}), "
                f"{interp}.[/dim]"
            )


def _contrast_remote(
    remote_url, domain_prompts, layers, commitment_data, seed, model_name_hint,
):
    """Run contrast analysis via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.contrast import run_contrast_remote
    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    model_name = health["model"]

    err_console.print(f"GPU: {device_name} via {remote_url}")

    # Flatten all prompts in domain order
    all_prompts = []
    for domain, prompts in domain_prompts.items():
        for p in prompts:
            all_prompts.append(p["prompt"])

    mlp_data_by_prompt: list[dict] = [{} for _ in all_prompts]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            "Fetching MLP deltas...", total=len(all_prompts),
        )

        for event in worker.forward_mlp_deltas_stream(
            all_prompts, layers=layers, seed=seed,
        ):
            etype = event.get("type")
            if etype == "progress":
                progress.update(
                    task,
                    completed=event.get("current", 0),
                    description=(
                        f"Fetching MLP deltas "
                        f"{event.get('current', 0)}"
                        f"/{event.get('total', len(all_prompts))}..."
                    ),
                )
            elif etype == "deltas":
                pidx = event["prompt_idx"]
                layer = event["layer"]
                dtype = np.float16 if event.get("dtype") == "float16" else np.float32
                mlp_in = np.frombuffer(
                    base64.b64decode(event["mlp_input"]),
                    dtype=dtype,
                ).astype(np.float32).copy()
                mlp_out = np.frombuffer(
                    base64.b64decode(event["mlp_output"]),
                    dtype=dtype,
                ).astype(np.float32).copy()

                mlp_data_by_prompt[pidx][str(layer)] = {
                    "mlp_input": mlp_in,
                    "mlp_output": mlp_out,
                }

        progress.update(task, description="Loading lm_head locally...")

        model_obj, tokenizer = load_model(model_name, device="cpu")
        lm_head, final_ln = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        progress.update(task, description="Computing contrast metrics...")

        cells, domain_deltas = run_contrast_remote(
            mlp_data_by_prompt,
            lm_head_weight,
            tokenizer,
            domain_prompts,
            layers,
            commitment_data=commitment_data,
        )
        progress.update(task, description="Done.")

    return cells, domain_deltas


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--prompt", "prompt_text", default=None, help="Single prompt to analyze.")
@click.option(
    "--dataset-builtin", default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset", "dataset_path", default=None,
    help="Custom JSONL dataset path.",
)
@click.option("--layer", required=True, type=int, help="Layer index to analyze.")
@click.option(
    "--target", required=True,
    help="Target: 'answer', 'competitor', 'both', or a literal token.",
)
@click.option(
    "--method", default="gradient", type=click.Choice(["gradient", "ablation"]),
    help="Attribution method.",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def attribute(
    model,
    db,
    prompt_text,
    dataset_builtin,
    dataset_path,
    layer,
    target,
    method,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Compute which input tokens most influence MLP behavior at a layer."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.attribute import (
        AttributionRun,
        aggregate_attributions,
        attribution_run_to_dict,
        generate_attribution_html,
    )

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if prompt_text is None and dataset_builtin is None and dataset_path is None:
        raise click.UsageError(
            "Must provide --prompt, --dataset-builtin, or --dataset."
        )

    # Build prompt list
    prompts: list[dict] = []
    dataset_name = None

    if prompt_text is not None:
        prompts = [{"prompt": prompt_text, "answer": ""}]
    elif dataset_builtin is not None:
        from neurotrace.datasets import get_builtin_dataset

        prompts = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    elif dataset_path is not None:
        from neurotrace.datasets import load_dataset

        prompts = load_dataset(dataset_path)
        dataset_name = dataset_path

    # Load commitment data for competitor/answer token resolution
    db_conn = TraceDB(db)
    commitment_data: dict[str, dict] = {}
    try:
        for run_info in db_conn.list_commitment_runs():
            results = db_conn.read_commitment_results(run_info["run_id"])
            for r in results:
                commitment_data[r["prompt"]] = {
                    "competitor_token": r["competitor_token"] or "",
                    "answer": r["answer"],
                }
    except Exception:
        pass

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # Determine target directions to run
    target_directions = []
    if target == "both":
        target_directions = ["answer", "competitor"]
    else:
        target_directions = [target]

    all_results = []

    for target_dir in target_directions:
        if remote is not None:
            dir_results = _attribute_remote(
                remote, prompts, layer, target_dir, method,
                commitment_data, seed, model,
            )
            from neurotrace.remote import RemoteWorker

            worker = RemoteWorker(remote)
            health = worker.health()
            model_name = health["model"]
        else:
            from neurotrace.attribute import (
                run_attribution_ablation_local,
                run_attribution_gradient_local,
            )
            from neurotrace.models import get_architecture, load_model

            device = _resolve_device(device)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=err_console,
            ) as progress:
                task = progress.add_task("Loading model...", total=None)
                model_obj, tokenizer = load_model(model, device=device)
                model_obj = _maybe_load_adapter(model_obj, adapter)
                arch = get_architecture(model_obj.config.model_type)
                model_name = model

                progress.update(
                    task,
                    description=f"Attribution: {len(prompts)} prompts @ L{layer}",
                    total=len(prompts),
                )

                dir_results = []
                for i, entry in enumerate(prompts):
                    progress.update(
                        task, completed=i,
                        description=f"Processing {i + 1}/{len(prompts)}...",
                    )

                    p = entry["prompt"]
                    answer = entry.get("answer", "")

                    # Resolve target token
                    target_token, target_token_id = _resolve_target_token(
                        tokenizer, p, answer, target_dir, commitment_data,
                    )
                    if target_token_id is None:
                        continue

                    if method == "gradient":
                        result = run_attribution_gradient_local(
                            model_obj, tokenizer, arch,
                            p, layer, target_token_id, target_token,
                            seed=seed,
                        )
                    else:
                        result = run_attribution_ablation_local(
                            model_obj, tokenizer, arch,
                            p, layer, target_token_id, target_token,
                            seed=seed,
                        )
                    dir_results.append(result)

                progress.update(task, description="Done.", completed=len(prompts))

        all_results.extend(dir_results)

    # Build run
    run = AttributionRun(
        run_id=run_id,
        layer=layer,
        target_direction=target,
        method=method,
        model_name=model_name,
        dataset=dataset_name,
        results=all_results,
        created_at=created_at,
    )

    # Save to DB
    db_conn.write_attribution_run(
        run_id=run_id,
        layer=layer,
        target_direction=target,
        method=method,
        model_name=model_name,
        dataset=dataset_name,
    )
    for r in all_results:
        for ta in r.token_attributions:
            db_conn.write_attribution_result(
                run_id=run_id,
                prompt=r.prompt,
                token_position=ta.position,
                token_text=ta.token,
                attribution_score=ta.attribution,
                target_token=r.target_token,
                target_token_id=r.target_token_id,
            )
    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        html = generate_attribution_html(run)
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    result_dict = attribution_run_to_dict(run)
    if output_json:
        click.echo(json.dumps(result_dict, indent=2))
        return

    # Terminal output
    if len(all_results) == 1:
        r = all_results[0]
        console.print(
            f'\n[bold]Attribution:[/bold] "{r.prompt}" → L{r.layer} MLP '
            f'→ "{r.target_token}"'
        )
        console.print(f"Method: {r.method}\n")

        max_attr = max(ta.attribution for ta in r.token_attributions)
        for ta in r.token_attributions:
            bar_len = int(10 * ta.attribution / max_attr) if max_attr > 0 else 0
            bar = "█" * bar_len + "░" * (10 - bar_len)
            marker = " ← strongest driver" if ta.attribution == max_attr else ""
            style = "bold" if ta.attribution == max_attr else ""
            console.print(
                f"  {ta.token:<12} {bar}  {ta.attribution:.3f}{marker}",
                style=style,
            )
    elif len(all_results) > 1:
        console.print(
            f"\n[bold]Token Attribution Summary:[/bold] "
            f"{dataset_name or 'custom'} @ L{layer} → {target}"
        )
        console.print(f"Method: {method} | {len(all_results)} prompts\n")

        agg = aggregate_attributions(all_results)
        sorted_tokens = sorted(
            agg.items(), key=lambda x: x[1]["avg_attribution"], reverse=True,
        )

        table = Table()
        table.add_column("Token")
        table.add_column("Avg Attribution", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("Max", justify="right")

        for token, stats in sorted_tokens[:15]:
            table.add_row(
                token.strip() or "▁",
                f"{stats['avg_attribution']:.3f}",
                f"{int(stats['count'])}",
                f"{stats['max_attribution']:.3f}",
            )
        console.print(table)
    else:
        console.print("[yellow]No attribution results computed.[/yellow]")


def _resolve_target_token(
    tokenizer, prompt, answer, target_dir, commitment_data,
):
    """Resolve target direction to (token_text, token_id) pair."""
    if target_dir == "answer":
        if not answer:
            answer = commitment_data.get(prompt, {}).get("answer", "")
        if not answer:
            return None, None
        ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(answer, add_special_tokens=False)
        if ids:
            return answer, ids[0]
        return None, None
    elif target_dir == "competitor":
        comp = commitment_data.get(prompt, {}).get("competitor_token", "")
        if not comp:
            return None, None
        ids = tokenizer.encode(" " + comp, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(comp, add_special_tokens=False)
        if ids:
            return comp, ids[0]
        return None, None
    else:
        # Literal token text
        ids = tokenizer.encode(" " + target_dir, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(target_dir, add_special_tokens=False)
        if ids:
            return target_dir, ids[0]
        return None, None


def _attribute_remote(
    remote_url, prompts, layer, target_dir, method,
    commitment_data, seed, model_name_hint,
):
    """Run attribution analysis via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.attribute import (
        build_attribution_result,
        run_attribution_ablation_remote,
    )
    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    model_name = health["model"]

    err_console.print(f"GPU: {device_name} via {remote_url}")

    # Load model locally for tokenizer and lm_head
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading tokenizer & lm_head...", total=None)
        model_obj, tokenizer = load_model(model_name, device="cpu")
        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        # Resolve target tokens for each prompt
        prompt_texts = []
        target_token_ids = []
        target_tokens = []
        valid_entries = []

        for entry in prompts:
            p = entry["prompt"]
            answer = entry.get("answer", "")
            tok_text, tok_id = _resolve_target_token(
                tokenizer, p, answer, target_dir, commitment_data,
            )
            if tok_id is not None:
                prompt_texts.append(p)
                target_token_ids.append(tok_id)
                target_tokens.append(tok_text)
                valid_entries.append(entry)

        if not prompt_texts:
            progress.update(task, description="No valid prompts.")
            return []

        results = []

        if method == "gradient":
            progress.update(
                task,
                description=(
                    f"Gradient attribution"
                    f" ({len(prompt_texts)} prompts)..."
                ),
                total=len(prompt_texts),
            )

            for event in worker.attribute_gradients_stream(
                prompt_texts, layer, target_token_ids, seed=seed,
            ):
                etype = event.get("type")
                if etype == "progress":
                    progress.update(
                        task, completed=event.get("current", 0),
                        description=(
                            f"Gradient attribution "
                            f"{event.get('current', 0)}"
                            f"/{event.get('total', len(prompt_texts))}..."
                        ),
                    )
                elif etype == "attribution":
                    tokens = [ta["token"] for ta in event["token_attributions"]]
                    attrs = [ta["attribution"] for ta in event["token_attributions"]]
                    results.append(build_attribution_result(
                        prompt=event["prompt"],
                        tokens=tokens,
                        attributions=attrs,
                        layer=layer,
                        target_token=event["target_token"],
                        target_token_id=event["target_token_id"],
                        method="gradient",
                    ))

        else:
            # Ablation method: use forward_mlp_deltas_stream
            # Generate N+1 prompts per original (baseline + one per token)
            progress.update(
                task,
                description="Preparing ablation prompts...",
            )

            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id or 0

            for pi, p in enumerate(prompt_texts):
                progress.update(
                    task,
                    description=f"Ablation {pi + 1}/{len(prompt_texts)}...",
                    total=len(prompt_texts),
                    completed=pi,
                )

                token_ids = tokenizer.encode(p)
                tokens = [tokenizer.decode([tid]) for tid in token_ids]

                # Build ablated prompts
                all_ablation_prompts = [p]  # baseline first
                for i in range(len(token_ids)):
                    ablated = token_ids.copy()
                    ablated[i] = pad_token_id
                    all_ablation_prompts.append(tokenizer.decode(ablated))

                # Fetch MLP deltas for all variants
                mlp_data = [{} for _ in all_ablation_prompts]
                for event in worker.forward_mlp_deltas_stream(
                    all_ablation_prompts, layers=[layer], seed=seed,
                ):
                    etype = event.get("type")
                    if etype == "deltas":
                        pidx_inner = event["prompt_idx"]
                        dtype = (
                            np.float16
                            if event.get("dtype") == "float16"
                            else np.float32
                        )
                        mlp_in = np.frombuffer(
                            base64.b64decode(event["mlp_input"]),
                            dtype=dtype,
                        ).astype(np.float32).copy()
                        mlp_out = np.frombuffer(
                            base64.b64decode(event["mlp_output"]),
                            dtype=dtype,
                        ).astype(np.float32).copy()
                        mlp_data[pidx_inner] = {
                            "mlp_input": mlp_in, "mlp_output": mlp_out,
                        }

                baseline = mlp_data[0]
                ablated = mlp_data[1:]

                if baseline and all(d for d in ablated):
                    result = run_attribution_ablation_remote(
                        baseline, ablated, lm_head_weight,
                        tokens, p, layer,
                        target_token_ids[pi], target_tokens[pi],
                    )
                    results.append(result)

        progress.update(task, description="Done.")

    return results


# ---------------------------------------------------------------------------
# token-trace command
# ---------------------------------------------------------------------------


@cli.command("token-trace")
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--prompt", "prompt_text", default=None, help="Single prompt to trace.")
@click.option("--answer", default=None, help="Expected answer token.")
@click.option("--subject", default=None, help="Subject entity in prompt.")
@click.option(
    "--dataset-builtin", default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset", "dataset_path", default=None,
    type=click.Path(exists=True),
    help="Path to JSONL/JSON dataset file.",
)
@click.option(
    "--layers", "layer_spec", default=None,
    help="Comma-separated layer indices (default: all).",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def token_trace(
    model,
    db,
    prompt_text,
    answer,
    subject,
    dataset_builtin,
    dataset_path,
    layer_spec,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Full position x layer MLP delta analysis (token-trace)."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.token_trace import (
        TokenTraceRun,
        aggregate_by_position_type,
        generate_token_trace_html_batch,
        generate_token_trace_html_single,
        token_trace_run_to_dict,
    )

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if prompt_text is None and dataset_builtin is None and dataset_path is None:
        raise click.UsageError(
            "Must provide --prompt, --dataset-builtin, or --dataset."
        )

    # Build prompt list
    prompts: list[dict] = []
    dataset_name = None

    if prompt_text is not None:
        if answer is None:
            raise click.UsageError("--answer is required with --prompt.")
        prompts = [{"prompt": prompt_text, "answer": answer, "subject": subject}]
    elif dataset_builtin is not None:
        from neurotrace.datasets import get_builtin_dataset

        prompts = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    elif dataset_path is not None:
        from neurotrace.datasets import load_dataset

        prompts = load_dataset(dataset_path)
        dataset_name = dataset_path

    # Load commitment data for competitor projections
    db_conn = TraceDB(db)
    commitment_data: dict[str, dict] = {}
    try:
        for run_info in db_conn.list_commitment_runs():
            results = db_conn.read_commitment_results(run_info["run_id"])
            for r in results:
                commitment_data[r["prompt"]] = {
                    "competitor_token": r["competitor_token"] or "",
                }
    except Exception:
        pass

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    if remote is not None:
        model_name, all_results, layers = _token_trace_remote(
            remote, prompts, layer_spec, commitment_data, seed, model,
        )
    else:
        model_name, all_results, layers = _token_trace_local(
            model, prompts, layer_spec, commitment_data, seed,
            device, adapter,
        )

    # Build run
    run = TokenTraceRun(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        layers=layers,
        prompt_count=len(all_results),
        results=all_results,
        created_at=created_at,
    )

    # Save to DB
    db_conn.write_token_trace_run(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        layers=json.dumps(layers),
        prompt_count=len(all_results),
    )
    for result in all_results:
        for entry in result.entries:
            db_conn.write_token_trace_result(
                run_id=run_id,
                prompt=result.prompt,
                layer=entry.layer,
                token_position=entry.position,
                token_text=entry.token,
                is_subject=entry.is_subject,
                is_last=entry.is_last,
                answer_projection=entry.answer_projection,
                competitor_projection=entry.competitor_projection,
                delta_magnitude=entry.delta_magnitude,
            )
    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        if len(all_results) == 1:
            html = generate_token_trace_html_single(all_results[0], run)
        else:
            from neurotrace.datasets import RELATION_KEYWORDS

            relation_kw = RELATION_KEYWORDS.get(dataset_name or "", set())
            html = generate_token_trace_html_batch(run, relation_kw or None)
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    result_dict = token_trace_run_to_dict(run)
    if output_json:
        click.echo(json.dumps(result_dict, indent=2))
        return

    # Terminal output
    if len(all_results) == 1:
        r = all_results[0]
        console.print(
            f'\n[bold]Token-Trace:[/bold] "{r.prompt}" → {r.answer}\n'
        )

        # Table: tokens x layers
        table = Table(title="Answer projection per position × layer")
        table.add_column("", style="dim")
        tokens_list: list[str] = []
        seen: set[int] = set()
        for e in r.entries:
            if e.position not in seen:
                tokens_list.append(e.token.strip()[:10] or "▁")
                seen.add(e.position)
        for tok in tokens_list:
            table.add_column(tok, justify="right")

        for layer in layers:
            row = [f"L{layer}"]
            for pos in range(len(tokens_list)):
                match = [
                    e for e in r.entries
                    if e.layer == layer and e.position == pos
                ]
                if match:
                    val = match[0].answer_projection
                    row.append(f"{val:+.2f}")
                else:
                    row.append("—")
            table.add_row(*row)

        console.print(table)

        peak = max(r.entries, key=lambda e: e.answer_projection)
        trough = min(r.entries, key=lambda e: e.answer_projection)
        console.print(
            f'\nPeak: "{peak.token.strip()}" @ L{peak.layer} '
            f'({peak.answer_projection:+.4f})'
        )
        console.print(
            f'Min:  "{trough.token.strip()}" @ L{trough.layer} '
            f'({trough.answer_projection:+.4f})'
        )
    elif len(all_results) > 1:
        console.print(
            f"\n[bold]Token-Trace Summary:[/bold] "
            f"{dataset_name or 'custom'} ({len(all_results)} prompts) × "
            f"{len(layers)} layers\n"
        )

        from neurotrace.datasets import RELATION_KEYWORDS

        relation_kw = RELATION_KEYWORDS.get(dataset_name or "", set())
        agg = aggregate_by_position_type(all_results, layers, relation_kw or None)

        import numpy as np

        table = Table(title="Position-type analysis (aggregated)")
        table.add_column("Position")
        table.add_column("Avg Proj", justify="right")
        table.add_column("Peak Layer", justify="right")
        table.add_column("Role")

        for pos_type in ["subject", "relation", "other", "last"]:
            if pos_type not in agg:
                continue
            layer_data = agg[pos_type]
            means = [
                layer_data[li]["mean"]
                for li in layers if li in layer_data
            ]
            if not means:
                continue
            avg = float(np.mean(means))
            peak_l = max(
                (li for li in layers if li in layer_data),
                key=lambda li: layer_data[li]["mean"],
            )
            role = "Negligible"
            if avg > 0.01:
                role = "Positive (factual signal)"
            elif avg < -0.01:
                role = "Suppression"

            table.add_row(
                pos_type,
                f"{avg:+.4f}",
                f"L{peak_l}",
                role,
            )

        console.print(table)
    else:
        console.print("[yellow]No token-trace results computed.[/yellow]")


def _token_trace_local(
    model_name, prompts, layer_spec, commitment_data, seed,
    device_str, adapter,
):
    """Run token-trace locally."""
    from neurotrace.models import get_architecture, get_lm_head_and_norm, load_model
    from neurotrace.token_trace import run_token_trace_local

    device_str = _resolve_device(device_str)
    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        model_obj, tokenizer = load_model(model_name, device=device_str)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        arch = get_architecture(model_obj.config.model_type)

        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        num_layers = len(arch.get_layers(model_obj))
        if layer_spec:
            layers = [int(x.strip()) for x in layer_spec.split(",")]
        else:
            layers = list(range(num_layers))

        progress.update(
            task,
            description=f"Token-trace: {len(prompts)} prompts × {len(layers)} layers",
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Processing {i + 1}/{len(prompts)}...",
            )
            result = run_token_trace_local(
                model_obj, tokenizer, arch,
                prompt=entry["prompt"],
                answer=entry["answer"],
                subject=entry.get("subject"),
                layers=layers,
                lm_head_weight=lm_head_weight,
                commitment_data=commitment_data,
                seed=seed,
            )
            all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, all_results, layers


@cli.command("attention-trace")
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--prompt", "prompt_text", default=None, help="Single prompt to trace.")
@click.option("--answer", default=None, help="Expected answer token.")
@click.option(
    "--dataset-builtin", default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset", "dataset_path", default=None,
    type=click.Path(exists=True),
    help="Path to JSONL/JSON dataset file.",
)
@click.option(
    "--layers", "layer_spec", default=None,
    help="Comma-separated layer indices (default: all).",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def attention_trace(
    model,
    db,
    prompt_text,
    answer,
    dataset_builtin,
    dataset_path,
    layer_spec,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Decompose per-head attention contributions to the answer direction."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.attention_trace import (
        AttentionTraceRun,
        attention_trace_run_to_dict,
        generate_attention_trace_html_batch,
        generate_attention_trace_html_single,
    )

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if prompt_text is None and dataset_builtin is None and dataset_path is None:
        raise click.UsageError(
            "Must provide --prompt, --dataset-builtin, or --dataset."
        )

    # Build prompt list
    prompts: list[dict] = []
    dataset_name = None

    if prompt_text is not None:
        if answer is None:
            raise click.UsageError("--answer is required with --prompt.")
        prompts = [{"prompt": prompt_text, "answer": answer}]
    elif dataset_builtin is not None:
        from neurotrace.datasets import get_builtin_dataset

        prompts = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    elif dataset_path is not None:
        from neurotrace.datasets import load_dataset

        prompts = load_dataset(dataset_path)
        dataset_name = dataset_path

    db_conn = TraceDB(db)
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    if remote is not None:
        model_name, all_results, layers = _attention_trace_remote(
            remote, prompts, layer_spec, seed, model,
        )
    else:
        model_name, all_results, layers = _attention_trace_local(
            model, prompts, layer_spec, seed, device, adapter,
        )

    # Build run
    run = AttentionTraceRun(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        layers=layers,
        prompt_count=len(all_results),
        results=all_results,
        created_at=created_at,
    )

    # Save to DB
    db_conn.write_attention_trace_run(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        layers=json.dumps(layers),
        prompt_count=len(all_results),
    )
    for result in all_results:
        for entry in result.entries:
            db_conn.write_attention_trace_result(
                run_id=run_id,
                prompt=result.prompt,
                layer=entry.layer,
                head_idx=entry.head_idx,
                answer_projection=entry.answer_projection,
                magnitude=entry.magnitude,
            )

    # Look up MLP total from token-trace if available (single prompt only)
    mlp_total = None
    if len(all_results) == 1:
        try:
            rows = db_conn._conn.execute(
                "SELECT SUM(answer_projection) FROM token_trace_results"
                " WHERE prompt = ? AND is_last = true",
                [all_results[0].prompt],
            ).fetchone()
            if rows and rows[0] is not None:
                mlp_total = float(rows[0])
        except Exception:
            pass

    # Look up vulnerable prompts from heatmap data (batch)
    vulnerable_prompts: set[str] | None = None
    if len(all_results) > 1:
        try:
            heatmap_runs = db_conn.get_all_heatmap_runs()
            if heatmap_runs:
                import json as _json

                cells = _json.loads(heatmap_runs[0]["cells"])
                vuln = set()
                for cell in cells:
                    if cell.get("flip_direction") == "broke":
                        vuln.add(cell["prompt"])
                if vuln:
                    vulnerable_prompts = vuln
        except Exception:
            pass

    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        if len(all_results) == 1:
            html = generate_attention_trace_html_single(
                all_results[0], run, mlp_total=mlp_total,
            )
        else:
            html = generate_attention_trace_html_batch(
                run, vulnerable_prompts=vulnerable_prompts,
            )
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    result_dict = attention_trace_run_to_dict(run)
    if output_json:
        click.echo(json.dumps(result_dict, indent=2))
        return

    # Terminal output
    if len(all_results) == 1:
        r = all_results[0]
        console.print(
            f'\n[bold]Attention-Trace:[/bold] "{r.prompt}" → {r.answer}\n'
        )

        # Top 10 heads table
        table = Table(title="Top attention head contributions")
        table.add_column("Head")
        table.add_column("Answer Proj", justify="right")
        table.add_column("Magnitude", justify="right")

        sorted_entries = sorted(
            r.entries, key=lambda e: abs(e.answer_projection), reverse=True,
        )
        for e in sorted_entries[:10]:
            proj_style = "green" if e.answer_projection > 0 else "red"
            table.add_row(
                f"L{e.layer}.H{e.head_idx}",
                f"[{proj_style}]{e.answer_projection:+.2f}[/{proj_style}]",
                f"{e.magnitude:.2f}",
            )
        console.print(table)

        attn_total = sum(e.answer_projection for e in r.entries)
        summary = f"\nTotal attention: {attn_total:+.2f}"
        if mlp_total is not None:
            net = attn_total + mlp_total
            summary += (
                f" | Total MLP (from token-trace): {mlp_total:+.2f}"
                f" | Net: {net:+.2f}"
            )
        console.print(summary)

    elif len(all_results) > 1:
        import numpy as np

        console.print(
            f"\n[bold]Attention-Trace Summary:[/bold] "
            f"{dataset_name or 'custom'} ({len(all_results)} prompts)\n"
        )

        # Top 10 heads by mean projection
        head_projs: dict[tuple[int, int], list[float]] = {}
        for r in all_results:
            for e in r.entries:
                key = (e.layer, e.head_idx)
                head_projs.setdefault(key, []).append(e.answer_projection)

        head_means = [
            ((layer, head), float(np.mean(vals)))
            for (layer, head), vals in head_projs.items()
        ]
        head_means.sort(key=lambda x: abs(x[1]), reverse=True)

        table = Table(title="Top 10 heads by mean answer projection")
        table.add_column("Head")
        table.add_column("Mean Proj", justify="right")

        if vulnerable_prompts:
            table.add_column(
                f"Robust ({len(all_results) - len(vulnerable_prompts)})",
                justify="right",
            )
            table.add_column(
                f"Vulnerable ({len(vulnerable_prompts)})",
                justify="right",
            )

        for (layer, head), mean_proj in head_means[:10]:
            proj_style = "green" if mean_proj > 0 else "red"
            row = [
                f"L{layer}.H{head}",
                f"[{proj_style}]{mean_proj:+.4f}[/{proj_style}]",
            ]
            if vulnerable_prompts:
                r_vals = [
                    e.answer_projection
                    for r in all_results if r.prompt not in vulnerable_prompts
                    for e in r.entries
                    if e.layer == layer and e.head_idx == head
                ]
                v_vals = [
                    e.answer_projection
                    for r in all_results if r.prompt in vulnerable_prompts
                    for e in r.entries
                    if e.layer == layer and e.head_idx == head
                ]
                row.append(f"{np.mean(r_vals):+.4f}" if r_vals else "—")
                row.append(f"{np.mean(v_vals):+.4f}" if v_vals else "—")
            table.add_row(*row)
        console.print(table)

        # Aggregate table
        if vulnerable_prompts:
            robust_results = [
                r for r in all_results if r.prompt not in vulnerable_prompts
            ]
            vuln_results = [
                r for r in all_results if r.prompt in vulnerable_prompts
            ]

            agg_table = Table(title="Aggregate")
            agg_table.add_column("Metric")
            agg_table.add_column(
                f"Robust ({len(robust_results)})", justify="right",
            )
            agg_table.add_column(
                f"Vulnerable ({len(vuln_results)})", justify="right",
            )

            r_totals = [
                sum(e.answer_projection for e in r.entries) for r in robust_results
            ] if robust_results else [0.0]
            v_totals = [
                sum(e.answer_projection for e in r.entries) for r in vuln_results
            ]
            agg_table.add_row(
                "Total attn proj",
                f"{np.mean(r_totals):+.2f}",
                f"{np.mean(v_totals):+.2f}",
            )

            r_counts = [
                sum(1 for e in r.entries if e.answer_projection > 0.5)
                for r in robust_results
            ] if robust_results else [0]
            v_counts = [
                sum(1 for e in r.entries if e.answer_projection > 0.5)
                for r in vuln_results
            ]
            agg_table.add_row(
                "Num heads > +0.5",
                f"{np.mean(r_counts):.1f}",
                f"{np.mean(v_counts):.1f}",
            )
            console.print(agg_table)
    else:
        console.print("[yellow]No attention-trace results computed.[/yellow]")


def _attention_trace_local(
    model_name, prompts, layer_spec, seed, device_str, adapter,
):
    """Run attention-trace locally."""
    from neurotrace.attention_trace import run_attention_trace_local
    from neurotrace.models import get_architecture, get_lm_head_and_norm, load_model

    device_str = _resolve_device(device_str)
    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        model_obj, tokenizer = load_model(model_name, device=device_str)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        arch = get_architecture(model_obj.config.model_type)

        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        num_layers = len(arch.get_layers(model_obj))
        if layer_spec:
            layers = [int(x.strip()) for x in layer_spec.split(",")]
        else:
            layers = list(range(num_layers))

        progress.update(
            task,
            description=(
                f"Attention-trace: {len(prompts)} prompts"
                f" × {len(layers)} layers"
            ),
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Processing {i + 1}/{len(prompts)}...",
            )
            result = run_attention_trace_local(
                model_obj, tokenizer, arch,
                prompt=entry["prompt"],
                answer=entry["answer"],
                layers=layers,
                lm_head_weight=lm_head_weight,
                seed=seed,
            )
            all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, all_results, layers


def _attention_trace_remote(
    remote_url, prompts, layer_spec, seed, model_name_hint,
):
    """Run attention-trace via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.attention_trace import run_attention_trace_remote
    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    model_name = health["model"]
    num_layers = health["num_layers"]

    err_console.print(f"GPU: {device_name} via {remote_url}")

    if layer_spec:
        layers = [int(x.strip()) for x in layer_spec.split(",")]
    else:
        layers = list(range(num_layers))

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading tokenizer & lm_head...", total=None)
        model_obj, tokenizer = load_model(model_name, device="cpu")
        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        progress.update(
            task,
            description=(
                f"Attention-trace: {len(prompts)} prompts"
                f" × {len(layers)} layers"
            ),
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Attention-trace {i + 1}/{len(prompts)}...",
            )

            prompt_text = entry["prompt"]

            # Fetch per-head contributions from remote
            layer_contributions: dict[int, np.ndarray] = {}
            for event in worker.attention_contributions_stream(
                prompt_text, layers, seed=seed,
            ):
                etype = event.get("type")
                if etype == "layer-contributions":
                    layer_idx = event["layer"]
                    shape = event["shape"]
                    dtype = (
                        np.float16
                        if event.get("dtype") == "float16"
                        else np.float32
                    )
                    arr = np.frombuffer(
                        base64.b64decode(event["contributions"]),
                        dtype=dtype,
                    ).astype(np.float32).reshape(shape).copy()
                    layer_contributions[layer_idx] = arr

            result = run_attention_trace_remote(
                layer_contributions=layer_contributions,
                tokenizer=tokenizer,
                prompt=prompt_text,
                answer=entry["answer"],
                layers=layers,
                lm_head_weight=lm_head_weight,
            )
            all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, all_results, layers


def _token_trace_remote(
    remote_url, prompts, layer_spec, commitment_data, seed, model_name_hint,
):
    """Run token-trace via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import RemoteWorker
    from neurotrace.token_trace import run_token_trace_remote

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    model_name = health["model"]
    num_layers = health["num_layers"]

    err_console.print(f"GPU: {device_name} via {remote_url}")

    if layer_spec:
        layers = [int(x.strip()) for x in layer_spec.split(",")]
    else:
        layers = list(range(num_layers))

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading tokenizer & lm_head...", total=None)
        model_obj, tokenizer = load_model(model_name, device="cpu")
        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        progress.update(
            task,
            description=f"Token-trace: {len(prompts)} prompts × {len(layers)} layers",
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Token-trace {i + 1}/{len(prompts)}...",
            )

            prompt_text = entry["prompt"]

            # Fetch all-position MLP deltas from remote
            all_position_deltas: dict[int, np.ndarray] = {}
            for event in worker.forward_mlp_deltas_all_positions_stream(
                prompt_text, layers, seed=seed,
            ):
                etype = event.get("type")
                if etype == "layer-deltas":
                    layer_idx = event["layer"]
                    shape = event["shape"]
                    dtype = (
                        np.float16
                        if event.get("dtype") == "float16"
                        else np.float32
                    )
                    arr = np.frombuffer(
                        base64.b64decode(event["deltas"]),
                        dtype=dtype,
                    ).astype(np.float32).reshape(shape).copy()
                    all_position_deltas[layer_idx] = arr

            result = run_token_trace_remote(
                all_position_deltas=all_position_deltas,
                tokenizer=tokenizer,
                prompt=prompt_text,
                answer=entry["answer"],
                subject=entry.get("subject"),
                layers=layers,
                lm_head_weight=lm_head_weight,
                commitment_data=commitment_data,
            )
            all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, all_results, layers


# ---------------------------------------------------------------------------
# diagnose command
# ---------------------------------------------------------------------------


@cli.command("diagnose")
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--prompt", "prompt_text", default=None,
    help="Single prompt to diagnose.",
)
@click.option("--answer", default=None, help="Expected answer token.")
@click.option("--subject", default=None, help="Subject entity in prompt.")
@click.option(
    "--dataset-builtin", default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset", "dataset_path", default=None,
    type=click.Path(exists=True),
    help="Path to JSONL/JSON dataset file.",
)
@click.option(
    "--layers", "layer_spec", default=None,
    help="Comma-separated layer indices (default: all).",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def diagnose(
    model,
    db,
    prompt_text,
    answer,
    subject,
    dataset_builtin,
    dataset_path,
    layer_spec,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Combined attention-trace + token-trace circuit diagnosis."""
    import uuid

    from neurotrace.diagnose import (
        DiagnosisResult,
        diagnosis_to_dict,
        generate_diagnose_html_batch,
        generate_diagnose_html_single,
        run_diagnosis,
    )

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if prompt_text is None and dataset_builtin is None and dataset_path is None:
        raise click.UsageError(
            "Must provide --prompt, --dataset-builtin, or --dataset."
        )

    # Build prompt list
    prompts: list[dict] = []
    dataset_name = None

    if prompt_text is not None:
        if answer is None:
            raise click.UsageError("--answer is required with --prompt.")
        prompts = [{"prompt": prompt_text, "answer": answer, "subject": subject}]
    elif dataset_builtin is not None:
        from neurotrace.datasets import get_builtin_dataset

        prompts = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    elif dataset_path is not None:
        from neurotrace.datasets import load_dataset

        prompts = load_dataset(dataset_path)
        dataset_name = dataset_path

    # Load commitment data for competitor projections in token-trace
    db_conn = TraceDB(db)
    commitment_data: dict[str, dict] = {}
    try:
        for run_info in db_conn.list_commitment_runs():
            results = db_conn.read_commitment_results(run_info["run_id"])
            for r in results:
                commitment_data[r["prompt"]] = {
                    "competitor_token": r["competitor_token"] or "",
                }
    except Exception:
        pass

    run_id = str(uuid.uuid4())

    if remote is not None:
        model_name, attn_results, token_results, layers = _diagnose_remote(
            remote, prompts, layer_spec, commitment_data, seed, model,
        )
    else:
        model_name, attn_results, token_results, layers = _diagnose_local(
            model, prompts, layer_spec, commitment_data, seed,
            device, adapter,
        )

    # Run diagnosis on each prompt
    diagnoses: list[DiagnosisResult] = []
    for attn_r, token_r in zip(attn_results, token_results):
        diag = run_diagnosis(attn_r, token_r)
        diagnoses.append(diag)

    # Save to DB
    db_conn.write_diagnosis_run(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        prompt_count=len(diagnoses),
    )
    for diag in diagnoses:
        repair_json_str = None
        if diag.repair is not None:
            repair_json_str = json.dumps({
                "target_heads": diag.repair.target_heads,
                "target_layers": diag.repair.target_layers,
                "suggested_fix": diag.repair.suggested_fix,
                "suggested_modules": diag.repair.suggested_modules,
                "alternative_fix": diag.repair.alternative_fix,
                "alternative_layers": diag.repair.alternative_layers,
                "alternative_modules": diag.repair.alternative_modules,
            })
        db_conn.write_diagnosis_result(
            run_id=run_id,
            prompt=diag.prompt,
            answer=diag.answer,
            total_attention=diag.circuit.total_attention,
            total_mlp=diag.suppression.total_mlp,
            top3_share=diag.circuit.top3_share,
            circuit_type=diag.circuit.circuit_type,
            signal_strength=diag.verdict.signal_strength,
            vulnerability=diag.verdict.vulnerability,
            confidence=diag.verdict.confidence,
            repair_json=repair_json_str,
        )
    db_conn.close()

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        if len(diagnoses) == 1:
            html = generate_diagnose_html_single(
                diagnoses[0], attn_results[0], token_results[0],
                layers, model_name,
            )
        else:
            html = generate_diagnose_html_batch(
                diagnoses, dataset_name, model_name,
            )
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    if output_json:
        for diag in diagnoses:
            click.echo(json.dumps(diagnosis_to_dict(diag)))
        return

    # Terminal output
    if len(diagnoses) == 1:
        diag = diagnoses[0]
        console.print(
            f'\n[bold]Diagnosis:[/bold] "{diag.prompt}" \u2192 {diag.answer}\n'
        )

        # Circuit table
        circuit_table = Table(title="Circuit")
        circuit_table.add_column("Metric")
        circuit_table.add_column("Value")

        active_str = str(len(diag.circuit.active_heads))
        if diag.circuit.active_heads:
            top_entries = ", ".join(
                f"L{h['layer']}.H{h['head']} {h['projection']:+.2f}"
                for h in diag.circuit.active_heads[:3]
            )
            if len(diag.circuit.active_heads) > 3:
                active_str += f" ({top_entries}, ...)"
            else:
                active_str += f" ({top_entries})"
        circuit_table.add_row("Active heads", active_str)
        circuit_table.add_row(
            "Circuit type",
            f"{diag.circuit.circuit_type.capitalize()}"
            f" (top3: {diag.circuit.top3_share:.0%})",
        )
        circuit_table.add_row(
            "Total attention", f"{diag.circuit.total_attention:+.2f}",
        )
        circuit_table.add_row(
            "Signal strength", diag.verdict.signal_strength.capitalize(),
        )
        console.print(circuit_table)

        # Suppression table
        supp_table = Table(title="Suppression")
        supp_table.add_column("Metric")
        supp_table.add_column("Value")
        supp_table.add_row("Total MLP", f"{diag.suppression.total_mlp:+.2f}")
        supp_table.add_row("Subject MLP", f"{diag.suppression.subject_mlp:+.2f}")
        if diag.suppression.worst_layer is not None:
            wl = diag.suppression.worst_layer
            wv = diag.suppression.worst_layer_value
            worst_str = f"L{wl} ({wv:+.2f})"
        else:
            worst_str = "\u2014"
        supp_table.add_row("Worst layer", worst_str)
        supp_table.add_row(
            "Suppression ratio", f"{diag.suppression.suppression_ratio:.1f}x",
        )
        console.print(supp_table)

        # Verdict table
        v_style = {
            "robust": "green", "moderate": "yellow",
            "vulnerable": "red", "absent": "red",
        }.get(diag.verdict.vulnerability, "white")
        verdict_table = Table(title="Verdict")
        verdict_table.add_column("")
        verdict_table.add_column("")
        verdict_table.add_row(
            "Vulnerability",
            f"[{v_style}]{diag.verdict.vulnerability.upper()}[/{v_style}]",
        )
        verdict_table.add_row("Reason", diag.verdict.reason)
        verdict_table.add_row("Confidence", diag.verdict.confidence.capitalize())
        console.print(verdict_table)

        # Repair prescription
        if diag.repair:
            heads_str = ", ".join(
                f"L{h['layer']}.H{h['head']}"
                for h in diag.repair.target_heads
            )
            layers_str = ", ".join(
                f"L{ly}" for ly in diag.repair.target_layers
            )
            console.print("\n[bold]Repair Prescription[/bold]")
            console.print(
                f"  Target: {layers_str} attention "
                f"({', '.join(diag.repair.suggested_modules)})"
            )
            console.print(f"  Heads to strengthen: {heads_str}")
            if diag.repair.alternative_layers:
                alt_layers_str = ", ".join(
                    f"L{ly}" for ly in diag.repair.alternative_layers
                )
                console.print(
                    f"  Alternative: {alt_layers_str} MLP "
                    f"({', '.join(diag.repair.alternative_modules)})"
                )

    elif len(diagnoses) > 1:
        console.print(
            f"\n[bold]Diagnosis Summary:[/bold] "
            f"{dataset_name or 'custom'} ({len(diagnoses)} prompts)\n"
        )

        # Verdict summary
        counts: dict[str, int] = {}
        for d in diagnoses:
            counts[d.verdict.vulnerability] = counts.get(d.verdict.vulnerability, 0) + 1

        summary_table = Table(title="Verdict Summary")
        summary_table.add_column("Verdict")
        summary_table.add_column("Count", justify="right")
        for v in ["robust", "moderate", "vulnerable", "absent"]:
            if v in counts:
                v_style = {
                    "robust": "green", "moderate": "yellow",
                    "vulnerable": "red", "absent": "red",
                }.get(v, "white")
                summary_table.add_row(
                    f"[{v_style}]{v.capitalize()}[/{v_style}]",
                    str(counts[v]),
                )
        console.print(summary_table)

        # Most common repair targets
        layer_counts: dict[int, int] = {}
        for d in diagnoses:
            if d.repair:
                for ly in d.repair.target_layers:
                    layer_counts[ly] = layer_counts.get(ly, 0) + 1
        if layer_counts:
            top_layers = sorted(
                layer_counts.items(), key=lambda x: x[1], reverse=True,
            )[:5]
            console.print("\nMost common repair targets:")
            for layer, count in top_layers:
                console.print(f"  L{layer} attention: {count} prompts")

        # Per-prompt table
        detail_table = Table()
        detail_table.add_column("Prompt")
        detail_table.add_column("Answer")
        detail_table.add_column("Total Attn", justify="right")
        detail_table.add_column("Total MLP", justify="right")
        detail_table.add_column("Circuit")
        detail_table.add_column("Verdict")

        for d in diagnoses:
            v_style = {
                "robust": "green", "moderate": "yellow",
                "vulnerable": "red", "absent": "red",
            }.get(d.verdict.vulnerability, "white")
            detail_table.add_row(
                d.prompt[:50],
                d.answer,
                f"{d.circuit.total_attention:+.2f}",
                f"{d.suppression.total_mlp:+.2f}",
                d.circuit.circuit_type,
                f"[{v_style}]{d.verdict.vulnerability.capitalize()}[/{v_style}]",
            )
        console.print(detail_table)
    else:
        console.print("[yellow]No diagnosis results computed.[/yellow]")


def _diagnose_local(
    model_name, prompts, layer_spec, commitment_data, seed,
    device_str, adapter,
):
    """Run diagnose locally -- both attention-trace and token-trace."""
    from neurotrace.attention_trace import run_attention_trace_local
    from neurotrace.models import get_architecture, get_lm_head_and_norm, load_model
    from neurotrace.token_trace import run_token_trace_local

    device_str = _resolve_device(device_str)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        model_obj, tokenizer = load_model(model_name, device=device_str)
        model_obj = _maybe_load_adapter(model_obj, adapter)
        arch = get_architecture(model_obj.config.model_type)

        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        num_layers = len(arch.get_layers(model_obj))
        if layer_spec:
            layers = [int(x.strip()) for x in layer_spec.split(",")]
        else:
            layers = list(range(num_layers))

        progress.update(
            task,
            description=f"Diagnose: {len(prompts)} prompts x {len(layers)} layers",
            total=len(prompts),
        )

        attn_results = []
        token_results = []

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Diagnosing {i + 1}/{len(prompts)}...",
            )

            attn_r = run_attention_trace_local(
                model_obj, tokenizer, arch,
                prompt=entry["prompt"],
                answer=entry["answer"],
                layers=layers,
                lm_head_weight=lm_head_weight,
                seed=seed,
            )
            attn_results.append(attn_r)

            token_r = run_token_trace_local(
                model_obj, tokenizer, arch,
                prompt=entry["prompt"],
                answer=entry["answer"],
                subject=entry.get("subject"),
                layers=layers,
                lm_head_weight=lm_head_weight,
                commitment_data=commitment_data,
                seed=seed,
            )
            token_results.append(token_r)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, attn_results, token_results, layers


def _diagnose_remote(
    remote_url, prompts, layer_spec, commitment_data, seed, model_name_hint,
):
    """Run diagnose via remote GPU worker."""
    import base64

    import numpy as np

    from neurotrace.attention_trace import run_attention_trace_remote
    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import RemoteWorker
    from neurotrace.token_trace import run_token_trace_remote

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    model_name = health["model"]
    num_layers = health["num_layers"]

    err_console.print(f"GPU: {device_name} via {remote_url}")

    if layer_spec:
        layers = [int(x.strip()) for x in layer_spec.split(",")]
    else:
        layers = list(range(num_layers))

    attn_results = []
    token_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading tokenizer & lm_head...", total=None)
        model_obj, tokenizer = load_model(model_name, device="cpu")
        lm_head, _ = get_lm_head_and_norm(model_obj)
        lm_head_weight = lm_head.weight.data.cpu().float().numpy()

        progress.update(
            task,
            description=f"Diagnose: {len(prompts)} prompts x {len(layers)} layers",
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Diagnosing {i + 1}/{len(prompts)}...",
            )

            prompt_text = entry["prompt"]

            # Attention-trace: fetch per-head contributions
            layer_contributions: dict[int, np.ndarray] = {}
            for event in worker.attention_contributions_stream(
                prompt_text, layers, seed=seed,
            ):
                etype = event.get("type")
                if etype == "layer-contributions":
                    layer_idx = event["layer"]
                    shape = event["shape"]
                    dtype = (
                        np.float16
                        if event.get("dtype") == "float16"
                        else np.float32
                    )
                    arr = np.frombuffer(
                        base64.b64decode(event["contributions"]),
                        dtype=dtype,
                    ).astype(np.float32).reshape(shape).copy()
                    layer_contributions[layer_idx] = arr

            attn_r = run_attention_trace_remote(
                layer_contributions=layer_contributions,
                tokenizer=tokenizer,
                prompt=prompt_text,
                answer=entry["answer"],
                layers=layers,
                lm_head_weight=lm_head_weight,
            )
            attn_results.append(attn_r)

            # Token-trace: fetch all-position MLP deltas
            all_position_deltas: dict[int, np.ndarray] = {}
            for event in worker.forward_mlp_deltas_all_positions_stream(
                prompt_text, layers, seed=seed,
            ):
                etype = event.get("type")
                if etype == "layer-deltas":
                    layer_idx = event["layer"]
                    shape = event["shape"]
                    dtype = (
                        np.float16
                        if event.get("dtype") == "float16"
                        else np.float32
                    )
                    arr = np.frombuffer(
                        base64.b64decode(event["deltas"]),
                        dtype=dtype,
                    ).astype(np.float32).reshape(shape).copy()
                    all_position_deltas[layer_idx] = arr

            token_r = run_token_trace_remote(
                all_position_deltas=all_position_deltas,
                tokenizer=tokenizer,
                prompt=prompt_text,
                answer=entry["answer"],
                subject=entry.get("subject"),
                layers=layers,
                lm_head_weight=lm_head_weight,
                commitment_data=commitment_data,
            )
            token_results.append(token_r)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, attn_results, token_results, layers


# ---------------------------------------------------------------------------
# decompose command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--dataset-builtin",
    default=None,
    help="Built-in dataset name (e.g. 'capitals').",
)
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    type=click.Path(exists=True),
    help="Custom JSONL dataset path.",
)
@click.option("--prompt", default=None, help="Single prompt to decompose.")
@click.option("--answer", default=None, help="Expected answer for --prompt.")
@click.option(
    "--competitors", default=None,
    help="Comma-separated competitor tokens (auto-detected if omitted).",
)
@click.option(
    "--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877)."
)
@click.option(
    "--device", default="cpu", help="Device: cpu, cuda, directml, auto."
)
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def decompose(
    model,
    db,
    dataset_builtin,
    dataset_path,
    prompt,
    answer,
    competitors,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """Logit Prism decomposition — exact additive logit attribution per component."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.decompose import (
        DecomposeRun,
        decompose_run_to_dict,
        generate_decompose_html,
        run_decompose_local,
    )

    # Validate inputs
    has_dataset = dataset_path is not None or dataset_builtin is not None
    has_single = prompt is not None
    if not has_dataset and not has_single:
        raise click.UsageError(
            "Must provide --dataset/--dataset-builtin or --prompt/--answer."
        )
    if has_dataset and has_single:
        raise click.UsageError(
            "Cannot provide both dataset and --prompt/--answer."
        )
    if has_single and answer is None:
        raise click.UsageError("--answer is required with --prompt.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Parse competitors
    competitor_list = None
    if competitors:
        competitor_list = [c.strip() for c in competitors.split(",") if c.strip()]

    # Build prompt list
    if has_single:
        prompts = [{"prompt": prompt, "answer": answer}]
        dataset_name = "single"
    else:
        from neurotrace.datasets import get_builtin_dataset, load_dataset

        if dataset_builtin is not None:
            dataset = get_builtin_dataset(dataset_builtin)
            dataset_name = dataset_builtin
        else:
            dataset = load_dataset(dataset_path)
            dataset_name = dataset_path
        prompts = [{"prompt": d["prompt"], "answer": d["answer"]} for d in dataset]

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    all_results = []

    if remote is not None:
        all_results = _decompose_remote(
            remote, prompts, competitor_list, seed,
        )
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        health = worker.health()
        model_name = health["model"]
    else:
        from neurotrace.models import load_model

        device = _resolve_device(device)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=err_console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter)
            model_name = model

            progress.update(
                task,
                description=f"Decomposing {len(prompts)} prompts...",
                total=len(prompts),
            )

            for p_idx, entry in enumerate(prompts):
                progress.update(
                    task,
                    completed=p_idx,
                    description=f"Prompt {p_idx + 1}/{len(prompts)}",
                )
                results = run_decompose_local(
                    model_obj,
                    tokenizer,
                    entry["prompt"],
                    entry["answer"],
                    competitors=competitor_list,
                    seed=seed,
                )
                all_results.extend(results)

            progress.update(task, description="Done.", completed=len(prompts))

    # Determine unique prompt count
    unique_prompts = len({r.prompt for r in all_results}) if all_results else 0

    run = DecomposeRun(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        num_prompts=unique_prompts or len(prompts),
        results=all_results,
        created_at=created_at,
    )

    # Save to DB
    db_conn = TraceDB(db)
    db_conn.write_decompose_run(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        prompt_count=run.num_prompts,
    )
    for r in all_results:
        db_conn.write_decompose_result(
            run_id=run_id,
            prompt=r.prompt,
            answer=r.answer,
            competitor=r.competitor,
            answer_logit=r.answer_logit,
            competitor_logit=r.competitor_logit,
            margin=r.margin,
            embedding_margin=r.embedding_margin,
            component_json=json.dumps(r.component_margins),
            reconstruction_error=r.reconstruction_error,
        )
    db_conn.close()

    run_dict = decompose_run_to_dict(run)

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        html = generate_decompose_html(run)
        with open(html_path, "w") as f:
            f.write(html)
        err_console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    if output_json:
        click.echo(json.dumps(run_dict, indent=2))
        return

    # Console output
    console.print(
        f"\n[bold]Logit Prism Decomposition:[/bold] {dataset_name} "
        f"({len(prompts)} prompts)"
    )
    console.print(f"Model: {model_name}\n")

    for r in all_results:
        err_str = "✓" if r.reconstruction_error < 0.01 else "⚠"
        console.print(
            f'[bold]"{r.prompt}"[/bold]  '
            f"reconstruction error: {r.reconstruction_error:.4f} {err_str}"
        )

        table = Table()
        table.add_column("Component", min_width=12)
        table.add_column(r.answer, justify="right")
        table.add_column(r.competitor, justify="right")
        table.add_column("Margin", justify="right")

        # Embedding
        em = r.embedding_margin
        em_style = "green" if em >= 0 else "red"
        table.add_row(
            "Embedding",
            f"{r.answer_decomposition.embedding:+.4f}",
            f"{r.competitor_decomposition.embedding:+.4f}",
            f"[{em_style}]{em:+.4f}[/{em_style}]",
        )

        # Per-layer
        for al, cl, cm in zip(
            r.answer_decomposition.layers,
            r.competitor_decomposition.layers,
            r.component_margins,
        ):
            attn_m = cm["attn_margin"]
            mlp_m = cm["mlp_margin"]
            as_ = "green" if attn_m >= 0 else "red"
            ms = "green" if mlp_m >= 0 else "red"
            table.add_row(
                f"L{al.layer} attn",
                f"{al.attention:+.4f}",
                f"{cl.attention:+.4f}",
                f"[{as_}]{attn_m:+.4f}[/{as_}]",
            )
            table.add_row(
                f"L{al.layer} mlp",
                f"{al.mlp:+.4f}",
                f"{cl.mlp:+.4f}",
                f"[{ms}]{mlp_m:+.4f}[/{ms}]",
            )

        # Total
        m_style = "green" if r.margin >= 0 else "red"
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{r.answer_logit:+.4f}[/bold]",
            f"[bold]{r.competitor_logit:+.4f}[/bold]",
            f"[bold][{m_style}]{r.margin:+.4f}[/{m_style}][/bold]",
        )

        console.print(table)

        # Top margin contributors
        sorted_margins = sorted(
            r.component_margins,
            key=lambda x: x["attn_margin"] + x["mlp_margin"],
        )
        console.print("\n  Largest negative margin contributors:")
        for cm in sorted_margins[:3]:
            total_m = cm["attn_margin"] + cm["mlp_margin"]
            if total_m < 0:
                console.print(
                    f"    L{cm['layer']}: {total_m:+.4f}"
                )
        console.print()


def _decompose_remote(remote_url, prompts, competitor_list, seed):
    """Run decompose via remote GPU worker."""
    from neurotrace.decompose import run_decompose_remote
    from neurotrace.remote import RemoteWorker

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))

    err_console.print(f"GPU: {device_name} via {remote_url}")

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            "Decomposing...", total=len(prompts),
        )

        for p_idx, entry in enumerate(prompts):
            progress.update(
                task,
                completed=p_idx,
                description=f"Prompt {p_idx + 1}/{len(prompts)}",
            )

            prompt_text = entry["prompt"]
            answer_text = entry["answer"]

            # Determine competitors for this prompt
            comps = competitor_list
            if not comps:
                comps = ["the", "a", "is"]  # fallback

            tokens = [answer_text] + comps

            remote_data = {}
            for event in worker.decompose_stream(
                prompt_text, tokens, seed=seed,
            ):
                etype = event.get("type")
                if etype == "decomposition":
                    remote_data = event.get("decompositions", {})

            for comp in comps:
                if comp in remote_data and answer_text in remote_data:
                    result = run_decompose_remote(
                        remote_data, prompt_text, answer_text, comp,
                    )
                    all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return all_results


# ---------------------------------------------------------------------------
# repair command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--prompt", default=None, help="Single prompt to repair.")
@click.option("--answer", default=None, help="Expected answer token.")
@click.option(
    "--competitor", default=None,
    help="Competitor token (auto-detected if omitted).",
)
@click.option(
    "--target-layer", default=None, type=int,
    help="Target layer (auto-detected if omitted).",
)
@click.option(
    "--target-component", default="mlp",
    type=click.Choice(["mlp", "attention"]),
    help="Target component.",
)
@click.option(
    "--target-margin", default=0.0, type=float,
    help="Target margin for component (default: 0.0 = neutralize).",
)
@click.option(
    "--dataset-builtin", default=None,
    help="Built-in dataset name for batch repair.",
)
@click.option(
    "--dataset", "dataset_path", default=None,
    type=click.Path(exists=True),
    help="Custom JSONL dataset path for batch repair.",
)
@click.option(
    "--verify-dataset", default=None,
    help="Built-in dataset for regression checking.",
)
@click.option("--undo", is_flag=True, help="Undo last edit on remote worker.")
@click.option("--save", "save_path", default=None, help="Save edited model to path.")
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def repair(
    model,
    db,
    prompt,
    answer,
    competitor,
    target_layer,
    target_component,
    target_margin,
    dataset_builtin,
    dataset_path,
    verify_dataset,
    undo,
    save_path,
    remote,
    device,
    html_path,
    output_json,
    adapter,
    seed,
):
    """ROME rank-one weight repair using Logit Prism decomposition."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.repair import (
        RepairRun,
        generate_repair_html,
        repair_run_to_dict,
    )

    # Handle --undo
    if undo:
        if remote is None:
            raise click.UsageError("--undo requires --remote.")
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        result = worker.repair_undo()
        console.print(f"[green]Undo: {result}[/green]")
        return

    # Handle --save
    if save_path is not None:
        if remote is None:
            raise click.UsageError("--save requires --remote.")
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        result = worker.repair_save(save_path)
        console.print(f"[green]Saved: {result}[/green]")
        return

    # Validate inputs
    has_dataset = dataset_path is not None or dataset_builtin is not None
    has_single = prompt is not None
    if not has_dataset and not has_single:
        raise click.UsageError(
            "Must provide --prompt/--answer or --dataset/--dataset-builtin."
        )
    if has_dataset and has_single:
        raise click.UsageError(
            "Cannot provide both dataset and --prompt/--answer."
        )
    if has_single and answer is None:
        raise click.UsageError("--answer is required with --prompt.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Build prompt list
    if has_single:
        prompts = [{"prompt": prompt, "answer": answer}]
        dataset_name = None
    else:
        from neurotrace.datasets import get_builtin_dataset, load_dataset

        if dataset_builtin is not None:
            dataset = get_builtin_dataset(dataset_builtin)
            dataset_name = dataset_builtin
        else:
            dataset = load_dataset(dataset_path)
            dataset_name = dataset_path
        prompts = [{"prompt": d["prompt"], "answer": d["answer"]} for d in dataset]

    # Build verify prompts
    verify_prompts = None
    if verify_dataset:
        from neurotrace.datasets import get_builtin_dataset

        verify_prompts = [
            {"prompt": d["prompt"], "answer": d["answer"]}
            for d in get_builtin_dataset(verify_dataset)
        ]

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    all_results = []

    if remote is not None:
        all_results = _repair_remote(
            remote, prompts, competitor, target_layer, target_component,
            target_margin, verify_prompts, seed,
        )
        from neurotrace.remote import RemoteWorker

        worker = RemoteWorker(remote)
        health = worker.health()
        model_name = health["model"]
    else:
        from neurotrace.models import load_model
        from neurotrace.repair import run_repair_local

        device = _resolve_device(device)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=err_console,
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter)
            model_name = model

            progress.update(
                task,
                description=f"Repairing {len(prompts)} prompts...",
                total=len(prompts),
            )

            for p_idx, entry in enumerate(prompts):
                progress.update(
                    task,
                    completed=p_idx,
                    description=f"Repair {p_idx + 1}/{len(prompts)}",
                )
                result = run_repair_local(
                    model_obj,
                    tokenizer,
                    entry["prompt"],
                    entry["answer"],
                    competitor=competitor,
                    target_layer=target_layer,
                    target_component=target_component,
                    target_margin=target_margin,
                    verify_prompts=verify_prompts,
                    seed=seed,
                )
                all_results.append(result)

            progress.update(task, description="Done.", completed=len(prompts))

    run = RepairRun(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        prompt_count=len(all_results),
        results=all_results,
        created_at=created_at,
    )

    # Save to DB
    db_conn = TraceDB(db)
    db_conn.write_repair_run(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        prompt_count=len(all_results),
    )
    for r in all_results:
        regressions_found = sum(
            1 for reg in r.regressions
            if reg.status == "regression"
        )
        db_conn.write_repair_result(
            run_id=run_id,
            prompt=r.prompt,
            answer=r.answer,
            competitor=r.competitor,
            target_layer=r.target_layer,
            target_component=r.target_component,
            before_margin=r.before.margin,
            after_margin=r.after.margin,
            before_prob=r.before.answer_prob,
            after_prob=r.after.answer_prob,
            edit_norm=r.edit.norm,
            regressions_checked=len(r.regressions),
            regressions_found=regressions_found,
            status=r.status,
        )
    db_conn.close()

    run_dict = repair_run_to_dict(run)

    # HTML output
    if html_path:
        import os

        os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
        html = generate_repair_html(run)
        with open(html_path, "w") as f:
            f.write(html)
        err_console.print(f"[green]Report saved to {html_path}[/green]")

    # JSON output
    if output_json:
        click.echo(json.dumps(run_dict, indent=2))
        return

    # Console output
    repaired = sum(1 for r in all_results if r.status == "repaired")
    skipped = sum(1 for r in all_results if r.status == "skipped")
    regressed = sum(1 for r in all_results if r.status == "regression")

    if len(all_results) == 1:
        r = all_results[0]
        console.print(
            f'\n[bold]Repair:[/bold] "{r.prompt}" → {r.answer}\n'
        )
        console.print("[bold] Before[/bold]")
        b = r.before
        console.print(
            f"  {r.answer} logit: {b.answer_logit:.2f}"
            f" | Prob: {b.answer_prob:.2f}"
        )
        console.print(
            f'  Competitor ("{r.competitor}"):'
            f" {b.competitor_logit:.2f}"
        )
        console.print(f"  Margin: {b.margin:+.2f}")
        comp = r.target_component.upper()
        console.print(
            f"  Worst component: L{r.target_layer} {comp}"
            f" (margin: {b.component_margin:+.2f})"
        )
        console.print("\n[bold] Edit[/bold]")
        console.print(
            f"  Target: L{r.edit.layer} {r.edit.matrix}"
            f" (rank-{r.edit.rank} update,"
            f" norm: {r.edit.norm:.3f})"
        )
        console.print("\n[bold] After[/bold]")
        a = r.after
        console.print(
            f"  {r.answer} logit: {a.answer_logit:.2f}"
            f" | Prob: {a.answer_prob:.2f}"
        )
        console.print(
            f'  Competitor ("{r.competitor}"):'
            f" {a.competitor_logit:.2f}"
        )
        console.print(f"  Margin: {a.margin:+.2f}")
        console.print(
            f"  L{r.target_layer} {comp} margin:"
            f" {b.component_margin:+.2f}"
            f" → {a.component_margin:+.2f}"
        )

        if r.regressions:
            n_reg = sum(
                1 for reg in r.regressions
                if reg.status == "regression"
            )
            n_total = len(r.regressions)
            console.print(
                f"\n[bold] Regression check"
                f" ({n_total} prompts)[/bold]"
            )
            if n_reg == 0:
                max_change = max(
                    (
                        abs(reg.after_prob - reg.before_prob)
                        for reg in r.regressions
                    ),
                    default=0,
                )
                console.print(
                    f"  [green]✓[/green] 0 regressions"
                    f" (max prob change:"
                    f" {max_change:+.2f})"
                )
            else:
                console.print(
                    f"  [red]✗[/red] {n_reg} regressions"
                )
                for reg in r.regressions:
                    if reg.status == "regression":
                        console.print(
                            f"    {reg.prompt[:40]}:"
                            f" {reg.before_prob:.2f}"
                            f" → {reg.after_prob:.2f}"
                        )
        console.print()
    else:
        console.print(
            f"\n[bold]Repair Summary:[/bold] {dataset_name or 'batch'} "
            f"({len(all_results)} prompts)"
        )
        console.print(f"  Repaired: {repaired}")
        console.print(f"  Skipped: {skipped}")
        console.print(f"  Regressions caught: {regressed}")

        table = Table()
        table.add_column("Prompt", max_width=40)
        table.add_column("Answer")
        table.add_column("Before", justify="right")
        table.add_column("After", justify="right")
        table.add_column("Status")
        for r in all_results:
            style_map = {
                "repaired": "green",
                "skipped": "yellow",
                "regression": "red",
            }
            s_style = style_map.get(r.status, "")
            table.add_row(
                r.prompt[:40],
                r.answer,
                f"{r.before.margin:+.2f}",
                f"{r.after.margin:+.2f}",
                f"[{s_style}]{r.status.upper()}[/{s_style}]",
            )
        console.print(table)
        console.print()


def _repair_remote(remote_url, prompts, competitor, target_layer, target_component,
                   target_margin, verify_prompts, seed):
    """Run repair via remote GPU worker."""
    from neurotrace.remote import RemoteWorker
    from neurotrace.repair import build_repair_result_from_remote

    worker = RemoteWorker(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))

    err_console.print(f"GPU: {device_name} via {remote_url}")

    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            "Repairing...", total=len(prompts),
        )

        for p_idx, entry in enumerate(prompts):
            progress.update(
                task,
                completed=p_idx,
                description=f"Repair {p_idx + 1}/{len(prompts)}",
            )

            remote_data = None
            for event in worker.repair_stream(
                prompt=entry["prompt"],
                answer=entry["answer"],
                competitor=competitor,
                target_layer=target_layer,
                target_component=target_component,
                target_margin=target_margin,
                verify_prompts=verify_prompts,
                seed=seed,
            ):
                etype = event.get("type")
                if etype == "progress":
                    progress.update(
                        task,
                        description=f"Repair {p_idx + 1}: {event.get('status', '')}",
                    )
                elif etype == "result":
                    remote_data = event

            if remote_data:
                result = build_repair_result_from_remote(remote_data)
                all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return all_results
