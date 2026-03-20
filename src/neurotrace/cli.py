"""Click CLI for NeuroTrace — trace, list, inspect, diff, predict commands."""

import json

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.storage import TraceDB

console = Console()
err_console = Console(stderr=True)


def _resolve_trace_id(db: TraceDB, trace_id: str) -> str:
    """Resolve label, prefix, 'latest', or full trace_id."""
    return db.resolve_trace_id(trace_id)


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
def trace(model, prompt, prompts_file, db, label, seed, capture_mode, layer_stride):
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

        model_obj, tokenizer = load_model(model)
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
@click.option(
    "--top-k", default=5, type=int, help="Top predictions per layer."
)
@click.option(
    "--changes-only", is_flag=True, help="Only show layers where top-1 changed."
)
@click.option("--layers", default=None, help="Comma-separated layer indices to show.")
@click.option(
    "--track", default=None, help="Token string to track across all layers."
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def predict(db, trace_id, top_k, changes_only, layers, track, output_json):
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

        model_obj, tokenizer = load_model(model_name)
        progress.update(task, description="Projecting layers...")

    lm_head = model_obj.lm_head
    final_ln = None
    if hasattr(model_obj, "model") and hasattr(model_obj.model, "norm"):
        final_ln = model_obj.model.norm

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
            res_tensor = torch.tensor(
                snap.residual_out, dtype=torch.float32
            ).unsqueeze(0)
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
            prev_top1 = next(
                (tid for tid, r in prev_ranking.items() if r == 0), None
            )
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
        for rank, (tid, prob, s, ann) in enumerate(zip(
            entry["top_k_ids"],
            entry["top_k_probs"],
            entry["top_k_strings"],
            entry["annotations"],
        )):
            ann_str = f"  [dim]{ann}[/dim]" if ann else ""
            console.print(
                f"  {rank + 1}. {repr(s):20s} ({prob:.3f}){ann_str}"
            )
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
        results.append({
            "token_id": tid,
            "string": decoded,
            "raw_token": raw_token,
            "hex_bytes": hex_bytes,
            "is_special": tid in special_ids,
        })
    return results


@cli.command()
@click.option(
    "--model", required=True, help="HuggingFace model name or path."
)
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
        raise click.UsageError(
            "Must provide --tokens or --from-trace."
        )
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
@click.option(
    "--model", required=True, help="HuggingFace model name or path."
)
@click.option("--prompt-a", required=True, help="First prompt.")
@click.option("--prompt-b", required=True, help="Second prompt.")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--light", is_flag=True, help="Use light capture mode.")
@click.option(
    "--flagged-only", is_flag=True, help="Show only flagged layers."
)
@click.option(
    "--head-detail", is_flag=True, help="Show critical head details."
)
@click.option(
    "--json", "output_json", is_flag=True, help="Output as JSON."
)
def compare(
    model, prompt_a, prompt_b, db, seed, light, flagged_only,
    head_detail, output_json,
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
            model, prompt, seed, capture_mode,
        )
        if existing is not None:
            status_console.print(
                f"[dim]Reusing existing trace {existing[:8]}[/dim]"
            )
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
        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Model loaded.")

        tracer = Tracer(
            model_obj, tokenizer, capture_mode=capture_mode,
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
            "token_legend": {
                str(k): v for k, v in token_lookup.items()
            },
            "layer_metrics": [
                {
                    "layer_index": m.layer_index,
                    "cosine_similarity": m.cosine_similarity,
                    "top1_changed": m.top1_changed,
                    "kl_divergence": m.kl_divergence,
                    "flagged": m.flagged,
                    "trace_a_top1": m.trace_a_top1,
                    "trace_a_top1_str": token_lookup.get(
                        m.trace_a_top1, "?"
                    ),
                    "trace_a_top1_prob": m.trace_a_top1_prob,
                    "trace_b_top1": m.trace_b_top1,
                    "trace_b_top1_str": token_lookup.get(
                        m.trace_b_top1, "?"
                    ),
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

    table = Table(
        title=f"Compare: {trace_a_id[:8]} vs {trace_b_id[:8]}"
    )
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
    flagged_count = sum(
        1 for m in diff_result.layer_metrics if m.flagged
    )
    console.print(
        f"\n[bold]Flagged layers:[/bold] "
        f"{flagged_count}/{len(diff_result.layer_metrics)}"
    )
    if diff_result.first_divergence_layer is not None:
        console.print(
            f"[bold]First divergence:[/bold]"
            f" layer {diff_result.first_divergence_layer}"
        )

    if head_detail and diff_result.critical_heads:
        console.print(
            "\n[bold]Critical Heads (top JS divergence):[/bold]"
        )
        head_table = Table()
        head_table.add_column("Layer", justify="right")
        head_table.add_column("Head", justify="right")
        head_table.add_column("JS Divergence", justify="right")
        for layer_idx, head_idx, js_div in diff_result.critical_heads:
            head_table.add_row(
                str(layer_idx), str(head_idx), f"{js_div:.6f}"
            )
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
    "--zero-layers", default=None,
    help="Zero entire layer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--zero-heads", default=None,
    help="Zero specific attention heads. E.g. '20:7,20:12'.",
)
@click.option(
    "--scale-layer", default=None,
    help="Scale layer contributions. E.g. '20:0.5,21:2.0'.",
)
@click.option(
    "--zero-mlp", default=None,
    help="Zero MLP sublayer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--scale-mlp", default=None,
    help="Scale MLP sublayer outputs. E.g. '20:0.5' or '20:0.3,21:0.7'.",
)
@click.option(
    "--baseline", default=None,
    help="Baseline trace ID, label, or prefix. If omitted, runs a clean trace.",
)
@click.option("--label", default=None, help="Label for the ablated trace.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def ablate(
    db, model, prompt, zero_layers, zero_heads, scale_layer,
    zero_mlp, scale_mlp, baseline, label, seed,
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
        zero_layers=zl, zero_heads=zh, scale_layers=sl,
        zero_mlp=zm, scale_mlp=sm,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        model_obj, tokenizer = load_model(model)
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
            model_obj, tokenizer, prompt, spec,
            baseline=baseline_trace, label=label, seed=seed,
        )
        progress.update(task, description="Storing traces...")

        # Store baseline if we created it
        if baseline is None:
            db_conn.write_trace(result.baseline_trace)
            console.print(
                f"[green]Baseline stored: {result.baseline_trace.metadata.trace_id[:8]}[/green]"
            )

        # Store ablated trace with interventions metadata
        db_conn.write_trace(result.ablated_trace, interventions=spec.to_json())
        console.print(
            f"[green]Ablated stored: {result.ablated_trace.metadata.trace_id[:8]}[/green]"
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
            or any(l == lc.layer_index for l, _ in spec.zero_heads)
            or any(l == lc.layer_index for l, _ in spec.scale_layers)
            or lc.layer_index in spec.zero_mlp
            or any(l == lc.layer_index for l, _ in spec.scale_mlp)
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

    has_residuals = any(
        s.residual_out is not None for s in result.layer_snapshots
    )
    if not has_residuals:
        return None, None

    from neurotrace.models import load_model

    try:
        model_obj, tokenizer = load_model(result.metadata.model_name)
    except (OSError, ValueError):
        return None, None
    lm_head = model_obj.lm_head
    final_ln = None
    if hasattr(model_obj, "model") and hasattr(model_obj.model, "norm"):
        final_ln = model_obj.model.norm

    layer_preds = []
    prev_ranking: dict[int, int] = {}
    # Store full prob vectors to build token tracks later
    prob_vectors: dict[int, torch.Tensor] = {}

    for snap in result.layer_snapshots:
        if snap.residual_out is None:
            continue

        with torch.no_grad():
            res = torch.tensor(
                snap.residual_out, dtype=torch.float32
            ).unsqueeze(0)
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

        layer_preds.append({
            "layer_index": snap.layer_index,
            "top_k_ids": topk_ids,
            "top_k_probs": topk_probs,
            "top_k_strings": topk_strings,
            "annotations": annotations,
        })

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
            token_tracks.append({
                "token": tok_str,
                "token_id": tid,
                "probs": track_probs,
                "layers": [lp["layer_index"] for lp in layer_preds],
            })

    return layer_preds, token_tracks


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option(
    "--trace-id", default=None, help="Single trace report: trace ID or label."
)
@click.option(
    "--trace-a", default=None, help="Comparison report: first trace ID."
)
@click.option(
    "--trace-b", default=None, help="Comparison report: second trace ID."
)
@click.option(
    "-o", "--output", default="report.html", help="Output file path."
)
@click.option(
    "--open", "open_browser", is_flag=True,
    help="Open report in browser after generating.",
)
@click.option(
    "--full-attention", is_flag=True,
    help="Include attention heatmaps for all layers.",
)
@click.option(
    "--no-attention", is_flag=True,
    help="Skip attention heatmaps entirely.",
)
@click.option(
    "--upload", is_flag=True,
    help="Upload report to CarbonFiles after generating.",
)
@click.option(
    "--bucket", default=None,
    help="Existing CarbonFiles bucket ID for upload.",
)
def report(
    db, trace_id, trace_a, trace_b, output, open_browser,
    full_attention, no_attention, upload, bucket,
):
    """Generate a self-contained HTML report from one or two traces."""
    import os
    import webbrowser

    from neurotrace.report import generate_comparison_report, generate_report

    # Validate options
    if trace_id and (trace_a or trace_b):
        raise click.UsageError(
            "Cannot use --trace-id with --trace-a/--trace-b."
        )
    if not trace_id and not (trace_a and trace_b):
        raise click.UsageError(
            "Provide --trace-id or both --trace-a and --trace-b."
        )
    if (trace_a and not trace_b) or (trace_b and not trace_a):
        raise click.UsageError(
            "Must provide both --trace-a and --trace-b."
        )

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
                task = progress.add_task(
                    "Computing predictions...", total=None
                )
                layer_preds, token_tracks = _compute_layer_predictions(
                    result
                )
                progress.update(task, description="Generating report...")

                html = generate_report(
                    result, layer_stats, layer_preds, token_tracks,
                    full_attention, no_attention,
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
                        for tid, s in zip(
                            lp["top_k_ids"], lp["top_k_strings"]
                        ):
                            token_lookup[tid] = s
                if preds_b:
                    for lp in preds_b:
                        for tid, s in zip(
                            lp["top_k_ids"], lp["top_k_strings"]
                        ):
                            token_lookup[tid] = s
                # Also add from diff metrics
                for m in diff_result.layer_metrics:
                    if m.trace_a_top1 not in token_lookup:
                        token_lookup[m.trace_a_top1] = str(m.trace_a_top1)
                    if m.trace_b_top1 not in token_lookup:
                        token_lookup[m.trace_b_top1] = str(m.trace_b_top1)

                progress.update(
                    task, description="Generating report..."
                )
                html = generate_comparison_report(
                    result_a, result_b, stats_a, stats_b,
                    preds_a, preds_b, tracks_a, tracks_b,
                    diff_result, token_lookup,
                    full_attention, no_attention,
                )
    except ValueError as e:
        db_conn.close()
        raise click.ClickException(str(e))

    with open(output, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(output) / 1024
    console.print(
        f"[green]Report written to {output} ({size_kb:.0f} KB)[/green]"
    )

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
        raise click.BadParameter(
            f"Expected format L:start:end:step, got {value!r}"
        )
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
        raise click.BadParameter(
            f"Expected format L:start-end, got {value!r}"
        )
    layer = int(parts[0])
    head_range = parts[1].split("-")
    if len(head_range) != 2:
        raise click.BadParameter(
            f"Expected format L:start-end, got {value!r}"
        )
    start = int(head_range[0])
    end = int(head_range[1])
    return layer, list(range(start, end + 1))


@cli.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt", required=True, help="Prompt text to trace.")
@click.option(
    "--baseline", default=None,
    help="Baseline trace ID, label, or prefix. If omitted, runs a clean trace.",
)
@click.option("--label", default=None, help="Base label for sweep traces.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--sweep-scale-mlp", default=None,
    help="Sweep MLP scale factor. Format: L:start:end:step (e.g. '20:0.1:0.9:0.1').",
)
@click.option(
    "--sweep-zero-heads", "sweep_zero_heads_opt", default=None,
    help="Sweep zero-heads one at a time. Format: L:start-end (e.g. '20:0-31').",
)
@click.option(
    "--sweep-zero-mlp", default=None,
    help="Sweep zero-mlp one layer at a time. Format: L1,L2,L3 (e.g. '18,19,20').",
)
@click.option(
    "--sweep-scale-layer", default=None,
    help="Sweep layer scale factor. Format: L:start:end:step (e.g. '20:0.1:0.9:0.1').",
)
@click.option(
    "--zero-layers", default=None,
    help="Fixed intervention: zero entire layer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--zero-heads", default=None,
    help="Fixed intervention: zero specific attention heads. E.g. '20:7,20:12'.",
)
@click.option(
    "--scale-layer", default=None,
    help="Fixed intervention: scale layer contributions. E.g. '20:0.5,21:2.0'.",
)
@click.option(
    "--zero-mlp", default=None,
    help="Fixed intervention: zero MLP sublayer outputs. E.g. '20' or '20,21'.",
)
@click.option(
    "--scale-mlp", default=None,
    help="Fixed intervention: scale MLP sublayer outputs. E.g. '20:0.5'.",
)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def sweep(
    db, model, prompt, baseline, label, seed,
    sweep_scale_mlp, sweep_zero_heads_opt, sweep_zero_mlp, sweep_scale_layer,
    zero_layers, zero_heads, scale_layer, zero_mlp, scale_mlp,
    output_json,
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
    sweep_flags = [sweep_scale_mlp, sweep_zero_heads_opt, sweep_zero_mlp, sweep_scale_layer]
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
        sweep_description = f"scale-mlp layer {target_layer} ({values[0]} → {values[-1]}, step {values[1] - values[0] if len(values) > 1 else 0})"
        for v in values:
            sm = fixed_sm + [(target_layer, v)]
            spec = AblationSpec(
                zero_layers=fixed_zl, zero_heads=fixed_zh,
                scale_layers=fixed_sl, zero_mlp=fixed_zm, scale_mlp=sm,
            )
            sweep_specs.append((str(v), spec))

    elif sweep_scale_layer:
        target_layer, values = _parse_sweep_range(sweep_scale_layer)
        sweep_target_layer = target_layer
        sweep_description = f"scale-layer layer {target_layer} ({values[0]} → {values[-1]}, step {values[1] - values[0] if len(values) > 1 else 0})"
        for v in values:
            sl = fixed_sl + [(target_layer, v)]
            spec = AblationSpec(
                zero_layers=fixed_zl, zero_heads=fixed_zh,
                scale_layers=sl, zero_mlp=fixed_zm, scale_mlp=fixed_sm,
            )
            sweep_specs.append((str(v), spec))

    elif sweep_zero_heads_opt:
        target_layer, heads = _parse_sweep_zero_heads(sweep_zero_heads_opt)
        sweep_target_layer = target_layer
        sweep_description = f"zero-heads layer {target_layer} (head {heads[0]} → {heads[-1]})"
        for h in heads:
            zh = fixed_zh + [(target_layer, h)]
            spec = AblationSpec(
                zero_layers=fixed_zl, zero_heads=zh,
                scale_layers=fixed_sl, zero_mlp=fixed_zm, scale_mlp=fixed_sm,
            )
            sweep_specs.append((f"h{h}", spec))

    elif sweep_zero_mlp:
        layers = parse_zero_layers(sweep_zero_mlp)
        sweep_target_layer = layers[0] if layers else -1
        sweep_description = f"zero-mlp layers {','.join(map(str, layers))}"
        for l_idx in layers:
            zm = fixed_zm + [l_idx]
            spec = AblationSpec(
                zero_layers=fixed_zl, zero_heads=fixed_zh,
                scale_layers=fixed_sl, zero_mlp=zm, scale_mlp=fixed_sm,
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

        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Model loaded.")

        db_conn = TraceDB(db)

        # Resolve or run baseline
        baseline_trace = None
        if baseline is not None:
            try:
                baseline_id = _resolve_trace_id(db_conn, baseline)
                baseline_trace = db_conn.read_trace(baseline_id)
                progress.update(
                    task, description=f"Loaded baseline {baseline_id[:8]}.",
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
                    f"[green]Baseline stored: {baseline_trace.metadata.trace_id[:8]}[/green]"
                )

        # Run sweep
        results: list[dict] = []
        for i, (sweep_val, spec) in enumerate(sweep_specs):
            desc = f"Sweep {i + 1}/{len(sweep_specs)}: {sweep_val}..."
            progress.update(task, description=desc)

            trace_label = f"{label}-{sweep_val}" if label else f"sweep-{sweep_val}"
            ablation_result = run_ablation(
                model_obj, tokenizer, prompt, spec,
                baseline=baseline_trace, label=trace_label, seed=seed,
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

            results.append({
                "sweep_value": sweep_val,
                "trace_id": trace_id,
                "final_token": ablation_result.ablated_final_token,
                "final_prob": ablation_result.ablated_final_prob,
                "layer_preds": layer_preds,
                "spec": spec.describe(),
            })

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
                        str(l): r["layer_preds"].get(l, {})
                        for l in display_layers_sorted
                    },
                }
                for r in results
            ],
        }
        click.echo(json.dumps(json_output, indent=2, default=str))
        return

    # Rich table output
    console.print(f"\n[bold]Sweep: {sweep_description}[/bold]")
    console.print(
        f"[dim]Baseline: {baseline_trace.metadata.trace_id[:8]}[/dim]\n"
    )

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
    "--dataset", "dataset_path", default=None,
    help="Path to JSON dataset file.",
)
@click.option(
    "--dataset-builtin", default=None,
    help="Use a built-in dataset (e.g. 'capitals').",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option(
    "--sabotage-threshold", default=0.5, type=float,
    help="Fraction of peak prob drop to flag as sabotage.",
)
@click.option(
    "--final-threshold", default=0.3, type=float,
    help="Minimum final probability to avoid weak flag.",
)
@click.option(
    "--save-traces", is_flag=True,
    help="Store all traces to database.",
)
@click.option(
    "--save-flagged", is_flag=True,
    help="Store only flagged traces to database.",
)
@click.option("--details", is_flag=True, help="Show layer details for flagged prompts.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def scan(
    db, model, dataset_path, dataset_builtin, seed,
    sabotage_threshold, final_threshold,
    save_traces, save_flagged, details, output_json,
):
    """Scan a dataset for sabotaged predictions."""
    from neurotrace.datasets import get_builtin_dataset, load_dataset
    from neurotrace.scan import run_scan

    if dataset_path is None and dataset_builtin is None:
        raise click.UsageError(
            "Must provide either --dataset or --dataset-builtin."
        )
    if dataset_path is not None and dataset_builtin is not None:
        raise click.UsageError(
            "Cannot provide both --dataset and --dataset-builtin."
        )

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

        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Model loaded.")

        db_conn = None
        if save_traces or save_flagged:
            db_conn = TraceDB(db)

        def progress_cb(i, total, prompt):
            desc = f"Scanning {i + 1}/{total}: {prompt[:40]}..."
            progress.update(task, description=desc)

        scan_result = run_scan(
            model_obj, tokenizer, dataset, dataset_name,
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
            status_str, prompt_display, r.answer,
            final_display, peak_display, flags_display,
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
                console.print(
                    f'[bold]\u26a0 "{r.prompt}" \u2192 {r.answer}[/bold]'
                )
                if r.ranks and r.probs:
                    # Show compact layer-by-layer
                    parts = []
                    peak_idx = r.probs.index(r.peak_prob) if r.peak_prob in r.probs else None
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
@click.option("--prompt", required=True, help="Target prompt.")
@click.option("--layer", required=True, type=int, help="MLP layer to analyze.")
@click.option(
    "--contrast", default=None,
    help="Contrast prompt (model gets this right).",
)
@click.option("--top-n", default=20, type=int, help="Top N neurons.")
@click.option(
    "--save-profile", default=None, help="Label to save profile.",
)
@click.option(
    "--ablate", "ablate_mode", is_flag=True,
    help="Enable ablation mode.",
)
@click.option(
    "--neurons", "neuron_str", default=None,
    help="Neuron indices: '0,1,2' or '100-200'.",
)
@click.option(
    "--from-profile", default=None,
    help="Use top-N neurons from a saved profile.",
)
@click.option(
    "--group-size", default=1, type=int,
    help="Ablate in groups of N (default: 1).",
)
@click.option("--baseline", default=None, help="Baseline trace ID for comparison.")
@click.option("--label", default=None, help="Label prefix for ablation traces.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def neurons(
    db, model, prompt, layer, contrast, top_n, save_profile,
    ablate_mode, neuron_str, from_profile, group_size,
    baseline, label, seed, output_json,
):
    """Neuron-level MLP attribution: profile or ablate individual neurons."""
    if ablate_mode:
        _neurons_ablate(
            db, model, prompt, layer, top_n, neuron_str, from_profile,
            group_size, baseline, label, seed, output_json,
        )
    else:
        _neurons_profile(
            db, model, prompt, layer, contrast, top_n,
            save_profile, seed, output_json,
        )


def _neurons_profile(
    db, model, prompt, layer, contrast, top_n,
    save_profile, seed, output_json,
):
    """Profile mode: capture and rank MLP intermediate neuron activations."""
    from neurotrace.neurons import profile_neurons

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Profiling neurons...")

        profile = profile_neurons(
            model_obj, tokenizer, prompt, layer,
            top_n=top_n, contrast_prompt=contrast,
            seed=seed, label=save_profile,
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
                    profile.diff_activations
                    or [None] * len(profile.neuron_indices),
                )
            ],
        }
        click.echo(json.dumps(output, indent=2, default=str))
        return

    # Rich table output
    console.print(
        f"\n[bold]Neuron Profile:[/bold] Layer {profile.layer} MLP"
    )
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
    db, model, prompt, layer, top_n, neuron_str, from_profile,
    group_size, baseline, label, seed, output_json,
):
    """Ablate mode: zero specific neurons and measure prediction impact."""
    from neurotrace.neurons import ablate_neurons, parse_neurons

    # Determine which neurons to ablate
    if neuron_str is None and from_profile is None:
        raise click.UsageError(
            "Ablation mode requires --neurons or --from-profile."
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model

        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Model loaded.")

        db_conn = TraceDB(db)

        # Resolve neuron indices
        if from_profile:
            progress.update(task, description="Loading profile...")
            profile = db_conn.load_neuron_profile(from_profile)
            if profile is None:
                db_conn.close()
                raise click.ClickException(
                    f"Profile not found: {from_profile}"
                )
            neuron_indices = profile.neuron_indices[:top_n]
        else:
            neuron_indices = parse_neurons(neuron_str)

        # Build groups
        if group_size <= 0:
            group_size = 1
        neuron_groups = []
        for i in range(0, len(neuron_indices), group_size):
            neuron_groups.append(neuron_indices[i:i + group_size])

        # Resolve baseline
        baseline_trace = None
        if baseline is not None:
            try:
                baseline_id = _resolve_trace_id(db_conn, baseline)
                baseline_trace = db_conn.read_trace(baseline_id)
                progress.update(
                    task, description=f"Loaded baseline {baseline_id[:8]}."
                )
            except ValueError as e:
                db_conn.close()
                raise click.ClickException(str(e))

        # Run ablations
        progress.update(
            task,
            description=f"Ablating {len(neuron_groups)} group(s)...",
        )
        baseline_created, results = ablate_neurons(
            model_obj, tokenizer, prompt, layer,
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
            interventions = json.dumps({
                "zero_neurons": {"layer": layer, "neurons": r.neurons}
            })
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
    console.print(
        f"\n[bold]Neuron Ablation:[/bold] Layer {layer} MLP"
    )
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
