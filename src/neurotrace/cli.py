"""Click CLI for NeuroTrace — trace, list, inspect, diff, predict commands."""

import json
import sys

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.storage import TraceDB

console = Console()
err_console = Console(stderr=True)


def _resolve_trace_id(db: TraceDB, trace_id: str) -> str:
    """Resolve 'latest' alias to actual trace_id."""
    if trace_id == "latest":
        return db.get_latest_trace_id()
    return trace_id


@click.group()
def cli() -> None:
    """NeuroTrace: interpretability toolkit for tracing transformer inference."""


@cli.command()
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt", default=None, help="Prompt text to trace.")
@click.option("--prompts-file", default=None, type=click.Path(exists=True),
              help="File with one prompt per line.")
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--label", default=None, help="Optional label for the trace.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--capture-mode", default="full", type=click.Choice(["full", "light"]),
              help="Capture mode.")
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

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=err_console) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model
        from neurotrace.tracer import Tracer

        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Model loaded.")

        tracer = Tracer(model_obj, tokenizer, capture_mode=capture_mode,
                        layer_stride=layer_stride)
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
@click.option("--head", default=None, type=int, help="Show specific head (requires --layer).")
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
    console.print(f"[bold]Layers:[/bold] {meta.num_layers}  [bold]Heads:[/bold] {meta.num_heads}")
    console.print(f"[bold]Capture:[/bold] {meta.capture_mode}  [bold]Stride:[/bold] {meta.layer_stride}")
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
@click.option("--cosine-threshold", default=0.95, type=float,
              help="Cosine similarity threshold for flagging.")
@click.option("--kl-threshold", default=0.5, type=float,
              help="KL divergence threshold for flagging.")
@click.option("--sort-by", default="layer", type=click.Choice(["layer", "cosine", "kl"]),
              help="Sort order for diff table.")
@click.option("--flagged-only", is_flag=True, help="Show only flagged layers.")
@click.option("--head-detail", is_flag=True, help="Show critical head details.")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON.")
def diff(db, trace_a, trace_b, cosine_threshold, kl_threshold, sort_by,
         flagged_only, head_detail, output_json):
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

    diff_result = compute_diff(result_a, result_b,
                               cosine_threshold=cosine_threshold,
                               kl_threshold=kl_threshold)

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

    table = Table(title=f"Diff: {diff_result.trace_a_id[:8]} vs {diff_result.trace_b_id[:8]}")
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
    console.print(f"\n[bold]Flagged layers:[/bold] {flagged_count}/{len(diff_result.layer_metrics)}")
    if diff_result.first_divergence_layer is not None:
        console.print(f"[bold]First divergence:[/bold] layer {diff_result.first_divergence_layer}")

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
@click.option("--model", required=True, help="HuggingFace model name or path.")
@click.option("--prompt", required=True, help="Prompt text.")
@click.option("--top-k", default=5, type=int, help="Number of top predictions per position.")
def predict(model, prompt, top_k):
    """Show per-position token predictions (no DB write)."""
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=err_console) as progress:
        task = progress.add_task("Loading model...", total=None)
        from neurotrace.models import load_model
        from neurotrace.tracer import Tracer

        model_obj, tokenizer = load_model(model)
        progress.update(task, description="Running forward pass...")

        tracer = Tracer(model_obj, tokenizer, capture_mode="light")
        result = tracer.trace(prompt, top_k=top_k)
        progress.update(task, description="Done.")

    tokens = result.metadata.tokens
    for pred in result.token_predictions:
        pos = pred.position
        token_str = tokens[pos] if pos < len(tokens) else "?"
        console.print(f"\n[bold]Position {pos}[/bold] (input: [cyan]{repr(token_str)}[/cyan])")
        for tok, prob, s in zip(pred.top_k_tokens, pred.top_k_probs, pred.top_k_strings):
            bar_len = int(prob * 40)
            bar = "#" * bar_len
            console.print(f"  {prob:6.3f} [green]{bar}[/green] {repr(s)} (id={tok})")
