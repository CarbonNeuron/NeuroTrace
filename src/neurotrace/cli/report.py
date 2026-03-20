"""Report CLI command."""


import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from neurotrace.cli import _resolve_trace_id, console, err_console
from neurotrace.storage import TraceDB


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


@click.command()
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
