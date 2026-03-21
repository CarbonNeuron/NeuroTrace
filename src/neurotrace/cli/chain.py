"""CLI command for chain trace (multi-hop reasoning)."""

import json

import click
from rich.table import Table

from neurotrace.cli import console, err_console


@click.command("chain")
@click.argument("prompt_text", required=False, default=None)
@click.option("--intermediate", default=None, help="Intermediate bridging token.")
@click.option("--answer", default=None, help="Final answer token.")
@click.option(
    "--dataset", "dataset_name", default=None,
    help="Built-in dataset name (e.g. 'multi_hop').",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--model", "model_name", default=None, help="HuggingFace model name (local mode).")
@click.option("--device", default="cpu", help="Device: cpu, cuda, auto.")
@click.option("--output", "output_format", default="table", type=click.Choice(["table", "json"]))
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.pass_context
def chain(ctx, prompt_text, intermediate, answer, dataset_name, remote, model_name, device, output_format, seed):
    """Trace multi-hop reasoning through the residual stream."""
    from neurotrace.chain import (
        ChainResult,
        chain_result_to_dict,
        run_chain_batch,
        run_chain_remote,
    )
    from neurotrace.datasets import get_builtin_dataset

    # Resolve model name from parent context
    if model_name is None:
        model_name = ctx.obj.get("model") if ctx.obj else None

    # Validate args
    if prompt_text is None and dataset_name is None:
        raise click.UsageError("Must provide a prompt or --dataset.")

    if prompt_text is not None and (intermediate is None or answer is None):
        raise click.UsageError("Single prompt mode requires --intermediate and --answer.")

    # Build entries
    entries: list[dict] = []
    if dataset_name:
        entries = get_builtin_dataset(dataset_name)
        # Validate multi_hop fields
        for i, e in enumerate(entries):
            if "intermediate" not in e or "answer" not in e:
                raise click.UsageError(
                    f"Dataset entry {i} missing 'intermediate' or 'answer' field."
                )
    else:
        entries = [{"prompt": prompt_text, "intermediate": intermediate, "answer": answer}]

    # Run
    results: list[ChainResult] = []

    if remote:
        from neurotrace.remote import WorkerClient

        client = WorkerClient(remote)
        err_console.print(f"[dim]Connected to {remote}[/dim]")

        def run_fn(p, i, a):
            return run_chain_remote(client, p, i, a, seed=seed)

    else:
        if model_name is None:
            raise click.UsageError("Must provide --model (local mode) or --remote.")

        from neurotrace.cli import _resolve_device
        from neurotrace.models import load_model

        resolved_device = _resolve_device(device)
        err_console.print(f"[dim]Loading {model_name} on {resolved_device}...[/dim]")
        model_obj, tokenizer = load_model(model_name, device=resolved_device)

        from neurotrace.chain import run_chain_local

        def run_fn(p, i, a):
            return run_chain_local(model_obj, tokenizer, p, i, a, seed=seed)

    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Tracing chains...", total=len(entries))
        for entry in entries:
            result = run_fn(entry["prompt"], entry["intermediate"], entry["answer"])
            results.append(result)
            progress.advance(task)

    # Output
    if output_format == "json":
        output = [chain_result_to_dict(r) for r in results]
        console.print(json.dumps(output, indent=2))
        return

    # Table output
    if len(results) == 1:
        _print_single(results[0])
    else:
        _print_batch(results)


def _status_color(status: str) -> str:
    colors = {
        "COMPLETE": "green",
        "PARTIAL": "yellow",
        "BROKEN": "red",
        "SHORTCUT": "cyan",
    }
    return colors.get(status, "white")


def _print_single(r):
    """Print detailed single-prompt chain trace."""
    console.print(f"\n[bold]Chain Trace:[/bold] {r.prompt}")
    console.print(
        f"  Intermediate: [bold]{r.intermediate}[/bold]  "
        f"Answer: [bold]{r.answer}[/bold]"
    )

    color = _status_color(r.chain_status)
    console.print(f"  Status: [{color}]{r.chain_status}[/{color}]")
    console.print()

    table = Table(title="Per-Layer Trace")
    table.add_column("Layer", justify="right", style="dim")
    table.add_column(f"{r.intermediate} (logit)", justify="right")
    table.add_column(f"{r.intermediate} (rank)", justify="right")
    table.add_column(f"{r.answer} (logit)", justify="right")
    table.add_column(f"{r.answer} (rank)", justify="right")
    table.add_column("Gap", justify="right")

    for i in range(r.num_layers):
        int_logit = r.intermediate_logits[i] if i < len(r.intermediate_logits) else 0.0
        ans_logit = r.answer_logits[i] if i < len(r.answer_logits) else 0.0
        int_rank = r.intermediate_ranks[i] if i < len(r.intermediate_ranks) else 999
        ans_rank = r.answer_ranks[i] if i < len(r.answer_ranks) else 999
        gap = int_logit - ans_logit

        # Color coding
        int_style = "green" if int_rank <= 10 else ("yellow" if int_logit < 0 else "")
        ans_style = "green" if ans_rank <= 10 else ("yellow" if ans_logit < 0 else "")

        annotation = ""
        if i == r.intermediate_commit_layer:
            annotation += " <- int commits"
        if i == r.answer_commit_layer:
            annotation += " <- ans commits"

        int_rank_str = str(int_rank) if int_rank < 999 else ">50"
        ans_rank_str = str(ans_rank) if ans_rank < 999 else ">50"

        table.add_row(
            str(i),
            f"[{int_style}]{int_logit:+.2f}[/{int_style}]" if int_style else f"{int_logit:+.2f}",
            f"[{int_style}]{int_rank_str}[/{int_style}]" if int_style else int_rank_str,
            f"[{ans_style}]{ans_logit:+.2f}[/{ans_style}]" if ans_style else f"{ans_logit:+.2f}",
            f"[{ans_style}]{ans_rank_str}[/{ans_style}]" if ans_style else ans_rank_str,
            f"{gap:+.2f}{annotation}",
        )

    console.print(table)

    # Summary metrics
    console.print(f"\n[bold]Metrics:[/bold]")
    console.print(f"  Intermediate commit layer: {r.intermediate_commit_layer}")
    console.print(f"  Answer commit layer:       {r.answer_commit_layer}")
    console.print(f"  Handoff gap:               {r.handoff_gap}")
    console.print(f"  Intermediate peak rank:    {r.intermediate_peak_rank}")
    console.print(f"  Answer peak rank:          {r.answer_peak_rank}")


def _print_batch(results):
    """Print batch summary table."""
    table = Table(title="Chain Trace — Batch Results")
    table.add_column("Prompt", max_width=50)
    table.add_column("Intermediate")
    table.add_column("Answer")
    table.add_column("Status", justify="center")
    table.add_column("Int Commit", justify="right")
    table.add_column("Ans Commit", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Int Peak", justify="right")
    table.add_column("Ans Peak", justify="right")

    for r in results:
        color = _status_color(r.chain_status)
        table.add_row(
            r.prompt[:50],
            r.intermediate,
            r.answer,
            f"[{color}]{r.chain_status}[/{color}]",
            str(r.intermediate_commit_layer) if r.intermediate_commit_layer is not None else "-",
            str(r.answer_commit_layer) if r.answer_commit_layer is not None else "-",
            str(r.handoff_gap) if r.handoff_gap is not None else "-",
            str(r.intermediate_peak_rank),
            str(r.answer_peak_rank),
        )

    console.print(table)

    # Summary stats
    statuses = [r.chain_status for r in results]
    for s in ["COMPLETE", "PARTIAL", "BROKEN", "SHORTCUT"]:
        count = statuses.count(s)
        if count:
            color = _status_color(s)
            console.print(f"  [{color}]{s}[/{color}]: {count}/{len(results)}")
