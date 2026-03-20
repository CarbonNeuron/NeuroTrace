"""Bench CLI command."""

import json

import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from neurotrace.cli import _maybe_load_adapter, _resolve_device, console, err_console
from neurotrace.storage import TraceDB


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option(
    "--datasets", "dataset_list", default=None,
    help="Comma-separated dataset names (default: all).",
)
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--html", "html_path", default=None, help="HTML report output path.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option(
    "--ppl-samples", default=100, type=int,
    help="Max samples for perplexity computation.",
)
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
def bench(
    db,
    model,
    dataset_list,
    remote,
    device,
    html_path,
    output_json,
    ppl_samples,
    seed,
    adapter,
):
    """Run benchmark: baseline, per-domain repair, combined."""
    from neurotrace.bench import (
        bench_run_to_dict,
        generate_bench_html,
        run_bench_local,
        run_bench_remote,
    )
    from neurotrace.datasets import list_builtin_datasets

    # Determine datasets
    if dataset_list:
        dataset_names = [d.strip() for d in dataset_list.split(",")]
    else:
        dataset_names = list_builtin_datasets()

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Starting benchmark...", total=None)

        def progress_cb(phase, msg):
            progress.update(task, description=msg)

        if remote is not None:
            from neurotrace.remote import RemoteWorker

            worker = RemoteWorker(remote)
            health = worker.health()
            gpu_info = health.get("device_name", health.get("device", "unknown"))
            err_console.print(f"GPU: {gpu_info} via {remote}")

            bench_result = run_bench_remote(
                remote, dataset_names,
                seed=seed, ppl_samples=ppl_samples,
                progress_callback=progress_cb,
            )
        else:
            from neurotrace.models import load_model

            device = _resolve_device(device)
            progress.update(task, description="Loading model...")
            model_obj, tokenizer = load_model(model, device=device)
            model_obj = _maybe_load_adapter(model_obj, adapter)

            bench_result = run_bench_local(
                model_obj, tokenizer, dataset_names,
                seed=seed, ppl_samples=ppl_samples,
                progress_callback=progress_cb,
            )

        progress.update(task, description="Saving results...")

    # Save to DB
    db_conn = TraceDB(db)
    db_conn.write_bench_run(
        run_id=bench_result.run_id,
        model_name=bench_result.model_name,
        datasets=",".join(bench_result.datasets),
        total_edits=bench_result.total_edits,
        baseline_ppl=bench_result.baseline_ppl,
        combined_ppl=bench_result.combined_ppl,
    )
    for dr in bench_result.domain_results:
        db_conn.write_bench_domain_result(
            run_id=bench_result.run_id,
            dataset=dr.dataset,
            baseline_correct=dr.baseline_correct,
            baseline_total=dr.baseline_total,
            repaired_correct=dr.repaired_correct,
            repaired_total=dr.repaired_total,
            baseline_ppl=dr.baseline_ppl,
            repaired_ppl=dr.repaired_ppl,
            edit_count=dr.edit_count,
            total_edit_norm=dr.total_edit_norm,
        )
        for pr in dr.prompt_results:
            db_conn.write_bench_prompt_result(
                run_id=bench_result.run_id,
                dataset=dr.dataset,
                prompt=pr.prompt,
                answer=pr.answer,
                baseline_correct=pr.baseline_correct,
                repaired_correct=pr.repaired_correct,
                before_margin=pr.before_margin,
                after_margin=pr.after_margin,
                before_prob=pr.before_prob,
                after_prob=pr.after_prob,
                layer=pr.layer,
                component=pr.component,
                edit_norm=pr.edit_norm,
            )
    db_conn.close()

    # Output
    if output_json:
        click.echo(json.dumps(bench_run_to_dict(bench_result), indent=2, default=str))
        return

    if html_path:
        html = generate_bench_html(bench_result)
        with open(html_path, "w") as f:
            f.write(html)
        console.print(f"\nReport saved to {html_path}")

    # Console output
    console.print("\n[bold]NeuroTrace Benchmark[/bold]")
    console.print(f"Model: {bench_result.model_name}\n")

    console.print("[bold]Phase 1: Baseline[/bold]")
    for dr in bench_result.domain_results:
        console.print(
            f" {dr.dataset:20s} {dr.baseline_correct}/{dr.baseline_total}"
            f" ({dr.baseline_acc:.1%})"
        )
    console.print(f" {'Perplexity':20s} {bench_result.baseline_ppl:.2f}\n")

    console.print("[bold]Phase 2: Per-Domain Repair[/bold]")
    for dr in bench_result.domain_results:
        ppl_change = dr.repaired_ppl - dr.baseline_ppl
        console.print(
            f" {dr.dataset:20s} -> {dr.repaired_correct}/{dr.repaired_total}"
            f" ({dr.repaired_acc:.1%})"
            f"  {dr.acc_delta:+.1%}"
            f"  PPL: {dr.repaired_ppl:.2f} ({ppl_change:+.2f})"
            f"  edits: {dr.edit_count}"
            f"  norm: {dr.total_edit_norm:.2f}"
        )

    if bench_result.combined_domain_results:
        console.print("\n[bold]Phase 3: Combined (all domains)[/bold]")
        for cdr in bench_result.combined_domain_results:
            console.print(
                f" {cdr.dataset:20s} {cdr.repaired_correct}/{cdr.repaired_total}"
                f" ({cdr.repaired_acc:.1%})"
            )
        ppl_change = bench_result.combined_ppl - bench_result.baseline_ppl
        console.print(
            f" {'Perplexity':20s} {bench_result.combined_ppl:.2f}"
            f" ({ppl_change:+.2f})"
        )
        console.print(
            f" Total edits: {bench_result.total_edits}"
            f"  Total norm: {bench_result.total_edit_norm:.2f}"
        )
