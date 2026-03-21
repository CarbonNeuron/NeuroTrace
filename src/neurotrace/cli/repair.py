"""Repair CLI commands: fingerprint, repair."""

import json

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.cli import _maybe_load_adapter, _resolve_device, console, err_console
from neurotrace.storage import TraceDB


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option(
    "--dataset-builtin", default=None,
    help="Built-in dataset name for fingerprinting.",
)
@click.option(
    "--dataset", "dataset_path", default=None,
    type=click.Path(exists=True),
    help="Custom JSONL dataset path.",
)
@click.option("--list", "list_runs", is_flag=True, help="Show fingerprint stats.")
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
def fingerprint(
    db,
    model,
    dataset_builtin,
    dataset_path,
    list_runs,
    remote,
    device,
    adapter,
    seed,
    output_json,
):
    """Build MLP key vector fingerprints for analytical regression checking."""
    import uuid

    from neurotrace.fingerprint import (
        FingerprintRun,
        compute_alignment_stats,
        compute_fingerprint_local,
        serialize_f16_tensor,
    )

    db_conn = TraceDB(db)

    # Handle --list
    if list_runs:
        runs = db_conn.list_fingerprint_runs()
        if not runs:
            console.print("[dim]No fingerprint runs found.[/dim]")
            db_conn.close()
            return

        if output_json:
            click.echo(json.dumps(runs, indent=2, default=str))
            db_conn.close()
            return

        table = Table()
        table.add_column("Run ID", max_width=12)
        table.add_column("Dataset")
        table.add_column("Model", max_width=30)
        table.add_column("Prompts", justify="right")
        table.add_column("Created")
        for run in runs:
            table.add_row(
                str(run["id"])[:12],
                run["dataset"],
                str(run["model_name"])[:30],
                str(run["prompt_count"]),
                str(run["created_at"]),
            )
        console.print(table)
        db_conn.close()
        return

    # Validate inputs
    if dataset_builtin is None and dataset_path is None:
        raise click.UsageError(
            "Must provide --dataset-builtin or --dataset."
        )
    if dataset_builtin is not None and dataset_path is not None:
        raise click.UsageError(
            "Cannot provide both --dataset-builtin and --dataset."
        )
    if remote is None and model is None:
        raise click.UsageError(
            "Must provide --model (local mode) or --remote."
        )

    # Load dataset
    from neurotrace.datasets import get_builtin_dataset, load_dataset

    if dataset_builtin is not None:
        dataset = get_builtin_dataset(dataset_builtin)
        dataset_name = dataset_builtin
    else:
        dataset = load_dataset(dataset_path)
        dataset_name = dataset_path

    prompts = [{"prompt": d["prompt"], "answer": d["answer"]} for d in dataset]
    run_id = str(uuid.uuid4())
    all_fingerprints = []

    if remote is not None:
        all_fingerprints = _fingerprint_remote(
            remote, prompts, seed, dataset_name,
        )
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
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
                description=f"Fingerprinting {len(prompts)} prompts...",
                total=len(prompts),
            )

            for p_idx, entry in enumerate(prompts):
                progress.update(
                    task,
                    completed=p_idx,
                    description=(
                        f"Fingerprint {p_idx + 1}/{len(prompts)}"
                    ),
                )
                fp = compute_fingerprint_local(
                    model_obj, tokenizer,
                    entry["prompt"], entry["answer"],
                    seed=seed,
                )
                all_fingerprints.append(fp)

            progress.update(
                task, description="Done.", completed=len(prompts),
            )

    # Save to DB
    db_conn.write_fingerprint_run(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        prompt_count=len(all_fingerprints),
    )
    for fp in all_fingerprints:
        db_conn.write_fingerprint(
            run_id=run_id,
            prompt=fp.prompt,
            answer=fp.answer,
            competitor=fp.competitor,
            answer_logit=fp.answer_logit,
            competitor_logit=fp.competitor_logit,
            margin=fp.margin,
            key_vectors_blob=serialize_f16_tensor(fp.key_vectors),
            p_answer_blob=serialize_f16_tensor(fp.p_answer),
            p_competitor_blob=serialize_f16_tensor(fp.p_competitor),
        )
    db_conn.close()

    # Build run object for output
    run = FingerprintRun(
        run_id=run_id,
        dataset=dataset_name,
        model_name=model_name,
        prompt_count=len(all_fingerprints),
        fingerprints=all_fingerprints,
    )

    if output_json:
        out = {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "model_name": run.model_name,
            "prompt_count": run.prompt_count,
            "fingerprints": [
                {
                    "prompt": fp.prompt,
                    "answer": fp.answer,
                    "competitor": fp.competitor,
                    "answer_logit": fp.answer_logit,
                    "competitor_logit": fp.competitor_logit,
                    "margin": fp.margin,
                }
                for fp in all_fingerprints
            ],
        }
        click.echo(json.dumps(out, indent=2))
        return

    # Compute alignment stats
    stats = compute_alignment_stats(all_fingerprints)

    # Compute total storage size
    total_bytes = 0
    for fp in all_fingerprints:
        total_bytes += fp.key_vectors.nbytes
        total_bytes += fp.p_answer.nbytes
        total_bytes += fp.p_competitor.nbytes

    if total_bytes >= 1024 * 1024:
        size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{total_bytes / 1024:.1f} KB"

    kv_shape = all_fingerprints[0].key_vectors.shape if all_fingerprints else (0, 0)
    p_shape = all_fingerprints[0].p_answer.shape if all_fingerprints else (0,)

    console.print(
        f"\n[bold]Fingerprint:[/bold] {dataset_name}"
        f" ({len(all_fingerprints)} prompts)"
    )
    if remote:
        console.print(f"GPU: via {remote}")
    console.print(f"\nStored {len(all_fingerprints)} fingerprints ({size_str})")
    console.print(
        f"  Key vectors: {kv_shape[0]} layers"
        f" x {kv_shape[1] if len(kv_shape) > 1 else '?'} dims x float16"
    )
    console.print(
        f"  Projection vectors: {p_shape[0]} dims x float16"
    )

    if stats.get("max_pair"):
        console.print("\nAlignment analysis:")
        console.print(
            f"  Avg cross-prompt alignment:"
            f" {stats['avg_alignment']:.2f}"
        )
        max_p = stats["max_pair"]
        min_p = stats["min_pair"]
        console.print(
            f"  Max alignment pair:"
            f" {max_p[0]}↔{max_p[1]}"
            f" ({stats['max_alignment']:.2f})"
        )
        console.print(
            f"  Min alignment pair:"
            f" {min_p[0]}↔{min_p[1]}"
            f" ({stats['min_alignment']:.2f})"
        )
    console.print()


def _fingerprint_remote(remote_url, prompts, seed, dataset_name):
    """Run fingerprinting via remote GPU worker."""
    from neurotrace.fingerprint import build_fingerprint_from_remote
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url)
    health = worker.health()
    device_name = health.get("device_name", health.get("device", "unknown"))
    err_console.print(f"GPU: {device_name} via {remote_url}")

    all_fingerprints = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task(
            "Fingerprinting...", total=len(prompts),
        )

        for p_idx, entry in enumerate(prompts):
            progress.update(
                task,
                completed=p_idx,
                description=f"Fingerprint {p_idx + 1}/{len(prompts)}",
            )

            for event in worker.fingerprint_stream([entry], seed):
                etype = event.get("type")
                if etype == "result":
                    for fp_data in event.get("fingerprints", []):
                        fp = build_fingerprint_from_remote(fp_data)
                        all_fingerprints.append(fp)

        progress.update(
            task, description="Done.", completed=len(prompts),
        )

    return all_fingerprints


@click.command()
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
@click.option(
    "--fast-regression", is_flag=True,
    help="Use fingerprints for analytical regression checking (no forward passes).",
)
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
    fast_regression,
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
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
        result = worker.repair_undo()
        console.print(f"[green]Undo: {result}[/green]")
        return

    # Handle --save
    if save_path is not None:
        if remote is None:
            raise click.UsageError("--save requires --remote.")
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
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
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
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

    # Fast regression checking via fingerprints
    if fast_regression and all_results:
        from neurotrace.fingerprint import (
            check_regressions_fast,
        )
        from neurotrace.repair import RegressionResult

        db_conn_fp = TraceDB(db)
        try:
            latest_fp_run = db_conn_fp.get_latest_fingerprint_run_id()
            fps = db_conn_fp.load_fingerprints(latest_fp_run)
        except ValueError:
            fps = []
            err_console.print(
                "[yellow]No fingerprints found — run"
                " 'neurotrace fingerprint' first.[/yellow]"
            )
        db_conn_fp.close()

        if fps:
            err_console.print(
                f"[dim]Analytical regression check:"
                f" {len(fps)} fingerprints[/dim]"
            )
            for r in all_results:
                if r.status == "skipped":
                    continue
                # We need k_star and delta — extract from repair.py
                # For local mode, re-compute them
                if remote is None:
                    from neurotrace.repair import (
                        compute_correction_delta,
                        compute_key_vector,
                    )

                    k_star = compute_key_vector(
                        model_obj, tokenizer, r.prompt,
                        r.target_layer, seed,
                    )
                    delta = compute_correction_delta(
                        model_obj, tokenizer, r.answer,
                        r.competitor,
                        target_margin - r.before.component_margin,
                    )
                    k_np = k_star.detach().cpu().numpy()
                    d_np = delta.detach().cpu().numpy()

                    affected = check_regressions_fast(
                        d_np, k_np, r.target_layer, fps,
                    )

                    for ar in affected:
                        status = "regression" if ar.regression else "ok"
                        r.regressions.append(RegressionResult(
                            prompt=ar.prompt,
                            answer=ar.answer,
                            before_prob=ar.current_margin,
                            after_prob=ar.new_margin,
                            status=status,
                        ))

                    if any(ar.regression for ar in affected):
                        r.status = "regression"

            err_console.print(
                "[dim]  (analytical, fingerprint-based,"
                " no forward pass)[/dim]"
            )

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
    """Run repair via remote GPU worker using v2 rome_edit()."""
    from neurotrace.remote import WorkerClient
    from neurotrace.repair import (
        RepairAfter,
        RepairBefore,
        RepairEdit,
        RepairResult,
        build_repair_result_from_remote,
    )

    worker = WorkerClient(remote_url)
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

            prompt_text = entry["prompt"]
            answer_text = entry["answer"]
            layer = target_layer

            # Auto-detect layer via decompose if not specified
            if layer is None:
                try:
                    decomp = worker.decompose(prompt_text, answer_text, seed=seed)
                    # Find worst layer (most negative margin)
                    worst = min(decomp.layers, key=lambda ld: ld.mlp_logit)
                    layer = worst.layer
                except Exception:
                    layer = health["num_layers"] // 2  # fallback to middle

            # Extract subject from prompt (heuristic: last noun-like word before verb)
            subject = _extract_subject(prompt_text)

            try:
                result = worker.rome_edit(
                    prompt_text, subject, answer_text, layer, seed=seed,
                )

                before = RepairBefore(
                    answer_logit=0.0,
                    competitor_logit=0.0,
                    margin=result.pre_margin,
                    component_margin=result.pre_margin,
                    answer_prob=result.pre_prob,
                )
                after = RepairAfter(
                    answer_logit=0.0,
                    competitor_logit=0.0,
                    margin=result.post_margin,
                    component_margin=result.post_margin,
                    answer_prob=result.post_prob,
                )
                edit = RepairEdit(
                    matrix=f"L{layer}.mlp",
                    layer=layer,
                    rank=1,
                    norm=0.0,
                )
                status = "repaired" if result.success else "skipped"

                all_results.append(RepairResult(
                    prompt=prompt_text,
                    answer=answer_text,
                    competitor=competitor or "",
                    target_layer=layer,
                    target_component=target_component,
                    before=before,
                    after=after,
                    edit=edit,
                    regressions=[],
                    status=status,
                ))
            except Exception:
                # Fallback to legacy repair_stream
                remote_data = None
                for event in worker.repair_stream(
                    prompt=prompt_text,
                    answer=answer_text,
                    competitor=competitor,
                    target_layer=target_layer,
                    target_component=target_component,
                    target_margin=target_margin,
                    verify_prompts=verify_prompts,
                    seed=seed,
                ):
                    etype = event.get("type")
                    if etype == "result":
                        remote_data = event
                if remote_data:
                    result = build_repair_result_from_remote(remote_data)
                    all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return all_results


def _extract_subject(prompt: str) -> str:
    """Extract subject entity from a prompt heuristic."""
    # Common patterns: "The capital of X is", "The X of Y is"
    # Try to find the last capitalized word or proper noun
    words = prompt.split()
    # Look for pattern "of X is" or "of X"
    for i, w in enumerate(words):
        if w.lower() == "of" and i + 1 < len(words):
            # Take the next word(s) until a stop word
            subject_parts = []
            for j in range(i + 1, len(words)):
                if words[j].lower() in ("is", "are", "was", "were", "has", "have"):
                    break
                subject_parts.append(words[j].rstrip(".,;:"))
            if subject_parts:
                return " ".join(subject_parts)
    # Fallback: return the prompt itself
    return prompt
