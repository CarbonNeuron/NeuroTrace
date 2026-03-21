"""Domain CLI commands: experiment, heatmap, commitment, contrast, attribute."""

import json

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.cli import _maybe_load_adapter, _resolve_device, console, err_console
from neurotrace.storage import TraceDB


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name or path.")
@click.option("--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877).")
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
    remote,
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
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if remote is not None:
        from neurotrace.remote import WorkerClient
        worker = WorkerClient(remote)
        health = worker.health()
        if model is None:
            model = health["model"]
        err_console.print(f"GPU: {health.get('device_name', 'unknown')} via {remote}")

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


@click.command()
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
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
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
    """Run heatmap via remote GPU worker using v2 inference primitives."""
    from neurotrace.heatmap import HeatmapCell, check_correct
    from neurotrace.remote import Hook, WorkerClient

    worker = WorkerClient(remote_url)
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

            # Baseline: forward pass with no hooks
            baseline = worker.forward(
                prompt, raw=True, top_k=1, seed=seed,
            )
            baseline_token = (
                baseline.top_tokens[0].token if baseline.top_tokens else ""
            )
            baseline_prob = (
                baseline.top_tokens[0].prob if baseline.top_tokens else 0.0
            )
            baseline_correct = check_correct(baseline_token, answer)
            progress.update(inner_task, completed=1)

            # Ablate each MLP layer
            for layer_idx in range(num_layers):
                hooks = [Hook(
                    layer=layer_idx, component="mlp", action="zero",
                )]
                result = worker.hooked(
                    prompt, hooks, raw=True, top_k=1, seed=seed,
                )
                token = (
                    result.top_tokens[0].token if result.top_tokens else ""
                )
                prob = (
                    result.top_tokens[0].prob if result.top_tokens else 0.0
                )
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
                        layer=layer_idx,
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
                progress.update(inner_task, completed=layer_idx + 2)

            progress.update(outer_task, advance=1)

    return cells


@click.command()
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
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
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
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url)
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

@click.command()
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
        from neurotrace.remote import WorkerClient

        worker = WorkerClient(remote)
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
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url)
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


@click.command()
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
            from neurotrace.remote import WorkerClient

            worker = WorkerClient(remote)
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
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url)
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
