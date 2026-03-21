"""Analysis CLI commands: ablate, sweep, scan, finetune, neurons, probe, circuit."""

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


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name or path.")
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
@click.option("--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877).")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--raw", "use_raw", is_flag=True, default=None,
              help="Raw inference (no chat template). Default when --remote is used.")
@click.option("--chat", "use_chat", is_flag=True, default=False,
              help="Force chat template mode (override raw default for --remote).")
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
    remote,
    device,
    use_raw,
    use_chat,
):
    """Run inference with targeted components disabled and compare to baseline."""
    if use_raw and use_chat:
        raise click.UsageError("Cannot use both --raw and --chat.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if remote is not None:
        from neurotrace.remote import WorkerClient
        worker = WorkerClient(remote)
        health = worker.health()
        model = health["model"]
        err_console.print(
            f"[dim]Note: ablate uses local model loading."
            f" Model resolved from worker: {model}[/dim]"
        )

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


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name or path.")
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
@click.option("--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877).")
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
    remote,
    device,
):
    """Run multiple ablations in a single model load, sweeping a parameter range."""
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if remote is not None:
        from neurotrace.remote import WorkerClient
        worker = WorkerClient(remote)
        health = worker.health()
        model = health["model"]
        err_console.print(
            f"[dim]Note: sweep uses local model loading."
            f" Model resolved from worker: {model}[/dim]"
        )

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


def _scan_remote(
    remote_url, dataset, dataset_name, seed,
    sabotage_threshold, final_threshold,
    save_traces, save_flagged, details, output_json, db,
    raw=True,
):
    """Run scan via remote GPU worker."""
    from neurotrace.remote import WorkerClient
    from neurotrace.scan import PromptResult, ScanResult

    worker = WorkerClient(remote_url)
    health = worker.health()
    model_name = health["model"]
    device_name = health.get("device_name", health.get("device", "unknown"))
    err_console.print(f"GPU: {device_name} via {remote_url}")

    prompt_results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Scanning...", total=len(dataset))

        for i, entry in enumerate(dataset):
            desc = (
                f"Scanning {i+1}/{len(dataset)}:"
                f" {entry['prompt'][:40]}..."
            )
            progress.update(task, completed=i, description=desc)

            trace_data = worker.trace(entry["prompt"], seed=seed, top_k=5, raw=raw)

            # Extract per-layer predictions for the expected answer
            answer = entry["answer"]
            ranks = []
            probs = []

            for layer_data in trace_data["layers"]:
                top_tokens = layer_data["top_tokens"]
                # Find answer rank and prob
                found = False
                for rank_idx, tt in enumerate(top_tokens):
                    token_text = tt["token"].strip().lstrip("\u2581").lower()
                    if answer.strip().lower().startswith(token_text) and token_text:
                        ranks.append(rank_idx + 1)
                        probs.append(tt["prob"])
                        found = True
                        break
                if not found:
                    ranks.append(999)
                    probs.append(0.0)

            # Final prediction
            final_token = trace_data["final_token"]
            final_prob = trace_data["final_prob"]

            # Find final rank
            final_layers = trace_data["layers"]
            if final_layers:
                final_rank = 1  # It's the final prediction
            else:
                final_rank = 999

            # Compute peak prob and layer
            peak_prob = max(probs) if probs else 0.0
            peak_layer = probs.index(peak_prob) if probs and peak_prob > 0 else None

            # Compute commitment layer (first layer where answer prob > threshold)
            commitment_layer = None
            for li, p in enumerate(probs):
                if p > 0.1:
                    commitment_layer = li
                    break

            # Detect sabotage layers
            sabotage_layers = []
            if peak_prob > 0:
                for li, p in enumerate(probs):
                    if li > 0 and peak_layer is not None and li > peak_layer:
                        drop = (peak_prob - p) / peak_prob
                        if drop >= sabotage_threshold:
                            sabotage_layers.append(li)

            # Classify
            final_token_clean = final_token.strip().lstrip("\u2581").lower()
            answer_clean = answer.strip().lower()

            flags = []
            if answer_clean.startswith(final_token_clean) and final_token_clean:
                if sabotage_layers:
                    status = "sabotaged"
                    flags.append("sabotage_detected")
                elif final_prob < final_threshold:
                    status = "weak"
                    flags.append("weak_confidence")
                else:
                    status = "correct"
            else:
                if sabotage_layers:
                    status = "sabotaged"
                    flags.append("sabotage_detected")
                    flags.append("wrong_final")
                else:
                    status = "wrong"
                    flags.append("wrong_final")

            prompt_results.append(PromptResult(
                prompt=entry["prompt"],
                answer=answer,
                final_token=final_token,
                final_prob=final_prob,
                final_rank=final_rank,
                peak_prob=peak_prob,
                peak_layer=peak_layer,
                commitment_layer=commitment_layer,
                sabotage_layers=sabotage_layers,
                flags=flags,
                status=status,
                ranks=ranks,
                probs=probs,
            ))

        progress.update(task, description="Done.", completed=len(dataset))

    return ScanResult(
        model_name=model_name,
        dataset_name=dataset_name,
        prompt_results=prompt_results,
    )


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name or path.")
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
@click.option("--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877).")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--raw", "use_raw", is_flag=True, default=None,
              help="Raw inference (no chat template). Default when --remote is used.")
@click.option("--chat", "use_chat", is_flag=True, default=False,
              help="Force chat template mode (override raw default for --remote).")
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
    remote,
    device,
    use_raw,
    use_chat,
):
    """Scan a dataset for sabotaged predictions."""
    from neurotrace.datasets import get_builtin_dataset, load_dataset
    from neurotrace.scan import run_scan

    if use_raw and use_chat:
        raise click.UsageError("Cannot use both --raw and --chat.")
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Resolve raw mode: default True for --remote, False for local
    raw = use_raw if use_raw is not None else (remote is not None and not use_chat)

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

    if remote is not None:
        db_conn = None
        if save_traces or save_flagged:
            db_conn = TraceDB(db)
        scan_result = _scan_remote(
            remote, dataset, dataset_name, seed, sabotage_threshold, final_threshold,
            save_traces, save_flagged, details, output_json, db_conn, raw=raw,
        )
        if db_conn:
            db_conn.close()
    else:
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


@click.command()
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


@click.command()
@click.option("--db", required=True, help="Path to DuckDB database file.")
@click.option("--model", default=None, help="HuggingFace model name or path.")
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
@click.option("--remote", default=None, help="GPU worker URL (e.g., http://172.30.0.1:8877).")
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
    remote,
    device,
):
    """Neuron-level MLP attribution: profile or ablate individual neurons."""
    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    if remote is not None:
        from neurotrace.remote import WorkerClient
        worker = WorkerClient(remote)
        health = worker.health()
        model = health["model"]
        err_console.print(
            f"[dim]Note: neurons uses local model loading."
            f" Model resolved from worker: {model}[/dim]"
        )

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


@click.command()
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


@click.command()
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
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote, timeout=600.0)
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


@click.command("probe-universal")
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

    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url)
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
