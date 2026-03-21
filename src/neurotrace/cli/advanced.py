"""Advanced tracing CLI commands: token-trace, attention-trace, diagnose, decompose."""

import json

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.cli import _maybe_load_adapter, _resolve_device, console, err_console
from neurotrace.storage import TraceDB


@click.command("token-trace")
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
            f'\n[bold]Token-Trace:[/bold] "{r.prompt}" \u2192 {r.answer}\n'
        )

        # Table: tokens x layers
        table = Table(title="Answer projection per position \u00d7 layer")
        table.add_column("", style="dim")
        tokens_list: list[str] = []
        seen: set[int] = set()
        for e in r.entries:
            if e.position not in seen:
                tokens_list.append(e.token.strip()[:10] or "\u2581")
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
                    row.append("\u2014")
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
            f"{dataset_name or 'custom'} ({len(all_results)} prompts) \u00d7 "
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
            description=(
                f"Token-trace: {len(prompts)} prompts"
                f" x {len(layers)} layers"
            ),
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


@click.command("attention-trace")
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
            f'\n[bold]Attention-Trace:[/bold] "{r.prompt}" \u2192 {r.answer}\n'
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
                row.append(f"{np.mean(r_vals):+.4f}" if r_vals else "\u2014")
                row.append(f"{np.mean(v_vals):+.4f}" if v_vals else "\u2014")
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
                f" \u00d7 {len(layers)} layers"
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
    """Run attention-trace via remote GPU worker using v2 attention()."""
    from neurotrace.attention_trace import AttentionTraceEntry, AttentionTraceResult
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url)
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
        task = progress.add_task(
            f"Attention-trace: {len(prompts)} prompts",
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Attention-trace {i + 1}/{len(prompts)}...",
            )

            prompt_text = entry["prompt"]
            answer_text = entry["answer"]

            try:
                attn_result = worker.attention(
                    prompt_text, answer_text, seed=seed,
                )

                # Build AttentionTraceResult from per-head contributions
                entries = []
                for hc in attn_result.heads:
                    if hc.layer in layers:
                        entries.append(AttentionTraceEntry(
                            prompt=prompt_text,
                            layer=hc.layer,
                            head_idx=hc.head,
                            answer_projection=hc.logit_contribution,
                            magnitude=abs(hc.logit_contribution),
                        ))

                all_results.append(AttentionTraceResult(
                    prompt=prompt_text,
                    answer=answer_text,
                    entries=entries,
                ))
            except Exception:
                # Fallback to legacy attention_contributions_stream
                import base64

                import numpy as np

                from neurotrace.attention_trace import run_attention_trace_remote
                from neurotrace.models import get_lm_head_and_norm, load_model

                m, t = load_model(model_name, device="cpu")
                lm, _ = get_lm_head_and_norm(m)
                lm_w = lm.weight.data.cpu().float().numpy()

                layer_contributions: dict[int, np.ndarray] = {}
                for event in worker.attention_contributions_stream(
                    prompt_text, layers, seed=seed,
                ):
                    etype = event.get("type")
                    if etype == "layer-contributions":
                        layer_idx = event["layer"]
                        shape = event["shape"]
                        dt = (
                            np.float16
                            if event.get("dtype") == "float16"
                            else np.float32
                        )
                        arr = np.frombuffer(
                            base64.b64decode(event["contributions"]),
                            dtype=dt,
                        ).astype(np.float32).reshape(shape).copy()
                        layer_contributions[layer_idx] = arr

                result = run_attention_trace_remote(
                    layer_contributions=layer_contributions,
                    tokenizer=t,
                    prompt=prompt_text,
                    answer=answer_text,
                    layers=layers,
                    lm_head_weight=lm_w,
                )
                all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return model_name, all_results, layers


def _token_trace_remote(
    remote_url, prompts, layer_spec, commitment_data, seed, model_name_hint,
):
    """Run token-trace via remote GPU worker using v2 hooked()."""
    import base64

    import numpy as np

    from neurotrace.models import get_lm_head_and_norm, load_model
    from neurotrace.remote import WorkerClient
    from neurotrace.token_trace import run_token_trace_remote

    worker = WorkerClient(remote_url)
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
                f"Token-trace: {len(prompts)} prompts"
                f" x {len(layers)} layers"
            ),
            total=len(prompts),
        )

        for i, entry in enumerate(prompts):
            progress.update(
                task, completed=i,
                description=f"Token-trace {i + 1}/{len(prompts)}...",
            )

            prompt_text = entry["prompt"]

            # Fetch all-position MLP deltas via legacy stream
            # (v2 hooked() doesn't yet support all-position capture)
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


@click.command("diagnose")
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
    from neurotrace.remote import WorkerClient
    from neurotrace.token_trace import run_token_trace_remote

    worker = WorkerClient(remote_url)
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
@click.option("--raw", "use_raw", is_flag=True, default=None,
              help="Raw inference (no chat template). Default when --remote is used.")
@click.option("--chat", "use_chat", is_flag=True, default=False,
              help="Force chat template mode (override raw default for --remote).")
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
    use_raw,
    use_chat,
):
    """Logit Prism — exact additive logit attribution per component."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.decompose import (
        DecomposeRun,
        decompose_run_to_dict,
        generate_decompose_html,
        run_decompose_local,
    )

    if use_raw and use_chat:
        raise click.UsageError("Cannot use both --raw and --chat.")

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

    # Resolve raw mode: default True for --remote, False for local
    raw = use_raw if use_raw is not None else (remote is not None and not use_chat)

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
            remote, prompts, competitor_list, seed, raw=raw,
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
        err_str = "\u2713" if r.reconstruction_error < 0.01 else "\u26a0"
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


def _decompose_remote(remote_url, prompts, competitor_list, seed, raw=True):
    """Run decompose via remote GPU worker using v2 decompose()."""
    from neurotrace.decompose import (
        DecomposeResult as LocalDecomposeResult,
    )
    from neurotrace.decompose import (
        LayerContribution,
        TokenDecomposition,
    )
    from neurotrace.remote import WorkerClient

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

            # Get decomposition from worker
            decomp = worker.decompose(
                prompt_text, answer_text, raw=raw, seed=seed,
            )

            # Build answer decomposition from worker result
            a_layers = [
                LayerContribution(
                    layer=ld.layer,
                    attention=ld.attn_logit,
                    mlp=ld.mlp_logit,
                )
                for ld in decomp.layers
            ]
            a_decomp = TokenDecomposition(
                token_str=answer_text,
                token_id=decomp.answer_token_id,
                final_logit=decomp.total_logit,
                embedding=0.0,
                layers=a_layers,
                reconstruction_error=decomp.reconstruction_error,
                norm_scale=1.0,
            )

            # Build results for each competitor
            comps = competitor_list or []
            if not comps and decomp.competitors:
                comps = [c.token for c in decomp.competitors]
            if not comps:
                comps = ["the", "a", "is"]

            for comp_info in decomp.competitors:
                if competitor_list and comp_info.token not in competitor_list:
                    continue
                c_layers = [
                    LayerContribution(layer=ld.layer, attention=0.0, mlp=0.0)
                    for ld in decomp.layers
                ]
                c_decomp = TokenDecomposition(
                    token_str=comp_info.token,
                    token_id=0,
                    final_logit=comp_info.total_logit,
                    embedding=0.0,
                    layers=c_layers,
                    reconstruction_error=0.0,
                    norm_scale=1.0,
                )

                component_margins = [
                    {
                        "layer": ld.layer,
                        "attn_margin": ld.attn_logit,
                        "mlp_margin": ld.mlp_logit,
                    }
                    for ld in decomp.layers
                ]

                all_results.append(LocalDecomposeResult(
                    prompt=prompt_text,
                    answer=answer_text,
                    competitor=comp_info.token,
                    answer_logit=decomp.total_logit,
                    competitor_logit=comp_info.total_logit,
                    margin=decomp.total_logit - comp_info.total_logit,
                    embedding_margin=0.0,
                    component_margins=component_margins,
                    reconstruction_error=decomp.reconstruction_error,
                    answer_decomposition=a_decomp,
                    competitor_decomposition=c_decomp,
                ))

            # If no competitors matched from the worker, use fallback
            if not all_results or all_results[-1].prompt != prompt_text:
                # Fallback: use legacy stream endpoint
                from neurotrace.decompose import run_decompose_remote as _legacy
                comps_to_try = competitor_list or ["the", "a", "is"]
                tokens = [answer_text] + comps_to_try
                remote_data = {}
                for event in worker.decompose_stream(
                    prompt_text, tokens, seed=seed, raw=raw,
                ):
                    etype = event.get("type")
                    if etype == "decomposition":
                        remote_data = event.get("decompositions", {})
                for comp in comps_to_try:
                    if comp in remote_data and answer_text in remote_data:
                        result = _legacy(
                            remote_data, prompt_text, answer_text, comp,
                        )
                        all_results.append(result)

        progress.update(task, description="Done.", completed=len(prompts))

    return all_results
