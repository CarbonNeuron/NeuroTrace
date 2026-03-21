"""Discover CLI command: automated knowledge gap detection."""

import json
import os

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.cli import _maybe_load_adapter, _resolve_device, console, err_console


@click.command()
@click.option(
    "--topic", default=None,
    help="Built-in topic: world_capitals, chemical_elements, "
    "country_currencies, country_languages, planet_properties.",
)
@click.option("--source", default="wikidata", help="wikidata or file.")
@click.option(
    "--sparql-file", default=None, type=click.Path(exists=True),
    help="Custom SPARQL query file or JSONL facts file.",
)
@click.option(
    "--template", default=None,
    help="Custom prompt template (requires --answer-field).",
)
@click.option(
    "--answer-field", default=None,
    help="Field name for answer in template.",
)
@click.option("--limit", default=100, type=int, help="Max facts.")
@click.option("--db", required=True, help="DuckDB database path.")
@click.option(
    "--model", "model_name", default=None,
    help="HuggingFace model name (local mode).",
)
@click.option("--remote", default=None, help="GPU worker URL.")
@click.option(
    "--heal", "do_heal", is_flag=True,
    help="Auto-heal discovered failures.",
)
@click.option(
    "--fingerprint", "fingerprint_path", default=None,
    type=click.Path(),
    help="Fingerprint cache for regression checking.",
)
@click.option(
    "--regression-threshold", default=0.3, type=float,
    help="Cosine similarity threshold.",
)
@click.option(
    "--max-edits", default=50, type=int,
    help="Max ROME edits before stopping.",
)
@click.option(
    "--ppl-threshold", default=5.0, type=float,
    help="Abort if perplexity exceeds this.",
)
@click.option(
    "--save", default=None, type=click.Path(),
    help="Save healed model weights.",
)
@click.option(
    "--dry-run", is_flag=True, help="Scan only, no healing.",
)
@click.option("--json", "output_json", is_flag=True, help="JSON output format.")
@click.option("--report", "report_path", default=None, help="HTML report path.")
@click.option("--upload", is_flag=True, help="Upload report to CarbonFiles.")
@click.option("--verbose", is_flag=True, help="Show per-prompt details.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--raw", "use_raw", is_flag=True, default=None,
              help="Raw inference (no chat template). Default when --remote is used.")
@click.option("--chat", "use_chat", is_flag=True, default=False,
              help="Force chat template mode (override raw default for --remote).")
@click.pass_context
def discover(
    ctx,
    topic,
    source,
    sparql_file,
    template,
    answer_field,
    limit,
    db,
    model_name,
    remote,
    do_heal,
    fingerprint_path,
    regression_threshold,
    max_edits,
    ppl_threshold,
    save,
    dry_run,
    output_json,
    report_path,
    upload,
    verbose,
    device,
    adapter,
    seed,
    use_raw,
    use_chat,
):
    """Discover knowledge gaps via structured fact extraction and scanning."""
    from neurotrace.discover import (
        expand_custom_template,
        fetch_topic_facts,
        load_facts_from_file,
        query_wikidata,
        run_discover,
    )

    if use_raw and use_chat:
        raise click.UsageError("Cannot use both --raw and --chat.")

    # Use model from group context if not specified
    if model_name is None:
        model_name = ctx.obj.get("model") if ctx.obj else None

    if remote is None and model_name is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Resolve raw mode: default True for --remote, False for local
    raw = use_raw if use_raw is not None else (remote is not None and not use_chat)

    # Step 1: Get facts
    facts: list[dict] = []

    if source == "file":
        if sparql_file is None:
            raise click.UsageError("--sparql-file required when --source is file.")
        facts = load_facts_from_file(sparql_file)
        err_console.print(f"Loaded {len(facts)} facts from {sparql_file}")
    elif source == "wikidata":
        if sparql_file is not None:
            # Custom SPARQL query
            with open(sparql_file) as f:
                custom_sparql = f.read()
            if template is None or answer_field is None:
                raise click.UsageError(
                    "Custom SPARQL requires --template and --answer-field."
                )
            custom_sparql = custom_sparql.replace("{limit}", str(limit))
            err_console.print("Querying Wikidata with custom SPARQL...")
            bindings = query_wikidata(custom_sparql)
            facts = expand_custom_template(bindings, template, answer_field)
            err_console.print(f"Got {len(facts)} facts from Wikidata")
        elif topic is not None:
            err_console.print(f"Querying Wikidata for {topic}...")
            facts = fetch_topic_facts(topic, limit=limit)
            err_console.print(f"Got {len(facts)} facts")
        else:
            raise click.UsageError(
                "Must provide --topic or --sparql-file for wikidata source."
            )
    else:
        raise click.UsageError(f"Unknown source: {source!r}. Use 'wikidata' or 'file'.")

    if not facts:
        err_console.print("[yellow]No facts found.[/yellow]")
        return

    if topic is None:
        topic = "custom"

    # Step 2: Run discovery pipeline
    if remote is not None:
        _discover_remote(
            remote, facts, topic, source, db, do_heal, fingerprint_path,
            regression_threshold, max_edits, save, dry_run,
            output_json, report_path, upload, verbose, seed, raw=raw,
        )
        return

    # Local mode
    from neurotrace.models import load_model

    device = _resolve_device(device)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        model_obj, tokenizer = load_model(model_name, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)

        def progress_cb(phase, msg):
            progress.update(task, description=msg)
            if verbose:
                err_console.print(f"[dim]{phase}: {msg}[/dim]")

        progress.update(task, description="Running discovery pipeline...")

        result = run_discover(
            model_obj,
            tokenizer,
            facts,
            topic,
            source,
            db_path=db,
            heal=do_heal,
            fingerprint_path=fingerprint_path,
            regression_threshold=regression_threshold,
            max_edits=max_edits,
            dry_run=dry_run,
            save_path=save,
            seed=seed,
            progress_callback=progress_cb,
        )

        progress.update(task, description="Done.")

    _output_result(
        result, output_json, report_path, upload, verbose,
    )


def _discover_remote(
    remote_url, facts, topic, source, db, do_heal, fingerprint_path,
    regression_threshold, max_edits, save, dry_run,
    output_json, report_path, upload, verbose, seed, raw=True,
):
    """Run discover pipeline via remote GPU worker."""
    import time
    import uuid

    from neurotrace.discover import (
        DiscoveryFact,
        DiscoveryResult,
        ensure_discoveries_table,
        insert_discovery,
    )
    from neurotrace.remote import WorkerClient

    worker = WorkerClient(remote_url, timeout=600.0)
    health = worker.health()
    model_name = health["model"]
    err_console.print(f"GPU: {health.get('device_name', 'unknown')} via {remote_url}")

    start_time = time.time()
    ensure_discoveries_table(db)

    discovery_facts: list[DiscoveryFact] = []
    baseline_correct = 0
    healed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Scanning...", total=None)

        # Baseline scan using forward() with v2 primitives
        for i, entry in enumerate(facts):
            progress.update(
                task,
                description=f"Scan {i + 1}/{len(facts)}: {entry['prompt'][:30]}",
            )

            result = worker.forward(
                entry["prompt"], raw=raw, top_k=5, seed=seed,
            )
            final_token = result.top_tokens[0].token if result.top_tokens else ""
            prob = result.top_tokens[0].prob if result.top_tokens else 0.0
            final_clean = final_token.strip().lstrip("\u2581").lower()
            answer_clean = entry["answer"].strip().lower()
            is_correct = answer_clean.startswith(final_clean) and bool(final_clean)

            fact = DiscoveryFact(
                id=str(uuid.uuid4())[:8],
                prompt=entry["prompt"],
                expected_answer=entry["answer"],
                topic=topic,
                source=source,
                baseline_correct=is_correct,
                baseline_prob=prob,
            )
            discovery_facts.append(fact)
            if is_correct:
                baseline_correct += 1

        # Heal if requested
        if do_heal and not dry_run:
            from neurotrace.cli.repair import _extract_subject

            failures = [f for f in discovery_facts if not f.baseline_correct]
            edits_done = 0

            for i, fact in enumerate(failures):
                if edits_done >= max_edits:
                    break

                progress.update(
                    task,
                    description=f"Heal {i + 1}/{len(failures)}: {fact.prompt[:30]}",
                )

                try:
                    subject = _extract_subject(fact.prompt)
                    layer = health["num_layers"] // 2
                    edit_result = worker.rome_edit(
                        fact.prompt, subject, fact.expected_answer,
                        layer, raw=raw, seed=seed,
                    )
                    if edit_result.success:
                        fact.healed = True
                        fact.healed_prob = edit_result.post_prob
                        healed_count += 1
                        edits_done += 1
                except Exception:
                    pass

            if save and edits_done > 0:
                try:
                    worker.repair_save(save)
                except Exception as e:
                    err_console.print(f"[red]Save failed: {e}[/red]")

        progress.update(task, description="Done.")

    # Store in DB
    for fact in discovery_facts:
        insert_discovery(db, fact)

    duration = time.time() - start_time

    result = DiscoveryResult(
        run_id=str(uuid.uuid4())[:8],
        topic=topic,
        source=source,
        model_name=model_name,
        total_facts=len(facts),
        baseline_correct=baseline_correct,
        baseline_wrong=len(facts) - baseline_correct,
        healed_count=healed_count,
        regression_count=0,
        facts=discovery_facts,
        duration_seconds=duration,
    )

    _output_result(result, output_json, report_path, upload, verbose)


def _output_result(result, output_json, report_path, upload, verbose):
    """Output discovery result to console, JSON, or HTML."""
    from neurotrace.discover import (
        discovery_result_to_dict,
        generate_discover_html,
    )

    # HTML report
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        html = generate_discover_html(result)
        with open(report_path, "w") as f:
            f.write(html)
        err_console.print(f"[green]Report saved to {report_path}[/green]")

        if upload:
            try:
                from neurotrace.upload import upload_report

                url = upload_report(report_path)
                err_console.print(f"[green]Uploaded: {url}[/green]")
            except Exception as e:
                err_console.print(f"[red]Upload failed: {e}[/red]")

    # JSON output
    if output_json:
        click.echo(json.dumps(discovery_result_to_dict(result), indent=2))
        return

    # Console output
    total = result.total_facts
    baseline_acc = result.baseline_correct / total if total > 0 else 0.0
    healed_total = result.baseline_correct + result.healed_count
    healed_acc = healed_total / total if total > 0 else 0.0

    console.print(f"\n[bold]Discover:[/bold] {result.topic} ({total} facts)")
    console.print(f"  Baseline: {result.baseline_correct}/{total} ({baseline_acc:.1%})")

    if result.healed_count > 0:
        console.print(f"  Healed:   {healed_total}/{total} ({healed_acc:.1%})")
        console.print(f"  Regressions: {result.regression_count}")

    console.print(f"  Duration: {result.duration_seconds:.1f}s")
    console.print()

    # Facts table
    table = Table()
    table.add_column("Prompt", max_width=50)
    table.add_column("Expected")
    table.add_column("Baseline", justify="right")
    table.add_column("Status")
    if result.healed_count > 0:
        table.add_column("Healed", justify="right")

    for f in result.facts:
        status_style = "green" if f.baseline_correct else "red"
        row = [
            f.prompt[:50],
            f.expected_answer,
            f"{f.baseline_prob:.2%}",
            f"[{status_style}]"
            f"{'CORRECT' if f.baseline_correct else 'WRONG'}"
            f"[/{status_style}]",
        ]
        if result.healed_count > 0:
            if f.healed:
                row.append(f"[yellow]{f.healed_prob:.2%}[/yellow]")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)
    console.print()
