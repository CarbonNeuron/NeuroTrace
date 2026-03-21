"""Heal CLI command: autonomous self-repair loop."""

import json

import click
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from neurotrace.cli import _maybe_load_adapter, _resolve_device, console, err_console


@click.command()
@click.option("--model", default=None, help="HuggingFace model name (local mode).")
@click.option(
    "--dataset", default="capitals",
    help="Built-in dataset name or JSON file path.",
)
@click.option("--db", default="demo.db", help="DuckDB database path.")
@click.option(
    "--remote", default=None,
    help="GPU worker URL (e.g., http://172.30.0.1:8877).",
)
@click.option("--output", default=None, help="Directory to save patched model weights.")
@click.option(
    "--max-edits", default=500, type=int,
    help="Safety cap on total ROME edits.",
)
@click.option(
    "--regression-threshold", default=0.05, type=float,
    help="Max allowed probability drop on correct prompts.",
)
@click.option(
    "--ppl-threshold", default=0.1, type=float,
    help="Max allowed perplexity increase.",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Run full pipeline but don't save weights.",
)
@click.option("--json", "output_json", is_flag=True, help="JSON output.")
@click.option("--report", "report_path", default=None, help="Path for HTML report.")
@click.option("--upload", is_flag=True, help="Upload HTML report to CarbonFiles.")
@click.option("--verbose", is_flag=True, help="Print each step as it runs.")
@click.option("--device", default="cpu", help="Device: cpu, cuda, directml, auto.")
@click.option("--adapter", default=None, help="Path to LoRA adapter directory.")
@click.option("--seed", default=42, type=int, help="Random seed.")
def heal(
    model,
    dataset,
    db,
    remote,
    output,
    max_edits,
    regression_threshold,
    ppl_threshold,
    dry_run,
    output_json,
    report_path,
    upload,
    verbose,
    device,
    adapter,
    seed,
):
    """Autonomous self-repair: scan, diagnose, repair, verify."""
    import os

    from neurotrace.heal import generate_heal_html, heal_result_to_dict, run_heal_local

    if remote is None and model is None:
        raise click.UsageError("Must provide --model (local mode) or --remote.")

    # Load dataset
    if os.path.isfile(dataset):
        from neurotrace.datasets import load_dataset as load_ds

        ds = load_ds(dataset)
        dataset_name = dataset
    else:
        from neurotrace.datasets import get_builtin_dataset

        ds = get_builtin_dataset(dataset)
        dataset_name = dataset

    if remote is not None:
        _heal_remote(
            remote, ds, dataset_name, db, output, max_edits,
            regression_threshold, ppl_threshold, dry_run,
            output_json, report_path, upload, verbose, seed,
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

        model_obj, tokenizer = load_model(model, device=device)
        model_obj = _maybe_load_adapter(model_obj, adapter)

        def progress_cb(phase, msg):
            progress.update(task, description=msg)
            if verbose:
                err_console.print(f"[dim]{phase}: {msg}[/dim]")

        progress.update(task, description="Running heal pipeline...")

        result = run_heal_local(
            model_obj,
            tokenizer,
            ds,
            dataset_name,
            db_path=db,
            max_edits=max_edits,
            regression_threshold=regression_threshold,
            ppl_threshold=ppl_threshold,
            dry_run=dry_run,
            seed=seed,
            output_path=output,
            progress_callback=progress_cb,
        )

        progress.update(task, description="Done.")

    # HTML report
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        html = generate_heal_html(result)
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
        click.echo(json.dumps(heal_result_to_dict(result), indent=2))
        return

    # Console output
    baseline_acc = (
        result.baseline_correct / result.baseline_total
        if result.baseline_total > 0
        else 0.0
    )
    healed_acc = (
        result.healed_correct / result.healed_total
        if result.healed_total > 0
        else 0.0
    )

    total = result.baseline_total
    console.print(
        f"\n[bold]Heal:[/bold] {dataset_name}"
        f" ({total} prompts)"
    )
    bc = result.baseline_correct
    hc = result.healed_correct
    console.print(f"  Baseline: {bc}/{total} ({baseline_acc:.0%})")
    console.print(f"  Healed:   {hc}/{total} ({healed_acc:.0%})")
    console.print()
    ea = result.edits_applied
    lo = result.edits_lobotomized
    rb = result.edits_rolled_back
    sk = result.edits_skipped
    console.print(
        f"  Edits: {ea} applied, {lo} lobotomized,"
        f" {rb} rolled back, {sk} skipped"
    )

    if result.regressions_found:
        md = result.max_regression_delta
        console.print(
            f"  Regressions: {result.regressions_found}"
            f" (max delta: {md:+.4f})"
        )

    if result.ppl_delta is not None:
        console.print(
            f"  Perplexity: {result.ppl_before:.2f}"
            f" → {result.ppl_after:.2f}"
            f" ({result.ppl_delta:+.2f})"
        )

    if result.catastrophic_undo:
        console.print("  [red]CATASTROPHIC: PPL increase > 1.0, all edits undone[/red]")

    if result.output_path:
        console.print(f"  Saved to: {result.output_path}")

    console.print(f"  Duration: {result.duration_seconds:.1f}s")
    console.print()

    # Prompt table
    table = Table()
    table.add_column("Prompt", max_width=40)
    table.add_column("Answer")
    table.add_column("Baseline", justify="right")
    table.add_column("Action")
    table.add_column("Edit")
    table.add_column("Result", justify="right")
    table.add_column("Final")

    action_styles = {
        "healed": "green",
        "lobotomized": "magenta",
        "already_correct": "yellow",
        "wrong": "red",
        "rolled_back": "bright_red",
        "skipped": "dim",
        "failed": "red",
    }

    for pr in result.prompt_results:
        style = action_styles.get(pr.action, "")
        if pr.before_token and pr.after_token:
            edit_str = f"{pr.before_token} → {pr.after_token}"
        else:
            edit_str = "-"
        table.add_row(
            pr.prompt[:40],
            pr.answer,
            f"{pr.baseline_prob:.2%}",
            f"[{style}]{pr.action.upper()}[/{style}]",
            edit_str,
            f"{pr.result_prob:.2%}",
            pr.final_status,
        )

    console.print(table)
    console.print()


def _heal_remote(
    remote_url, dataset, dataset_name, db, output, max_edits,
    regression_threshold, ppl_threshold, dry_run,
    output_json, report_path, upload, verbose, seed,
):
    """Run heal pipeline via remote GPU worker.

    Uses the same endpoints as repair/scan/fingerprint/bench.
    """
    import os
    import time

    from neurotrace.heal import (
        HealResult,
        PromptHealResult,
        generate_heal_html,
        heal_result_to_dict,
    )
    from neurotrace.remote import WorkerClient

    from neurotrace.remote import WorkerError

    worker = WorkerClient(remote_url, timeout=600.0)
    health = worker.health()
    model_name = health["model"]
    device_name = health.get("device_name", health.get("device", "unknown"))
    err_console.print(f"GPU: {device_name} via {remote_url}")

    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=err_console,
    ) as progress:
        task = progress.add_task("Heal pipeline...", total=None)

        # Step 1: Baseline scan via forward() with layer_predictions
        progress.update(task, description="Baseline scan...")
        baseline_results = []
        for entry in dataset:
            result = worker.forward(
                entry["prompt"], raw=True, top_k=5,
                layer_predictions=False, seed=seed,
            )
            # Whitespace-skip: if top-1 is whitespace (e.g. " "),
            # append it and re-run to get the actual answer token.
            effective_prompt = entry["prompt"]
            if (
                result.top_tokens
                and result.top_tokens[0].token.strip().lstrip("\u2581\u0120") == ""
            ):
                effective_prompt = entry["prompt"] + result.top_tokens[0].token
                result = worker.forward(
                    effective_prompt, raw=True, top_k=5,
                    layer_predictions=False, seed=seed,
                )
            # Check if top-1 token matches answer
            final_token = result.top_tokens[0].token if result.top_tokens else ""
            prob = result.top_tokens[0].prob if result.top_tokens else 0.0
            final_clean = final_token.strip().lstrip("\u2581\u0120").lower()
            answer_clean = entry["answer"].strip().lower()
            is_correct = (
                final_clean == answer_clean
                or (len(final_clean) >= 2 and answer_clean.startswith(final_clean))
            )
            # Multi-token verification: if top-1 is a prefix of the
            # answer but not a complete match, use generate() to check.
            if not is_correct and final_clean and answer_clean.startswith(final_clean):
                gen = worker.generate(
                    effective_prompt, raw=True,
                    max_tokens=len(entry["answer"]) + 2,
                    temperature=0.0, seed=seed,
                )
                generated = "".join(
                    t.token.replace("\u2581", "").replace("\u0120", "")
                    for t in gen.tokens
                ).strip().lower()
                if generated.startswith(answer_clean) or answer_clean in generated:
                    is_correct = True
            margin = prob if is_correct else -prob
            status = "correct" if is_correct else "wrong"
            baseline_results.append({
                "prompt": entry["prompt"],
                "answer": entry["answer"],
                "prob": prob,
                "margin": margin,
                "status": status,
                "before_token": final_token.strip().lstrip("\u2581\u0120"),
            })

        baseline_correct = sum(1 for r in baseline_results if r["status"] == "correct")
        baseline_wrong = sum(1 for r in baseline_results if r["status"] == "wrong")
        total = len(dataset)

        # Identify healable prompts (wrong ones for remote mode)
        healable = [r for r in baseline_results if r["status"] == "wrong"]
        healable.sort(key=lambda x: x["margin"])

        prompt_results = []
        for r in baseline_results:
            if r["status"] == "correct":
                prompt_results.append(PromptHealResult(
                    prompt=r["prompt"],
                    answer=r["answer"],
                    baseline_prob=r["prob"],
                    baseline_status="correct",
                    action="already_correct",
                    result_prob=r["prob"],
                    final_status="correct",
                ))

        # Step 2: Sequential repair
        edits_healed = 0
        edits_failed = 0
        edits_lobotomized = 0
        edits_rolled_back = 0
        edits_skipped = 0

        from neurotrace.cli.analysis import (
            _is_answer_prefix,
            _is_whitespace_token,
            _token_matches_answer,
            _verify_via_generate,
        )

        def _verify_answer_after_edit(prompt, answer):
            """Run scan logic (whitespace-skip + multi-token) on post-edit model.

            Returns (is_answer_top1, answer_prob, after_token_str).
            """
            post_fwd = worker.forward(
                prompt, raw=True, top_k=10,
                layer_predictions=False, seed=seed,
            )
            effective = prompt
            if (
                post_fwd.top_tokens
                and _is_whitespace_token(post_fwd.top_tokens[0].token)
            ):
                effective = prompt + post_fwd.top_tokens[0].token
                post_fwd = worker.forward(
                    effective, raw=True, top_k=10,
                    layer_predictions=False, seed=seed,
                )

            top1 = post_fwd.top_tokens[0] if post_fwd.top_tokens else None
            after_tok = (
                top1.token.strip().lstrip("\u2581\u0120") if top1 else "?"
            )
            top1_prob = top1.prob if top1 else 0.0

            # Check if expected answer is top-1
            if top1 and _token_matches_answer(top1.token, answer):
                return True, top1_prob, after_tok

            # Multi-token: top-1 is a prefix of answer → generate to verify
            if top1 and _is_answer_prefix(top1.token, answer):
                gen_match, gen_prob = _verify_via_generate(
                    worker, effective, answer, top1_prob,
                    raw=True, seed=seed,
                )
                if gen_match:
                    return True, gen_prob, after_tok

            # Answer not top-1 — find its prob in top_tokens if present
            ans_prob = 0.0
            for tt in post_fwd.top_tokens:
                if _token_matches_answer(tt.token, answer):
                    ans_prob = tt.prob
                    break
            return False, ans_prob, after_tok

        for i, entry in enumerate(healable):
            if edits_healed >= max_edits:
                for remaining in healable[i:]:
                    prompt_results.append(PromptHealResult(
                        prompt=remaining["prompt"],
                        answer=remaining["answer"],
                        baseline_prob=remaining["prob"],
                        baseline_status="wrong",
                        action="skipped",
                        result_prob=remaining["prob"],
                        final_status="wrong",
                    ))
                    edits_skipped += 1
                break

            progress.update(
                task,
                description=f"Repair {i + 1}/{len(healable)}: {entry['prompt'][:30]}",
            )

            # Use rome_edit for repair
            try:
                from neurotrace.cli.repair import _extract_subject

                subject = _extract_subject(entry["prompt"])
                # Auto-detect layer: use middle layer as default
                layer = health["num_layers"] // 2

                edit_result = worker.rome_edit(
                    entry["prompt"], subject, entry["answer"],
                    layer, seed=seed,
                )
                if edit_result.success:
                    # Verify: is the expected answer now top-1?
                    is_top1, post_prob, after_token = _verify_answer_after_edit(
                        entry["prompt"], entry["answer"],
                    )
                    before_tok = entry.get("before_token", "?")

                    if is_top1 and post_prob > entry["prob"]:
                        # HEALED: answer is top-1 and prob improved
                        edits_healed += 1
                        prompt_results.append(PromptHealResult(
                            prompt=entry["prompt"],
                            answer=entry["answer"],
                            baseline_prob=entry["prob"],
                            baseline_status="wrong",
                            action="healed",
                            result_prob=post_prob,
                            final_status="correct",
                            target_layer=layer,
                            edit_norm=0.0,
                            before_token=before_tok,
                            after_token=after_token,
                        ))
                    elif post_prob < entry["prob"]:
                        # LOBOTOMIZED: prob got worse
                        edits_lobotomized += 1
                        try:
                            worker.edit_undo()
                            edits_rolled_back += 1
                        except Exception:
                            pass  # undo not available
                        prompt_results.append(PromptHealResult(
                            prompt=entry["prompt"],
                            answer=entry["answer"],
                            baseline_prob=entry["prob"],
                            baseline_status="wrong",
                            action="lobotomized",
                            result_prob=post_prob,
                            final_status="wrong",
                            target_layer=layer,
                            edit_norm=0.0,
                            rollback_reason="auto-rollback: result < baseline",
                            before_token=before_tok,
                            after_token=after_token,
                        ))
                    else:
                        # FAILED: edit didn't make answer top-1, roll back
                        edits_failed += 1
                        try:
                            worker.edit_undo()
                            edits_rolled_back += 1
                        except Exception:
                            pass  # undo not available
                        prompt_results.append(PromptHealResult(
                            prompt=entry["prompt"],
                            answer=entry["answer"],
                            baseline_prob=entry["prob"],
                            baseline_status="wrong",
                            action="failed",
                            result_prob=post_prob,
                            final_status="wrong",
                            target_layer=layer,
                            edit_norm=0.0,
                            rollback_reason="auto-rollback: answer not top-1",
                            before_token=before_tok,
                            after_token=after_token,
                        ))
                else:
                    edits_skipped += 1
                    prompt_results.append(PromptHealResult(
                        prompt=entry["prompt"],
                        answer=entry["answer"],
                        baseline_prob=entry["prob"],
                        baseline_status="wrong",
                        action="skipped",
                        result_prob=entry["prob"],
                        final_status="wrong",
                    ))
            except WorkerError as e:
                if verbose:
                    err_console.print(f"[red]  Edit failed: {e}[/red]")
                edits_skipped += 1
                prompt_results.append(PromptHealResult(
                    prompt=entry["prompt"],
                    answer=entry["answer"],
                    baseline_prob=entry["prob"],
                    baseline_status="wrong",
                    action="failed",
                    result_prob=entry["prob"],
                    final_status="wrong",
                    rollback_reason=str(e),
                ))
            except Exception as e:
                if verbose:
                    err_console.print(
                        f"[red]  Edit error: {e}[/red]",
                    )
                edits_skipped += 1
                prompt_results.append(PromptHealResult(
                    prompt=entry["prompt"],
                    answer=entry["answer"],
                    baseline_prob=entry["prob"],
                    baseline_status="wrong",
                    action="failed",
                    result_prob=entry["prob"],
                    final_status="wrong",
                    rollback_reason=str(e),
                ))

        # Step 3: Perplexity check
        ppl_after = None
        ppl_before = None
        ppl_delta = None

        if edits_healed > 0:
            progress.update(task, description="Computing perplexity...")
            try:
                for event in worker.perplexity_stream(50):
                    if event.get("type") == "result":
                        ppl_after = event["perplexity"]
            except Exception:
                pass

        # Save if requested
        if output and not dry_run and edits_healed > 0:
            progress.update(task, description=f"Saving to {output}...")
            try:
                worker.repair_save(output)
            except Exception as e:
                err_console.print(f"[red]Save failed: {e}[/red]")

        progress.update(task, description="Done.")

    duration = time.time() - start_time

    result = HealResult(
        model_name=model_name,
        dataset_name=dataset_name,
        baseline_total=total,
        baseline_correct=baseline_correct,
        baseline_wrong=baseline_wrong,
        baseline_sabotaged=0,
        baseline_weak=0,
        healed_total=total,
        healed_correct=baseline_correct + edits_healed,
        healed_wrong=baseline_wrong - edits_healed,
        healed_sabotaged=0,
        healed_weak=0,
        edits_attempted=len(healable),
        edits_applied=edits_healed,
        edits_lobotomized=edits_lobotomized,
        edits_rolled_back=edits_rolled_back,
        edits_skipped=edits_skipped,
        regressions_checked=0,
        regressions_found=0,
        max_regression_delta=0.0,
        ppl_before=ppl_before,
        ppl_after=ppl_after,
        ppl_delta=ppl_delta,
        output_path=output if not dry_run else None,
        duration_seconds=duration,
        prompt_results=prompt_results,
    )

    # HTML report
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        html = generate_heal_html(result)
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

    if output_json:
        click.echo(json.dumps(heal_result_to_dict(result), indent=2))
        return

    # Console output
    baseline_acc = baseline_correct / total if total > 0 else 0.0
    healed_acc = result.healed_correct / total if total > 0 else 0.0

    console.print(f"\n[bold]Heal:[/bold] {dataset_name} ({total} prompts)")
    console.print(f"  Baseline: {baseline_correct}/{total} ({baseline_acc:.0%})")
    console.print(f"  Healed:   {result.healed_correct}/{total} ({healed_acc:.0%})")
    console.print(
        f"  Edits: {edits_healed} healed,"
        f" {edits_failed} failed,"
        f" {edits_lobotomized} lobotomized,"
        f" {edits_rolled_back} rolled back,"
        f" {edits_skipped} skipped"
    )
    if result.ppl_after is not None:
        console.print(f"  Perplexity (after): {result.ppl_after:.2f}")
    console.print(f"  Duration: {duration:.1f}s")
    console.print()

    table = Table()
    table.add_column("Prompt", max_width=40)
    table.add_column("Answer")
    table.add_column("Baseline", justify="right")
    table.add_column("Action")
    table.add_column("Edit")
    table.add_column("Result", justify="right")

    action_styles = {
        "healed": "green",
        "lobotomized": "magenta",
        "already_correct": "yellow",
        "wrong": "red",
        "rolled_back": "bright_red",
        "skipped": "dim",
        "failed": "red",
    }

    for pr in prompt_results:
        style = action_styles.get(pr.action, "")
        if pr.before_token and pr.after_token:
            edit_str = f"{pr.before_token} → {pr.after_token}"
        else:
            edit_str = "-"
        table.add_row(
            pr.prompt[:40],
            pr.answer,
            f"{pr.baseline_prob:.2%}",
            f"[{style}]{pr.action.upper()}[/{style}]",
            edit_str,
            f"{pr.result_prob:.2%}",
        )

    console.print(table)
    console.print()
