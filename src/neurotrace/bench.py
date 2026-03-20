"""Bench command: comprehensive evaluation pipeline for ROME repair generalization."""

from __future__ import annotations

import html as _html
import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptBenchResult:
    """Per-prompt result from bench repair+measure."""

    prompt: str
    answer: str
    baseline_correct: bool
    repaired_correct: bool
    competitor: str = ""
    before_margin: float = 0.0
    after_margin: float = 0.0
    before_prob: float = 0.0
    after_prob: float = 0.0
    layer: int | None = None
    component: str | None = None
    edit_norm: float | None = None


@dataclass
class DomainBenchResult:
    """Per-domain aggregated result."""

    dataset: str
    baseline_correct: int
    baseline_total: int
    repaired_correct: int
    repaired_total: int
    baseline_ppl: float
    repaired_ppl: float
    edit_count: int
    total_edit_norm: float
    prompt_results: list[PromptBenchResult] = field(default_factory=list)

    @property
    def baseline_acc(self) -> float:
        if not self.baseline_total:
            return 0.0
        return self.baseline_correct / self.baseline_total

    @property
    def repaired_acc(self) -> float:
        if not self.repaired_total:
            return 0.0
        return self.repaired_correct / self.repaired_total

    @property
    def acc_delta(self) -> float:
        return self.repaired_acc - self.baseline_acc

    @property
    def ppl_delta(self) -> float:
        return self.repaired_ppl - self.baseline_ppl


@dataclass
class BenchRun:
    """Full benchmark run result."""

    run_id: str
    model_name: str
    datasets: list[str]
    baseline_ppl: float
    domain_results: list[DomainBenchResult]
    combined_ppl: float
    total_edits: int
    total_edit_norm: float
    combined_domain_results: list[DomainBenchResult] = field(default_factory=list)
    created_at: str = ""


# ---------------------------------------------------------------------------
# Perplexity computation (local)
# ---------------------------------------------------------------------------


def compute_perplexity_local(
    model,
    tokenizer,
    max_samples: int = 100,
    max_length: int = 512,
) -> dict:
    """Compute perplexity on WikiText-2 test set."""
    import torch

    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    except Exception:
        text = (
            "The tower is 324 metres tall, about the same "
            "height as an 81-storey building, and the "
            "tallest structure in Paris. Its base is square,"
            " measuring 125 metres on each side. "
            "During its construction, the Eiffel Tower "
            "surpassed the Washington Monument to become "
            "the tallest man-made structure in the world. "
            * 50
        )

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(next(model.parameters()).device)

    model_max_length = getattr(model.config, "max_position_embeddings", 2048) or 2048
    actual_max_length = min(model_max_length, max_length)
    stride = actual_max_length // 2

    nlls = []
    total_tokens = 0

    max_pos = min(input_ids.size(1), max_samples * actual_max_length)
    for begin in range(0, max_pos, stride):
        end = min(begin + actual_max_length, input_ids.size(1))
        input_chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits[:, :-1, :].contiguous()
            targets = input_chunk[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="mean",
            )

        chunk_tokens = targets.numel()
        nlls.append(loss.item() * chunk_tokens)
        total_tokens += chunk_tokens

        if end == input_ids.size(1):
            break

    avg_loss = sum(nlls) / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss)

    return {
        "perplexity": ppl,
        "loss": avg_loss,
        "tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Scan accuracy helper
# ---------------------------------------------------------------------------


def scan_accuracy(
    model,
    tokenizer,
    dataset: list[dict],
    seed: int = 42,
) -> tuple[int, int, list[bool]]:
    """Run scan and return (correct_count, total, per_prompt_correct_list)."""
    from neurotrace.scan import run_scan

    result = run_scan(
        model, tokenizer, dataset, "bench",
        seed=seed, sabotage_threshold=0.5, final_threshold=0.3,
    )
    correct_list = [
        r.status in ("correct", "weak") and r.final_rank == 1
        for r in result.prompt_results
    ]
    return sum(correct_list), len(correct_list), correct_list


# ---------------------------------------------------------------------------
# Run bench locally
# ---------------------------------------------------------------------------


def run_bench_local(
    model,
    tokenizer,
    dataset_names: list[str],
    seed: int = 42,
    target_margin: float = 0.0,
    ppl_samples: int = 100,
    progress_callback=None,
) -> BenchRun:
    """Run the full benchmark pipeline locally."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.datasets import get_builtin_dataset
    from neurotrace.repair import run_repair_local

    run_id = str(uuid.uuid4())
    model_name = model.config._name_or_path
    created_at = datetime.now(timezone.utc).isoformat()

    datasets_map = {}
    for name in dataset_names:
        datasets_map[name] = get_builtin_dataset(name)

    # Phase 1: Baseline
    if progress_callback:
        progress_callback("phase1", "Computing baseline...")

    baseline_accuracies = {}
    baseline_correct_lists = {}
    for name, ds in datasets_map.items():
        if progress_callback:
            progress_callback("scan", f"Baseline scan: {name}")
        correct, total, correct_list = scan_accuracy(model, tokenizer, ds, seed)
        baseline_accuracies[name] = (correct, total)
        baseline_correct_lists[name] = correct_list

    if progress_callback:
        progress_callback("ppl", "Computing baseline perplexity...")
    baseline_ppl_data = compute_perplexity_local(model, tokenizer, ppl_samples)
    baseline_ppl = baseline_ppl_data["perplexity"]

    # Phase 2: Per-domain repair
    domain_results = []
    # Store original weights for undo
    import copy

    original_state = copy.deepcopy(model.state_dict())

    # Domain order for combined: science first, then history, capitals, math
    domain_order = ["science_symbols", "history_dates", "capitals", "math_simple"]
    ordered_names = [n for n in domain_order if n in dataset_names]
    ordered_names += [n for n in dataset_names if n not in ordered_names]

    for name in dataset_names:
        if progress_callback:
            progress_callback("phase2", f"Per-domain repair: {name}")

        # Reset model to original state
        model.load_state_dict(copy.deepcopy(original_state))

        ds = datasets_map[name]
        prompt_results = []
        edit_count = 0
        total_edit_norm = 0.0

        # Sort by weakest margin first (wrong ones first)
        baseline_list = baseline_correct_lists[name]

        # Repair wrong prompts
        for i, entry in enumerate(ds):
            if progress_callback:
                msg = f"Repairing {name} {i + 1}/{len(ds)}"
                progress_callback("repair", msg)

            result = run_repair_local(
                model, tokenizer,
                entry["prompt"], entry["answer"],
                target_margin=target_margin,
                seed=seed,
            )

            if result.status != "skipped":
                edit_count += 1
                total_edit_norm += result.edit.norm

            prompt_results.append(PromptBenchResult(
                prompt=entry["prompt"],
                answer=entry["answer"],
                baseline_correct=baseline_list[i],
                repaired_correct=True,  # will re-scan
                competitor=result.competitor,
                before_margin=result.before.margin,
                after_margin=result.after.margin,
                before_prob=result.before.answer_prob,
                after_prob=result.after.answer_prob,
                layer=result.target_layer,
                component=result.target_component,
                edit_norm=result.edit.norm,
            ))

        # Re-scan after repairs
        if progress_callback:
            progress_callback("scan", f"Post-repair scan: {name}")
        repaired_correct, repaired_total, repaired_list = scan_accuracy(
            model, tokenizer, ds, seed,
        )
        for j, pr in enumerate(prompt_results):
            pr.repaired_correct = repaired_list[j]

        # Perplexity after this domain's repairs
        if progress_callback:
            progress_callback("ppl", f"Post-repair perplexity: {name}")
        repaired_ppl_data = compute_perplexity_local(model, tokenizer, ppl_samples)

        bc, bt = baseline_accuracies[name]
        domain_results.append(DomainBenchResult(
            dataset=name,
            baseline_correct=bc,
            baseline_total=bt,
            repaired_correct=repaired_correct,
            repaired_total=repaired_total,
            baseline_ppl=baseline_ppl,
            repaired_ppl=repaired_ppl_data["perplexity"],
            edit_count=edit_count,
            total_edit_norm=total_edit_norm,
            prompt_results=prompt_results,
        ))

    # Phase 3: Combined repair
    if progress_callback:
        progress_callback("phase3", "Combined repair (all domains)...")

    model.load_state_dict(copy.deepcopy(original_state))

    combined_edit_count = 0
    combined_edit_norm = 0.0

    for name in ordered_names:
        ds = datasets_map[name]
        for entry in ds:
            if progress_callback:
                progress_callback("repair", f"Combined: {entry['prompt'][:30]}")
            result = run_repair_local(
                model, tokenizer,
                entry["prompt"], entry["answer"],
                target_margin=target_margin,
                seed=seed,
            )
            if result.status != "skipped":
                combined_edit_count += 1
                combined_edit_norm += result.edit.norm

    # Scan all domains after combined repair
    combined_domain_results = []
    for name in dataset_names:
        ds = datasets_map[name]
        if progress_callback:
            progress_callback("scan", f"Combined scan: {name}")
        cc, ct, _ = scan_accuracy(model, tokenizer, ds, seed)
        bc, bt = baseline_accuracies[name]
        combined_domain_results.append(DomainBenchResult(
            dataset=name,
            baseline_correct=bc,
            baseline_total=bt,
            repaired_correct=cc,
            repaired_total=ct,
            baseline_ppl=baseline_ppl,
            repaired_ppl=0.0,  # filled below
            edit_count=0,
            total_edit_norm=0.0,
        ))

    if progress_callback:
        progress_callback("ppl", "Combined perplexity...")
    combined_ppl_data = compute_perplexity_local(model, tokenizer, ppl_samples)
    combined_ppl = combined_ppl_data["perplexity"]
    for cdr in combined_domain_results:
        cdr.repaired_ppl = combined_ppl

    # Restore original weights
    model.load_state_dict(original_state)

    return BenchRun(
        run_id=run_id,
        model_name=model_name,
        datasets=dataset_names,
        baseline_ppl=baseline_ppl,
        domain_results=domain_results,
        combined_ppl=combined_ppl,
        total_edits=combined_edit_count,
        total_edit_norm=combined_edit_norm,
        combined_domain_results=combined_domain_results,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Run bench via remote worker
# ---------------------------------------------------------------------------


def run_bench_remote(
    remote_url: str,
    dataset_names: list[str],
    seed: int = 42,
    target_margin: float = 0.0,
    ppl_samples: int = 100,
    progress_callback=None,
) -> BenchRun:
    """Run the full benchmark pipeline via remote GPU worker."""
    import uuid
    from datetime import datetime, timezone

    from neurotrace.datasets import get_builtin_dataset
    from neurotrace.remote import RemoteWorker
    from neurotrace.repair import build_repair_result_from_remote

    worker = RemoteWorker(remote_url, timeout=600.0)
    health = worker.health()
    model_name = health["model"]

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    datasets_map = {}
    for name in dataset_names:
        datasets_map[name] = get_builtin_dataset(name)

    def remote_scan(ds: list[dict]) -> tuple[int, int, list[bool]]:
        """Scan via remote /infer-style approach using repair_stream to check."""
        correct_list = []
        for entry in ds:
            correct = False
            for event in worker.repair_stream(
                entry["prompt"], entry["answer"],
                target_margin=-999.0,  # no actual repair
                seed=seed,
            ):
                if event.get("type") == "result":
                    correct = event["before"]["answer_prob"] > 0.01
                    margin = event["before"]["margin"]
                    correct = margin > 0
            correct_list.append(correct)
            # Undo the dummy edit
            try:
                worker.repair_undo()
            except Exception:
                pass
        return sum(correct_list), len(correct_list), correct_list

    def remote_ppl() -> float:
        """Get perplexity from remote worker."""
        for event in worker.perplexity_stream(ppl_samples):
            if event.get("type") == "result":
                return event["perplexity"]
        return 0.0

    # Phase 1: Baseline
    if progress_callback:
        progress_callback("phase1", "Computing baseline...")

    # Undo all edits first
    _undo_all(worker)

    baseline_accuracies = {}
    baseline_correct_lists = {}
    for name, ds in datasets_map.items():
        if progress_callback:
            progress_callback("scan", f"Baseline scan: {name}")
        correct, total, correct_list = remote_scan(ds)
        baseline_accuracies[name] = (correct, total)
        baseline_correct_lists[name] = correct_list

    if progress_callback:
        progress_callback("ppl", "Baseline perplexity...")
    baseline_ppl = remote_ppl()

    # Phase 2: Per-domain repair
    domain_results = []
    domain_order = ["science_symbols", "history_dates", "capitals", "math_simple"]
    ordered_names = [n for n in domain_order if n in dataset_names]
    ordered_names += [n for n in dataset_names if n not in ordered_names]

    for name in dataset_names:
        if progress_callback:
            progress_callback("phase2", f"Per-domain repair: {name}")

        _undo_all(worker)
        ds = datasets_map[name]
        prompt_results = []
        edit_count = 0
        total_edit_norm = 0.0
        baseline_list = baseline_correct_lists[name]

        for i, entry in enumerate(ds):
            if progress_callback:
                progress_callback(
                    "repair", f"Repairing {name} {i + 1}/{len(ds)}"
                )

            repair_result = None
            for event in worker.repair_stream(
                entry["prompt"], entry["answer"],
                target_margin=target_margin,
                seed=seed,
            ):
                if event.get("type") == "result":
                    repair_result = build_repair_result_from_remote(event)

            if repair_result and repair_result.status != "skipped":
                edit_count += 1
                total_edit_norm += repair_result.edit.norm

            pr = PromptBenchResult(
                prompt=entry["prompt"],
                answer=entry["answer"],
                baseline_correct=baseline_list[i],
                repaired_correct=True,
                competitor=repair_result.competitor if repair_result else "",
                before_margin=repair_result.before.margin if repair_result else 0.0,
                after_margin=repair_result.after.margin if repair_result else 0.0,
                before_prob=repair_result.before.answer_prob if repair_result else 0.0,
                after_prob=repair_result.after.answer_prob if repair_result else 0.0,
                layer=repair_result.target_layer if repair_result else None,
                component=repair_result.target_component if repair_result else None,
                edit_norm=repair_result.edit.norm if repair_result else None,
            )
            prompt_results.append(pr)

        # Re-scan
        if progress_callback:
            progress_callback("scan", f"Post-repair scan: {name}")
        rc, rt, rl = remote_scan(ds)
        for j, pr in enumerate(prompt_results):
            pr.repaired_correct = rl[j]

        if progress_callback:
            progress_callback("ppl", f"Post-repair perplexity: {name}")
        repaired_ppl = remote_ppl()

        bc, bt = baseline_accuracies[name]
        domain_results.append(DomainBenchResult(
            dataset=name,
            baseline_correct=bc,
            baseline_total=bt,
            repaired_correct=rc,
            repaired_total=rt,
            baseline_ppl=baseline_ppl,
            repaired_ppl=repaired_ppl,
            edit_count=edit_count,
            total_edit_norm=total_edit_norm,
            prompt_results=prompt_results,
        ))

    # Phase 3: Combined repair
    if progress_callback:
        progress_callback("phase3", "Combined repair...")

    _undo_all(worker)

    combined_edit_count = 0
    combined_edit_norm = 0.0

    for name in ordered_names:
        ds = datasets_map[name]
        for entry in ds:
            if progress_callback:
                progress_callback("repair", f"Combined: {entry['prompt'][:30]}")
            for event in worker.repair_stream(
                entry["prompt"], entry["answer"],
                target_margin=target_margin,
                seed=seed,
            ):
                if event.get("type") == "result":
                    r = build_repair_result_from_remote(event)
                    if r.status != "skipped":
                        combined_edit_count += 1
                        combined_edit_norm += r.edit.norm

    combined_domain_results = []
    for name in dataset_names:
        ds = datasets_map[name]
        if progress_callback:
            progress_callback("scan", f"Combined scan: {name}")
        cc, ct, _ = remote_scan(ds)
        bc, bt = baseline_accuracies[name]
        combined_domain_results.append(DomainBenchResult(
            dataset=name,
            baseline_correct=bc,
            baseline_total=bt,
            repaired_correct=cc,
            repaired_total=ct,
            baseline_ppl=baseline_ppl,
            repaired_ppl=0.0,
            edit_count=0,
            total_edit_norm=0.0,
        ))

    if progress_callback:
        progress_callback("ppl", "Combined perplexity...")
    combined_ppl = remote_ppl()
    for cdr in combined_domain_results:
        cdr.repaired_ppl = combined_ppl

    _undo_all(worker)

    return BenchRun(
        run_id=run_id,
        model_name=model_name,
        datasets=dataset_names,
        baseline_ppl=baseline_ppl,
        domain_results=domain_results,
        combined_ppl=combined_ppl,
        total_edits=combined_edit_count,
        total_edit_norm=combined_edit_norm,
        combined_domain_results=combined_domain_results,
        created_at=created_at,
    )


def _undo_all(worker) -> None:
    """Undo all edits on the remote worker."""
    while True:
        try:
            result = worker.repair_undo()
            if result.get("edits_remaining", 0) == 0:
                break
        except Exception:
            break


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def bench_run_to_dict(run: BenchRun) -> dict:
    """Serialize a BenchRun to a JSON-compatible dict."""
    return {
        "run_id": run.run_id,
        "model_name": run.model_name,
        "datasets": run.datasets,
        "baseline_ppl": run.baseline_ppl,
        "combined_ppl": run.combined_ppl,
        "total_edits": run.total_edits,
        "total_edit_norm": run.total_edit_norm,
        "created_at": run.created_at,
        "domain_results": [
            {
                "dataset": dr.dataset,
                "baseline_correct": dr.baseline_correct,
                "baseline_total": dr.baseline_total,
                "repaired_correct": dr.repaired_correct,
                "repaired_total": dr.repaired_total,
                "baseline_ppl": dr.baseline_ppl,
                "repaired_ppl": dr.repaired_ppl,
                "edit_count": dr.edit_count,
                "total_edit_norm": dr.total_edit_norm,
                "baseline_acc": dr.baseline_acc,
                "repaired_acc": dr.repaired_acc,
                "acc_delta": dr.acc_delta,
                "ppl_delta": dr.ppl_delta,
                "prompt_results": [
                    {
                        "prompt": pr.prompt,
                        "answer": pr.answer,
                        "baseline_correct": pr.baseline_correct,
                        "repaired_correct": pr.repaired_correct,
                        "competitor": pr.competitor,
                        "before_margin": pr.before_margin,
                        "after_margin": pr.after_margin,
                        "before_prob": pr.before_prob,
                        "after_prob": pr.after_prob,
                        "layer": pr.layer,
                        "component": pr.component,
                        "edit_norm": pr.edit_norm,
                    }
                    for pr in dr.prompt_results
                ],
            }
            for dr in run.domain_results
        ],
        "combined_domain_results": [
            {
                "dataset": dr.dataset,
                "baseline_correct": dr.baseline_correct,
                "baseline_total": dr.baseline_total,
                "repaired_correct": dr.repaired_correct,
                "repaired_total": dr.repaired_total,
                "repaired_ppl": dr.repaired_ppl,
            }
            for dr in run.combined_domain_results
        ],
    }


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_GREEN = "#4caf50"
_RED = "#f44336"
_BLUE = "#64b5f6"
_BG = "#1a1a2e"
_BG2 = "#16213e"
_BG3 = "#0f3460"
_TEXT = "#e0e0e0"
_DIM = "#8a8a9a"
_ACCENT = "#e8956a"
_YELLOW = "#ffca28"


def _esc(s: Any) -> str:
    return _html.escape(str(s))


def _bench_css() -> str:
    return f"""
    :root {{
        --bg: {_BG}; --bg2: {_BG2}; --bg3: {_BG3};
        --text: {_TEXT}; --dim: {_DIM};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont,
            'Segoe UI', Helvetica, Arial, sans-serif;
        font-size: 14px; line-height: 1.6; padding: 2rem;
        max-width: 1600px; margin: 0 auto;
    }}
    h1 {{ color: {_ACCENT}; font-size: 1.8rem; margin-bottom: 0.5rem; }}
    h2 {{
        color: {_BLUE}; font-size: 1.3rem; margin: 2rem 0 1rem;
        border-bottom: 1px solid var(--bg3); padding-bottom: 0.5rem;
    }}
    h3 {{ color: {_TEXT}; font-size: 1.1rem; margin: 1.5rem 0 0.5rem; }}
    .meta {{
        background: var(--bg2); padding: 1.5rem;
        border-radius: 8px; margin-bottom: 2rem;
    }}
    .meta-grid {{
        display: grid; gap: 0.5rem 2rem;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    }}
    .meta-label {{ color: var(--dim); font-size: 0.85rem; }}
    .meta-value {{
        color: var(--text);
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th {{
        background: var(--bg3); color: var(--dim); text-align: left;
        padding: 0.5rem 0.75rem; font-weight: 600; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    td {{
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid var(--bg3);
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
    }}
    .positive {{ color: {_GREEN}; }}
    .negative {{ color: {_RED}; }}
    .neutral {{ color: {_YELLOW}; }}
    .bar-chart {{
        display: flex; align-items: flex-end; gap: 8px;
        height: 200px; padding: 1rem 0;
    }}
    .bar-group {{
        display: flex; flex-direction: column; align-items: center;
        flex: 1;
    }}
    .bar {{
        width: 100%; min-width: 40px; border-radius: 4px 4px 0 0;
    }}
    .bar-label {{
        font-size: 0.75rem; color: var(--dim);
        margin-top: 4px; text-align: center;
    }}
    .bar-value {{
        font-size: 0.75rem; color: var(--text);
        margin-bottom: 2px; text-align: center;
    }}
    details {{
        background: var(--bg2); border-radius: 8px;
        padding: 1rem; margin-bottom: 1rem;
    }}
    summary {{
        cursor: pointer; font-weight: 600; color: {_BLUE};
        padding: 0.5rem 0;
    }}
    .findings {{
        background: var(--bg2); padding: 1.5rem;
        border-radius: 8px; margin-top: 2rem;
    }}
    .findings li {{ margin: 0.5rem 0; }}
    @media print {{
        body {{ background: white; color: #222; padding: 1rem; }}
        .meta, details, .findings {{ background: #f5f5f5; color: #222; }}
        th {{ background: #e0e0e0; color: #333; }}
        td {{ border-color: #ccc; color: #222; }}
        h1 {{ color: #333; }} h2 {{ color: #555; }}
    }}
    """


def generate_bench_html(run: BenchRun) -> str:
    """Generate self-contained HTML report for a bench run."""
    parts = []
    ppl_change = run.combined_ppl - run.baseline_ppl

    # Header
    parts.append(f"""
    <h1>NeuroTrace Benchmark Report</h1>
    <div class="meta">
        <div class="meta-grid">
            <div><span class="meta-label">Model</span><br>
                <span class="meta-value">\
{_esc(run.model_name)}</span></div>
            <div><span class="meta-label">Date</span><br>
                <span class="meta-value">\
{_esc(run.created_at[:10])}</span></div>
            <div><span class="meta-label">Total Edits</span><br>
                <span class="meta-value">\
{run.total_edits}</span></div>
            <div><span class="meta-label">Baseline PPL</span><br>
                <span class="meta-value">\
{run.baseline_ppl:.2f}</span></div>
            <div><span class="meta-label">Combined PPL</span><br>
                <span class="meta-value">\
{run.combined_ppl:.2f}</span></div>
            <div><span class="meta-label">PPL Change</span><br>
                <span class="meta-value">\
{ppl_change:+.2f}</span></div>
        </div>
    </div>
    """)

    # Domain comparison table
    parts.append("<h2>Domain Comparison</h2>")
    parts.append(
        "<table><thead><tr>"
        "<th>Dataset</th><th>Baseline</th><th>Repaired</th>"
        "<th>Delta</th><th>PPL Change</th>"
        "<th>Edits</th><th>Norm</th>"
        "</tr></thead><tbody>"
    )
    for dr in run.domain_results:
        if dr.acc_delta > 0:
            delta_cls = "positive"
        elif dr.acc_delta < 0:
            delta_cls = "negative"
        else:
            delta_cls = "neutral"
        if dr.ppl_delta > 0.1:
            ppl_cls = "negative"
        elif dr.ppl_delta < 0:
            ppl_cls = "positive"
        else:
            ppl_cls = "neutral"
        ba = f"{dr.baseline_correct}/{dr.baseline_total}"
        ra = f"{dr.repaired_correct}/{dr.repaired_total}"
        parts.append(
            f"<tr>"
            f"<td>{_esc(dr.dataset)}</td>"
            f"<td>{ba} ({dr.baseline_acc:.1%})</td>"
            f"<td>{ra} ({dr.repaired_acc:.1%})</td>"
            f'<td class="{delta_cls}">{dr.acc_delta:+.1%}</td>'
            f'<td class="{ppl_cls}">{dr.ppl_delta:+.2f}</td>'
            f"<td>{dr.edit_count}</td>"
            f"<td>{dr.total_edit_norm:.2f}</td>"
            f"</tr>"
        )
    parts.append("</tbody></table>")

    # Perplexity bar chart (SVG)
    domain_ppls = [dr.repaired_ppl for dr in run.domain_results]
    all_ppls = [run.baseline_ppl] + domain_ppls + [run.combined_ppl]
    max_ppl = max(all_ppls) * 1.15 if all_ppls else 10
    chart_h = 180
    chart_w = max(400, len(all_ppls) * 80)
    bar_w = min(50, chart_w // (len(all_ppls) + 1))

    parts.append(f'<h2>Perplexity</h2><svg width="{chart_w}" height="{chart_h + 40}" '
                 f'xmlns="http://www.w3.org/2000/svg">')
    labels = ["Baseline"] + [dr.dataset for dr in run.domain_results] + ["Combined"]
    colors = [_BLUE] + [_ACCENT] * len(run.domain_results) + [_GREEN]

    for i, (ppl, label, color) in enumerate(zip(all_ppls, labels, colors)):
        x = 10 + i * (bar_w + 20)
        h = (ppl / max_ppl) * chart_h if max_ppl > 0 else 0
        y = chart_h - h
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" '
            f'fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{x + bar_w // 2}" y="{y - 4}" text-anchor="middle" '
            f'fill="{_TEXT}" font-size="10">{ppl:.2f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w // 2}" y="{chart_h + 14}" text-anchor="middle" '
            f'fill="{_DIM}" font-size="9">{_esc(label[:10])}</text>'
        )
    parts.append("</svg>")

    # Combined results
    if run.combined_domain_results:
        parts.append("<h2>Combined Repair (All Domains)</h2>")
        parts.append(
            "<table><thead><tr>"
            "<th>Dataset</th><th>Correct</th><th>Total</th><th>Accuracy</th>"
            "</tr></thead><tbody>"
        )
        for cdr in run.combined_domain_results:
            parts.append(
                f"<tr><td>{_esc(cdr.dataset)}</td>"
                f"<td>{cdr.repaired_correct}</td>"
                f"<td>{cdr.repaired_total}</td>"
                f"<td>{cdr.repaired_acc:.1%}</td></tr>"
            )
        parts.append("</tbody></table>")
        parts.append(
            f"<p>Combined PPL: {run.combined_ppl:.2f} "
            f"({run.combined_ppl - run.baseline_ppl:+.2f}) | "
            f"Total edits: {run.total_edits} | "
            f"Total norm: {run.total_edit_norm:.2f}</p>"
        )

    # Per-domain detail (collapsible)
    for dr in run.domain_results:
        if not dr.prompt_results:
            continue
        n = len(dr.prompt_results)
        parts.append(
            f"<details><summary>{_esc(dr.dataset)}"
            f" - {n} prompts</summary>"
        )
        parts.append(
            "<table><thead><tr>"
            "<th>Prompt</th><th>Answer</th><th>Baseline</th>"
            "<th>Repaired</th><th>Before Margin</th><th>After Margin</th>"
            "<th>Layer</th><th>Edit Norm</th>"
            "</tr></thead><tbody>"
        )
        for pr in dr.prompt_results:
            bc = "positive" if pr.baseline_correct else "negative"
            rc = "positive" if pr.repaired_correct else "negative"
            parts.append(
                f"<tr>"
                f"<td>{_esc(pr.prompt[:40])}</td>"
                f"<td>{_esc(pr.answer)}</td>"
                f'<td class="{bc}">{"Y" if pr.baseline_correct else "N"}</td>'
                f'<td class="{rc}">{"Y" if pr.repaired_correct else "N"}</td>'
                f"<td>{pr.before_margin:+.2f}</td>"
                f"<td>{pr.after_margin:+.2f}</td>"
                f"<td>{pr.layer if pr.layer is not None else '-'}</td>"
                f"<td>{f'{pr.edit_norm:.4f}' if pr.edit_norm is not None else '-'}</td>"
                f"</tr>"
            )
        parts.append("</tbody></table></details>")

    # Key findings
    parts.append('<div class="findings"><h2>Key Findings</h2><ul>')
    effective = sum(
        1 for dr in run.domain_results if dr.acc_delta > 0.01
    )
    total_d = len(run.domain_results)
    parts.append(
        f"<li>ROME repair effective on "
        f"{effective}/{total_d} domains</li>"
    )
    ppl_increase = run.combined_ppl - run.baseline_ppl
    ppl_pct = (ppl_increase / run.baseline_ppl * 100) if run.baseline_ppl > 0 else 0
    parts.append(
        f"<li>Perplexity impact: {ppl_increase:+.2f} ({ppl_pct:.1f}% increase)</li>"
    )
    for dr in run.domain_results:
        if dr.acc_delta > 0.01:
            state = "improved"
        elif dr.acc_delta < -0.01:
            state = "degraded"
        else:
            state = "unchanged"
        parts.append(
            f"<li>{_esc(dr.dataset)}: {state} "
            f"({dr.baseline_acc:.0%} &rarr; {dr.repaired_acc:.0%})</li>"
        )
    parts.append("</ul></div>")

    body = "\n".join(parts)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>NeuroTrace Benchmark - {_esc(run.model_name)}</title>\n"
        f"<style>{_bench_css()}</style>\n"
        f"</head>\n<body>\n{body}\n</body>\n</html>"
    )
