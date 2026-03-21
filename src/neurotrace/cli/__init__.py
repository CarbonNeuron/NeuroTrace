"""NeuroTrace CLI — modular package."""

import click
from rich.console import Console

from neurotrace.storage import TraceDB

console = Console()
err_console = Console(stderr=True)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to best available, pass others through."""
    if device != "auto":
        return device
    try:
        import torch_directml  # noqa: F401

        return "privateuseone:0"
    except ImportError:
        pass
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_trace_id(db: TraceDB, trace_id: str) -> str:
    """Resolve label, prefix, 'latest', or full trace_id."""
    return db.resolve_trace_id(trace_id)


def _maybe_load_adapter(model, adapter_path: str | None):
    """Load and merge a LoRA adapter if path is provided."""
    if adapter_path is None:
        return model
    from neurotrace.finetune import load_adapter

    return load_adapter(model, adapter_path)


@click.group()
@click.option("--model", default=None, help="Model name (from worker if --remote)")
@click.pass_context
def cli(ctx: click.Context, model: str | None) -> None:
    """NeuroTrace: interpretability toolkit for tracing transformer inference."""
    ctx.ensure_object(dict)
    ctx.obj["model"] = model


# Import and register commands from submodules
from neurotrace.cli.trace import (  # noqa: E402
    compare,
    decode,
    diff,
    inspect,
    list_traces,
    predict,
    trace,
)

cli.add_command(trace)
cli.add_command(list_traces)
cli.add_command(inspect)
cli.add_command(diff)
cli.add_command(predict)
cli.add_command(decode)
cli.add_command(compare)

from neurotrace.cli.analysis import (  # noqa: E402
    ablate,
    circuit,
    finetune,
    neurons,
    probe,
    probe_universal,
    scan,
    sweep,
)

cli.add_command(ablate)
cli.add_command(sweep)
cli.add_command(scan)
cli.add_command(finetune)
cli.add_command(neurons)
cli.add_command(probe)
cli.add_command(circuit)
cli.add_command(probe_universal)

from neurotrace.cli.domain import (  # noqa: E402
    attribute,
    commitment,
    contrast,
    experiment,
    heatmap,
)

cli.add_command(experiment)
cli.add_command(heatmap)
cli.add_command(commitment)
cli.add_command(contrast)
cli.add_command(attribute)

from neurotrace.cli.advanced import (  # noqa: E402
    attention_trace,
    decompose,
    diagnose,
    token_trace,
)

cli.add_command(token_trace)
cli.add_command(attention_trace)
cli.add_command(diagnose)
cli.add_command(decompose)

from neurotrace.cli.repair import fingerprint, repair  # noqa: E402

cli.add_command(fingerprint)
cli.add_command(repair)

from neurotrace.cli.bench import bench  # noqa: E402

cli.add_command(bench)

from neurotrace.cli.report import report  # noqa: E402

cli.add_command(report)

from neurotrace.cli.worker import worker  # noqa: E402

cli.add_command(worker)

from neurotrace.cli.heal import heal  # noqa: E402

cli.add_command(heal)

from neurotrace.cli.discover import discover  # noqa: E402

cli.add_command(discover)

from neurotrace.cli.chain import chain  # noqa: E402

cli.add_command(chain)
