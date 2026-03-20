"""Worker management commands for the GPU inference server."""

import click

from neurotrace.cli import err_console


@click.group()
def worker():
    """Manage the GPU worker."""


@worker.command()
@click.option("--remote", required=True, help="Worker URL (e.g. http://172.30.0.1:8877)")
def version(remote: str):
    """Show GPU worker version and status."""
    from neurotrace.remote import RemoteWorker

    w = RemoteWorker(remote)
    info = w.worker_version()

    dirty = " [dirty]" if info.get("dirty") else ""
    uptime = info.get("uptime_seconds", 0)
    hours = uptime // 3600
    minutes = (uptime % 3600) // 60

    click.echo(f"Worker: {info.get('device_name', 'unknown')}")
    click.echo(f"Commit: {info.get('commit', '?')} ({info.get('branch', '?')}){dirty}")
    click.echo(f"Model:  {info.get('model', '?')}")
    click.echo(f"Torch:  {info.get('torch', '?')} / Python {info.get('python', '?')}")
    click.echo(f"Uptime: {hours}h {minutes}m")


@worker.command()
@click.option("--remote", required=True, help="Worker URL (e.g. http://172.30.0.1:8877)")
@click.option(
    "--model", default=None,
    help="New model to load (e.g. Qwen/Qwen2.5-3B-Instruct).",
)
@click.option(
    "--dtype", default=None,
    type=click.Choice(["auto", "float16", "float32", "bfloat16"]),
    help="Model dtype.",
)
def reload(remote: str, model: str | None, dtype: str | None):
    """Change the model and/or dtype on the GPU worker without restarting."""
    if model is None and dtype is None:
        raise click.UsageError("Must provide --model and/or --dtype.")

    from neurotrace.remote import RemoteWorker

    w = RemoteWorker(remote, timeout=600.0)

    for event in w.reload_stream(model=model, dtype=dtype):
        status = event.get("status", "")
        message = event.get("message", "")

        if status == "unloading":
            err_console.print(f"  [dim]{message}[/dim]")
        elif status == "loading":
            err_console.print(f"  [yellow]{message}[/yellow]")
        elif "model" in event and "dtype" in event and status != "unloading":
            # Done event
            new_model = event.get("model", "?")
            new_dtype = event.get("dtype", "?")
            vram_mb = event.get("vram_mb", 0)
            click.echo(f"Reloaded: {new_model} ({new_dtype})")
            if vram_mb:
                click.echo(f"VRAM: {vram_mb} MB")
        elif message and not status:
            err_console.print(f"  [red]{message}[/red]")


@worker.command()
@click.option("--remote", required=True, help="Worker URL (e.g. http://172.30.0.1:8877)")
def update(remote: str):
    """Pull latest code and restart the GPU worker."""
    from neurotrace.remote import RemoteWorker

    w = RemoteWorker(remote)

    with err_console.status("Updating worker..."):
        for event in w.worker_update_stream():
            status = event.get("status", "")
            message = event.get("message", "")

            if status == "pulling":
                err_console.print(f"  [dim]{message}[/dim]")
            elif status == "pulled":
                err_console.print(f"  {message}")
            elif status == "restarting":
                err_console.print(f"  [yellow]{message}[/yellow]")
            elif "old_commit" in event:
                old = event.get("old_commit", "?")
                new = event.get("new_commit", "?")
                changed = event.get("changed", False)
                if changed:
                    click.echo(f"Updated: {old} -> {new}")
                else:
                    click.echo(f"Already up to date ({old})")
            elif "message" in event and status == "":
                err_console.print(f"  [red]{message}[/red]")
