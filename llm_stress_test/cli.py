"""Click CLI entry point for llm-stress-test."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import box

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_context_lengths(value: str | None) -> list[int]:
    """Parse '4k,16k,32k' → [4, 16, 32]."""
    defaults = [1, 2, 4, 8, 16, 32, 64, 128]
    if not value:
        return defaults
    result = []
    for part in value.split(","):
        part = part.strip().lower().rstrip("k")
        result.append(int(part))
    return result


def _fmt_cost(usd: float) -> str:
    if usd < 0.001:
        return "<$0.001"
    return f"${usd:.4f}"


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option()
def main():
    """LLM Context Window Stress Tester — needle-in-a-haystack benchmarking."""


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", "-m", required=True, help="Model name (e.g. gpt-4o, claude-3-5-sonnet, llama3).")
@click.option("--max-context", default=None, help="Max context to test, e.g. 32k. Default: all sizes.")
@click.option("--context-lengths", default=None, help="Comma-separated list, e.g. '4k,16k,32k'.")
@click.option("--depths", default=None, help="Comma-separated depth percents, e.g. '0,25,50,75,100'.")
@click.option("--trials", "-t", default=3, show_default=True, help="Trials per cell.")
@click.option("--output", "-o", default="results", show_default=True, help="Output directory.")
@click.option("--quick", is_flag=True, help="Quick mode: 4 context lengths × 3 depths.")
@click.option("--cost-estimate", is_flag=True, help="Show cost estimate and exit without running.")
@click.option("--ollama-url", default="http://localhost:11434", help="Ollama base URL.")
@click.option("--no-html", is_flag=True, help="Skip interactive HTML heatmap.")
@click.option("--no-png", is_flag=True, help="Skip PNG heatmap.")
@click.option("--demo", is_flag=True, help="Synthetic demo run — no API key needed.")
def run(model, max_context, context_lengths, depths, trials, output, quick,
        cost_estimate, ollama_url, no_html, no_png, demo):
    """Run the needle-in-a-haystack stress test."""
    from .harness import (
        CONTEXT_LENGTHS_K, DEPTH_PERCENTS, QUICK_CONTEXT_LENGTHS_K, QUICK_DEPTH_PERCENTS,
        estimate_total_cost, run_demo, run_stress_test,
    )
    from .models import get_adapter

    # Resolve context lengths
    if quick:
        ctx_k = QUICK_CONTEXT_LENGTHS_K
        depth_pcts = QUICK_DEPTH_PERCENTS
    elif context_lengths:
        ctx_k = _parse_context_lengths(context_lengths)
        depth_pcts = [float(d) for d in depths.split(",")] if depths else DEPTH_PERCENTS
    else:
        ctx_k = _parse_context_lengths(None)
        depth_pcts = [float(d) for d in depths.split(",")] if depths else DEPTH_PERCENTS

    # Apply max-context cap
    if max_context:
        cap_k = int(max_context.lower().rstrip("k"))
        ctx_k = [k for k in ctx_k if k <= cap_k]

    # Demo mode — skip adapter setup and cost check entirely
    if demo:
        console.print(
            Panel(
                "[bold yellow]Demo mode[/bold yellow] — synthetic data, no API calls.\n"
                f"Model: [cyan]{model}[/cyan]  |  Cells: {len(ctx_k) * len(depth_pcts)}",
                expand=False,
            )
        )
        total_cells = len(ctx_k) * len(depth_pcts)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(f"Simulating [cyan]{model}[/cyan]", total=total_cells)

            def on_cell_demo(done, total, k, depth, cell):
                icon = "[green]✓[/green]" if cell.score >= 0.9 else (
                    "[yellow]~[/yellow]" if cell.score >= 0.5 else "[red]✗[/red]"
                )
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Simulating [cyan]{model}[/cyan] | "
                        f"ctx=[bold]{k}k[/bold] depth=[bold]{int(depth)}%[/bold] "
                        f"score={icon}[bold]{cell.score:.2f}[/bold]"
                    ),
                )

            result, json_path = run_demo(
                model=model,
                context_lengths_k=ctx_k,
                depth_percents=depth_pcts,
                trials=trials,
                output_dir=output,
                quick=quick,
                progress_callback=on_cell_demo,
            )
        _finish_run(console, result, json_path, output, no_png, no_html)
        return

    try:
        adapter = get_adapter(model, ollama_url=ollama_url)
    except (EnvironmentError, ImportError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # Cost estimate
    est = estimate_total_cost(adapter, ctx_k, depth_pcts, trials)
    _print_cost_panel(model, ctx_k, depth_pcts, trials, est)

    if cost_estimate:
        return

    total_cells = len(ctx_k) * len(depth_pcts)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"Testing [cyan]{model}[/cyan]", total=total_cells)

        def on_cell(done, total, k, depth, cell):
            icon = "[green]✓[/green]" if cell.score >= 0.9 else (
                "[yellow]~[/yellow]" if cell.score >= 0.5 else "[red]✗[/red]"
            )
            progress.update(
                task,
                advance=1,
                description=(
                    f"Testing [cyan]{model}[/cyan] | "
                    f"ctx=[bold]{k}k[/bold] depth=[bold]{int(depth)}%[/bold] "
                    f"score={icon}[bold]{cell.score:.2f}[/bold]"
                ),
            )

        try:
            result, json_path = run_stress_test(
                model=model,
                context_lengths_k=ctx_k,
                depth_percents=depth_pcts,
                trials=trials,
                output_dir=output,
                quick=quick,
                ollama_url=ollama_url,
                progress_callback=on_cell,
            )
        except RuntimeError as exc:
            progress.stop()
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
            console.print(
                "\n[dim]Tip: make sure your API key is exported, e.g.  "
                "export OPENAI_API_KEY=sk-...[/dim]"
            )
            sys.exit(1)

    _finish_run(console, result, json_path, output, no_png, no_html)


def _finish_run(console, result, json_path, output, no_png, no_html):
    from .visualize import print_ascii_heatmap, save_html, save_png

    console.print()
    console.print(f"[bold green]Run complete![/bold green] Results saved to [cyan]{json_path}[/cyan]")
    total_tok = result.total_input_tokens + result.total_output_tokens
    demo_note = "  [dim](demo — no real tokens)[/dim]" if result.metadata.get("demo") else ""
    console.print(
        f"  Total tokens used: [bold]{total_tok:,}[/bold]"
        f"  |  Elapsed: [bold]{result.elapsed_seconds:.1f}s[/bold]{demo_note}"
    )

    console.print()
    print_ascii_heatmap(result)

    if not no_png:
        try:
            png = save_png(result, output)
            console.print(f"\n[bold]PNG heatmap:[/bold] {png}")
        except Exception as exc:
            console.print(f"[yellow]PNG generation failed:[/yellow] {exc}")

    if not no_html:
        try:
            html = save_html(result, output)
            console.print(f"[bold]HTML heatmap:[/bold] {html}")
        except Exception as exc:
            console.print(f"[yellow]HTML generation failed:[/yellow] {exc}")


# ---------------------------------------------------------------------------
# plot command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="results", show_default=True)
@click.option("--no-html", is_flag=True)
@click.option("--no-png", is_flag=True)
def plot(json_file, output, no_html, no_png):
    """Regenerate heatmaps from a saved JSON result file."""
    from .harness import load_result
    from .visualize import print_ascii_heatmap, save_html, save_png

    result = load_result(json_file)
    console.print(f"Loaded results for [cyan]{result.model}[/cyan] ({result.timestamp[:10]})")
    console.print(f"  {len(result.cells)} cells, {result.trials_per_cell} trials each\n")

    print_ascii_heatmap(result)

    if not no_png:
        try:
            png = save_png(result, output)
            console.print(f"\n[bold]PNG:[/bold] {png}")
        except Exception as exc:
            console.print(f"[yellow]PNG failed:[/yellow] {exc}")

    if not no_html:
        try:
            html = save_html(result, output)
            console.print(f"[bold]HTML:[/bold] {html}")
        except Exception as exc:
            console.print(f"[yellow]HTML failed:[/yellow] {exc}")


# ---------------------------------------------------------------------------
# compare command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("json_files", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--output", "-o", default="results", show_default=True)
def compare(json_files, output):
    """Side-by-side heatmap comparison of two or more model result files."""
    from .harness import load_result
    from .visualize import save_comparison_html

    if len(json_files) < 2:
        console.print("[red]Provide at least two JSON result files to compare.[/red]")
        sys.exit(1)

    results = [load_result(f) for f in json_files]
    for r in results:
        console.print(f"  Loaded [cyan]{r.model}[/cyan] — {len(r.cells)} cells")

    try:
        html = save_comparison_html(results, output)
        console.print(f"\n[bold]Comparison HTML:[/bold] {html}")
    except Exception as exc:
        console.print(f"[red]Comparison failed:[/red] {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# models command
# ---------------------------------------------------------------------------

@main.command("list-models")
def list_models():
    """List known supported models."""
    from .models import list_known_models

    table = Table(title="Known Models", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Model", style="cyan")
    table.add_column("Provider")
    table.add_column("Max Context")

    from .models import _MAX_CONTEXT, _COST_TABLE
    for m in list_known_models():
        if m.startswith("gpt"):
            provider = "OpenAI"
        elif m.startswith("claude"):
            provider = "Anthropic"
        else:
            provider = "Ollama (local)"
        max_ctx = _MAX_CONTEXT.get(m, "?")
        max_ctx_str = f"{max_ctx // 1000}k" if isinstance(max_ctx, int) else str(max_ctx)
        table.add_row(m, provider, max_ctx_str)

    console.print(table)
    console.print("\n[dim]For Ollama models, pass any model name — it will use http://localhost:11434[/dim]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_cost_panel(model, ctx_k, depth_pcts, trials, est):
    lines = [
        f"[bold]Model:[/bold]              {model}",
        f"[bold]Context lengths:[/bold]    {', '.join(str(k)+'k' for k in ctx_k)}",
        f"[bold]Depths:[/bold]             {', '.join(str(int(d))+'%' for d in depth_pcts)}",
        f"[bold]Trials per cell:[/bold]    {trials}",
        f"[bold]Total cells:[/bold]        {len(ctx_k) * len(depth_pcts)}",
        f"[bold]Est. input tokens:[/bold]  {est['total_input_tokens']:,}",
        f"[bold]Est. cost:[/bold]          {_fmt_cost(est['estimated_usd'])}",
        f"[bold]Model max context:[/bold]  {est['model_max_context'] // 1000}k",
    ]
    if est["cells_skipped"]:
        lines.append(f"[yellow]Cells skipped (over model limit): {est['cells_skipped']}[/yellow]")

    console.print(
        Panel("\n".join(lines), title="[bold cyan]Run Configuration[/bold cyan]", expand=False)
    )
