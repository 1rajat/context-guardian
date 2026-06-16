"""Click CLI entry point for context-guardian."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def main():
    """context-guardian — real-time context loss detection for LLM apps."""


@main.command("watch")
@click.argument("script", type=click.Path(exists=True))
@click.option("--model", "-m", default="", help="Model hint (inferred from calls if omitted).")
@click.option("--report", default="end", type=click.Choice(["inline", "end", "silent", "json"]),
              show_default=True)
@click.option("--app", default="default", show_default=True, help="Session app name.")
def watch(script, model, report, app):
    """Monkey-patch a Python script and report context risks after it exits."""
    import runpy
    from context_guardian.watcher import ContextWatcher

    watcher = ContextWatcher(model=model, report=report, app=app)
    with watcher.watch():
        runpy.run_path(script, run_name="__main__")

    if report == "end":
        watcher.report()


@main.group("session")
def session_group():
    """Manage persistent session history."""


@session_group.command("show")
@click.option("--app", default="default", show_default=True)
def session_show(app):
    """Show session summary for an app."""
    from context_guardian.session import Session
    s = Session(app)
    s.print_summary()


@session_group.command("clear")
@click.option("--app", default="default", show_default=True)
@click.confirmation_option(prompt="Clear session history?")
def session_clear(app):
    """Clear session history for an app."""
    from context_guardian.session import Session
    s = Session(app)
    s.clear()
    console.print(f"[green]Session '[bold]{app}[/bold]' cleared.[/green]")


@session_group.command("list")
def session_list():
    """List all apps with saved session history."""
    from context_guardian.session import Session
    apps = Session.list_apps()
    if not apps:
        console.print("[dim]No sessions found.[/dim]")
        return
    for a in apps:
        console.print(f"  [cyan]{a}[/cyan]")


@main.command("report")
@click.argument("jsonl_file", type=click.Path(exists=True))
def guardian_report(jsonl_file):
    """Generate a report from a saved JSONL session file."""
    import json
    from context_guardian.reporter import print_table
    from context_guardian.analyzer import AnalysisResult

    records = []
    for line in Path(jsonl_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(AnalysisResult(
                model=d.get("model", "unknown"),
                total_tokens=d.get("total_tokens", 0),
                context_tokens=d.get("context_tokens", 0),
                question=d.get("question_snippet", ""),
                position_risk=d.get("position_risk", "low"),
                risk_depth=d.get("risk_depth_pct", 0) / 100,
                faithfulness_score=d.get("faithfulness_score", 1.0),
                fix_suggestion=d.get("fix_suggestion", ""),
            ))
        except Exception:
            pass

    if not records:
        console.print("[yellow]No valid records found in file.[/yellow]")
        return

    print_table(records)


@main.command("insights")
@click.option("--app", default="default", show_default=True, help="Session app name.")
def insights(app):
    """Show learned blind-spot insights for an app."""
    from context_guardian.session import _SESSION_DIR
    from rich.markdown import Markdown

    insights_file = _SESSION_DIR / f"{app}_insights.md"
    if not insights_file.exists():
        console.print(
            f"[yellow]No insights yet for '[bold]{app}[/bold]'.[/yellow]\n"
            "[dim]Insights are generated after 50+ calls. "
            "Run your app with Guardian to accumulate history.[/dim]"
        )
        return
    md = Markdown(insights_file.read_text(encoding="utf-8"))
    console.print(md)
