"""Output formatting for context-guardian analysis results.

Modes:
    inline  — rich panel printed after every call (only when risk > low)
    table   — summary table printed at session end
    json    — machine-readable dict
    silent  — no output, session-only
    html    — browser heatmap (delegates to llm_stress_test.visualize)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import AnalysisResult

_RISK_COLORS = {
    "low": "green",
    "medium": "yellow",
    "high": "red",
    "critical": "bold red",
}

_RISK_ICONS = {
    "low": "✓",
    "medium": "~",
    "high": "⚠",
    "critical": "✖",
}


# ---------------------------------------------------------------------------
# Inline panel
# ---------------------------------------------------------------------------

def print_inline(result: "AnalysisResult") -> None:
    """Print the rich panel for a single call. Skipped when risk is low and faithfulness is high."""
    if result.position_risk == "low" and result.faithfulness_score >= 0.8:
        return

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console(stderr=False)
        risk_color = _RISK_COLORS.get(result.position_risk, "white")
        icon = _RISK_ICONS.get(result.position_risk, "?")

        lines: list[str] = []

        risk_label = result.position_risk.upper()
        lines.append(
            f"[{risk_color}]{icon}  {risk_label} RISK[/{risk_color}]"
            f"  │  Depth: [bold]{result.risk_depth_pct:.0f}%[/bold]"
            f"  │  Tokens: [bold]{result.context_tokens:,}[/bold]"
            f"  │  Model: {result.model}"
        )

        faith_color = "green" if result.faithfulness_score >= 0.8 else (
            "yellow" if result.faithfulness_score >= 0.6 else "red"
        )
        lines.append(
            f"Faithfulness: [{faith_color}]{result.faithfulness_score:.2f}[/{faith_color}]"
            + ("  [dim]— answer may not use context[/dim]" if result.faithfulness_flag else "")
        )

        if result.fix_suggestion:
            lines.append(f"[cyan]💡 {result.fix_suggestion}[/cyan]")

        console.print(
            Panel(
                "\n".join(lines),
                title="[bold dim]context-guardian[/bold dim]",
                border_style=risk_color,
                expand=False,
            )
        )
    except ImportError:
        _print_inline_plain(result)


def _print_inline_plain(result: "AnalysisResult") -> None:
    icon = _RISK_ICONS.get(result.position_risk, "?")
    print(
        f"[context-guardian] {icon} {result.position_risk.upper()} "
        f"| depth={result.risk_depth_pct:.0f}% "
        f"| tokens={result.context_tokens:,} "
        f"| faith={result.faithfulness_score:.2f} "
        f"| {result.fix_suggestion}"
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(results: list["AnalysisResult"]) -> None:
    """Print end-of-session summary table."""
    if not results:
        return

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title="[bold]context-guardian — Session Summary[/bold]",
            box=box.ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Tokens", justify="right")
        table.add_column("Risk", justify="center")
        table.add_column("Faithfulness", justify="center")
        table.add_column("Depth", justify="center")
        table.add_column("Model")

        for i, r in enumerate(results, 1):
            risk_color = _RISK_COLORS.get(r.position_risk, "white")
            faith_color = "green" if r.faithfulness_score >= 0.8 else (
                "yellow" if r.faithfulness_score >= 0.6 else "red"
            )
            table.add_row(
                str(i),
                f"{r.context_tokens:,}",
                f"[{risk_color}]{r.position_risk.upper()}[/{risk_color}]",
                f"[{faith_color}]{r.faithfulness_score:.2f}[/{faith_color}]",
                f"{r.risk_depth_pct:.0f}%",
                r.model,
            )

        console.print(table)

    except ImportError:
        _print_table_plain(results)


def _print_table_plain(results: list["AnalysisResult"]) -> None:
    header = f"{'#':>3}  {'Tokens':>8}  {'Risk':>8}  {'Faith':>6}  {'Depth':>6}  Model"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, 1):
        print(
            f"{i:>3}  {r.context_tokens:>8,}  {r.position_risk.upper():>8}"
            f"  {r.faithfulness_score:>6.2f}  {r.risk_depth_pct:>5.0f}%  {r.model}"
        )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def to_json(result: "AnalysisResult") -> dict:
    return {
        "call_index": result.call_index,
        "model": result.model,
        "total_tokens": result.total_tokens,
        "context_tokens": result.context_tokens,
        "question_snippet": result.question[:120],
        "position_risk": result.position_risk,
        "risk_depth_pct": round(result.risk_depth_pct, 1),
        "faithfulness_score": round(result.faithfulness_score, 4),
        "faithfulness_flag": result.faithfulness_flag,
        "fix_suggestion": result.fix_suggestion,
    }


def dump_json(results: list["AnalysisResult"]) -> str:
    return json.dumps([to_json(r) for r in results], indent=2)
