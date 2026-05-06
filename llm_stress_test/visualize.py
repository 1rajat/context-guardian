"""Heatmap generation — PNG via matplotlib and interactive HTML via plotly."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .harness import RunResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_matrix(result: RunResult) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (matrix, col_labels, row_labels) where matrix[row][col] = score."""
    ctx_k = sorted(set(c.context_length_k for c in result.cells))
    depths = sorted(set(c.depth_pct for c in result.cells))

    col_labels = [f"{k}k" for k in ctx_k]
    row_labels = [f"{int(d)}%" for d in depths]

    cell_map = {(c.context_length_k, c.depth_pct): c for c in result.cells}

    matrix = np.full((len(depths), len(ctx_k)), np.nan)
    for ri, d in enumerate(depths):
        for ci, k in enumerate(ctx_k):
            cell = cell_map.get((k, d))
            if cell is not None and cell.error is None:
                matrix[ri, ci] = cell.score

    return matrix, col_labels, row_labels, ctx_k, depths, cell_map


def _score_to_hex(score: float) -> str:
    """Map 0.0→red, 1.0→green using a smooth gradient."""
    r = int(255 * (1 - score))
    g = int(200 * score)
    return f"#{r:02x}{g:02x}00"


# ---------------------------------------------------------------------------
# PNG heatmap (matplotlib)
# ---------------------------------------------------------------------------

def save_png(result: RunResult, output_dir: str = "results") -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
    except ImportError:
        raise ImportError("matplotlib not installed. Run: pip install matplotlib")

    matrix, col_labels, row_labels, ctx_k, depths, cell_map = _build_matrix(result)

    fig, ax = plt.subplots(figsize=(max(10, len(ctx_k) * 1.4), max(6, len(depths) * 0.9)))

    # Custom colormap: red → yellow → green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#d73027", "#fee08b", "#1a9850"], N=256
    )
    cmap.set_bad(color="#cccccc")  # grey for skipped cells

    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Context Length (tokens)", fontsize=13, labelpad=10)
    ax.set_ylabel("Needle Depth (position in document)", fontsize=13, labelpad=10)

    model_display = result.model
    ax.set_title(
        f"Needle-in-Haystack: {model_display}\n{result.timestamp[:10]}",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    # Annotate each cell with the score
    for ri in range(len(depths)):
        for ci in range(len(ctx_k)):
            val = matrix[ri, ci]
            if np.isnan(val):
                ax.text(ci, ri, "—", ha="center", va="center", fontsize=10, color="#888")
            else:
                text_color = "white" if val < 0.4 else "black"
                ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Retrieval Score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_model = result.model.replace("/", "_").replace(":", "_")
    ts = result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    png_file = out_path / f"heatmap_{safe_model}_{ts}.png"
    fig.savefig(png_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_file


# ---------------------------------------------------------------------------
# Interactive HTML heatmap (plotly)
# ---------------------------------------------------------------------------

def save_html(result: RunResult, output_dir: str = "results") -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly not installed. Run: pip install plotly")

    matrix, col_labels, row_labels, ctx_k, depths, cell_map = _build_matrix(result)

    # Build hover text
    hover = []
    for ri, d in enumerate(depths):
        row_hover = []
        for ci, k in enumerate(ctx_k):
            cell = cell_map.get((k, d))
            if cell is None or np.isnan(matrix[ri, ci]):
                row_hover.append("Skipped (exceeds model limit)")
            else:
                row_hover.append(
                    f"<b>Score:</b> {cell.score:.3f}<br>"
                    f"<b>Exact match rate:</b> {cell.exact_match_rate:.0%}<br>"
                    f"<b>Hallucination rate:</b> {cell.hallucination_rate:.0%}<br>"
                    f"<b>Context length:</b> {k}k tokens<br>"
                    f"<b>Needle depth:</b> {d}%<br>"
                    f"<b>Actual tokens:</b> {cell.actual_tokens:,}<br>"
                    f"<b>Trials:</b> {cell.trials}"
                )
        hover.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale=[
                [0.0, "#d73027"],
                [0.5, "#fee08b"],
                [1.0, "#1a9850"],
            ],
            zmin=0.0,
            zmax=1.0,
            text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in matrix],
            texttemplate="%{text}",
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            colorbar=dict(title="Score", tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Needle-in-Haystack: <b>{result.model}</b>  |  {result.timestamp[:10]}",
            font=dict(size=18),
            x=0.5,
        ),
        xaxis_title="Context Length (tokens)",
        yaxis_title="Needle Depth (position in document)",
        width=900,
        height=500 + len(depths) * 30,
        font=dict(size=12),
        plot_bgcolor="white",
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_model = result.model.replace("/", "_").replace(":", "_")
    ts = result.timestamp[:19].replace(":", "").replace("-", "").replace("T", "_")
    html_file = out_path / f"heatmap_{safe_model}_{ts}.html"
    fig.write_html(str(html_file), include_plotlyjs="cdn")
    return html_file


# ---------------------------------------------------------------------------
# Side-by-side comparison heatmap
# ---------------------------------------------------------------------------

def save_comparison_html(results: list[RunResult], output_dir: str = "results") -> Path:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly not installed.")

    n = len(results)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[r.model for r in results],
        shared_yaxes=True,
    )

    colorscale = [
        [0.0, "#d73027"],
        [0.5, "#fee08b"],
        [1.0, "#1a9850"],
    ]

    for idx, result in enumerate(results):
        matrix, col_labels, row_labels, ctx_k, depths, cell_map = _build_matrix(result)

        hover = []
        for ri, d in enumerate(depths):
            row_hover = []
            for ci, k in enumerate(ctx_k):
                cell = cell_map.get((k, d))
                if cell is None or np.isnan(matrix[ri, ci]):
                    row_hover.append("Skipped")
                else:
                    row_hover.append(
                        f"<b>Model:</b> {result.model}<br>"
                        f"<b>Score:</b> {cell.score:.3f}<br>"
                        f"<b>Context:</b> {k}k | Depth: {d}%"
                    )
            hover.append(row_hover)

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=col_labels,
                y=row_labels,
                colorscale=colorscale,
                zmin=0.0,
                zmax=1.0,
                text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row] for row in matrix],
                texttemplate="%{text}",
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
                showscale=(idx == n - 1),
            ),
            row=1,
            col=idx + 1,
        )

    model_names = " vs ".join(r.model for r in results)
    fig.update_layout(
        title=dict(text=f"Model Comparison: {model_names}", font=dict(size=16), x=0.5),
        height=500,
        width=600 * n,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = results[0].timestamp[:10].replace("-", "")
    safe_names = "_vs_".join(r.model.replace("/", "_").replace(":", "_") for r in results)
    html_file = out_path / f"compare_{safe_names}_{ts}.html"
    fig.write_html(str(html_file), include_plotlyjs="cdn")
    return html_file


# ---------------------------------------------------------------------------
# ASCII heatmap (terminal)
# ---------------------------------------------------------------------------

def print_ascii_heatmap(result: RunResult) -> None:
    """Print a compact ASCII heatmap to stdout using rich if available."""
    matrix, col_labels, row_labels, ctx_k, depths, cell_map = _build_matrix(result)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        table = Table(
            title=f"[bold]Needle-in-Haystack Results — {result.model}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Depth \\ Ctx", style="bold", justify="center")
        for lbl in col_labels:
            table.add_column(lbl, justify="center")

        for ri, lbl in enumerate(row_labels):
            row_cells = [lbl]
            for ci in range(len(col_labels)):
                val = matrix[ri, ci]
                if np.isnan(val):
                    row_cells.append("[dim]—[/dim]")
                elif val >= 0.9:
                    row_cells.append(f"[bold green]{val:.2f}[/bold green]")
                elif val >= 0.5:
                    row_cells.append(f"[yellow]{val:.2f}[/yellow]")
                else:
                    row_cells.append(f"[bold red]{val:.2f}[/bold red]")
            table.add_row(*row_cells)

        console.print(table)

    except ImportError:
        # Fallback plain ASCII
        header = "Depth \\ Ctx | " + " | ".join(f"{l:>5}" for l in col_labels)
        print(header)
        print("-" * len(header))
        for ri, lbl in enumerate(row_labels):
            vals = []
            for ci in range(len(col_labels)):
                v = matrix[ri, ci]
                vals.append(f"{v:.2f}" if not np.isnan(v) else "  — ")
            print(f"{lbl:>11} | " + " | ".join(f"{v:>5}" for v in vals))
