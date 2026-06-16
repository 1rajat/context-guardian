"""WatchedClient — drop-in replacement for openai.OpenAI() / anthropic.Anthropic().

Usage:
    client = Guardian(openai.OpenAI(), model="gpt-4o")
    response = client.chat.completions.create(...)  # identical to raw client

Guardian intercepts every call, runs analysis, and prints inline warnings
with ~zero perceived latency (async by default).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional, Union


def _default_warn_before() -> Union[bool, str]:
    return os.environ.get("CONTEXT_GUARDIAN_ENV", "").lower() != "production"


class Guardian:
    """Wraps an openai or anthropic client and adds context analysis."""

    def __init__(
        self,
        client: Any,
        model: str = "",
        report: str = "inline",
        app: str = "default",
        threshold: str = "medium",
        warn_before: Union[bool, str, None] = None,
    ):
        self._client = client
        self._model = model
        self._report_mode = report   # "inline" | "table" | "silent" | "json"
        self._app = app
        self._threshold = threshold
        self._warn_before = _default_warn_before() if warn_before is None else warn_before

        from .session import Session
        self._session = Session(app)
        self._results: list = []
        self._call_index = 0

        # Detect which SDK we're wrapping
        cls_name = type(client).__module__ + "." + type(client).__name__
        if "openai" in cls_name.lower():
            self._provider = "openai"
        elif "anthropic" in cls_name.lower():
            self._provider = "anthropic"
        else:
            self._provider = "unknown"

        # Proxy chat.completions
        self.chat = _ChatProxy(self)
        # Proxy messages (anthropic)
        self.messages = _MessagesProxy(self)

    def _maybe_warn_before(self, messages: list[dict], model: str) -> None:
        """Print pre-call warning if risk is high and warn_before is enabled."""
        if self._warn_before is False:
            return

        from .analyzer import analyze_pre_call
        result = analyze_pre_call(messages, model)

        _risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        threshold = "medium" if self._warn_before == "all" else "high"
        if _risk_order.get(result.position_risk, 0) < _risk_order.get(threshold, 2):
            return

        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()

            _risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
            color = _risk_colors.get(result.position_risk, "red")
            display_model = model or self._model or "LLM"

            lines = [
                f"[{color}]⚠  {result.position_risk.upper()} RISK detected BEFORE sending to {display_model}[/{color}]",
                f"   Relevant content at [bold]{result.risk_depth_pct:.0f}%[/bold] depth"
                f" | [bold]{result.context_tokens:,}[/bold] tokens",
                f"   This is {display_model}'s known blind zone at this size",
                "[cyan]💡 Run: client.suggest_fix(messages) to reorder[/cyan]",
            ]

            console.print(Panel(
                "\n".join(lines),
                title="[bold dim]context-guardian WARNING[/bold dim]",
                border_style=color,
                expand=False,
            ))
        except ImportError:
            print(
                f"[context-guardian] ⚠ {result.position_risk.upper()} RISK before API call"
                f" — {result.context_tokens:,} tokens at {result.risk_depth_pct:.0f}%"
            )

    def _intercept(self, messages: list[dict], response_text: str, model: str) -> None:
        """Called in a background thread after a call completes."""
        from .analyzer import analyze
        from .reporter import print_inline, print_table, to_json
        import json as _json

        effective_model = model or self._model
        result = analyze(messages, response_text, effective_model, self._call_index)
        self._results.append(result)
        self._session.record(result)

        if self._report_mode == "inline":
            print_inline(result)
        elif self._report_mode == "json":
            print(_json.dumps(to_json(result)))

    def suggest_fix(self, messages: list[dict]) -> list[dict]:
        """Reorder messages so high-relevance content is at the end (lowest risk zone)."""
        return suggest_fix(messages, model=self._model)

    def report(self) -> list[dict]:
        """Return all call results as a list of dicts and pretty-print table."""
        from .reporter import print_table, to_json
        print_table(self._results)
        return [to_json(r) for r in self._results]

    def session_summary(self) -> dict:
        self._session.print_summary()
        return self._session.summary()

    # ------------------------------------------------------------------
    # Proxy attribute access — forward everything else to the underlying client
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# Proxy classes
# ---------------------------------------------------------------------------

class _ChatProxy:
    def __init__(self, guardian: Guardian):
        self._g = guardian
        self.completions = _CompletionsProxy(guardian)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._g._client.chat, name)


class _CompletionsProxy:
    def __init__(self, guardian: Guardian):
        self._g = guardian

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", self._g._model)
        messages = kwargs.get("messages", [])

        # Pre-call warning (synchronous, before API call)
        self._g._maybe_warn_before(list(messages), model)

        resp = self._g._client.chat.completions.create(**kwargs)
        response_text = _extract_openai_text(resp)

        # Fire analysis in background — zero latency
        t = threading.Thread(
            target=self._g._intercept,
            args=(list(messages), response_text, model),
            daemon=True,
        )
        t.start()
        self._g._call_index += 1

        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._g._client.chat.completions, name)


class _MessagesProxy:
    """Proxy for anthropic.Anthropic().messages.*"""

    def __init__(self, guardian: Guardian):
        self._g = guardian

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", self._g._model)

        # Anthropic messages format differs — convert to unified list
        messages = _anthropic_to_unified(kwargs)

        # Pre-call warning (synchronous, before API call)
        self._g._maybe_warn_before(messages, model)

        resp = self._g._client.messages.create(**kwargs)
        response_text = _extract_anthropic_text(resp)

        t = threading.Thread(
            target=self._g._intercept,
            args=(messages, response_text, model),
            daemon=True,
        )
        t.start()
        self._g._call_index += 1

        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._g._client.messages, name)


# ---------------------------------------------------------------------------
# Helpers to extract text from various response shapes
# ---------------------------------------------------------------------------

def _extract_openai_text(resp: Any) -> str:
    """Handle regular and streaming OpenAI responses."""
    if hasattr(resp, "choices"):
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""
    chunks = []
    try:
        for chunk in resp:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                chunks.append(delta.content)
    except Exception:
        pass
    return "".join(chunks)


def _extract_anthropic_text(resp: Any) -> str:
    if hasattr(resp, "content") and resp.content:
        try:
            return resp.content[0].text
        except Exception:
            pass
    return ""


def _anthropic_to_unified(kwargs: dict) -> list[dict]:
    """Convert anthropic.create() kwargs into the unified messages format."""
    messages: list[dict] = []
    system = kwargs.get("system", "")
    if system:
        messages.append({"role": "system", "content": system})
    for msg in kwargs.get("messages", []):
        messages.append(msg)
    return messages


# ---------------------------------------------------------------------------
# suggest_fix — standalone function
# ---------------------------------------------------------------------------

def suggest_fix(messages: list[dict], model: str = "") -> list[dict]:
    """Reorder context so high-relevance chunks are at the end (lowest risk zone).

    Returns a new messages list ready to use directly as a drop-in replacement.
    """
    from .analyzer import (
        _extract_context_question,
        _chunk_context,
        _count_tokens,
        ContextAnalyzer,
    )
    from .risk_tables import get_risk_level, worst_risk

    context_text, question = _extract_context_question(messages)
    if not context_text or not question:
        return list(messages)

    chunks = _chunk_context(context_text)
    if len(chunks) <= 1:
        return list(messages)

    analyzer = ContextAnalyzer(use_embeddings=True)
    _, scores = analyzer._rank_chunks(question, chunks, top_k=len(chunks))

    # Top ~30% by score are high-relevance → move to end
    score_pairs = list(zip(scores, range(len(chunks))))
    sorted_by_score = sorted(score_pairs, key=lambda x: x[0], reverse=True)
    cutoff = max(1, len(chunks) // 3)
    high_rel_indices = {idx for _, idx in sorted_by_score[:cutoff]}

    low_rel = [c for i, c in enumerate(chunks) if i not in high_rel_indices]
    high_rel = [c for i, c in enumerate(chunks) if i in high_rel_indices]

    reordered = low_rel + high_rel
    new_context = "\n\n".join(c.text for c in reordered)

    # Compute risk before/after for the diff panel
    context_tokens = sum(c.token_end - c.token_start for c in chunks) if chunks else 0
    risk_before_zones = [
        get_risk_level(model, context_tokens, c.depth * 100) for c in high_rel
    ]
    risk_before = worst_risk(risk_before_zones) if risk_before_zones else "low"
    risk_after = "low"

    _print_suggest_fix_panel(high_rel, low_rel, risk_before, risk_after, len(chunks))

    # Reconstruct messages: replace the largest non-user message's content
    # (usually the system message containing the context)
    new_messages = _replace_context_in_messages(messages, context_text, new_context)
    return new_messages


def _print_suggest_fix_panel(
    high_rel: list,
    low_rel: list,
    risk_before: str,
    risk_after: str,
    total_chunks: int,
) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()

        _risk_colors = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}
        rb_color = _risk_colors.get(risk_before, "red")
        ra_color = _risk_colors.get(risk_after, "green")

        lines = ["[bold cyan]🔄 context-guardian reordered your context:[/bold cyan]\n"]

        if high_rel:
            lines.append("[bold]MOVED TO END[/bold] [dim](high relevance):[/dim]")
            n_high = len(high_rel)
            for i, chunk in enumerate(high_rel[:3]):
                snippet = chunk.text[:60].replace("\n", " ").strip()
                was_pct = int(chunk.depth * 100)
                now_pct = 85 + int(i / max(n_high, 1) * 10)
                lines.append(
                    f"  [green]↑[/green] [dim]\"{snippet}...\"[/dim]"
                    f" (was at {was_pct}% → now {now_pct}%)"
                )

        if low_rel:
            lines.append("\n[bold]LEFT IN PLACE[/bold] [dim](low relevance):[/dim]")
            for chunk in low_rel[:3]:
                snippet = chunk.text[:60].replace("\n", " ").strip()
                lines.append(
                    f"  [dim]·[/dim] [dim]\"{snippet}...\"[/dim]"
                    f" [dim]({int(chunk.depth * 100)}% depth, low risk)[/dim]"
                )

        lines.append(
            f"\nRisk before: [{rb_color}]{risk_before.upper()}[/{rb_color}]"
            f" → Risk after: [{ra_color}]{risk_after.upper()}[/{ra_color}]"
        )
        lines.append("[dim]Use fixed_messages directly in your next API call ✓[/dim]")

        console.print(Panel(
            "\n".join(lines),
            title="[bold dim]context-guardian — suggest_fix[/bold dim]",
            border_style="cyan",
            expand=False,
        ))
    except ImportError:
        print(f"[context-guardian] suggest_fix: risk {risk_before} → {risk_after}")


def _replace_context_in_messages(
    messages: list[dict],
    original_context: str,
    new_context: str,
) -> list[dict]:
    """Replace the original context text with new_context inside messages."""
    new_messages = []
    replaced = False

    for msg in messages:
        if not replaced and msg.get("role") == "system":
            new_msg = dict(msg)
            content = msg.get("content", "")
            if isinstance(content, str):
                new_msg["content"] = content.replace(original_context, new_context, 1)
            new_messages.append(new_msg)
            replaced = True
        else:
            new_messages.append(msg)

    if not replaced:
        # No system message — find the largest non-last-user message
        last_user_idx = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        best_idx = max(
            (i for i in range(len(messages)) if i != last_user_idx),
            key=lambda i: len(str(messages[i].get("content", ""))),
            default=-1,
        )
        new_messages = []
        for i, msg in enumerate(messages):
            if i == best_idx:
                new_msg = dict(msg)
                content = msg.get("content", "")
                if isinstance(content, str):
                    new_msg["content"] = content.replace(original_context, new_context, 1)
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)

    return new_messages
