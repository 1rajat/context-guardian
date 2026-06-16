"""@watch_llm decorator — wraps any function that makes LLM calls."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional


def watch_llm(
    model: str = "",
    report: str = "inline",
    app: str = "default",
) -> Callable:
    """Decorator that intercepts all LLM calls inside the decorated function.

    Args:
        model:  Model name hint (used when model cannot be inferred from the call).
        report: "inline" | "end" | "silent" | "json"
                inline  — print after every high-risk call
                end     — print summary table when function returns
                silent  — save to session only
                json    — print JSON after every call
        app:    App name for session memory (default: function name).
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from .watcher import ContextWatcher

            app_name = app if app != "default" else fn.__name__
            watcher = ContextWatcher(model=model, report=report, app=app_name)

            with watcher.watch():
                result = fn(*args, **kwargs)

            # "end" report is handled by ContextWatcher.__exit__
            # For other modes we just return
            return result

        return wrapper

    return decorator
