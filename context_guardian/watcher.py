"""ContextWatcher — context-manager API that monkey-patches openai/anthropic at runtime."""

from __future__ import annotations

import threading
from typing import Any, Optional


class ContextWatcher:
    """Intercepts all LLM calls made inside a `with watcher.watch():` block.

    Works by monkey-patching the installed openai and/or anthropic modules.
    The patch is applied on __enter__ and reverted on __exit__.
    """

    def __init__(
        self,
        model: str = "",
        report: str = "end",
        app: str = "default",
    ):
        self._model = model
        self._report_mode = report  # "inline" | "end" | "silent" | "json"
        self._app = app
        self._results: list = []
        self._call_index = 0
        self._lock = threading.Lock()

        from .session import Session
        self._session = Session(app)

        self._original_openai_create: Optional[Any] = None
        self._original_anthropic_create: Optional[Any] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def watch(self) -> "ContextWatcher":
        return self

    def __enter__(self) -> "ContextWatcher":
        self._patch()
        return self

    def __exit__(self, *_: Any) -> None:
        self._unpatch()
        if self._report_mode == "end":
            self._print_summary()

    # ------------------------------------------------------------------
    # Monkey-patching
    # ------------------------------------------------------------------

    def _patch(self) -> None:
        watcher = self

        # --- openai ---
        try:
            import openai
            from openai.resources.chat.completions import Completions

            original = Completions.create

            def patched_openai(self_comp: Any, **kwargs: Any) -> Any:
                resp = original(self_comp, **kwargs)
                watcher._handle_openai(kwargs, resp)
                return resp

            self._original_openai_create = original
            Completions.create = patched_openai  # type: ignore[method-assign]
        except ImportError:
            pass

        # --- anthropic ---
        try:
            import anthropic
            from anthropic.resources.messages import Messages

            original_ant = Messages.create

            def patched_anthropic(self_msg: Any, **kwargs: Any) -> Any:
                resp = original_ant(self_msg, **kwargs)
                watcher._handle_anthropic(kwargs, resp)
                return resp

            self._original_anthropic_create = original_ant
            Messages.create = patched_anthropic  # type: ignore[method-assign]
        except ImportError:
            pass

    def _unpatch(self) -> None:
        try:
            import openai
            from openai.resources.chat.completions import Completions
            if self._original_openai_create is not None:
                Completions.create = self._original_openai_create  # type: ignore[method-assign]
        except Exception:
            pass

        try:
            import anthropic
            from anthropic.resources.messages import Messages
            if self._original_anthropic_create is not None:
                Messages.create = self._original_anthropic_create  # type: ignore[method-assign]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_openai(self, kwargs: dict, resp: Any) -> None:
        from .guardian import _extract_openai_text
        messages = list(kwargs.get("messages", []))
        model = kwargs.get("model", self._model)
        response_text = _extract_openai_text(resp)
        self._record(messages, response_text, model)

    def _handle_anthropic(self, kwargs: dict, resp: Any) -> None:
        from .guardian import _anthropic_to_unified, _extract_anthropic_text
        messages = _anthropic_to_unified(kwargs)
        model = kwargs.get("model", self._model)
        response_text = _extract_anthropic_text(resp)
        self._record(messages, response_text, model)

    def _record(self, messages: list[dict], response_text: str, model: str) -> None:
        from .analyzer import analyze
        from .reporter import print_inline, to_json
        import json as _json

        with self._lock:
            idx = self._call_index
            self._call_index += 1

        result = analyze(messages, response_text, model or self._model, idx)

        with self._lock:
            self._results.append(result)
        self._session.record(result)

        if self._report_mode == "inline":
            print_inline(result)
        elif self._report_mode == "json":
            print(_json.dumps(to_json(result)))

    def _print_summary(self) -> None:
        from .reporter import print_table
        print_table(self._results)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """Return structured summary dict and pretty-print table."""
        from .reporter import print_table, to_json
        print_table(self._results)
        return {
            "calls": [to_json(r) for r in self._results],
            "total": len(self._results),
        }
