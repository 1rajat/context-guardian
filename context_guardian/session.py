"""Session memory — Layer 3 intelligence.

Persists every analyzed call to ~/.context_guardian/sessions/{app_name}.jsonl
and learns per-app failure patterns over time.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import AnalysisResult

_SESSION_DIR = Path.home() / ".context_guardian" / "sessions"

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# ---------------------------------------------------------------------------
# Persisted record shape
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    timestamp: str
    app_name: str
    model: str
    total_tokens: int
    context_tokens: int
    position_risk: str
    risk_depth_pct: float
    faithfulness_score: float
    fix_suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_result(cls, result: "AnalysisResult", app_name: str) -> "CallRecord":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            app_name=app_name,
            model=result.model,
            total_tokens=result.total_tokens,
            context_tokens=result.context_tokens,
            position_risk=result.position_risk,
            risk_depth_pct=round(result.risk_depth_pct, 1),
            faithfulness_score=round(result.faithfulness_score, 4),
            fix_suggestion=result.fix_suggestion,
        )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """Manages call history for a named app and emits learned warnings."""

    def __init__(self, app_name: str = "default"):
        self.app_name = app_name
        self._session_file = _SESSION_DIR / f"{app_name}.jsonl"
        self._runtime_calls: list[CallRecord] = []
        self._total_call_count = self._count_disk_calls()

    def _count_disk_calls(self) -> int:
        if not self._session_file.exists():
            return 0
        try:
            return sum(
                1 for line in self._session_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception:
            return 0

    def record(self, result: "AnalysisResult") -> None:
        rec = CallRecord.from_result(result, self.app_name)
        self._runtime_calls.append(rec)
        self._append_to_disk(rec)
        self._total_call_count += 1

        if self._total_call_count % 10 == 0:
            self._print_blind_spot_panel()
        if self._total_call_count % 50 == 0:
            self.save_insights()

    def _append_to_disk(self, rec: CallRecord) -> None:
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        with self._session_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    def load_history(self) -> list[CallRecord]:
        if not self._session_file.exists():
            return []
        records = []
        for line in self._session_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(CallRecord(**json.loads(line)))
            except Exception:
                pass
        return records

    # ------------------------------------------------------------------
    # Blind spot analysis (Layer 3 learning)
    # ------------------------------------------------------------------

    def _compute_blind_spot(self) -> dict:
        history = self.load_history()
        if len(history) < 5:
            return {}

        zones: dict[str, list[CallRecord]] = {
            "0-20": [], "20-40": [], "40-60": [], "60-80": [], "80-100": [],
        }
        for r in history:
            d = r.risk_depth_pct
            if d < 20:
                zones["0-20"].append(r)
            elif d < 40:
                zones["20-40"].append(r)
            elif d < 60:
                zones["40-60"].append(r)
            elif d < 80:
                zones["60-80"].append(r)
            else:
                zones["80-100"].append(r)

        zone_rates: dict[str, float] = {}
        for zone, calls in zones.items():
            if calls:
                fails = sum(1 for c in calls if _RISK_ORDER.get(c.position_risk, 0) >= 2)
                zone_rates[zone] = fails / len(calls)

        if not zone_rates:
            return {}

        worst_zone = max(zone_rates, key=lambda z: zone_rates[z])
        worst_rate = zone_rates[worst_zone]

        high_risk = [r for r in history if _RISK_ORDER.get(r.position_risk, 0) >= 2]
        threshold_tokens = (
            int(sum(r.context_tokens for r in high_risk) / len(high_risk))
            if high_risk else 0
        )

        risk_counts: dict[str, int] = {}
        for r in history:
            risk_counts[r.position_risk] = risk_counts.get(r.position_risk, 0) + 1
        most_common_risk = max(risk_counts, key=lambda k: risk_counts[k])

        return {
            "worst_zone": worst_zone,
            "worst_rate": worst_rate,
            "threshold_tokens": threshold_tokens,
            "most_common_risk": most_common_risk,
            "high_risk_count": len(high_risk),
            "total": len(history),
            "avg_faithfulness": sum(r.faithfulness_score for r in history) / len(history),
            "avg_context_tokens": int(sum(r.context_tokens for r in history) / len(history)),
            "models": list({r.model for r in history}),
            "zone_rates": zone_rates,
        }

    def _plain_english_summary(self, blind_spot: dict) -> str:
        if not blind_spot:
            return "Not enough data yet."
        zone = blind_spot["worst_zone"]
        rate = int(blind_spot["worst_rate"] * 100)
        tokens = blind_spot["threshold_tokens"]
        k = max(tokens // 1000, 1)
        return (
            f"Content at {zone}% depth fails {rate}% of the time "
            f"when context exceeds {k}k tokens"
        )

    def _print_blind_spot_panel(self) -> None:
        blind_spot = self._compute_blind_spot()
        if not blind_spot:
            return

        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()

            n = blind_spot["total"]
            summary = self._plain_english_summary(blind_spot)
            most_common = blind_spot["most_common_risk"].upper()
            high_count = blind_spot["high_risk_count"]
            avg_faith = blind_spot["avg_faithfulness"]
            threshold_k = max(blind_spot["threshold_tokens"] // 1000, 1)

            faith_note = " [dim]— model often ignores context[/dim]" if avg_faith < 0.65 else ""

            lines = [
                f"[bold]🧠 After {n} calls in [cyan]\"{self.app_name}\"[/cyan]:[/bold]",
                "",
                "   [bold]Your app's blind spot:[/bold]",
                f"   {summary}",
                "",
                f"   Most common risk: [bold]{most_common}[/bold] ({high_count}/{n} calls)",
                f"   Avg faithfulness: [bold]{avg_faith:.2f}[/bold]{faith_note}",
                "",
                "[cyan]💡 Top fix: use suggest_fix() before every call[/cyan]",
                f"[dim]   or switch to RAG for queries over {threshold_k}k tokens[/dim]",
            ]

            console.print(Panel(
                "\n".join(lines),
                title="[bold dim]context-guardian learned something[/bold dim]",
                border_style="cyan",
                expand=False,
            ))
        except ImportError:
            blind_spot_summary = self._plain_english_summary(blind_spot)
            print(f"[context-guardian] Blind spot after {blind_spot['total']} calls: {blind_spot_summary}")

    def save_insights(self) -> None:
        """Save full insight report to ~/.context_guardian/sessions/{app}_insights.md."""
        blind_spot = self._compute_blind_spot()
        history = self.load_history()
        n = len(history)
        if n == 0:
            return

        now = datetime.now()
        summary_str = self._plain_english_summary(blind_spot)

        high_risk = [r for r in history if _RISK_ORDER.get(r.position_risk, 0) >= 2]
        avg_faith = sum(r.faithfulness_score for r in history) / n
        avg_tokens = int(sum(r.context_tokens for r in history) / n)
        models = list({r.model for r in history})
        most_used_model = models[0] if models else "unknown"

        threshold_k = max(avg_tokens // 1000, 1)
        fix_rec = (
            f"Your queries consistently exceed {threshold_k}k tokens with relevant "
            f"content in the middle. Switch to RAG or always run "
            f"`suggest_fix()` before API calls."
        )

        risk_bars = {
            "low": "██░░░░░░░░",
            "medium": "████░░░░░░",
            "high": "████████░░",
            "critical": "██████████",
        }
        timeline_calls = history[-20:]
        start_idx = max(n - len(timeline_calls) + 1, 1)
        timeline_lines = [
            f"Call {start_idx + i:>3}:  {risk_bars.get(r.position_risk, '??????????')} {r.position_risk.upper()}"
            for i, r in enumerate(timeline_calls)
        ]

        content = f"""# context-guardian insights — {self.app_name}
Generated: {now.strftime('%Y-%m-%d')} | Calls analyzed: {n}

## Your App's Blind Spot
{summary_str}.

## Call History Summary
| Metric | Value |
|--------|-------|
| Total calls | {n} |
| High/Critical risk | {len(high_risk)} ({len(high_risk) * 100 // max(n, 1)}%) |
| Avg faithfulness | {avg_faith:.2f} |
| Avg context size | {avg_tokens:,} tokens |
| Most used model | {most_used_model} |

## Recommended Fix
{fix_rec}

## Risk Timeline (last {len(timeline_calls)} calls)
{chr(10).join(timeline_lines)}
"""
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        insights_file = _SESSION_DIR / f"{self.app_name}_insights.md"
        insights_file.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary (existing, enhanced)
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict summarising app-specific blind spots."""
        history = self.load_history()
        n = len(history)

        if n == 0:
            return {"app_name": self.app_name, "total_calls": 0, "message": "No history yet."}

        high_risk = [r for r in history if _RISK_ORDER.get(r.position_risk, 0) >= 2]
        faith_failures = [r for r in history if r.faithfulness_score < 0.6]

        zones = {"0-25": [], "25-50": [], "50-75": [], "75-100": []}
        for r in history:
            d = r.risk_depth_pct
            if d < 25:
                zones["0-25"].append(r)
            elif d < 50:
                zones["25-50"].append(r)
            elif d < 75:
                zones["50-75"].append(r)
            else:
                zones["75-100"].append(r)

        zone_failure_rates = {}
        for zone, calls in zones.items():
            if calls:
                fails = sum(1 for c in calls if _RISK_ORDER.get(c.position_risk, 0) >= 2)
                zone_failure_rates[zone] = round(fails / len(calls), 2)
            else:
                zone_failure_rates[zone] = None

        result: dict = {
            "app_name": self.app_name,
            "total_calls": n,
            "high_risk_calls": len(high_risk),
            "faithfulness_failures": len(faith_failures),
            "avg_faithfulness": round(sum(r.faithfulness_score for r in history) / n, 3),
            "zone_failure_rates": zone_failure_rates,
        }

        if n >= 50:
            blind_spots = [
                zone for zone, rate in zone_failure_rates.items()
                if rate is not None and rate > 0.5
            ]
            if blind_spots:
                result["personalized_warning"] = (
                    f"In YOUR {self.app_name} pipeline: content at "
                    f"{', '.join(blind_spots)}% depth fails "
                    f">{int(max(zone_failure_rates[z] for z in blind_spots) * 100)}% "
                    "of the time. Consider restructuring your context."
                )

        return result

    def print_summary(self) -> None:
        """Pretty-print the session summary using rich."""
        data = self.summary()
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            lines = [
                f"App: [bold]{data['app_name']}[/bold]  |  Calls: {data['total_calls']}",
                f"High-risk calls: [red]{data.get('high_risk_calls', 0)}[/red]"
                f"  |  Faith failures: [yellow]{data.get('faithfulness_failures', 0)}[/yellow]"
                f"  |  Avg faithfulness: {data.get('avg_faithfulness', 'N/A')}",
            ]

            rates = data.get("zone_failure_rates", {})
            if rates:
                zone_parts = []
                for zone, rate in rates.items():
                    if rate is None:
                        zone_parts.append(f"[dim]{zone}%: n/a[/dim]")
                    else:
                        color = "red" if rate > 0.5 else ("yellow" if rate > 0.25 else "green")
                        zone_parts.append(f"[{color}]{zone}%: {rate:.0%}[/{color}]")
                lines.append("Depth-zone failure rates: " + "  ".join(zone_parts))

            if "personalized_warning" in data:
                lines.append(f"\n[bold yellow]{data['personalized_warning']}[/bold yellow]")

            console.print(Panel(
                "\n".join(lines),
                title="[bold]context-guardian — Session Summary[/bold]",
                border_style="cyan",
                expand=False,
            ))
        except ImportError:
            print(json.dumps(data, indent=2))

    def clear(self) -> None:
        if self._session_file.exists():
            self._session_file.unlink()
        self._runtime_calls.clear()
        self._total_call_count = 0

    @classmethod
    def list_apps(cls) -> list[str]:
        if not _SESSION_DIR.exists():
            return []
        return [f.stem for f in _SESSION_DIR.glob("*.jsonl")]
