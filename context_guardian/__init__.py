"""context-guardian — real-time context loss detection for LLM applications."""

__version__ = "0.1.0"

from .guardian import Guardian, suggest_fix
from .watcher import ContextWatcher
from .decorators import watch_llm
from .analyzer import ContextAnalyzer, AnalysisResult, analyze
from .session import Session

__all__ = [
    "Guardian",
    "suggest_fix",
    "ContextWatcher",
    "watch_llm",
    "ContextAnalyzer",
    "AnalysisResult",
    "analyze",
    "Session",
]
