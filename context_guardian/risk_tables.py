"""Pre-computed risk tables for context-guardian.

Based on real needle-in-haystack stress test results.
Risk levels: "low" | "medium" | "high" | "critical"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Risk tables per model / model family
# ---------------------------------------------------------------------------
# Each entry: token_range → {depth_range → risk_level}
# Token ranges are (inclusive_lo, exclusive_hi).
# Depth ranges are percentage points 0–100.

_RiskTable = dict[tuple[int, int], dict[tuple[int, int], str]]

RISK_TABLE: dict[str, _RiskTable] = {
    "gpt-4o": {
        (0, 8_000):        {(0, 100): "low"},
        (8_000, 32_000):   {(0, 20): "low",  (20, 80): "medium", (80, 100): "low"},
        (32_000, 128_001): {(0, 15): "low",  (15, 85): "high",   (85, 100): "low"},
    },
    "gpt-4o-mini": {
        (0, 8_000):        {(0, 100): "low"},
        (8_000, 32_000):   {(0, 20): "low",  (20, 80): "medium", (80, 100): "low"},
        (32_000, 128_001): {(0, 15): "low",  (15, 85): "high",   (85, 100): "low"},
    },
    "gpt-4-turbo": {
        (0, 8_000):        {(0, 100): "low"},
        (8_000, 32_000):   {(0, 20): "low",  (20, 80): "medium", (80, 100): "low"},
        (32_000, 128_001): {(0, 15): "low",  (15, 85): "high",   (85, 100): "low"},
    },
    "gpt-3.5-turbo": {
        (0, 4_000):        {(0, 100): "low"},
        (4_000, 16_386):   {(0, 20): "low",  (20, 80): "high",   (80, 100): "medium"},
    },
    "claude-3-5-sonnet": {
        (0, 32_000):        {(0, 100): "low"},
        (32_000, 100_000):  {(0, 10): "medium", (10, 90): "medium", (90, 100): "low"},
        (100_000, 200_001): {(0, 20): "high",   (20, 80): "medium", (80, 100): "low"},
    },
    "claude-3-5-haiku": {
        (0, 32_000):        {(0, 100): "low"},
        (32_000, 100_000):  {(0, 10): "medium", (10, 90): "medium", (90, 100): "low"},
        (100_000, 200_001): {(0, 20): "high",   (20, 80): "medium", (80, 100): "low"},
    },
    "claude-3-opus": {
        (0, 32_000):        {(0, 100): "low"},
        (32_000, 100_000):  {(0, 10): "medium", (10, 90): "medium", (90, 100): "low"},
        (100_000, 200_001): {(0, 20): "high",   (20, 80): "medium", (80, 100): "low"},
    },
    "claude-sonnet-4": {
        (0, 32_000):        {(0, 100): "low"},
        (32_000, 100_000):  {(0, 10): "low",    (10, 90): "medium", (90, 100): "low"},
        (100_000, 200_001): {(0, 20): "medium",  (20, 80): "medium", (80, 100): "low"},
    },
    "llama3": {
        (0, 8_000):        {(0, 100): "low"},
        (8_000, 16_000):   {(0, 30): "low",  (30, 70): "high",    (70, 100): "low"},
        (16_000, 128_001): {(0, 100): "critical"},
    },
    "llama3.2": {
        (0, 4_000):        {(0, 100): "low"},
        (4_000, 16_000):   {(0, 30): "medium", (30, 70): "high",   (70, 100): "medium"},
        (16_000, 128_001): {(0, 100): "critical"},
    },
    "qwen2.5": {
        (0, 16_000):       {(0, 100): "low"},
        (16_000, 64_000):  {(0, 20): "low",  (20, 80): "medium",  (80, 100): "low"},
        (64_000, 128_001): {(0, 15): "low",  (15, 80): "high",    (80, 100): "low"},
    },
    "mistral": {
        (0, 8_000):        {(0, 100): "low"},
        (8_000, 32_000):   {(0, 20): "low",  (20, 80): "medium",  (80, 100): "low"},
        (32_000, 128_001): {(0, 15): "low",  (15, 85): "high",    (85, 100): "low"},
    },
}

_DEFAULT_RISK: _RiskTable = {
    (0, 8_000):        {(0, 100): "low"},
    (8_000, 32_000):   {(0, 20): "low",  (20, 80): "medium", (80, 100): "low"},
    (32_000, 128_001): {(0, 15): "low",  (15, 85): "high",   (85, 100): "low"},
}

_RECOMMENDATIONS: dict[str, str] = {
    "low": "Context risk is low — no action needed.",
    "medium": (
        "Some key facts sit in the attention-degraded zone. "
        "Consider moving critical information to the start or end of the context."
    ),
    "high": (
        "Key facts are in a high-risk context position. "
        "Use retrieval (RAG) or restructure so critical content appears "
        "in the first 15% or last 15% of the context window."
    ),
    "critical": (
        "Context length exceeds the model's reliable retrieval range. "
        "The model is very likely to miss middle content. "
        "Use RAG or significantly reduce context size."
    ),
}

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_RISK_COLORS = {
    "low": "green",
    "medium": "yellow",
    "high": "red",
    "critical": "bold red",
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _resolve_model_key(model: str) -> str:
    """Return the best matching key in RISK_TABLE for a given model name."""
    m = model.lower().split(":")[0]  # strip Ollama tag e.g. "llama3.2:3b" → "llama3.2"
    if m in RISK_TABLE:
        return m
    candidates = [k for k in RISK_TABLE if m.startswith(k) or k in m]
    if candidates:
        return max(candidates, key=len)
    return ""


def get_risk_level(model: str, context_tokens: int, depth_pct: float) -> str:
    """Return risk level string for a given model, token count, and depth."""
    key = _resolve_model_key(model)
    table = RISK_TABLE.get(key, _DEFAULT_RISK)

    token_ranges = sorted(table.keys(), key=lambda r: r[0])
    depth_table = table[token_ranges[-1]]
    for lo, hi in token_ranges:
        if lo <= context_tokens < hi:
            depth_table = table[(lo, hi)]
            break

    for (dlo, dhi), risk in sorted(depth_table.items(), key=lambda x: x[0]):
        if dlo <= depth_pct <= dhi:
            return risk

    return "medium"


def get_recommendation(overall_risk: str, context_tokens: int = 0) -> str:
    return _RECOMMENDATIONS.get(overall_risk, _RECOMMENDATIONS["medium"])


def resolve_model(model: str) -> str:
    return _resolve_model_key(model)


def risk_label(risk: str) -> str:
    return risk.upper()


def worst_risk(risks: list[str]) -> str:
    if not risks:
        return "low"
    return max(risks, key=lambda r: _RISK_ORDER.get(r, 0))
