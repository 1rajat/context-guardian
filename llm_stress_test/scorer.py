"""Scoring functions: exact match, fuzzy match, hallucination detection."""

from __future__ import annotations

import re

NEEDLE_VALUE = "FLAMINGO-7429"
# All plausible code-word patterns the model might hallucinate
_CODE_WORD_PATTERN = re.compile(r"\b[A-Z]{2,}-\d{3,}\b")


def _normalize(text: str) -> str:
    return text.upper().strip()


def exact_match(response: str, expected: str = NEEDLE_VALUE) -> bool:
    return expected.upper() in _normalize(response)


def fuzzy_score(response: str, expected: str = NEEDLE_VALUE) -> float:
    """Return 1.0 for exact match, 0.5 for partial (prefix or suffix), 0.0 otherwise."""
    norm_resp = _normalize(response)
    norm_exp = _normalize(expected)

    if norm_exp in norm_resp:
        return 1.0

    # Check each component separately (e.g. "FLAMINGO" present but not the number)
    parts = norm_exp.split("-")
    matched = sum(1 for p in parts if p in norm_resp)
    if matched == len(parts):
        return 1.0
    if matched > 0:
        return 0.5 * (matched / len(parts))
    return 0.0


def hallucination_check(response: str, expected: str = NEEDLE_VALUE) -> dict:
    """Detect if the model confidently returned a *different* code word."""
    norm_resp = _normalize(response)
    norm_exp = _normalize(expected)

    found_codes = _CODE_WORD_PATTERN.findall(response.upper())
    wrong_codes = [c for c in found_codes if c != norm_exp]

    return {
        "hallucinated": len(wrong_codes) > 0 and norm_exp not in norm_resp,
        "wrong_codes": wrong_codes,
        "stated_correct": norm_exp in norm_resp,
    }


def score_response(response: str, expected: str = NEEDLE_VALUE) -> dict:
    """Return a full scoring dict for a single model response."""
    fs = fuzzy_score(response, expected)
    hall = hallucination_check(response, expected)
    return {
        "score": fs,
        "exact_match": exact_match(response, expected),
        "fuzzy_score": fs,
        "hallucinated": hall["hallucinated"],
        "wrong_codes": hall["wrong_codes"],
        "response_snippet": response[:200].replace("\n", " "),
    }


def aggregate_scores(scores: list[dict]) -> dict:
    """Average multiple trial scores into a single cell result."""
    if not scores:
        return {"score": 0.0, "trials": 0, "exact_match_rate": 0.0, "hallucination_rate": 0.0}

    avg_score = sum(s["score"] for s in scores) / len(scores)
    exact_rate = sum(1 for s in scores if s["exact_match"]) / len(scores)
    hall_rate = sum(1 for s in scores if s["hallucinated"]) / len(scores)

    return {
        "score": avg_score,
        "trials": len(scores),
        "exact_match_rate": exact_rate,
        "hallucination_rate": hall_rate,
        "individual": scores,
    }
