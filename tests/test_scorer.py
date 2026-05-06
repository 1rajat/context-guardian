"""Tests for the scoring module."""

import pytest
from llm_stress_test.scorer import (
    NEEDLE_VALUE,
    aggregate_scores,
    exact_match,
    fuzzy_score,
    hallucination_check,
    score_response,
)


class TestExactMatch:
    def test_exact(self):
        assert exact_match(f"The answer is {NEEDLE_VALUE}") is True

    def test_case_insensitive(self):
        assert exact_match(f"the code word is {NEEDLE_VALUE.lower()}") is True

    def test_no_match(self):
        assert exact_match("I don't know") is False

    def test_wrong_code(self):
        assert exact_match("The code is PELICAN-1234") is False


class TestFuzzyScore:
    def test_full_match(self):
        assert fuzzy_score(f"The secret is {NEEDLE_VALUE}") == 1.0

    def test_no_match(self):
        assert fuzzy_score("I have no idea") == 0.0

    def test_partial_match(self):
        # Only one component present
        score = fuzzy_score("The word is FLAMINGO")
        assert 0 < score < 1.0


class TestHallucination:
    def test_correct_answer_no_hallucination(self):
        result = hallucination_check(f"The code is {NEEDLE_VALUE}")
        assert result["hallucinated"] is False
        assert result["stated_correct"] is True

    def test_wrong_code_hallucination(self):
        result = hallucination_check("The code is PELICAN-9999")
        assert result["hallucinated"] is True
        assert "PELICAN-9999" in result["wrong_codes"]

    def test_no_code_no_hallucination(self):
        result = hallucination_check("I cannot find a code word in the document.")
        assert result["hallucinated"] is False
        assert result["stated_correct"] is False


class TestScoreResponse:
    def test_perfect(self):
        result = score_response(f"The secret code word is {NEEDLE_VALUE}.")
        assert result["score"] == 1.0
        assert result["exact_match"] is True
        assert result["hallucinated"] is False

    def test_failure(self):
        result = score_response("The document does not contain any code word.")
        assert result["score"] == 0.0
        assert result["exact_match"] is False


class TestAggregateScores:
    def test_average(self):
        scores = [
            {"score": 1.0, "exact_match": True, "hallucinated": False},
            {"score": 0.0, "exact_match": False, "hallucinated": False},
            {"score": 1.0, "exact_match": True, "hallucinated": True},
        ]
        result = aggregate_scores(scores)
        assert abs(result["score"] - (2 / 3)) < 0.01
        assert abs(result["exact_match_rate"] - (2 / 3)) < 0.01
        assert abs(result["hallucination_rate"] - (1 / 3)) < 0.01

    def test_empty(self):
        result = aggregate_scores([])
        assert result["score"] == 0.0
        assert result["trials"] == 0
