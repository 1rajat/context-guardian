"""Tests for the filler text generator."""

import pytest
from llm_stress_test.filler import build_context, count_tokens, generate_filler


class TestGenerateFiller:
    def test_exact_token_count(self):
        for n in [10, 100, 500, 1000]:
            text = generate_filler(n)
            actual = count_tokens(text)
            assert actual == n, f"Expected {n} tokens, got {actual}"

    def test_returns_string(self):
        assert isinstance(generate_filler(50), str)

    def test_zero_tokens(self):
        text = generate_filler(0)
        assert count_tokens(text) == 0


class TestBuildContext:
    def test_needle_present(self):
        needle = "The secret code word is: FLAMINGO-7429"
        doc, _ = build_context(filler_tokens=200, needle=needle, depth_pct=50)
        assert "FLAMINGO-7429" in doc

    def test_depth_zero(self):
        needle = "MARKER_START"
        doc, _ = build_context(filler_tokens=100, needle=needle, depth_pct=0)
        assert doc.startswith("MARKER_START")

    def test_depth_hundred(self):
        needle = "MARKER_END"
        doc, _ = build_context(filler_tokens=100, needle=needle, depth_pct=100)
        assert doc.endswith("MARKER_END")

    def test_approximate_token_count(self):
        needle = "FLAMINGO-7429"
        target = 500
        _, actual = build_context(filler_tokens=target, needle=needle, depth_pct=50)
        # Within 5% of target
        assert abs(actual - target) / target < 0.05
