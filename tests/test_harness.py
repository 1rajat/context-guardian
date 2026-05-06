"""Tests for the harness module (no API calls — uses a mock adapter)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_stress_test.harness import RunResult, estimate_total_cost, load_result
from llm_stress_test.scorer import NEEDLE_VALUE


class _MockAdapter:
    """Always returns the correct answer."""

    model = "mock-model"

    def max_context(self):
        return 128_000

    def cost_per_million(self):
        return (0.0, 0.0)

    def estimate_cost(self, *args):
        return 0.0

    def complete(self, system, user, max_tokens=256):
        from llm_stress_test.models import ModelResponse
        return ModelResponse(
            content=f"The secret code word is {NEEDLE_VALUE}.",
            input_tokens=100,
            output_tokens=20,
            model=self.model,
        )


class TestEstimateCost:
    def test_zero_cost_for_free_model(self):
        adapter = _MockAdapter()
        est = estimate_total_cost(adapter, [4, 8], [25.0, 75.0], trials=2)
        assert est["estimated_usd"] == 0.0
        assert est["total_input_tokens"] > 0

    def test_skips_over_limit(self):
        adapter = _MockAdapter()
        adapter.max_context = lambda: 4_000  # only 4k allowed
        est = estimate_total_cost(adapter, [4, 8, 16], [50.0], trials=1)
        # 8k and 16k should be skipped
        assert est["cells_skipped"] == 2


class TestRunResultSerialization:
    def test_round_trip(self, tmp_path):
        from llm_stress_test.harness import CellResult

        rr = RunResult(
            model="test-model",
            timestamp="2024-01-01T00:00:00+00:00",
            context_lengths_k=[4, 8],
            depth_percents=[25.0, 75.0],
            trials_per_cell=2,
            total_input_tokens=1000,
            total_output_tokens=100,
            elapsed_seconds=12.3,
        )
        rr.cells.append(
            CellResult(
                context_length_k=4,
                depth_pct=25.0,
                actual_tokens=4000,
                score=1.0,
                exact_match_rate=1.0,
                hallucination_rate=0.0,
                trials=2,
            )
        )

        json_path = tmp_path / "test_run.json"
        json_path.write_text(json.dumps(rr.to_dict(), indent=2))

        loaded = load_result(str(json_path))
        assert loaded.model == rr.model
        assert len(loaded.cells) == 1
        assert loaded.cells[0].score == 1.0


class TestRunStressTest:
    def test_mock_run(self, tmp_path):
        """Smoke test run_stress_test with a mock adapter — no real API calls."""
        from llm_stress_test import harness

        with patch.object(harness, "get_adapter", return_value=_MockAdapter()):
            result, json_path = harness.run_stress_test(
                model="mock-model",
                context_lengths_k=[2],
                depth_percents=[50.0],
                trials=1,
                output_dir=str(tmp_path),
            )

        assert len(result.cells) == 1
        cell = result.cells[0]
        assert cell.score == 1.0
        assert cell.exact_match_rate == 1.0
        assert json_path.exists()
