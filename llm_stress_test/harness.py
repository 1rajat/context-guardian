"""Core needle-in-a-haystack test harness."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .filler import build_context, count_tokens
from .models import BaseAdapter, get_adapter
from .scorer import NEEDLE_VALUE, aggregate_scores, score_response

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
NEEDLE_SENTENCE = (
    f"The secret code word is: {NEEDLE_VALUE}"
)
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions based solely on the provided document. "
    "Be concise and direct."
)
USER_QUESTION = "What is the secret code word mentioned in the document?"

CONTEXT_LENGTHS_K = [1, 2, 4, 8, 16, 32, 64, 128]  # in thousands of tokens
DEPTH_PERCENTS = [0, 10, 25, 50, 75, 90, 100]

QUICK_CONTEXT_LENGTHS_K = [4, 16, 32, 64]
QUICK_DEPTH_PERCENTS = [10, 50, 90]


def _k_to_tokens(k: int) -> int:
    return k * 1000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CellResult:
    context_length_k: int
    depth_pct: float
    actual_tokens: int
    score: float
    exact_match_rate: float
    hallucination_rate: float
    trials: int
    individual_scores: list[dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RunResult:
    model: str
    timestamp: str
    context_lengths_k: list[int]
    depth_percents: list[float]
    trials_per_cell: int
    total_input_tokens: int
    total_output_tokens: int
    elapsed_seconds: float
    cells: list[CellResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def as_matrix(self) -> dict[tuple[int, float], CellResult]:
        return {(c.context_length_k, c.depth_pct): c for c in self.cells}

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        cells = [CellResult(**c) for c in d.pop("cells", [])]
        obj = cls(**d)
        obj.cells = cells
        return obj


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------
def estimate_total_cost(
    adapter: BaseAdapter,
    context_lengths_k: list[int],
    depth_percents: list[float],
    trials: int,
) -> dict:
    """Return a cost estimate without running any real queries."""
    total_input = 0
    skipped = 0
    max_ctx = adapter.max_context()

    for k in context_lengths_k:
        tokens = _k_to_tokens(k)
        if tokens > max_ctx:
            skipped += len(depth_percents) * trials
            continue
        for _ in depth_percents:
            total_input += tokens * trials

    total_output = len(context_lengths_k) * len(depth_percents) * trials * 100  # ~100 tok/response
    inp_cost, out_cost = adapter.cost_per_million()
    total_usd = (total_input * inp_cost + total_output * out_cost) / 1_000_000

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_usd": round(total_usd, 4),
        "cells_skipped": skipped,
        "model_max_context": max_ctx,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_stress_test(
    model: str,
    context_lengths_k: Optional[list[int]] = None,
    depth_percents: Optional[list[float]] = None,
    trials: int = 3,
    output_dir: str = "results",
    quick: bool = False,
    ollama_url: str = "http://localhost:11434",
    progress_callback: Optional[Callable[[int, int, int, float, CellResult], None]] = None,
) -> RunResult:
    """Run the full needle-in-haystack stress test.

    progress_callback(done, total, k, depth, cell_result) is called after each cell.
    """
    adapter = get_adapter(model, ollama_url=ollama_url)

    if quick:
        ctx_lengths = QUICK_CONTEXT_LENGTHS_K
        depths = QUICK_DEPTH_PERCENTS
    else:
        ctx_lengths = context_lengths_k or CONTEXT_LENGTHS_K
        depths = depth_percents or DEPTH_PERCENTS

    # Filter context lengths that exceed model maximum
    max_ctx = adapter.max_context()
    ctx_lengths = [k for k in ctx_lengths if _k_to_tokens(k) <= max_ctx]

    total_cells = len(ctx_lengths) * len(depths)
    cells_done = 0

    result = RunResult(
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        context_lengths_k=ctx_lengths,
        depth_percents=list(depths),
        trials_per_cell=trials,
        total_input_tokens=0,
        total_output_tokens=0,
        elapsed_seconds=0.0,
    )

    start = time.perf_counter()

    for k in ctx_lengths:
        target_tokens = _k_to_tokens(k)

        for depth in depths:
            # Build document with needle at the specified depth
            try:
                document, actual_tokens = build_context(
                    filler_tokens=target_tokens,
                    needle=NEEDLE_SENTENCE,
                    depth_pct=depth,
                )
            except Exception as exc:
                cell = CellResult(
                    context_length_k=k,
                    depth_pct=depth,
                    actual_tokens=target_tokens,
                    score=0.0,
                    exact_match_rate=0.0,
                    hallucination_rate=0.0,
                    trials=0,
                    error=str(exc),
                )
                result.cells.append(cell)
                cells_done += 1
                continue

            prompt_user = f"{document}\n\n---\n\n{USER_QUESTION}"
            trial_scores = []
            first_trial = True

            for _ in range(trials):
                resp = adapter.complete(
                    system=SYSTEM_PROMPT,
                    user=prompt_user,
                    max_tokens=150,
                )
                result.total_input_tokens += resp.input_tokens
                result.total_output_tokens += resp.output_tokens

                if resp.error:
                    # Fail fast on the first trial so the user sees a real error
                    # instead of a heatmap full of silent 0.00s.
                    if first_trial:
                        raise RuntimeError(
                            f"API call failed (model={model}, ctx={k}k, depth={depth}%): "
                            f"{resp.error}"
                        )
                    trial_scores.append({"score": 0.0, "exact_match": False,
                                         "fuzzy_score": 0.0, "hallucinated": False,
                                         "wrong_codes": [], "response_snippet": resp.error,
                                         "error": resp.error})
                else:
                    trial_scores.append(score_response(resp.content))

                first_trial = False

            agg = aggregate_scores(trial_scores)
            cell = CellResult(
                context_length_k=k,
                depth_pct=depth,
                actual_tokens=actual_tokens,
                score=agg["score"],
                exact_match_rate=agg["exact_match_rate"],
                hallucination_rate=agg["hallucination_rate"],
                trials=agg["trials"],
                individual_scores=trial_scores,
            )
            result.cells.append(cell)
            cells_done += 1

            if progress_callback:
                progress_callback(cells_done, total_cells, k, depth, cell)

    result.elapsed_seconds = time.perf_counter() - start

    # Persist results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(":", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_path / f"run_{safe_model}_{ts}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result, json_path


def load_result(path: str) -> RunResult:
    """Load a previously saved RunResult from JSON."""
    data = json.loads(Path(path).read_text())
    return RunResult.from_dict(data)


# ---------------------------------------------------------------------------
# Demo mode — synthetic degradation data (no API calls)
# ---------------------------------------------------------------------------

def _degradation_score(k: int, depth: float) -> float:
    """Simulate the lost-in-the-middle effect realistically.

    - Large context + middle depth = low score (the classic failure mode)
    - Small context = high score everywhere
    - Edges (0% / 100%) = high score regardless of length
    """
    import math, random

    # Context penalty: starts hurting above 8k, severe above 32k
    ctx_penalty = max(0.0, math.log2(k / 8) / math.log2(16)) if k > 8 else 0.0

    # Depth penalty: worst at 50%, good at 0% and 100% (primacy + recency)
    depth_factor = 1.0 - abs((depth / 100.0) - 0.5) * 2  # 0 at edges, 1 at center
    middle_penalty = depth_factor * ctx_penalty * 0.85

    base = max(0.0, 1.0 - middle_penalty)

    # Small natural noise per cell
    noise = random.gauss(0, 0.04)
    return max(0.0, min(1.0, base + noise))


def run_demo(
    model: str = "gpt-4o",
    context_lengths_k: Optional[list[int]] = None,
    depth_percents: Optional[list[float]] = None,
    trials: int = 3,
    output_dir: str = "results",
    quick: bool = False,
    progress_callback: Optional[Callable[[int, int, int, float, CellResult], None]] = None,
) -> tuple["RunResult", Path]:
    """Generate a synthetic run with realistic-looking degradation — no API calls."""
    import random
    import time as _time

    random.seed(42)

    ctx_lengths = (QUICK_CONTEXT_LENGTHS_K if quick else context_lengths_k or CONTEXT_LENGTHS_K)
    depths = (QUICK_DEPTH_PERCENTS if quick else depth_percents or DEPTH_PERCENTS)

    result = RunResult(
        model=f"{model} (demo)",
        timestamp=datetime.now(timezone.utc).isoformat(),
        context_lengths_k=ctx_lengths,
        depth_percents=list(depths),
        trials_per_cell=trials,
        total_input_tokens=0,
        total_output_tokens=0,
        elapsed_seconds=0.0,
        metadata={"demo": True},
    )

    total_cells = len(ctx_lengths) * len(depths)
    cells_done = 0
    start = _time.perf_counter()

    for k in ctx_lengths:
        for depth in depths:
            # Simulate per-trial noise
            trial_results = []
            for _ in range(trials):
                s = _degradation_score(k, depth)
                trial_results.append({
                    "score": s,
                    "exact_match": s >= 0.8,
                    "fuzzy_score": s,
                    "hallucinated": s < 0.2 and random.random() < 0.3,
                    "wrong_codes": [],
                    "response_snippet": "(demo)",
                })
                result.total_input_tokens += k * 1000
                result.total_output_tokens += 80

            avg_score = sum(t["score"] for t in trial_results) / trials
            cell = CellResult(
                context_length_k=k,
                depth_pct=depth,
                actual_tokens=k * 1000,
                score=avg_score,
                exact_match_rate=sum(1 for t in trial_results if t["exact_match"]) / trials,
                hallucination_rate=sum(1 for t in trial_results if t["hallucinated"]) / trials,
                trials=trials,
                individual_scores=trial_results,
            )
            result.cells.append(cell)
            cells_done += 1
            _time.sleep(0.05)  # tiny delay so progress bar is visible

            if progress_callback:
                progress_callback(cells_done, total_cells, k, depth, cell)

    result.elapsed_seconds = _time.perf_counter() - start

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace(":", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_path / f"demo_{safe_model}_{ts}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))

    return result, json_path
