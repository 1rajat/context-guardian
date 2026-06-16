"""THE BRAIN — three-layer context analysis engine.

Layer 1: Semantic Relevance Mapping   — finds where relevant content lives
Layer 2: Faithfulness Judge           — checks if the model used what it had
Layer 3: (session learning)           — wired in via session.py, not here

All analysis is LOCAL — no extra API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContextChunk:
    text: str
    depth: float        # 0.0 – 1.0 (position in context)
    token_start: int
    token_end: int


@dataclass
class AnalysisResult:
    model: str
    total_tokens: int
    context_tokens: int
    question: str

    # Layer 1
    position_risk: str          # "low" | "medium" | "high" | "critical"
    risk_depth: float           # 0.0–1.0 depth of most relevant chunk
    top_chunks: list[ContextChunk] = field(default_factory=list)
    raw_scores: list[float] = field(default_factory=list)

    # Layer 2
    faithfulness_score: float = 1.0  # 0.0–1.0

    fix_suggestion: str = ""
    call_index: int = 0

    @property
    def faithfulness_flag(self) -> bool:
        return self.faithfulness_score < 0.6

    @property
    def risk_depth_pct(self) -> float:
        return self.risk_depth * 100


# ---------------------------------------------------------------------------
# Token counting (graceful fallback)
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Context / question extraction
# ---------------------------------------------------------------------------

def _extract_context_question(messages: list[dict]) -> tuple[str, str]:
    """Split messages into (context_text, question_text).

    The last user-role message is the question; everything else is context.
    """
    def _content_str(msg: dict) -> str:
        c = msg.get("content", "")
        if isinstance(c, list):
            return " ".join(
                p.get("text", "") for p in c
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return str(c) if c else ""

    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i

    ctx_parts: list[str] = []
    question = ""
    for i, msg in enumerate(messages):
        text = _content_str(msg)
        if i == last_user_idx:
            question = text
        elif text.strip():
            ctx_parts.append(text)

    return "\n\n".join(ctx_parts), question


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_context(context_text: str, chunk_size: int = 512) -> list[ContextChunk]:
    """Split context into ~chunk_size-token chunks with depth positions."""
    if not context_text:
        return []

    # Split on double-newlines first, then by sentence if chunks are large
    paragraphs = re.split(r"\n{2,}", context_text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not para.strip():
            continue
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if _count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If the paragraph alone exceeds chunk_size, hard-split by words
            if _count_tokens(para) > chunk_size:
                words = para.split()
                sub = ""
                for word in words:
                    trial = (sub + " " + word).strip()
                    if _count_tokens(trial) <= chunk_size:
                        sub = trial
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = word
                if sub:
                    chunks.append(sub)
            else:
                current = para

    if current:
        chunks.append(current)

    if not chunks:
        return []

    total_chars = len(context_text)
    result: list[ContextChunk] = []
    char_pos = 0

    for chunk in chunks:
        start_pos = context_text.find(chunk[:50], char_pos)
        if start_pos == -1:
            start_pos = char_pos
        depth = start_pos / max(total_chars, 1)
        tok_start = _count_tokens(context_text[:start_pos])
        tok_end = tok_start + _count_tokens(chunk)
        result.append(ContextChunk(
            text=chunk,
            depth=min(max(depth, 0.0), 1.0),
            token_start=tok_start,
            token_end=tok_end,
        ))
        char_pos = start_pos + len(chunk)

    return result


# ---------------------------------------------------------------------------
# Faithfulness helpers
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAIM_MIN_LEN = 12


def _extract_claims(text: str) -> list[str]:
    """Extract sentences as candidate claims from a response."""
    sentences = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= _CLAIM_MIN_LEN]


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class ContextAnalyzer:
    """Semantic context analysis — all three intelligence layers."""

    def __init__(self, use_embeddings: bool = True):
        self._use_embeddings = use_embeddings

    def analyze(
        self,
        messages: list[dict],
        response: str,
        model: str,
        call_index: int = 0,
    ) -> AnalysisResult:
        from .risk_tables import get_risk_level, get_recommendation, worst_risk

        context_text, question = _extract_context_question(messages)

        all_text = " ".join(
            (msg.get("content") or "") if isinstance(msg.get("content"), str)
            else " ".join(
                p.get("text", "") for p in (msg.get("content") or [])
                if isinstance(p, dict)
            )
            for msg in messages
        )
        total_tokens = _count_tokens(all_text)
        context_tokens = _count_tokens(context_text) if context_text else 0

        empty = AnalysisResult(
            model=model,
            total_tokens=total_tokens,
            context_tokens=context_tokens,
            question=question[:200],
            position_risk="low",
            risk_depth=0.0,
            faithfulness_score=1.0,
            fix_suggestion=get_recommendation("low"),
            call_index=call_index,
        )

        if not context_text or not question:
            return empty

        # --- Layer 1: Semantic Relevance Mapping ---
        chunks = _chunk_context(context_text)
        if not chunks:
            return empty

        top_chunks, raw_scores = self._rank_chunks(question, chunks)
        risk_zones = [
            get_risk_level(model, context_tokens, chunk.depth * 100)
            for chunk in top_chunks
        ]
        position_risk = worst_risk(risk_zones) if risk_zones else "low"
        risk_depth = top_chunks[0].depth if top_chunks else 0.0

        # --- Layer 2: Faithfulness Judge ---
        faithfulness = self._check_faithfulness(response, chunks)

        fix = self._suggest_fix(position_risk, top_chunks, faithfulness, context_tokens)

        return AnalysisResult(
            model=model,
            total_tokens=total_tokens,
            context_tokens=context_tokens,
            question=question[:200],
            position_risk=position_risk,
            risk_depth=risk_depth,
            top_chunks=top_chunks[:3],
            raw_scores=raw_scores[:len(chunks)],
            faithfulness_score=faithfulness,
            fix_suggestion=fix,
            call_index=call_index,
        )

    def analyze_pre_call(
        self,
        messages: list[dict],
        model: str,
        call_index: int = 0,
    ) -> AnalysisResult:
        """Pre-call risk analysis — token count + semantic mapping only, no faithfulness check."""
        from .risk_tables import get_risk_level, get_recommendation, worst_risk

        context_text, question = _extract_context_question(messages)

        all_text = " ".join(
            (msg.get("content") or "") if isinstance(msg.get("content"), str)
            else " ".join(
                p.get("text", "") for p in (msg.get("content") or [])
                if isinstance(p, dict)
            )
            for msg in messages
        )
        total_tokens = _count_tokens(all_text)
        context_tokens = _count_tokens(context_text) if context_text else 0

        empty = AnalysisResult(
            model=model,
            total_tokens=total_tokens,
            context_tokens=context_tokens,
            question=question[:200],
            position_risk="low",
            risk_depth=0.0,
            faithfulness_score=1.0,
            fix_suggestion=get_recommendation("low"),
            call_index=call_index,
        )

        if not context_text or not question:
            return empty

        chunks = _chunk_context(context_text)
        if not chunks:
            return empty

        top_chunks, raw_scores = self._rank_chunks(question, chunks)
        risk_zones = [
            get_risk_level(model, context_tokens, chunk.depth * 100)
            for chunk in top_chunks
        ]
        position_risk = worst_risk(risk_zones) if risk_zones else "low"
        risk_depth = top_chunks[0].depth if top_chunks else 0.0
        fix = self._suggest_fix(position_risk, top_chunks, 1.0, context_tokens)

        return AnalysisResult(
            model=model,
            total_tokens=total_tokens,
            context_tokens=context_tokens,
            question=question[:200],
            position_risk=position_risk,
            risk_depth=risk_depth,
            top_chunks=top_chunks[:3],
            raw_scores=raw_scores[:len(chunks)],
            faithfulness_score=1.0,
            fix_suggestion=fix,
            call_index=call_index,
        )

    # ------------------------------------------------------------------
    # Layer 1 — semantic ranking
    # ------------------------------------------------------------------

    def _rank_chunks(
        self,
        question: str,
        chunks: list[ContextChunk],
        top_k: int = 3,
    ) -> tuple[list[ContextChunk], list[float]]:
        if self._use_embeddings:
            return self._rank_chunks_semantic(question, chunks, top_k)
        return self._rank_chunks_heuristic(question, chunks, top_k)

    def _rank_chunks_semantic(
        self,
        question: str,
        chunks: list[ContextChunk],
        top_k: int,
    ) -> tuple[list[ContextChunk], list[float]]:
        try:
            from .embedder import embed, embed_batch, cosine_similarity_1d
            q_emb = embed(question)
            c_embs = embed_batch([c.text for c in chunks])
            scores = cosine_similarity_1d(q_emb, c_embs).tolist()
            ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
            top = [c for _, c in ranked[:top_k]]
            return top, scores
        except Exception:
            return self._rank_chunks_heuristic(question, chunks, top_k)

    def _rank_chunks_heuristic(
        self,
        question: str,
        chunks: list[ContextChunk],
        top_k: int,
    ) -> tuple[list[ContextChunk], list[float]]:
        stop = frozenset({
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "have", "has", "do", "does", "did", "will", "would", "could",
            "should", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "and", "or", "but", "if", "when", "what", "how",
            "i", "you", "he", "she", "it", "we", "they", "tell", "say",
        })
        q_words = {w for w in re.findall(r"\b[a-z]{3,}\b", question.lower()) if w not in stop}
        scores = []
        for c in chunks:
            c_words = set(re.findall(r"\b[a-z]{3,}\b", c.text.lower()))
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            scores.append(float(overlap))

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        top = [c for _, c in ranked[:top_k]]
        return top, scores

    # ------------------------------------------------------------------
    # Layer 2 — faithfulness judge
    # ------------------------------------------------------------------

    def _check_faithfulness(
        self,
        response: str,
        chunks: list[ContextChunk],
    ) -> float:
        if not response or not chunks:
            return 1.0

        claims = _extract_claims(response)
        if not claims:
            return 1.0

        if self._use_embeddings:
            try:
                return self._faithfulness_semantic(claims, chunks)
            except Exception:
                pass
        return self._faithfulness_heuristic(claims, chunks)

    def _faithfulness_semantic(
        self,
        claims: list[str],
        chunks: list[ContextChunk],
    ) -> float:
        from .embedder import embed, embed_batch, cosine_similarity_1d
        ctx_embs = embed_batch([c.text for c in chunks])
        scores = []
        for claim in claims:
            c_emb = embed(claim)
            sims = cosine_similarity_1d(c_emb, ctx_embs)
            scores.append(float(np.max(sims)) if len(sims) > 0 else 0.0)
        return float(np.mean(scores)) if scores else 1.0

    def _faithfulness_heuristic(
        self,
        claims: list[str],
        chunks: list[ContextChunk],
    ) -> float:
        ctx_words: set[str] = set()
        for c in chunks:
            ctx_words |= set(re.findall(r"\b[a-z]{4,}\b", c.text.lower()))

        scores = []
        for claim in claims:
            c_words = set(re.findall(r"\b[a-z]{4,}\b", claim.lower()))
            if not c_words:
                continue
            overlap = len(c_words & ctx_words) / len(c_words)
            scores.append(overlap)
        return float(np.mean(scores)) if scores else 1.0

    # ------------------------------------------------------------------
    # Fix suggestions
    # ------------------------------------------------------------------

    def _suggest_fix(
        self,
        risk: str,
        top_chunks: list[ContextChunk],
        faithfulness: float,
        context_tokens: int,
    ) -> str:
        if context_tokens > 60_000 and risk in ("high", "critical"):
            return "Switch to RAG — full context loading is unreliable at this size"
        if risk == "critical":
            return "Switch to RAG — full context loading is failing at this size"
        if risk == "high":
            depth_pct = int(top_chunks[0].depth * 100) if top_chunks else 50
            return f"Move relevant content from {depth_pct}% depth to end of context"
        if risk == "medium":
            if faithfulness < 0.6:
                return "Context ignored — consider chunking or summarizing before this query"
            return "Consider moving critical content toward start or end of context"
        if faithfulness < 0.6:
            return "Model may have ignored context — verify response against source material"
        return "Context position looks safe for this model"


# Module-level singleton so callers don't need to instantiate
_default_analyzer = ContextAnalyzer(use_embeddings=True)


def analyze(
    messages: list[dict],
    response: str,
    model: str,
    call_index: int = 0,
) -> AnalysisResult:
    """Convenience wrapper around the default ContextAnalyzer."""
    return _default_analyzer.analyze(messages, response, model, call_index)


def analyze_pre_call(
    messages: list[dict],
    model: str,
    call_index: int = 0,
) -> AnalysisResult:
    """Convenience wrapper for pre-call analysis (no response/faithfulness check)."""
    return _default_analyzer.analyze_pre_call(messages, model, call_index)
