"""PDF Q&A — context-guardian live analysis demo.

Upload a PDF, ask questions, watch context-guardian's risk analysis in real time.
"""

from __future__ import annotations

import io

import requests
import streamlit as st
import openai
import pdfplumber

from context_guardian.analyzer import analyze, AnalysisResult, ContextChunk

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PDF Q&A · context-guardian",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "messages": [],        # list of {role, content} — conversational history
    "pdf_text": "",        # full extracted text
    "pdf_key": "",         # fingerprint to detect PDF change
    "last_analysis": None, # AnalysisResult | None
    "call_history": [],    # list[AnalysisResult]
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_OLLAMA_BASE = "http://localhost:11434"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — Ollama
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_ollama_models() -> list[str] | None:
    """Return list of pulled model names, or None if Ollama is not reachable."""
    try:
        resp = requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=2)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — PDF extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(raw_bytes: bytes) -> str:
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n\n".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — risk rendering
# ──────────────────────────────────────────────────────────────────────────────

_RISK_META = {
    "low":      {"color": "#2ecc71", "bg": "#2ecc7118", "emoji": "✅", "label": "LOW"},
    "medium":   {"color": "#f39c12", "bg": "#f39c1218", "emoji": "⚠️", "label": "MEDIUM"},
    "high":     {"color": "#e74c3c", "bg": "#e74c3c18", "emoji": "🔴", "label": "HIGH"},
    "critical": {"color": "#8e44ad", "bg": "#8e44ad18", "emoji": "💀", "label": "CRITICAL"},
}


def _risk_meta(risk: str) -> dict:
    return _RISK_META.get(risk, _RISK_META["medium"])


def render_risk_badge(risk: str) -> None:
    m = _risk_meta(risk)
    st.markdown(
        f"""<div style="background:{m['bg']};border-left:4px solid {m['color']};
        padding:12px 16px;border-radius:6px;margin-bottom:12px;">
        <span style="font-size:1.25em;font-weight:700;color:{m['color']};">
        {m['emoji']}&nbsp; {m['label']} RISK
        </span></div>""",
        unsafe_allow_html=True,
    )


def render_faithfulness_bar(score: float) -> None:
    color = "#2ecc71" if score >= 0.8 else ("#f39c12" if score >= 0.6 else "#e74c3c")
    label = "Good" if score >= 0.8 else ("Partial" if score >= 0.6 else "Poor")
    pct = int(score * 100)
    st.markdown(
        f"""<div style="margin-bottom:14px;">
        <span style="font-size:0.82em;color:#666;font-weight:500;">
        Faithfulness score
        </span>
        <div style="display:flex;align-items:center;gap:10px;margin-top:4px;">
          <div style="flex:1;background:#eee;border-radius:4px;height:8px;">
            <div style="width:{pct}%;background:{color};height:8px;border-radius:4px;"></div>
          </div>
          <span style="font-size:1.1em;font-weight:700;color:{color};min-width:36px;">
          {score:.2f}
          </span>
          <span style="font-size:0.82em;color:{color};">{label}</span>
        </div></div>""",
        unsafe_allow_html=True,
    )


def render_chunk_preview(chunk: ContextChunk) -> None:
    snippet = chunk.text[:400].replace("<", "&lt;").replace(">", "&gt;")
    depth_pct = int(chunk.depth * 100)
    st.markdown(
        f"**Most relevant chunk** "
        f"<span style='color:#888;font-size:0.85em;'>found at {depth_pct}% depth</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;
        padding:10px 12px;font-size:0.82em;font-family:monospace;white-space:pre-wrap;
        max-height:200px;overflow-y:auto;color:#333;line-height:1.5;">{snippet}…</div>""",
        unsafe_allow_html=True,
    )


def render_analysis_panel(result: AnalysisResult) -> None:
    render_risk_badge(result.position_risk)

    c1, c2 = st.columns(2)
    c1.metric(
        "Relevant depth",
        f"{result.risk_depth_pct:.0f}%",
        help=(
            "Where the most relevant passage sits in the context. "
            "~50% = buried in the middle (highest risk for most models)."
        ),
    )
    c2.metric(
        "Context tokens",
        f"{result.context_tokens:,}",
        help="Tokens in the context window, excluding the question.",
    )

    render_faithfulness_bar(result.faithfulness_score)

    if result.fix_suggestion:
        st.info(f"💡 **{result.fix_suggestion}**")

    if result.top_chunks:
        render_chunk_preview(result.top_chunks[0])


def render_history_row(call_num: int, r: AnalysisResult) -> None:
    m = _risk_meta(r.position_risk)
    st.markdown(
        f"{m['emoji']} **#{call_num}** "
        f"<span style='color:{m['color']};font-weight:600;'>{m['label']}</span>"
        f"&nbsp;·&nbsp;depth {r.risk_depth_pct:.0f}%"
        f"&nbsp;·&nbsp;faith {r.faithfulness_score:.2f}",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛡️ context-guardian")
    st.caption("PDF Q&A with live context analysis")
    st.divider()

    ollama_models = fetch_ollama_models()
    ollama_ready = False
    model = ""

    if ollama_models is None:
        st.error(
            "**Ollama not running**\n\n"
            "Start it with:\n```\nollama serve\n```",
        )
    elif len(ollama_models) == 0:
        st.warning(
            "Ollama is running but no models are pulled.\n\n"
            "Pull one with:\n```\nollama pull llama3.2\n```",
        )
    else:
        model = st.selectbox("Model", ollama_models, index=0)
        ollama_ready = True
        st.caption(f"Ollama at {_OLLAMA_BASE}")

    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_key != st.session_state.pdf_key:
            with st.spinner("Extracting text…"):
                raw = uploaded_file.read()
                try:
                    text = extract_pdf_text(raw)
                except Exception as e:
                    st.error(f"Failed to read PDF: {e}")
                    text = ""
            if text:
                st.session_state.pdf_text = text
                st.session_state.pdf_key = file_key
                st.session_state.messages = []
                st.session_state.last_analysis = None
                st.session_state.call_history = []
                st.toast("PDF loaded!", icon="📄")
            else:
                st.warning("Could not extract text from this PDF.")

    if st.session_state.pdf_text:
        words = len(st.session_state.pdf_text.split())
        chars = len(st.session_state.pdf_text)
        st.success(f"📄 {words:,} words · {chars:,} chars")
        with st.expander("Preview (first 400 chars)"):
            st.text(st.session_state.pdf_text[:400] + "…")

    st.divider()

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_analysis = None
        st.session_state.call_history = []
        st.rerun()

    st.divider()
    st.caption(
        "context-guardian analyzes every call locally — no extra API calls. "
        "It maps where relevant content sits in your context window and checks "
        "whether the model's answer is grounded in the document."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────

chat_col, analysis_col = st.columns([3, 2], gap="large")

# ──────────────────────────────────────────────────────────────────────────────
# Chat column
# ──────────────────────────────────────────────────────────────────────────────

with chat_col:
    st.subheader("Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    ready = ollama_ready and bool(st.session_state.pdf_text)
    hint = (
        "Ask a question about the PDF…" if ready
        else "Upload a PDF and select an Ollama model to begin"
    )

    user_input: str | None = st.chat_input(hint, disabled=not ready)

    if not ready:
        if not ollama_ready:
            st.warning("⬅️  Start Ollama and pull a model to use this app.")
        if not st.session_state.pdf_text:
            st.info("⬅️  Upload a PDF to start chatting.")

    if user_input and ready:
        # Append user message before API call so it appears in conversational history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Full PDF as system context — intentionally no chunking
        api_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer questions using only the document below.\n\n"
                    "DOCUMENT:\n"
                    + st.session_state.pdf_text
                    + "\n\nEND OF DOCUMENT"
                ),
            },
            *[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
        ]

        with st.chat_message("assistant"):
            answer = ""
            with st.spinner("Thinking…"):
                try:
                    oai = openai.OpenAI(
                        base_url=f"{_OLLAMA_BASE}/v1",
                        api_key="ollama",
                    )
                    resp = oai.chat.completions.create(
                        model=model,
                        messages=api_messages,
                        max_tokens=1024,
                    )
                    answer = resp.choices[0].message.content or ""
                except openai.APIConnectionError:
                    st.error(
                        "Could not reach Ollama. "
                        "Make sure it's running: `ollama serve`"
                    )
                    st.stop()
                except openai.BadRequestError as exc:
                    st.error(
                        f"Ollama rejected the request: {exc}\n\n"
                        "The PDF may be too large for this model's context window. "
                        "Try a model with a larger context (e.g. `ollama pull qwen2.5`)."
                    )
                    st.stop()
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")
                    st.stop()

            if not answer:
                st.warning("Model returned an empty response.")
                st.stop()

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # context-guardian analysis — runs locally, no API call
        with st.spinner("context-guardian analysing…"):
            try:
                result = analyze(
                    messages=api_messages,
                    response=answer,
                    model=model,
                    call_index=len(st.session_state.call_history),
                )
                st.session_state.last_analysis = result
                st.session_state.call_history.append(result)
            except Exception as exc:
                st.warning(f"context-guardian error: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# Analysis column
# ──────────────────────────────────────────────────────────────────────────────

with analysis_col:
    st.subheader("🛡️ context-guardian")

    if st.session_state.last_analysis is not None:
        render_analysis_panel(st.session_state.last_analysis)

        history = st.session_state.call_history
        if len(history) > 1:
            st.divider()
            shown = history[-5:]
            st.caption(f"Call history — last {len(shown)} of {len(history)}")
            for i, r in enumerate(reversed(shown)):
                render_history_row(len(history) - i, r)
    else:
        st.markdown(
            """<div style="text-align:center;padding:60px 20px;color:#999;">
            <div style="font-size:3.5em;margin-bottom:16px;">🛡️</div>
            <p style="font-weight:600;font-size:1.05em;margin-bottom:8px;color:#666;">
            Waiting for first call
            </p>
            <p style="font-size:0.9em;line-height:1.7;max-width:280px;margin:0 auto;">
            After you ask a question, context-guardian will report:<br>
            <strong>position risk · faithfulness score<br>
            token count · relevant chunk preview</strong>
            </p>
            </div>""",
            unsafe_allow_html=True,
        )
