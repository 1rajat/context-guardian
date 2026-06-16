# PDF Q&A — context-guardian tester

A Streamlit app that lets you chat with any PDF while **context-guardian** monitors every call and shows live risk analysis in a side panel.

The app intentionally sends the **full PDF as context** with no chunking — so you can watch context-guardian catch the "lost in the middle" problem in real time.

## Setup

```bash
cd guardian-tester

# Install everything (includes context-guardian from the local path)
pip install -r requirements.txt

# Run
streamlit run app.py
```

## What the app does

1. **Upload a PDF** — full text is extracted with pdfplumber
2. **Ask a question** — the entire PDF is sent to OpenAI as the system prompt
3. **See the answer** in the chat panel
4. **See context-guardian's analysis** in the right panel after every call

## Right panel: what each field means

| Field | What it tells you |
|---|---|
| **Risk level** | LOW / MEDIUM / HIGH / CRITICAL — based on where relevant content sits in the context |
| **Relevant depth** | The % position of the most relevant passage. ~50% = buried in the middle = highest risk |
| **Context tokens** | Total tokens sent as context (not counting your question) |
| **Faithfulness** | 0–1 score: how well the model's answer is grounded in the document |
| **Fix suggestion** | What to do if risk is high (move content, use RAG, etc.) |
| **Relevant chunk** | Preview of the passage context-guardian identified as most relevant to your question |

## Why no chunking?

The whole point of this demo is to show the **lost-in-the-middle** problem. Large PDFs often have the relevant answer buried at 40–60% depth in the context window — exactly where most LLMs pay the least attention. context-guardian finds it and tells you.

To fix it, you'd normally use RAG (chunk → embed → retrieve just the relevant section). This app deliberately doesn't do that so you can see the risk scores go up.

## Requirements

- Python 3.9+
- OpenAI API key (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, or `gpt-3.5-turbo`)
- context-guardian package at `../llm-stress-test` (installed via requirements.txt)
