# context-guardian

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> The first Python package that detects context blindness in your LLM app — silently, in real time.

Every LLM observability tool tracks cost, latency, and errors.  
Nobody tracks whether your model actually **read** what you gave it.

**context-guardian** watches every LLM call and tells you when:
- Your answer was buried in a position the model tends to ignore
- The model responded but silently ignored the context
- Your RAG pipeline is feeding content into a known blind spot

```
pip install context-guardian
```

---

## 2-line integration

```python
import openai
from context_guardian import Guardian

# Before:
# client = openai.OpenAI()

# After — one word change:
client = Guardian(openai.OpenAI(), model="gpt-4o")

# Everything else is identical
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": very_long_system_prompt},
        {"role": "user", "content": "What does clause 14 say about termination?"}
    ]
)
```

Guardian intercepts the call, runs all three analysis layers locally, and prints a warning:

```
╭─ context-guardian ─────────────────────────────────────────╮
│ ⚠  HIGH RISK  │ Depth: 44% │ Tokens: 28,432               │
│ Faithfulness: 0.71  — answer may not use context           │
│ 💡 Move relevant content from 44% depth to end of context  │
╰────────────────────────────────────────────────────────────╯
```

Zero perceived latency — analysis runs in a background thread.

---

## What it catches that nothing else does

Most observability tools tell you **what** was called and **how long** it took.  
context-guardian tells you **whether the model actually used what you gave it**.

| What it detects | How |
|---|---|
| Content in the model's attention blind zone | Semantic depth analysis + risk tables from real stress tests |
| Model responded but ignored context | Local faithfulness scoring (no extra API call) |
| Your app's specific failure patterns | Learns from your call history over time |

---

## Three APIs — pick the one that fits your code

### API 1 — Drop-in client wrapper

```python
from context_guardian import Guardian
import openai

client = Guardian(openai.OpenAI(), model="gpt-4o", report="inline")
response = client.chat.completions.create(...)
```

Works with `openai.OpenAI()` and `anthropic.Anthropic()`.

### API 2 — Context manager

```python
from context_guardian import ContextWatcher

watcher = ContextWatcher(model="gpt-4o", report="end")

with watcher.watch():
    r1 = client.chat.completions.create(...)
    r2 = client.chat.completions.create(...)

report = watcher.report()  # prints table + returns structured dict
```

### API 3 — Decorator

```python
from context_guardian import watch_llm

@watch_llm(model="gpt-4o", report="inline", app="my-contract-bot")
def review_contract(contract: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": contract + "\n\n" + question}]
    )
    return response.choices[0].message.content
```

---

## Report modes

| Mode | Latency added | When to use |
|---|---|---|
| `inline` (default) | ~150ms background | Development, debugging |
| `end` | ~0ms perceived | End-of-function summary |
| `silent` | ~0ms | CI/CD — logs to session only |
| `json` | ~150ms background | Machine-readable pipeline integration |

---

## How it works

### Layer 1 — Semantic Relevance Mapping

Uses `sentence-transformers` (all-MiniLM-L6-v2, runs locally, free, ~90MB) to embed both the user's question and every chunk of context. Finds where the most relevant content sits, then looks up that depth in pre-computed model risk tables derived from real needle-in-haystack stress tests.

> "Your most relevant content is at 44% depth → HIGH RISK for gpt-4o at 28k tokens"

### Layer 2 — Faithfulness Judge

After the model responds, runs a local check with no extra API call:
- Extracts key claims from the response
- Checks if each claim is grounded in the context via cosine similarity
- Scores 0.0–1.0: 1.0 = fully grounded, 0.0 = ignored context

This catches **silent hallucination** — when the model had the answer but didn't use it.

### Layer 3 — Session Memory

Persists every call to `~/.context_guardian/sessions/{app}.jsonl`. After 10+ calls it computes per-depth-zone failure rates for your specific app. After 50+ calls it emits personalised warnings:

> "In YOUR pipeline: content at 40–60% depth fails 67% of the time at >16k tokens"

```python
from context_guardian import Session
s = Session("my-contract-bot")
s.print_summary()
```

---

## CLI

```bash
# Run a Python script and report context risks after it exits
context-guardian watch script.py --model gpt-4o --report end

# Show session history
context-guardian session show --app my-contract-bot

# Clear session history
context-guardian session clear --app my-contract-bot

# List all apps with history
context-guardian session list

# Generate report from a saved JSONL session file
context-guardian report results/session.jsonl
```

---

## The stress tester (advanced)

The original needle-in-haystack CLI is still available — it's what generated the risk tables:

```bash
# Quick test against your Ollama model
llm-stress-test run --model llama3.2 --quick

# Demo mode — no API key needed
llm-stress-test run --model gpt-4o --demo

# Full test with heatmap
llm-stress-test run --model gpt-4o --context-lengths 4k,16k,32k,64k

# Cost estimate before running
llm-stress-test run --model gpt-4o --cost-estimate
```

See [RESULTS.md](RESULTS.md) for community-submitted benchmark results.

---

## Installation

```bash
pip install context-guardian

# sentence-transformers downloads all-MiniLM-L6-v2 on first use (~90MB, one-time)
```

**Requirements:** Python 3.9+, openai >= 1.0 or anthropic >= 0.20

---

## Contributing

```bash
git clone https://github.com/1rajat/context-guardian
cd context-guardian
pip install -e ".[dev]"
pytest
```

Submit stress test results to [RESULTS.md](RESULTS.md) — the more models we cover, the better the risk tables get.
