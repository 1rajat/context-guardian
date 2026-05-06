# llm-stress-test

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/your-org/llm-stress-test/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/llm-stress-test/actions)

**Needle-in-a-haystack context window benchmarking for any LLM.**

Hides a secret fact at different positions inside increasingly large documents, then asks the model to find it. Reveals exactly where — and at what context length — your model starts to fail.

<!-- Add heatmap GIF here -->

---

## What does this test?

LLMs don't read documents uniformly. Research shows that models perform worse when relevant information is buried in the **middle** of a long context — they over-attend to the beginning and end. This is the ["lost in the middle" problem](https://arxiv.org/abs/2307.03172).

This tool quantifies that degradation cell by cell:

- **X-axis** — context length (1k → 128k tokens)
- **Y-axis** — where in the document the key fact is hidden (0% = start, 100% = end)
- **Color** — retrieval score (green = found it, red = failed)

A perfect model is uniformly green. Real models show a red band in the middle at large context sizes.

---

## Installation

**Requirements:** Python 3.9+, git

```bash
git clone https://github.com/your-org/llm-stress-test
cd llm-stress-test
pip install -e .
```

Verify the install:

```bash
llm-stress-test --version
llm-stress-test list-models
```

---

## Quick start — pick your provider

### Option A — Ollama (free, local, no account needed)

Best way to start. No API key, no cost, runs entirely on your machine.

**1. Install Ollama**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**2. Pull a model and start the server**

```bash
ollama pull llama3.2:3b      # small, fast (~2 GB)
# or
ollama pull qwen2.5:14b      # more capable (~9 GB)

ollama serve                  # starts the local API on http://localhost:11434
```

**3. Run the stress test**

```bash
llm-stress-test run --model llama3.2:3b --quick --trials 1
```

---

### Option B — OpenAI

```bash
export OPENAI_API_KEY=sk-...

# Check cost first (no API calls made)
llm-stress-test run --model gpt-4o-mini --quick --cost-estimate

# Run
llm-stress-test run --model gpt-4o-mini --quick --trials 3
```

Supported models: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`

---

### Option C — Anthropic

```bash
export ANTHROPIC_API_KEY=sk-ant-...

llm-stress-test run --model claude-3-5-sonnet --quick --trials 3
```

Supported models: `claude-3-5-sonnet`, `claude-3-5-haiku-20241022`, `claude-3-opus`

---

## What you get after every run

Three files land in `results/` automatically:

| File | What it is |
|------|-----------|
| `run_{model}_{timestamp}.json` | Raw scores — every trial, every cell |
| `heatmap_{model}_{timestamp}.png` | Static PNG heatmap (150 DPI, shareable) |
| `heatmap_{model}_{timestamp}.html` | Interactive Plotly heatmap — hover cells for details |

Plus a live ASCII heatmap printed directly in the terminal as soon as the run finishes.

---

## More commands

```bash
# Full run — all context lengths up to 32k, 3 trials per cell
llm-stress-test run --model llama3.2:3b --max-context 32k --trials 3

# Custom context lengths and depths
llm-stress-test run --model gpt-4o --context-lengths 4k,16k,64k --depths 0,25,50,75,100

# Estimate API cost before spending money (OpenAI/Anthropic only)
llm-stress-test run --model gpt-4o --max-context 64k --cost-estimate

# Regenerate heatmaps from a saved JSON (no API calls)
llm-stress-test plot results/run_llama3.2_3b_20241215.json

# Side-by-side comparison of two models
llm-stress-test compare results/run_llama3.2_3b_*.json results/run_qwen2.5_14b_*.json

# List all known models with max context and provider
llm-stress-test list-models
```

---

## CLI reference

```
llm-stress-test run [OPTIONS]
  --model / -m          Model name (required)
  --max-context         Cap context size, e.g. 32k
  --context-lengths     Custom comma-separated list: 4k,16k,32k
  --depths              Custom depth percents: 0,25,50,75,100
  --trials / -t         Trials per cell (default: 3)
  --output / -o         Output directory (default: results/)
  --quick               4 lengths × 3 depths — fast sanity check
  --cost-estimate       Print cost estimate and exit without running
  --no-html             Skip interactive HTML heatmap
  --no-png              Skip PNG heatmap
  --ollama-url          Ollama base URL (default: http://localhost:11434)

llm-stress-test plot JSON_FILE
  Regenerate PNG + HTML from a saved result file.

llm-stress-test compare FILE1 FILE2 ...
  Side-by-side interactive heatmap of two or more result files.

llm-stress-test list-models
  List known models with provider and max context size.
```

---

## How scoring works

Each (context length × depth) cell runs `--trials` independent calls and averages the scores.

The needle is one unmistakable sentence hidden in the filler document:

> *"The secret code word is: FLAMINGO-7429"*

The model is asked: *"What is the secret code word mentioned in the document?"*

| Score | Meaning |
|-------|---------|
| 1.00 | Perfect — exact code word returned |
| 0.50 | Partial — found one component (e.g. "FLAMINGO" but wrong number) |
| 0.00 | Failure — code word not found |
| ⚠️ | Hallucination — model returned a *different* code word confidently |

---

## Context lengths and defaults

| Mode | Context lengths | Depths | Cells |
|------|----------------|--------|-------|
| `--quick` | 4k, 16k, 32k, 64k | 10%, 50%, 90% | 12 |
| Full | 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k | 0%, 10%, 25%, 50%, 75%, 90%, 100% | 56 |

Context lengths exceeding the model's maximum are skipped automatically.

---

## Adding a new model

Extend `BaseAdapter` in [llm_stress_test/models.py](llm_stress_test/models.py):

```python
class MyProviderAdapter(BaseAdapter):
    def complete(self, system: str, user: str, max_tokens: int = 256) -> ModelResponse:
        # call your API here
        return ModelResponse(content="...", input_tokens=0, output_tokens=0, model=self.model)
```

Then register it in `get_adapter()`.

---

## Community leaderboard

Run the test against your model and submit your results to [RESULTS.md](RESULTS.md) via PR. See that file for the table format.

---

## License

MIT — see [LICENSE](LICENSE).
