"""Compare context risk across models for the same query.

Runs the same scenario through gpt-4o and claude-3-5-sonnet,
prints a side-by-side comparison of depth risk and faithfulness.
Requires OPENAI_API_KEY and ANTHROPIC_API_KEY.
"""

import json
import openai
import anthropic
from context_guardian.analyzer import analyze
from context_guardian.reporter import to_json, print_table

# Same context for both models — key fact buried in the middle
PREAMBLE = "Background information about company operations. " * 150
KEY_FACT = "\nThe Q3 revenue target is $4.2 million, set by the CFO on June 15th.\n"
FILLER = "Additional operational details and supporting data. " * 150

CONTEXT = PREAMBLE + KEY_FACT + FILLER
QUESTION = "What is the Q3 revenue target and who set it?"

results = []

# --- OpenAI ---
try:
    oai_client = openai.OpenAI()
    oai_resp = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CONTEXT},
            {"role": "user", "content": QUESTION},
        ],
    )
    oai_text = oai_resp.choices[0].message.content
    oai_result = analyze(
        messages=[
            {"role": "system", "content": CONTEXT},
            {"role": "user", "content": QUESTION},
        ],
        response=oai_text,
        model="gpt-4o",
        call_index=0,
    )
    oai_result.model = "gpt-4o"
    results.append(oai_result)
    print(f"gpt-4o response: {oai_text[:200]}")
except Exception as e:
    print(f"OpenAI error: {e}")

# --- Anthropic ---
try:
    ant_client = anthropic.Anthropic()
    ant_resp = ant_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=CONTEXT,
        messages=[{"role": "user", "content": QUESTION}],
        max_tokens=256,
    )
    ant_text = ant_resp.content[0].text
    ant_result = analyze(
        messages=[
            {"role": "system", "content": CONTEXT},
            {"role": "user", "content": QUESTION},
        ],
        response=ant_text,
        model="claude-3-5-sonnet",
        call_index=1,
    )
    ant_result.model = "claude-3-5-sonnet"
    results.append(ant_result)
    print(f"claude-3-5-sonnet response: {ant_text[:200]}")
except Exception as e:
    print(f"Anthropic error: {e}")

# --- Comparison table ---
if results:
    print("\n--- Model Comparison ---")
    print_table(results)
    print("\n--- Raw scores ---")
    for r in results:
        print(json.dumps(to_json(r), indent=2))
