"""Silent mode — logs to session but prints nothing during calls.

Useful for CI/CD or production where you want observability without noise.
Run this, then check the session:

    context-guardian session show --app silent-demo
"""

import openai
from context_guardian import Guardian
from context_guardian.session import Session

client = Guardian(
    openai.OpenAI(),
    model="gpt-4o",
    report="silent",
    app="silent-demo",
)

LONG_CONTEXT = (
    "The API documentation describes various endpoints and their parameters. "
    * 200
    + "\nThe rate limit for the search endpoint is 100 requests per minute. "
    "Exceeding this limit returns a 429 status code with a Retry-After header."
    + " Additional text to push the key fact into the middle. " * 200
)

questions = [
    "What is the rate limit for the search endpoint?",
    "What happens when you exceed the rate limit?",
]

for q in questions:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LONG_CONTEXT},
            {"role": "user", "content": q},
        ],
    )
    print(f"Q: {q}")
    print(f"A: {response.choices[0].message.content}\n")

# Dump session at end
print("\n--- Session Summary ---")
client.session_summary()

# Or load from disk to inspect later
s = Session("silent-demo")
import json
print("\n--- Raw session data (last 3 records) ---")
history = s.load_history()
for rec in history[-3:]:
    print(json.dumps(rec.to_dict(), indent=2))
