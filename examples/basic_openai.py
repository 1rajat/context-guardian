"""Simplest possible example — wrap the OpenAI client and ask a question."""

import openai
from context_guardian import Guardian

CONTEXT = """
Project Alpha was launched in Q1 2024. It aims to reduce customer churn by 20%.
The primary stakeholder is Alice Chen from the growth team.

Clause 14 — Termination: Either party may terminate this agreement with 30 days
written notice. Termination without cause requires a $5,000 settlement fee.

The budget for Project Alpha is $150,000 for the fiscal year.
All invoices must be submitted by the 15th of each month.
""" * 30  # repeat to simulate realistic context depth

client = Guardian(openai.OpenAI(), model="gpt-4o")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": CONTEXT},
        {"role": "user", "content": "What does clause 14 say about termination fees?"},
    ],
)

print(response.choices[0].message.content)
