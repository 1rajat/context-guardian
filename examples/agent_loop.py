"""Multi-step agent loop — context-guardian monitors all calls, reports at end."""

import openai
from context_guardian import ContextWatcher

client = openai.OpenAI()

KNOWLEDGE_BASE = """
COMPANY HANDBOOK — Version 3.2

Chapter 1: Onboarding
New employees must complete onboarding within their first week.
All access requests go through IT via the internal portal at hr.internal/access.

Chapter 2: Expense Policy
Expenses under $50 can be submitted without manager approval.
Expenses between $50–$500 require direct manager approval.
Expenses over $500 require VP-level sign-off and a receipt.
Submit all expenses within 30 days of the purchase date.

""" + "\n".join([
    f"Chapter {i}: Standard policy section {i} with general guidelines and procedures."
    for i in range(3, 30)
]) + """

Chapter 30: Remote Work
Employees may work remotely up to 3 days per week with manager approval.
All remote work must be from a secure, private network connection.
Core hours are 10am–3pm in the employee's local timezone.

Chapter 31: PTO Policy
Employees accrue 15 days PTO per year in the first 3 years.
After 3 years, PTO accrual increases to 20 days per year.
Unused PTO up to 5 days may be carried over to the following year.
"""

AGENT_TASKS = [
    "What is the expense approval limit for direct manager approval?",
    "How many days of PTO do I get in year 4?",
    "What are the core hours for remote workers?",
]

watcher = ContextWatcher(model="gpt-4o", report="end", app="hr-agent")

with watcher.watch():
    for task in AGENT_TASKS:
        print(f"\nAgent task: {task}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"You are an HR assistant. Answer based on the handbook.\n\n{KNOWLEDGE_BASE}"},
                {"role": "user", "content": task},
            ],
        )
        print(f"Response: {response.choices[0].message.content}")

# Summary table is printed when the context manager exits
