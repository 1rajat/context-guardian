"""Legal document review — the canonical context-guardian use case."""

import openai
from context_guardian import watch_llm

client = openai.OpenAI()

CONTRACT = """
MASTER SERVICE AGREEMENT

This Master Service Agreement ("Agreement") is entered into as of January 1, 2024
between Acme Corp ("Client") and WidgetCo ("Provider").

1. SERVICES. Provider agrees to deliver software development services as described
   in each Statement of Work ("SOW") executed by both parties.

2. PAYMENT. Client shall pay Provider within 30 days of invoice receipt.
   Late payments accrue interest at 1.5% per month.

3. CONFIDENTIALITY. Both parties agree to maintain strict confidentiality of all
   proprietary information for a period of 5 years following termination.

4. INTELLECTUAL PROPERTY. All work product created under this Agreement shall be
   considered work-for-hire and ownership transfers to Client upon full payment.

""" + "\n".join([f"SECTION {i}. Lorem ipsum standard legal boilerplate clause {i}." for i in range(5, 50)]) + """

14. TERMINATION FOR CAUSE. Either party may terminate this Agreement immediately
    upon written notice if the other party materially breaches this Agreement and
    fails to cure such breach within 14 days of receiving written notice.
    The terminating party shall owe no settlement fee in cases of termination for cause.

15. TERMINATION WITHOUT CAUSE. Either party may terminate this Agreement with
    60 days written notice. A $10,000 early termination fee applies if Client
    terminates without cause within the first 12 months of the Agreement.

""" + "\n".join([f"SECTION {i}. Additional standard provisions and boilerplate text for section {i}." for i in range(16, 40)])


@watch_llm(model="gpt-4o", report="inline", app="contract-review")
def review_contract(contract: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a legal assistant. Review the contract carefully.\n\n{contract}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    questions = [
        "What is the termination fee if we cancel without cause in month 6?",
        "How long does the confidentiality obligation last after termination?",
        "Who owns the IP for work delivered under this agreement?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = review_contract(CONTRACT, q)
        print(f"A: {answer}")
