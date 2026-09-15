import sys

sys.path.append(".")

from src.decision import EscalationDecision


decision_maker = EscalationDecision()


tests = [
    {
        "name": "Strong refund case",
        "intent": "refund_return",
        "evidence_similarity": 0.80,
        "customer_message": "I want to return this item and get a refund.",
    },
    {
        "name": "Weak evidence",
        "intent": "order_issue",
        "evidence_similarity": 0.30,
        "customer_message": "I have a problem with something I ordered.",
    },
    {
        "name": "Explicit human request",
        "intent": "delivery_shipping",
        "evidence_similarity": 0.80,
        "customer_message": "My package is late. I want to speak to a human agent.",
    },
    {
        "name": "Account payment issue",
        "intent": "account_payment",
        "evidence_similarity": 0.80,
        "customer_message": "There is an unexpected charge on my account.",
    },
    {
        "name": "Strong delivery case",
        "intent": "delivery_shipping",
        "evidence_similarity": 0.80,
        "customer_message": "My package says delivered but I never received it.",
    },
]


for test in tests:
    result = decision_maker.decide(
        intent=test["intent"],
        evidence_similarity=test["evidence_similarity"],
        customer_message=test["customer_message"],
    )

    print("\n" + "=" * 60)
    print(test["name"])
    print("=" * 60)
    print("Customer:", test["customer_message"])
    print("Decision:", result["decision"])
    print("Reason:", result["reason"])