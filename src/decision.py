class EscalationDecision:

    def decide(
        self,
        intent,
        evidence_similarity,
        customer_message
    ):

        message_lower = customer_message.lower()

        # Rule 1: Weak historical evidence
        if evidence_similarity < 0.45:
            return {
                "decision": "escalate",
                "reason": "Insufficient historical evidence for a confident reply."
            }

        # Rule 2: Customer explicitly requests human support
        human_requests = [
            "human",
            "agent",
            "representative",
            "talk to someone",
            "speak to someone",
            "customer service"
        ]

        if any(phrase in message_lower for phrase in human_requests):
            return {
                "decision": "escalate",
                "reason": "Customer explicitly requested human support."
            }

        # Rule 3: Potentially sensitive account/payment issues
        if intent == "account_payment":
            return {
                "decision": "escalate",
                "reason": "Account or payment issues require human review."
            }

        # Otherwise, auto-handle
        return {
            "decision": "auto_handle",
            "reason": "Relevant historical evidence is available and no escalation trigger was detected."
        }