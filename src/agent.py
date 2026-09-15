from src.intent_classifier import AmazonIntentClassifier
from src.retriever import AmazonRetriever
from src.gemini_generator import GeminiReplyGenerator
from src.decision import EscalationDecision


class AmazonSupportAgent:

    def __init__(self):

        self.classifier = AmazonIntentClassifier()
        self.retriever = AmazonRetriever(
    "data/amazon_knowledge_base_clean.csv"
)
        self.generator = GeminiReplyGenerator()
        self.decision_maker = EscalationDecision()

    def analyze(self, message, top_k=3):

        # 1. Classify the customer message
        intent, confidence = self.classifier.predict_with_confidence(
            message
        )

        # 2. Retrieve similar historical conversations
        evidence = self.retriever.retrieve(
            message,
            top_k=top_k
        )

        # 3. Get the similarity of the strongest evidence
        evidence_similarity = float(
            evidence.iloc[0]["similarity"]
        )

        # 4. Decide whether to auto-handle or escalate
        decision = self.decision_maker.decide(
    intent=intent,
    evidence_similarity=evidence_similarity,
    customer_message=message
)

        # 5. Generate a reply
        reply = self.generator.generate_reply(
            message,
            evidence
        )

        return {
            "message": message,
            "intent": intent,
            "confidence": confidence,
            "evidence_similarity": evidence_similarity,
            "decision": decision["decision"],
            "decision_reason": decision["reason"],
            "reply": reply,
            "evidence": evidence
        }