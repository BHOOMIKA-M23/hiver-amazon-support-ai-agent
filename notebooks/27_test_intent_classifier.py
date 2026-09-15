import sys
sys.path.append(".")

from src.intent_classifier import AmazonIntentClassifier


# Create the classifier
classifier = AmazonIntentClassifier()


# Test messages
test_messages = [
    "My package says it was delivered but I never received it.",
    "I want to return this item and get a refund.",
    "I cannot log into my Amazon account.",
    "How do I connect my Fire Stick?",
    "I want to cancel my order.",
    "My Prime membership is not working.",
    "My card was charged twice."
]


print("\nINTENT CLASSIFICATION RESULTS")
print("=" * 70)


for message in test_messages:

    intent, confidence = (
        classifier.predict_with_confidence(message)
    )

    print("\nCUSTOMER:")
    print(message)

    print("PREDICTED INTENT:")
    print(intent)

    print("CONFIDENCE:")
    print(round(confidence, 3))

    print("-" * 70)