import sys
sys.path.append(".")

from src.gemini_generator import GeminiReplyGenerator


generator = GeminiReplyGenerator()


customer_message = "My package says it was delivered but I never received it."


historical_examples = [
    {
        "customer_message": "My package says it was delivered Monday but I never received it.",
        "amazon_reply": "I'm sorry to hear this! Have you had a chance to report this to our Customer Support team?"
    },
    {
        "customer_message": "My package says it was delivered and it wasn't?",
        "amazon_reply": "I'm sorry you didn't receive your order! Have you tried these steps to help locate your missing package?"
    }
]


import pandas as pd

historical_examples = pd.DataFrame(historical_examples)


reply = generator.generate_reply(
    customer_message,
    historical_examples
)


print("\nCUSTOMER MESSAGE")
print("----------------")
print(customer_message)

print("\nGEMINI GENERATED REPLY")
print("----------------------")
print(reply)