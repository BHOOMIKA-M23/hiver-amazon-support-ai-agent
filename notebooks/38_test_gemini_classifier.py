import pandas as pd
from google import genai
import os
import json


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# Load test set
# --------------------------------------------------

df = pd.read_csv("data/intent_test.csv")

# Use customer_message when available, otherwise use text
df["message"] = df["customer_message"].fillna(df["text"])

df = df.dropna(subset=["message", "intent"])

df["message"] = df["message"].astype(str)


# --------------------------------------------------
# Allowed intents
# --------------------------------------------------

intents = [
    "delivery_shipping",
    "order_issue",
    "refund_return",
    "account_payment",
    "product_help",
    "prime_issue",
    "other"
]


# --------------------------------------------------
# Build one prompt for all test examples
# --------------------------------------------------

messages_text = ""

for i, message in enumerate(df["message"], start=1):
    messages_text += f"\n{i}. {message}\n"


prompt = f"""
You are an intent classifier for Amazon customer support.

Classify every customer message into exactly ONE of these intents:

delivery_shipping
- Late, missing, delayed, tracking, carrier, or delivery problems.

order_issue
- Problems with placing, cancelling, changing, or processing an order.

refund_return
- Returns, refunds, replacements, return labels, or returning products.

account_payment
- Account access, login, charges, payment methods, gift cards,
  Amazon Pay, or account-related restrictions.

product_help
- Problems using, configuring, troubleshooting, or operating an
  Amazon product or device.

prime_issue
- Problems specifically related to Prime membership, Prime benefits,
  Prime Video, or Prime-specific services.

other
- Messages with no clear issue/request covered by the categories above,
  acknowledgements, thanks, complaints without a specific support issue,
  or requests for contact information.

IMPORTANT:
- Focus on the customer's main reason for contacting Amazon.
- Do not infer an intent from unrelated words.
- Return exactly one intent for each message.
- Keep the same order as the input.
- Return ONLY a JSON array of intent names.
- Do not include explanations.

CUSTOMER MESSAGES:
{messages_text}
"""


# --------------------------------------------------
# Send ONE request to Gemini
# --------------------------------------------------

print("\nGEMINI INTENT CLASSIFIER")
print("=" * 70)

response = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)

raw_output = response.output_text.strip()

print("\nGemini raw output:")
print(raw_output)


# --------------------------------------------------
# Parse predictions
# --------------------------------------------------

try:

    predictions = json.loads(raw_output)

except json.JSONDecodeError:

    # Handle markdown code fences if Gemini adds them
    cleaned = raw_output.replace("```json", "").replace("```", "").strip()

    predictions = json.loads(cleaned)


# --------------------------------------------------
# Validate predictions
# --------------------------------------------------

if len(predictions) != len(df):

    raise ValueError(
        f"Expected {len(df)} predictions but received {len(predictions)}."
    )


for prediction in predictions:

    if prediction not in intents:

        raise ValueError(
            f"Invalid intent returned by Gemini: {prediction}"
        )


# --------------------------------------------------
# Show predictions
# --------------------------------------------------

for i, (message, actual, predicted) in enumerate(
    zip(df["message"], df["intent"], predictions),
    start=1
):

    status = "CORRECT" if actual == predicted else "WRONG"

    print(f"\n{i}. {status}")
    print("Actual:   ", actual)
    print("Predicted:", predicted)
    print("Message:  ", message)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(
    df["intent"],
    predictions
)

print("\n\nGEMINI RESULTS")
print("=" * 70)

print("Accuracy:", round(accuracy, 3))

print("\nClassification Report:")

print(
    classification_report(
        df["intent"],
        predictions,
        labels=intents,
        zero_division=0
    )
)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results = pd.DataFrame({
    "message": df["message"],
    "actual_intent": df["intent"],
    "predicted_intent": predictions
})

results.to_csv(
    "data/gemini_intent_results.csv",
    index=False
)

print("\nResults saved to: data/gemini_intent_results.csv")