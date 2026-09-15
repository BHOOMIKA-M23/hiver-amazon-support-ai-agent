from google import genai
import os
import re


import time

class AmazonIntentClassifier:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(api_key=api_key)

        self.intents = [
            "delivery_shipping",
            "order_issue",
            "refund_return",
            "account_payment",
            "product_help",
            "prime_issue",
            "other"
        ]

    def predict(self, message):

        intent, confidence = self.predict_with_confidence(message)

        return intent

    def predict_with_confidence(self, message):

        prompt = f"""
You are an intent classifier for Amazon customer support.

Classify the customer message into exactly ONE of these intents:

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
- Return ONLY the intent name.
- Do not explain your answer.

CUSTOMER MESSAGE:
{message}
"""

        for attempt in range(5):
            try:
                response = self.client.interactions.create(
                    model="gemini-3.6-flash",
                    input=prompt
                )
                break
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    time.sleep((attempt + 1) * 5)
                else:
                    raise e

        prediction = response.output_text.strip()

        prediction = re.sub(
            r"[^a-z_]",
            "",
            prediction.lower()
        )

        if prediction not in self.intents:
            prediction = "other"

        return prediction, None