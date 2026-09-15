from google import genai
import os


import time

class GeminiReplyGenerator:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate_reply(
        self,
        customer_message,
        historical_examples
    ):

        evidence_text = ""

        for _, row in historical_examples.iterrows():

            evidence_text += (
                f"\nHistorical customer message:\n"
                f"{row['customer_message']}\n"
                f"Historical Amazon reply:\n"
                f"{row['amazon_reply']}\n"
            )

        prompt = f"""
You are an Amazon customer support assistant.

Your task is to draft a helpful customer support reply.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT EXAMPLES:
{evidence_text}

Instructions:
1. Use ONLY information supported by the historical support examples.
2. Do not invent policies, refunds, delivery dates, account information, or troubleshooting steps.
3. Do not create, modify, or guess URLs. If a URL appears in the historical evidence, you may use it; otherwise do not include a URL.
4. Do not claim that an action has already been completed unless the historical evidence supports that exact action.
5. Do not assume facts about the customer's order, account, payment, or delivery.
6. If the historical examples do not provide enough information to safely answer the customer, recommend contacting Amazon Customer Support rather than guessing.
7. Keep the reply concise, professional, empathetic, and suitable for customer support.
8. Base the response on the most relevant historical examples, not unrelated examples.
9. Return only the draft customer reply.
"""

        for attempt in range(5):
            try:
                interaction = self.client.interactions.create(
                    model="gemini-3.6-flash",
                    input=prompt
                )
                break
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    time.sleep((attempt + 1) * 5)
                else:
                    raise e

        return interaction.output_text.strip()