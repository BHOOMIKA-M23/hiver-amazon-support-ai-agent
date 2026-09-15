import pandas as pd

file_path = "data/amazon_support.csv"

df = pd.read_csv(file_path, low_memory=False)

customer = df[df["inbound"] == True].copy()

intent_keywords = {
    "delivery_issue": ["delivery", "delivered", "late", "missing"],
    "order_issue": ["order", "ordered", "cancel", "ordering"],
    "shipping_tracking": ["tracking", "shipping", "carrier"],
    "refund_return": ["refund", "return", "replacement"],
    "account_issue": ["account", "login", "password", "locked"],
    "payment_issue": ["payment", "charge", "charged", "gift card"],
    "product_issue": ["damaged", "broken", "defective", "product"],
    "prime_issue": ["prime"]
}

for intent, keywords in intent_keywords.items():

    pattern = "|".join(keywords)

    matches = customer[
        customer["text"]
        .astype(str)
        .str.contains(pattern, case=False, na=False)
    ]

    print("\n" + "=" * 60)
    print(intent.upper())
    print("Number of matching messages:", len(matches))
    print("=" * 60)

    for text in matches["text"].sample(
        n=min(5, len(matches)),
        random_state=42
    ):
        print("-", text)