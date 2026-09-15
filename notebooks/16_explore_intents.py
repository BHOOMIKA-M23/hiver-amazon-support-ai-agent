import pandas as pd

file_path = "data/amazon_clean.csv"

df = pd.read_csv(file_path)

messages = df["customer_message"].astype(str).str.lower()

intent_keywords = {

    "delivery_shipping": [
        "delivery",
        "delivered",
        "delivery date",
        "shipping",
        "shipped",
        "tracking",
        "package",
        "parcel",
        "courier",
        "arrived",
        "late",
        "missing"
    ],

    "order_issue": [
        "order",
        "ordered",
        "cancel",
        "cancelled",
        "canceled",
        "change my order",
        "pre-order",
        "preorder"
    ],

    "refund_return": [
        "refund",
        "return",
        "returned",
        "replacement",
        "replace",
        "money back"
    ],

    "account_payment": [
        "account",
        "login",
        "password",
        "hacked",
        "payment",
        "charged",
        "charge",
        "card",
        "billing"
    ],

    "product_issue": [
        "broken",
        "damaged",
        "defective",
        "doesn't work",
        "doesnt work",
        "not working",
        "faulty",
        "product"
    ],

    "prime_issue": [
        "prime",
        "prime membership",
        "prime member",
        "prime video",
        "prime music"
    ]
}

for intent, keywords in intent_keywords.items():

    mask = pd.Series(False, index=df.index)

    for keyword in keywords:
        mask = mask | messages.str.contains(
            keyword,
            na=False
        )

    matches = df[mask]

    print("\n" + "=" * 70)
    print(intent.upper())
    print("Matching messages:", len(matches))
    print("=" * 70)

    for message in matches["customer_message"].head(5):
        print("-", message)


print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("Total cleaned messages:", len(df))