import pandas as pd

input_file = "data/amazon_clean.csv"
output_file = "data/intent_labeling_sample.csv"

df = pd.read_csv(input_file)

# Select 50 random messages for manual labeling
sample = df.sample(
    n=50,
    random_state=42
).copy()

# Add an empty intent column
sample["intent"] = ""

# Keep only the columns needed for labeling
sample = sample[
    [
        "customer_tweet_id",
        "customer_message",
        "amazon_reply",
        "intent"
    ]
]

sample.to_csv(
    output_file,
    index=False
)

print("Labeling sample created!")
print("Number of messages:", len(sample))
print("Saved to:", output_file)

print("\nIntent values to use:")
print("delivery_shipping")
print("order_issue")
print("refund_return")
print("account_payment")
print("product_issue")
print("prime_issue")
print("other")