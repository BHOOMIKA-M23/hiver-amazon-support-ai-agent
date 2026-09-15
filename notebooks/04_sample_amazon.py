import pandas as pd

file_path = "data/amazon_support.csv"

df = pd.read_csv(file_path)

print("Total AmazonHelp rows:", len(df))

print("\nInbound vs Outbound:")
print(df["inbound"].value_counts())

print("\nSample customer messages:\n")

customer_messages = df[df["inbound"] == True]

sample = customer_messages.sample(
    n=50,
    random_state=42
)

for i, text in enumerate(sample["text"], 1):
    print(f"{i}. {text}")