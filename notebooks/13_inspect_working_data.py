import pandas as pd

file_path = "data/amazon_working.csv"

df = pd.read_csv(file_path)

print("Total rows:", len(df))

print("\nMissing values:")
print(df.isna().sum())

print("\nSample customer messages:\n")

for i, message in enumerate(df["customer_message"].head(20), start=1):
    print(f"{i}. {message}")
    print("-" * 70)