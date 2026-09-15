import pandas as pd

input_file = "data/amazon_working.csv"
output_file = "data/amazon_clean.csv"

df = pd.read_csv(input_file)

print("Original rows:", len(df))

# Remove extremely short messages
df["message_length"] = df["customer_message"].astype(str).str.len()

df = df[df["message_length"] >= 20].copy()

# Remove exact duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
)

# Remove helper column
df = df.drop(columns=["message_length"])

df = df.reset_index(drop=True)

print("Cleaned rows:", len(df))
print("Rows removed:", 10000 - len(df))

df.to_csv(
    output_file,
    index=False
)

print("Saved to:", output_file)

print("\nFinal missing values:")
print(df.isna().sum())