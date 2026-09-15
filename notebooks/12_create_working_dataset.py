import pandas as pd

input_file = "data/amazon_reply_pairs.csv"
output_file = "data/amazon_working.csv"

df = pd.read_csv(input_file)

print("Total pairs available:", len(df))

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_message"]
)

print("After removing duplicates:", len(df))

# Take a manageable sample
sample_size = min(10000, len(df))

working_df = df.sample(
    n=sample_size,
    random_state=42
)

working_df = working_df.reset_index(drop=True)

working_df.to_csv(
    output_file,
    index=False
)

print("Working dataset size:", len(working_df))
print("Saved to:", output_file)

print("\nColumns:")
print(list(working_df.columns))

print("\nFirst 5 rows:")
print(working_df.head())