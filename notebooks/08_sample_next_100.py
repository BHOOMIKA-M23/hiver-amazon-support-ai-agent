import pandas as pd

input_file = "data/amazon_support.csv"
output_file = "data/amazon_label_batch_2.csv"

# Load Amazon customer messages
df = pd.read_csv(input_file, low_memory=False)

# Keep only customer messages
customers = df[df["inbound"] == True].copy()

# Remove very short messages
customers = customers[customers["text"].astype(str).str.len() >= 15]

# Take a random sample of 100
sample = customers.sample(n=100, random_state=42)

# Keep only the columns we need
sample = sample[
    ["tweet_id", "text", "response_tweet_id"]
].copy()

# Add an empty intent column
sample["intent"] = ""

# Save
sample.to_csv(output_file, index=False)

print("Created:", output_file)
print("Number of examples:", len(sample))
print(sample.head(10))