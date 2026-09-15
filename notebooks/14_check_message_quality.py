import pandas as pd

file_path = "data/amazon_working.csv"

df = pd.read_csv(file_path)

messages = df["customer_message"].astype(str)

# Message length
df["message_length"] = messages.str.len()

print("Total messages:", len(df))

print("\nMessage length statistics:")
print(df["message_length"].describe())

print("\nMessages shorter than 20 characters:",
      (df["message_length"] < 20).sum())

print("Messages shorter than 50 characters:",
      (df["message_length"] < 50).sum())

print("Messages longer than 500 characters:",
      (df["message_length"] > 500).sum())

# Count messages containing common URL markers
url_count = messages.str.contains(
    "http|https|t.co",
    case=False,
    na=False
).sum()

print("\nMessages containing URLs:", url_count)

# Count @ mentions
mention_count = messages.str.contains(
    "@",
    na=False
).sum()

print("Messages containing @ mentions:", mention_count)

print("\nShortest 10 messages:")

shortest = df.sort_values("message_length").head(10)

for i, message in enumerate(shortest["customer_message"], start=1):
    print(f"{i}. {message}")