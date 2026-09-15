import pandas as pd

input_file = "data/amazon_conversations.csv"

df = pd.read_csv(input_file)

print("Total conversation pairs:", len(df))

# Create message lengths
df["customer_length"] = df["customer_message"].astype(str).str.len()
df["reply_length"] = df["amazon_reply"].astype(str).str.len()

print("\nCustomer message length:")
print(df["customer_length"].describe())

print("\nAmazon reply length:")
print(df["reply_length"].describe())

# Show very short customer messages
short_messages = df[
    df["customer_length"] <= 30
].head(10)

print("\nSHORT CUSTOMER MESSAGES:")
for _, row in short_messages.iterrows():
    print("\nCUSTOMER:")
    print(row["customer_message"])
    print("AMAZON:")
    print(row["amazon_reply"])
    print("-" * 60)

# Show longer, potentially useful conversations
long_messages = df[
    df["customer_length"] >= 100
].head(10)

print("\nLONGER CUSTOMER MESSAGES:")
for _, row in long_messages.iterrows():
    print("\nCUSTOMER:")
    print(row["customer_message"])
    print("AMAZON:")
    print(row["amazon_reply"])
    print("-" * 60)