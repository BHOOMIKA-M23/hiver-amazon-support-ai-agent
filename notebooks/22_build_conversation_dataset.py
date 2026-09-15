import pandas as pd

input_file = "data/twcs/twcs.csv"
output_file = "data/amazon_conversations.csv"

# --------------------------------------------------
# 1. Load the original Twitter customer-support data
# --------------------------------------------------

df = pd.read_csv(
    input_file,
    usecols=[
        "tweet_id",
        "inbound",
        "text",
        "created_at",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    low_memory=False
)

# Convert IDs to strings
df["tweet_id"] = df["tweet_id"].astype(str)

# Convert timestamps
df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce"
)

# --------------------------------------------------
# 2. Build lookup tables
# --------------------------------------------------

tweet_lookup = df.set_index("tweet_id")["text"].to_dict()

inbound_lookup = df.set_index("tweet_id")["inbound"].to_dict()

created_at_lookup = df.set_index("tweet_id")["created_at"].to_dict()

# --------------------------------------------------
# 3. Find Amazon customer messages
# --------------------------------------------------

customers = df[
    (df["inbound"] == True) &
    (df["text"].astype(str).str.contains(
        "@AmazonHelp",
        case=False,
        na=False
    ))
].copy()

customers = customers[
    customers["response_tweet_id"].notna()
].copy()

print("Customer messages with responses:", len(customers))

# --------------------------------------------------
# 4. Create customer → Amazon reply pairs
# --------------------------------------------------

conversation_rows = []

for _, customer in customers.iterrows():

    customer_id = str(customer["tweet_id"])
    customer_text = str(customer["text"])

    response_ids = str(
        customer["response_tweet_id"]
    ).split(",")

    replies = []

    for response_id in response_ids:

        response_id = response_id.strip()

        if response_id in tweet_lookup:

            # Only keep outbound/support responses
            if inbound_lookup.get(response_id) == False:

                replies.append({
                    "tweet_id": response_id,
                    "text": tweet_lookup[response_id],
                    "created_at": created_at_lookup.get(response_id)
                })

    # Sort replies chronologically
    replies = sorted(
        replies,
        key=lambda x: (
            pd.isna(x["created_at"]),
            x["created_at"]
        )
    )

    # Extract reply text
    reply_texts = [
        reply["text"]
        for reply in replies
    ]

    # Combine multiple response tweets into one reply
    if reply_texts:

        combined_reply = " ".join(reply_texts)

        conversation_rows.append({
            "customer_tweet_id": customer_id,
            "reply_tweet_id": ",".join(response_ids),
            "customer_message": customer_text,
            "amazon_reply": combined_reply
        })

# --------------------------------------------------
# 5. Convert to DataFrame
# --------------------------------------------------

conversations = pd.DataFrame(conversation_rows)

# Remove exact duplicate pairs
conversations = conversations.drop_duplicates(
    subset=[
        "customer_message",
        "amazon_reply"
    ]
)

# Remove missing messages
conversations = conversations.dropna(
    subset=[
        "customer_message",
        "amazon_reply"
    ]
)

# --------------------------------------------------
# 6. Save
# --------------------------------------------------

conversations.to_csv(
    output_file,
    index=False
)

print("Conversation pairs created:", len(conversations))
print("Saved to:", output_file)

# --------------------------------------------------
# 7. Show examples
# --------------------------------------------------

print("\nSample conversation pairs:\n")

for _, row in conversations.head(5).iterrows():

    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nAMAZON REPLY:")
    print(row["amazon_reply"])

    print("\n" + "-" * 70)