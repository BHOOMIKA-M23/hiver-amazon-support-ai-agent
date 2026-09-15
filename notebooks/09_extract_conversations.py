import pandas as pd

input_file = "data/twcs/twcs.csv"
output_file = "data/amazon_conversations.csv"

# Step 1: Find all tweets mentioning AmazonHelp
amazon_ids = set()

for chunk in pd.read_csv(
    input_file,
    chunksize=100000,
    low_memory=False
):

    matches = chunk[
        chunk["text"].astype(str).str.contains(
            "@AmazonHelp",
            case=False,
            na=False
        )
    ]

    amazon_ids.update(
        matches["tweet_id"].astype(str)
    )

print("AmazonHelp tweet IDs found:", len(amazon_ids))


# Step 2: Find tweets that are connected to AmazonHelp tweets
conversation_chunks = []

for chunk in pd.read_csv(
    input_file,
    chunksize=100000,
    low_memory=False
):

    tweet_ids = chunk["tweet_id"].astype(str)

    parent_ids = chunk[
        "in_response_to_tweet_id"
    ].fillna("").astype(str)

    # Keep:
    # 1. AmazonHelp tweets themselves
    # 2. Tweets replying to an AmazonHelp tweet

    matches = (
        tweet_ids.isin(amazon_ids)
        |
        parent_ids.isin(amazon_ids)
    )

    if matches.any():
        conversation_chunks.append(
            chunk[matches]
        )


conversations = pd.concat(
    conversation_chunks,
    ignore_index=True
)

conversations.to_csv(
    output_file,
    index=False
)

print(
    "Conversation tweets extracted:",
    len(conversations)
)

print(
    "Saved to:",
    output_file
)