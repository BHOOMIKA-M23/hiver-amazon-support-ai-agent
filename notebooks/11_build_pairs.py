import pandas as pd

file_path = "data/twcs/twcs.csv"
output_file = "data/amazon_reply_pairs.csv"

# Read the dataset
df = pd.read_csv(
    file_path,
    usecols=[
        "tweet_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    low_memory=False
)

df["tweet_id"] = df["tweet_id"].astype(str)

# Find AmazonHelp tweets
amazon_mask = df["text"].astype(str).str.contains(
    "@AmazonHelp",
    case=False,
    na=False
)

amazon_ids = set(
    df.loc[amazon_mask, "tweet_id"]
)

print("AmazonHelp tweets:", len(amazon_ids))

# Lookup table
tweet_lookup = df.set_index("tweet_id")

pairs = []

# Only customer tweets that mention AmazonHelp
customer_rows = df[
    (df["inbound"] == True)
    & amazon_mask
]

for _, row in customer_rows.iterrows():

    response_ids = str(row["response_tweet_id"])

    if response_ids == "nan":
        continue

    for response_id in response_ids.split(","):

        response_id = response_id.strip()

        if response_id not in tweet_lookup.index:
            continue

        reply = tweet_lookup.loc[response_id]

        # Make sure the response is from the support side
        if reply["inbound"] == False:

            pairs.append({
                "customer_tweet_id": row["tweet_id"],
                "customer_message": row["text"],
                "amazon_tweet_id": response_id,
                "amazon_reply": reply["text"]
            })

pairs_df = pd.DataFrame(pairs)

pairs_df.to_csv(
    output_file,
    index=False
)

print(
    "\nAmazonHelp customer → reply pairs:",
    len(pairs_df)
)

print("\nSample AmazonHelp pairs:\n")

for _, row in pairs_df.head(10).iterrows():

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAMAZON:")
    print(row["amazon_reply"])

    print("-" * 60)