import pandas as pd

input_file = "data/twcs/twcs.csv"

# Load only the columns we need
df = pd.read_csv(
    input_file,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    low_memory=False
)

# Find customer messages that directly mention AmazonHelp
amazon_customers = df[
    df["text"].astype(str).str.contains(
        "@AmazonHelp",
        case=False,
        na=False
    )
    & (df["inbound"] == True)
].copy()

print("Amazon customer messages:", len(amazon_customers))

# Keep messages that have a response tweet
with_response = amazon_customers[
    amazon_customers["response_tweet_id"].notna()
].copy()

print("Customer messages with response IDs:", len(with_response))

# Look at the first few examples
print("\nSample customer messages:")
print(
    with_response[
        [
            "tweet_id",
            "text",
            "response_tweet_id"
        ]
    ].head(10).to_string(index=False)
)

# Get the IDs of the expected replies
reply_ids = set()

for value in with_response["response_tweet_id"]:
    for tweet_id in str(value).split(","):
        tweet_id = tweet_id.strip()
        if tweet_id and tweet_id != "nan":
            reply_ids.add(tweet_id)

print("\nUnique response tweet IDs:", len(reply_ids))

# Find those replies in the original dataset
replies = df[
    df["tweet_id"].astype(str).isin(reply_ids)
].copy()

print("Replies found:", len(replies))

print("\nSample customer → Amazon reply pairs:")

for _, customer in with_response.head(5).iterrows():

    ids = str(customer["response_tweet_id"]).split(",")

    print("\nCUSTOMER:")
    print(customer["text"])

    for reply_id in ids:

        reply = replies[
            replies["tweet_id"].astype(str) == reply_id.strip()
        ]

        if len(reply) > 0:
            print("\nAMAZON REPLY:")
            print(reply.iloc[0]["text"])