import pandas as pd

file_path = "data/amazon_support.csv"

df = pd.read_csv(file_path, low_memory=False)

# Count how many tweets belong to each conversation
conversation_counts = (
    df["in_response_to_tweet_id"]
    .value_counts()
)

print("\nTotal AmazonHelp tweets:", len(df))

print(
    "Tweets that are replies to another tweet:",
    df["in_response_to_tweet_id"].notna().sum()
)

print(
    "Unique parent tweets:",
    df["in_response_to_tweet_id"].nunique()
)

print("\nMost common conversation sizes:")

print(conversation_counts.head(20))