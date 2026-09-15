import pandas as pd

file_path = "data/amazon_support.csv"

df = pd.read_csv(file_path, low_memory=False)

# Pick a conversation with several messages
conversation = df[
    df["in_response_to_tweet_id"] == 615
]

print("\nConversation:\n")

for _, row in conversation.iterrows():
    speaker = "CUSTOMER" if row["inbound"] else "AMAZON"

    print(f"\n{speaker}")
    print("-" * 40)
    print(row["text"])