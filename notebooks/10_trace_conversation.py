import pandas as pd

file_path = "data/twcs/twcs.csv"

target_ids = {"615", "616", "618"}

for chunk in pd.read_csv(
    file_path,
    chunksize=100000,
    low_memory=False
):

    chunk_ids = chunk["tweet_id"].astype(str)

    matches = chunk[
        chunk_ids.isin(target_ids)
    ]

    if len(matches) > 0:
        for _, row in matches.iterrows():

            speaker = (
                "CUSTOMER"
                if row["inbound"]
                else "AMAZON"
            )

            print("\n" + "=" * 60)
            print("TWEET ID:", row["tweet_id"])
            print("SPEAKER:", speaker)
            print("IN RESPONSE TO:", row["in_response_to_tweet_id"])
            print("RESPONSE TWEET:", row["response_tweet_id"])
            print("TEXT:")
            print(row["text"])

    if len(matches) == 3:
        break