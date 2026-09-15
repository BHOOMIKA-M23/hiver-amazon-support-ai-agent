import pandas as pd

file_path = "data/twcs/twcs.csv"

amazon = []

for chunk in pd.read_csv(file_path, chunksize=100000):
    matches = chunk[
        chunk["text"].astype(str).str.contains(
            "@AmazonHelp",
            case=False,
            na=False
        )
    ]

    amazon.extend(
        matches[
            [
                "tweet_id",
                "inbound",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id"
            ]
        ].values.tolist()
    )

    if len(amazon) >= 50:
        break

print("\nAmazonHelp conversation examples:\n")

for row in amazon[:30]:
    print(row)