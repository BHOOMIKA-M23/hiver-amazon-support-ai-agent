import pandas as pd

input_file = "data/twcs/twcs.csv"
output_file = "data/amazon_support.csv"

amazon_chunks = []

for chunk in pd.read_csv(input_file, chunksize=100000):

    matches = chunk[
        chunk["text"].astype(str).str.contains(
            "@AmazonHelp",
            case=False,
            na=False
        )
    ]

    if len(matches) > 0:
        amazon_chunks.append(matches)

amazon = pd.concat(amazon_chunks, ignore_index=True)

print("AmazonHelp rows:", len(amazon))

amazon.to_csv(output_file, index=False)

print("Saved to:", output_file)