import pandas as pd


input_file = "data/amazon_conversations.csv"
output_file = "data/amazon_knowledge_base.csv"


df = pd.read_csv(input_file, low_memory=False)


# Keep reasonably informative customer messages
df = df[
    df["customer_message"].fillna("").str.len() >= 50
]


# Keep reasonably informative Amazon replies
df = df[
    df["amazon_reply"].fillna("").str.len() >= 40
]


# Keep only replies that appear to come from Amazon support
df = df[
    df["amazon_reply"].str.contains(
        "@",
        na=False
    )
]


# Remove duplicate customer/reply pairs
df = df.drop_duplicates(
    subset=[
        "customer_message",
        "amazon_reply"
    ]
)


# Remove empty rows
df = df.dropna(
    subset=[
        "customer_message",
        "amazon_reply"
    ]
)


df.to_csv(
    output_file,
    index=False
)


print("Original conversation pairs:", len(pd.read_csv(input_file, low_memory=False)))

print("Clean knowledge-base pairs:", len(df))

print("Saved to:", output_file)