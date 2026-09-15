import pandas as pd


knowledge_base = pd.read_csv(
    "data/amazon_knowledge_base.csv",
    low_memory=False
)

golden = pd.read_csv(
    "data/amazon_golden_set.csv",
    low_memory=False
)


# Handle both golden-set message column names
golden["message"] = golden["customer_message"].fillna(
    golden["text"]
)


golden_messages = set(
    golden["message"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Remove golden examples from the knowledge base
clean_kb = knowledge_base[
    ~knowledge_base["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
    .isin(golden_messages)
].copy()


output_file = "data/amazon_knowledge_base_clean.csv"

clean_kb.to_csv(
    output_file,
    index=False
)


print("\nLEAKAGE-FREE KNOWLEDGE BASE")
print("===========================")

print("Original knowledge-base pairs:", len(knowledge_base))
print("Golden examples:", len(golden_messages))
print("Clean evaluation KB pairs:", len(clean_kb))
print("Removed overlapping pairs:", len(knowledge_base) - len(clean_kb))

print("\nSaved to:", output_file)
