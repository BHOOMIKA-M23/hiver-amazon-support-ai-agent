import pandas as pd


golden = pd.read_csv(
    "data/amazon_golden_set.csv",
    low_memory=False
)

knowledge_base = pd.read_csv(
    "data/amazon_knowledge_base.csv",
    low_memory=False
)


# The first batch stores the message in customer_message.
# The second batch stores it in text.
golden["message"] = golden["customer_message"].fillna(
    golden["text"]
)


golden_messages = set(
    golden["message"]
    .fillna("")
    .astype(str)
    .str.strip()
)


kb_messages = set(
    knowledge_base["customer_message"]
    .fillna("")
    .astype(str)
    .str.strip()
)


overlap = golden_messages.intersection(kb_messages)


print("\nRETRIEVAL LEAKAGE CHECK")
print("=======================")

print("Golden examples:", len(golden_messages))
print("Knowledge-base messages:", len(kb_messages))
print("Exact message overlap:", len(overlap))


if len(overlap) > 0:

    print("\nWARNING: Overlapping messages found.")

    print("\nExamples:")

    for message in list(overlap)[:10]:
        print("-", message)

else:

    print("\nNo exact message overlap found.")