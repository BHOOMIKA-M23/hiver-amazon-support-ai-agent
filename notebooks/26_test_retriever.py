import sys
sys.path.append(".")

from src.retriever import AmazonRetriever


# Create the retriever
retriever = AmazonRetriever()


# New customer message
message = (
    "My package says it was delivered "
    "but I never received it."
)


# Retrieve similar historical cases
results = retriever.retrieve(
    message,
    top_k=3
)


print("\nNEW CUSTOMER MESSAGE:")
print(message)

print("\nRETRIEVED HISTORICAL CASES:")

for _, row in results.iterrows():

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAMAZON RESPONSE:")
    print(row["amazon_reply"])

    print("\nSIMILARITY:")
    print(round(row["similarity"], 3))

    print("\n" + "=" * 70)