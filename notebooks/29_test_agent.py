import sys
sys.path.append(".")

from src.agent import AmazonSupportAgent


agent = AmazonSupportAgent()


message = "My package says it was delivered but I never received it."

result = agent.analyze(message)


print("\nCUSTOMER MESSAGE")
print("----------------")
print(result["message"])

print("\nPREDICTED INTENT")
print("----------------")
print(result["intent"])

print("\nCONFIDENCE")
print("----------------")
print(round(result["confidence"], 3))

print("\nHISTORICAL EVIDENCE")
print("-------------------")

for _, row in result["evidence"].iterrows():

    print("\nSimilarity:", round(row["similarity"], 3))
    print("Customer:", row["customer_message"])
    print("Amazon:", row["amazon_reply"])