import sys
sys.path.append(".")

from src.agent import AmazonSupportAgent


agent = AmazonSupportAgent()

message = "I want to return this item and get a refund."

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

print("\nEVIDENCE SIMILARITY")
print("-------------------")
print(round(result["evidence_similarity"], 3))

print("\nDECISION")
print("--------")
print(result["decision"])

print("\nDECISION REASON")
print("----------------")
print(result["decision_reason"])

print("\nGEMINI GENERATED REPLY")
print("----------------------")
print(result["reply"])

print("\nHISTORICAL EVIDENCE")
print("-------------------")

for _, row in result["evidence"].iterrows():

    print("\nSimilarity:", round(row["similarity"], 3))
    print("Customer:", row["customer_message"])
    print("Amazon:", row["amazon_reply"])