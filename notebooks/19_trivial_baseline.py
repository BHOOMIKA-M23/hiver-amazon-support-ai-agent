import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the golden dataset
df = pd.read_csv("data/amazon_golden_set.csv")

# Create one common message column
df["message"] = df["customer_message"].fillna(df["text"])

# Remove missing messages or labels
df = df.dropna(subset=["message", "intent"])

# Split the data
_, X_test, _, y_test = train_test_split(
    df["message"],
    df["intent"],
    test_size=0.2,
    random_state=42,
    stratify=df["intent"]
)

# Find the most common intent
most_common_intent = df["intent"].value_counts().idxmax()

# Predict the same intent for every test message
predictions = [most_common_intent] * len(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nTrivial Baseline Results")
print("------------------------")
print("Most common intent:", most_common_intent)
print("Testing examples:", len(y_test))
print("Accuracy:", round(accuracy, 3))