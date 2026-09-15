import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "data/amazon_golden_set.csv"

train_file = "data/intent_train.csv"
test_file = "data/intent_test.csv"

df = pd.read_csv(input_file, low_memory=False)

train, test = train_test_split(
    df,
    test_size=30,
    stratify=df["intent"],
    random_state=42
)

train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

print("\nGOLDEN SET SPLIT")
print("================")

print("Total examples:", len(df))
print("Training examples:", len(train))
print("Test examples:", len(test))

print("\nTraining distribution:")
print(train["intent"].value_counts())

print("\nTest distribution:")
print(test["intent"].value_counts())

print("\nSaved:")
print(train_file)
print(test_file)