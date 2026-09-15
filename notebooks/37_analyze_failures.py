import pandas as pd

df = pd.read_csv("data/intent_failures.csv")

print("\nFAILURE ANALYSIS")
print("=" * 70)

print("\nTotal wrong predictions:", len(df))

print("\nWrong prediction patterns:")
patterns = (
    df.groupby(["actual_intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(patterns.to_string(index=False))

print("\n\nINDIVIDUAL FAILURES")
print("=" * 70)

for i, row in df.iterrows():
    print(f"\n{i+1}.")
    print("Actual:   ", row["actual_intent"])
    print("Predicted:", row["predicted_intent"])
    print("Message:  ", row["message"])