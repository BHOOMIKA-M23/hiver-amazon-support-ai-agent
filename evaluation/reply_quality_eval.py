import pandas as pd
from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = PROJECT_ROOT / "data" / "amazon_golden_set.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_sample.csv"


# Load golden set
df = pd.read_csv(GOLDEN_PATH)

# We will use the 30 held-out examples already created
test_path = PROJECT_ROOT / "data" / "intent_test.csv"
test_df = pd.read_csv(test_path)

# Select a smaller evaluation sample
# Using all 30 held-out examples keeps the evaluation reproducible.
sample = test_df.copy()

# Keep useful columns
# Use customer_message when available.
# Some held-out rows store the original message in the text column.
sample["customer_message"] = sample["customer_message"].fillna(sample["text"])

# Keep useful columns
sample = sample[
    ["tweet_id", "customer_message", "intent"]
].copy()

# Add empty columns for later evaluation
sample["generated_reply"] = ""
sample["evidence_similarity"] = ""
sample["human_reply_quality"] = ""
sample["human_evidence_supported"] = ""
sample["judge_reply_quality"] = ""
sample["judge_evidence_supported"] = ""
sample["judge_reason"] = ""

# Save evaluation template
sample.to_csv(OUTPUT_PATH, index=False)

print("=" * 60)
print("REPLY QUALITY EVALUATION TEMPLATE")
print("=" * 60)
print(f"Evaluation examples: {len(sample)}")
print(f"Saved to: {OUTPUT_PATH}")
print()
print("Columns:")
for column in sample.columns:
    print(f"- {column}")