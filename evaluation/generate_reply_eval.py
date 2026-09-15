import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import AmazonSupportAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_sample.csv"


# Load evaluation examples
df = pd.read_csv(INPUT_PATH)

# Create the AI support agent
agent = AmazonSupportAgent()

print("=" * 60)
print("GENERATING REPLIES FOR EVALUATION")
print("=" * 60)

for i, row in df.iterrows():

    message = str(row["customer_message"])

    print(f"\nExample {i + 1}/ {len(df)}")
    print(f"Customer: {message}")

    result = agent.analyze(message)

    df.at[i, "generated_reply"] = result["reply"]
    df.at[i, "evidence_similarity"] = result["evidence_similarity"]

    print(f"Intent: {result['intent']}")
    print(f"Evidence similarity: {result['evidence_similarity']:.3f}")
    print(f"Reply: {result['reply']}")

    # Save after every example so progress is not lost
    df.to_csv(OUTPUT_PATH, index=False)

print()
print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Saved to: {OUTPUT_PATH}")