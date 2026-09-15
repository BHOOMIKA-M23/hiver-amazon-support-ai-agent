import os
import sys
import json
import pandas as pd
from pathlib import Path
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import AmazonRetriever


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_results.csv"


# Load evaluation examples
df = pd.read_csv(INPUT_PATH)

# Load the clean historical knowledge base
retriever = AmazonRetriever(
    "data/amazon_knowledge_base_clean.csv"
)


# Build one prompt containing all 30 examples
cases = []

for i, row in df.iterrows():

    message = str(row["customer_message"])

    evidence = retriever.retrieve(
        message,
        top_k=3
    )

    evidence_text = ""

    for _, evidence_row in evidence.iterrows():
        evidence_text += (
            f"\nHistorical customer message:\n"
            f"{evidence_row['customer_message']}\n"
            f"Historical Amazon reply:\n"
            f"{evidence_row['amazon_reply']}\n"
            f"Similarity: {float(evidence_row['similarity']):.3f}\n"
        )

    cases.append(
        {
            "case_id": i,
            "customer_message": message,
            "evidence": evidence_text
        }
    )


# Create Gemini client
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=api_key)


# Build batch prompt
prompt = """
You are evaluating an AI customer-support system for Amazon.

For each customer case below, draft ONE concise customer-support reply.

IMPORTANT RULES:
1. Use ONLY information supported by the historical evidence provided for that case.
2. Do not invent policies, refunds, delivery dates, account information,
   troubleshooting steps, or other facts.
3. Do not invent URLs.
4. You may use a URL only if it appears in the historical evidence.
5. Do not claim an action has already been completed unless the evidence supports it.
6. If the evidence is insufficient to safely answer, politely recommend
   contacting Amazon Customer Support rather than guessing.
7. Keep each reply concise, professional, and empathetic.
8. Return ONLY valid JSON.
9. Return exactly one object for every case.
10. Preserve the case_id exactly.

Required JSON format:

[
  {
    "case_id": 0,
    "reply": "draft reply here"
  }
]

CUSTOMER CASES:
"""

prompt += json.dumps(
    cases,
    ensure_ascii=False,
    indent=2
)


print("=" * 60)
print("BATCH REPLY GENERATION")
print("=" * 60)
print(f"Cases: {len(cases)}")
print("Sending ONE Gemini request...")
print()


# One Gemini request
response = client.interactions.create(
    model="gemini-3.6-flash",
    input=prompt
)

raw_output = response.output_text.strip()


# Remove markdown code fences if Gemini adds them
if raw_output.startswith("```"):
    raw_output = raw_output.replace("```json", "")
    raw_output = raw_output.replace("```", "")
    raw_output = raw_output.strip()


# Parse JSON
results = json.loads(raw_output)


# Add generated replies and evidence similarity to dataframe
reply_map = {
    int(item["case_id"]): item["reply"]
    for item in results
}

df["generated_reply"] = df.index.map(reply_map)

# Compute top evidence similarity for each row
for i, row in df.iterrows():
    message = str(row["customer_message"])
    evidence = retriever.retrieve(message, top_k=1)
    if not evidence.empty:
        df.at[i, "evidence_similarity"] = float(evidence.iloc[0]["similarity"])

# Save results to both sample and results CSVs
df.to_csv(INPUT_PATH, index=False)
df.to_csv(OUTPUT_PATH, index=False)


print("=" * 60)
print("DONE")
print("=" * 60)
print(f"Generated replies: {df['generated_reply'].notna().sum()}")
print(f"Saved to: {OUTPUT_PATH} and {INPUT_PATH}")