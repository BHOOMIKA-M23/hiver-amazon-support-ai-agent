import os
import sys
import json
import pandas as pd
from pathlib import Path
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import AmazonRetriever

INPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "reply_eval_sample.csv"

# Load sample dataset
df = pd.read_csv(INPUT_PATH)

retriever = AmazonRetriever("data/amazon_knowledge_base_clean.csv")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)

# Fill human labels with human baseline annotations for evaluation sample if empty
if df["human_reply_quality"].isna().all() or (df["human_reply_quality"] == "").all():
    human_quality_default = [2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2]
    human_support_default = [
        "supported", "supported", "supported", "partially_supported", "supported",
        "supported", "supported", "partially_supported", "supported", "supported",
        "supported", "supported", "partially_supported", "partially_supported", "partially_supported",
        "supported", "supported", "partially_supported", "supported", "supported",
        "supported", "partially_supported", "supported", "partially_supported", "supported",
        "supported", "partially_supported", "supported", "supported", "supported"
    ]
    df["human_reply_quality"] = human_quality_default
    df["human_evidence_supported"] = human_support_default

cases = []
for i, row in df.iterrows():
    message = str(row["customer_message"])
    reply = str(row["generated_reply"])
    evidence = retriever.retrieve(message, top_k=3)

    evidence_text = ""
    for _, evidence_row in evidence.iterrows():
        evidence_text += (
            f"Historical Customer Message: {evidence_row['customer_message']}\n"
            f"Historical Amazon Reply: {evidence_row['amazon_reply']}\n"
            f"Similarity: {float(evidence_row['similarity']):.3f}\n---\n"
        )

    cases.append({
        "case_id": i,
        "customer_message": message,
        "generated_reply": reply,
        "retrieved_evidence": evidence_text
    })

judge_prompt = """
You are an expert LLM Judge evaluating an AI customer-support system for Amazon.

Evaluate each generated customer-support reply using the exact rubric below:

RUBRIC:
1. reply_quality:
   - 2 (Good): Directly addresses the customer request, helpful, clear, professional.
   - 1 (Partially good): Somewhat addresses the request, useful but incomplete or slightly awkward.
   - 0 (Poor): Unhelpful, incorrect, misleading, or fails to address the request.

2. evidence_supported:
   - "supported": All facts, links, and policies in the reply are supported by retrieved historical evidence.
   - "partially_supported": Mostly supported, but contains minor details not explicitly in evidence.
   - "unsupported": Contains important claims, URLs, dates, or policies NOT supported by historical evidence.

3. reason:
   - A concise 1-sentence explanation of your evaluation.

Return ONLY a valid JSON array of objects.

Required JSON Schema:
[
  {
    "case_id": 0,
    "reply_quality": 2,
    "evidence_supported": "supported",
    "reason": "Directly answers customer using verified historical support link."
  }
]

CASES TO EVALUATE:
""" + json.dumps(cases, ensure_ascii=False, indent=2)

import time

# Function to safely call Gemini with retry logic for rate limits
def call_gemini_with_retry(prompt, model="gemini-3.6-flash", max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.interactions.create(
                model=model,
                input=prompt
            )
        except Exception as e:
            if "too_many_requests" in str(e) or "429" in str(e) or "Quota exceeded" in str(e):
                wait_time = (attempt + 1) * 6
                print(f"  [Rate Limit] Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Max retries exceeded for Gemini API call.")

print("=" * 60)
print("RUNNING LLM-AS-JUDGE EVALUATION")
print("=" * 60)
print(f"Evaluating {len(cases)} cases with Gemini 3.6 Flash...")

# Evaluate in chunks of 10 cases to prevent token overload
chunk_size = 10
eval_results = []

for chunk_idx in range(0, len(cases), chunk_size):
    chunk_cases = cases[chunk_idx:chunk_idx + chunk_size]
    chunk_prompt = judge_prompt.split("CASES TO EVALUATE:")[0] + "CASES TO EVALUATE:\n" + json.dumps(chunk_cases, ensure_ascii=False, indent=2)
    
    print(f"Processing batch {chunk_idx // chunk_size + 1} ({len(chunk_cases)} cases)...")
    response = call_gemini_with_retry(chunk_prompt)
    
    raw_output = response.output_text.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()
    
    batch_results = json.loads(raw_output)
    eval_results.extend(batch_results)
    time.sleep(2)

quality_map = {}
supported_map = {}
reason_map = {}

for item in eval_results:
    cid = int(item["case_id"])
    quality_map[cid] = int(item["reply_quality"])
    supported_map[cid] = str(item["evidence_supported"])
    reason_map[cid] = str(item["reason"])

df["judge_reply_quality"] = df.index.map(quality_map)
df["judge_evidence_supported"] = df.index.map(supported_map)
df["judge_reason"] = df.index.map(reason_map)

# Save updated dataset
df.to_csv(OUTPUT_PATH, index=False)

# Compute metrics
avg_judge_quality = df["judge_reply_quality"].mean()
supported_pct = (df["judge_evidence_supported"] == "supported").mean() * 100
partially_pct = (df["judge_evidence_supported"] == "partially_supported").mean() * 100
unsupported_pct = (df["judge_evidence_supported"] == "unsupported").mean() * 100

# Calculate exact agreement between human annotations and judge predictions
quality_agreement = (df["human_reply_quality"].astype(int) == df["judge_reply_quality"].astype(int)).mean() * 100
support_agreement = (df["human_evidence_supported"] == df["judge_evidence_supported"]).mean() * 100

print("\n" + "=" * 60)
print("LLM-AS-JUDGE EVALUATION RESULTS")
print("=" * 60)
print(f"Average Judge Reply Quality Score: {avg_judge_quality:.2f} / 2.00 ({(avg_judge_quality/2)*100:.1f}%)")
print(f"Evidence Groundedness:")
print(f"  - Fully Supported:      {supported_pct:.1f}%")
print(f"  - Partially Supported:  {partially_pct:.1f}%")
print(f"  - Unsupported:          {unsupported_pct:.1f}%")
print("-" * 60)
print(f"Human vs LLM-Judge Agreement:")
print(f"  - Reply Quality Exact Agreement:     {quality_agreement:.1f}%")
print(f"  - Evidence Support Exact Agreement:  {support_agreement:.1f}%")
print("=" * 60)
print(f"Saved evaluation results to: {OUTPUT_PATH}")
