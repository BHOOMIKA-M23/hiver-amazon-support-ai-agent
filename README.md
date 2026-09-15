# Hiver SDE Intern - AI Customer Support Agent

## Overview

This project builds an AI-powered customer support agent for AmazonHelp using historical customer-support conversations from Twitter.

The system:

- Classifies customer messages into support intents.
- Retrieves similar historical Amazon support conversations.
- Generates a grounded customer-support reply using Gemini.
- Decides whether to auto-handle the request or escalate it to a human.
- Evaluates the system using a manually labelled golden evaluation set.

## Dataset

The project uses the Customer Support on Twitter dataset from Kaggle.

For this project, the AmazonHelp brand was selected because it contains a large number of customer-support interactions and repeatable support issues.

After filtering the dataset:

- AmazonHelp tweets: 135,182
- Customer messages: 135,160
- Conversation pairs: 92,968
- Clean historical knowledge-base pairs: 77,003
- Leakage-free evaluation knowledge base: 76,902
- Manually labelled golden examples: 150

## Intent Taxonomy

The system classifies customer messages into seven intents:

| Intent              | Description                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| `delivery_shipping` | Late, missing, delayed, tracking, carrier, or delivery problems                 |
| `order_issue`       | Problems with placing, cancelling, changing, or processing an order             |
| `refund_return`     | Returns, refunds, replacements, or return labels                                |
| `account_payment`   | Account access, login, charges, payment methods, gift cards, or Amazon Pay      |
| `product_help`      | Product/device usage, configuration, troubleshooting, or product problems       |
| `prime_issue`       | Prime membership, Prime benefits, Prime Video, or Prime-specific services       |
| `other`             | Messages with no clear support issue, acknowledgements, or unsupported requests |

## System Architecture

The AI support agent follows four main stages:

1. **Intent Classification**

   Gemini classifies the incoming customer message into one of the seven support intents.

2. **Historical Evidence Retrieval**

   A TF-IDF retriever finds similar customer-support conversations from the cleaned AmazonHelp knowledge base.

3. **Reply Generation**

   Gemini generates a concise customer-support reply grounded only in the retrieved historical examples.

4. **Auto-handle or Escalate**

   Rule-based logic decides whether the request can be auto-handled or should be escalated to a human.

### Pipeline

```text
Customer Message
        |
        v
Intent Classification
        |
        v
Historical Evidence Retrieval
        |
        v
Gemini Reply Generation
        |
        v
Auto-handle / Escalate
```

## Evaluation Results

The system was evaluated on a held-out test set of 30 examples from the 150-example golden set.

### Intent Classification

| Method                       |  Accuracy | Macro F1 | Weighted F1 |
| ---------------------------- | --------: | -------: | ----------: |
| Majority Class Baseline      |     43.3% |        - |           - |
| TF-IDF + Logistic Regression |     46.7% |     0.21 |        0.40 |
| Gemini Classifier            | **93.3%** | **0.88** |    **0.93** |

The Gemini classifier correctly classified 28 out of 30 held-out examples.

The Gemini result shown here is the latest reproducible result. An earlier run produced a higher score, but it was not used in the final headline result.

### Escalation Logic

Five representative decision tests were performed:

- Strong refund case → Auto-handle
- Weak historical evidence → Escalate
- Explicit human-support request → Escalate
- Account/payment issue → Escalate
- Strong delivery case → Auto-handle

Result: **5/5 decision tests passed.**

## Limitations


The current evaluation has several limitations:

- Only 30 held-out examples were used for the final intent evaluation.
- The golden set contains only 150 manually labelled examples.
- The intent distribution is imbalanced, with `other` as the largest class.
- Short, ambiguous, multilingual, and overlapping support messages may remain difficult.
- Retrieval quality directly affects the quality and safety of generated replies.
- Human review of 30 generated replies rated 70.0% as Good and 30.0% as Partially Good.
- The LLM judge rated all 30 replies as Good, so automated judge results should be interpreted cautiously.

## Project Structure

## Project Structure

```text
HIVER/
├── data/
│   ├── amazon_golden_set.csv
│   ├── amazon_label_batch_2.csv
│   ├── gemini_intent_results.csv
│   ├── intent_failures.csv
│   ├── intent_labeling_sample.csv
│   ├── intent_test.csv
│   ├── intent_train.csv
│   ├── reply_eval_results.csv
│   ├── reply_eval_sample.csv
│   └── sample.csv
│
├── src/
│   ├── agent.py
│   ├── intent_classifier.py
│   ├── retriever.py
│   ├── gemini_generator.py
│   └── decision.py
│
├── notebooks/
│   └── data processing and evaluation scripts
│
├── evaluation/
│   ├── reply_quality_eval.py
│   ├── generate_reply_eval.py
│   ├── batch_reply_eval.py
│   ├── llm_judge_eval.py
│   └── reply_quality_rubric.txt
│
├── reports/
│   ├── final_report.md
│   ├── decision_log.txt
│   ├── decision_test_results.txt
│   ├── baseline_results.txt
│   └── final_failure_analysis.txt
│
├── .gitignore
├── requirements.txt
└── README.md

## Setup

### 1. Create and activate the virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn joblib scipy google-genai
```

### 3. Configure the Gemini API key

Set the `GEMINI_API_KEY` environment variable before running the Gemini-based components.

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Do not commit the API key to GitHub.

### 4. Run the system

The main agent can be imported from:

```python
from src.agent import AmazonSupportAgent

agent = AmazonSupportAgent()

result = agent.analyze(
    "My package says delivered but I never received it."
)

print(result["intent"])
print(result["decision"])
print(result["reply"])
```

## Reproducibility (< 5 Minute Verification)

You can reproduce all headline results (93.3% intent classification accuracy, grounded reply generation, escalation logic, and LLM-as-judge reply quality evaluation) in under 5 minutes:

```bash
# Set your Gemini API Key
export GEMINI_API_KEY="YOUR_API_KEY"

# 1. Batch generate grounded replies for held-out evaluation set
python evaluation/batch_reply_eval.py

# 2. Run automated LLM-as-Judge evaluation & human-judge agreement calculation
python evaluation/llm_judge_eval.py
```

The scripts print empirical performance metrics directly to stdout:

- **Gemini Intent Accuracy**: 93.3% (28/30 held-out cases correct vs 46.7% baseline)
- **LLM-Judge Reply Quality Score**: 2.00 / 2.00 (30/30 rated Good by the LLM judge)
- **Human Reply Quality**: 1.70 / 2.00 (21/30 Good, 9/30 Partially Good)
- **Human vs. LLM-Judge Agreement**: 70.0% Exact Agreement
- The LLM judge was more lenient than human review, so its score should be interpreted cautiously.
The Gemini API is required for the Gemini-based classification, reply-generation, and LLM-judge components.

## Safety and Grounding

Generated replies are grounded in retrieved historical AmazonHelp support examples.

The reply generator is instructed to:

- Use only information supported by the historical evidence.
- Avoid inventing policies, refunds, delivery dates, account information, or troubleshooting steps.
- Avoid guessing customer-specific information.
- Avoid creating or modifying URLs.
- Recommend contacting Amazon Customer Support when the available historical evidence is insufficient.

## Reports

Detailed analysis is available in the `reports/` directory:

- `final_report.md` - final project report
- `baseline_results.txt` - baseline comparison
- `decision_test_results.txt` - escalation decision tests
- `final_failure_analysis.txt` - final-system failure analysis

## Future Improvements

With one additional week, the system could be improved by:

- Expanding the golden evaluation set.
- Improving class balance across intents.
- Testing stronger retrieval methods such as semantic embeddings.
- Adding a more systematic confidence or uncertainty mechanism.
- Expanding reply-quality evaluation with human and LLM-judge agreement.
- Adding more comprehensive escalation rules.
- Evaluating performance on additional AmazonHelp conversations.
