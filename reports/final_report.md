# Amazon Customer Support AI Agent

## 1. Problem Framing

### Goal

Build an AI-powered customer support agent for AmazonHelp that can:

1. Classify incoming customer messages into a small set of support intents.
2. Draft a reply grounded in historically observed Amazon support responses.
3. Decide whether the case should be auto-handled or escalated to a human.

### Dataset

Source: Customer Support on Twitter dataset.

Brand selected: AmazonHelp.

Amazon-specific data extracted: 135,182 tweets.

Customer messages: 135,160.

Conversation pairs: 92,968.

Clean historical knowledge-base pairs: 77,003.

After removing golden-set overlap for evaluation:

Clean evaluation knowledge base: 76,902 pairs.

---

## 2. Intent Taxonomy

The system uses seven customer-support intents:

1. delivery_shipping
2. order_issue
3. refund_return
4. account_payment
5. product_help
6. prime_issue
7. other

The taxonomy was created by reviewing representative AmazonHelp conversations and grouping recurring customer problems.

---

## 3. System Architecture

Customer Message
↓
Intent Classifier
↓
Historical Conversation Retriever
↓
Gemini Reply Generator
↓
Escalation Decision
↓
Auto-handle / Escalate

The retriever uses TF-IDF similarity to find historically similar Amazon customer-support conversations.

The reply generator uses Gemini and is instructed to ground replies only in retrieved historical evidence.

---

## 4. Intent Classification Results

### Baseline 1: Majority Class

Accuracy: 43.3%

### Baseline 2: TF-IDF + Logistic Regression

Accuracy: 46.7%

Macro F1: 0.21

Weighted F1: 0.40

### Gemini Classifier

Accuracy: 93.3%

Macro F1: 0.88

Weighted F1: 0.93

Test set: 30 held-out examples.

Gemini classified 28 of 30 examples correctly.

---

## 5. Reply Quality Evaluation



A 30-example held-out evaluation set was generated using the full `AmazonSupportAgent` pipeline.

The generated replies were evaluated using Gemini 3.6 Flash as an automated LLM-as-judge following the rubric defined in `reply_quality_rubric.txt`. The rubric evaluates two dimensions: **Reply Quality** (0=Poor, 1=Partially Good, 2=Good) and **Evidence Support** (supported, partially_supported, unsupported).

### LLM-as-Judge Results

- **Average Reply Quality Score**: **2.00 / 2.00**
- **Replies rated Good by LLM judge**: **30/30 (100.0%)**
- **Evidence Fully Supported**: **30/30 (100.0%)**
- **Evidence Partially Supported**: **0/30 (0.0%)**
- **Evidence Unsupported**: **0/30 (0.0%)**

### Human Review

Human review of the same 30 examples gave a more conservative assessment:

- **Average Human Reply Quality Score**: **1.70 / 2.00**
- **Good**: **21/30 (70.0%)**
- **Partially Good**: **9/30 (30.0%)**
- **Poor**: **0/30 (0.0%)**

The main human-rated weaknesses were that some replies were too generic, did not directly address the customer's specific problem, or escalated to support without providing a useful first step.

### Human vs. LLM-Judge Agreement

Exact agreement between the human annotations and LLM-judge predictions was:

- **Reply Quality Exact Agreement**: **70.0%**
- **Evidence Support Exact Agreement**: **70.0%**

The LLM judge was noticeably more lenient than the human reviewer: it rated all 30 replies as Good, while the human reviewer rated 21 as Good and 9 as Partially Good.

Therefore, the automated judge score should not be interpreted as proof that all generated replies are high quality. Human review provides a more conservative view of response usefulness.

The 30-example sample is small, so these results should be treated as an initial evaluation rather than a production-quality estimate.

## 6. Escalation Logic

The system escalates when:

1. Historical evidence similarity is below 0.45.
2. The customer explicitly requests a human/agent/representative.
3. The intent is account_payment.

Otherwise, the system auto-handles the request when relevant historical evidence is available.

Decision-system test result:

5/5 representative decision tests passed.

---

## 7. Top Failure Modes

### Failure Mode 1: Generic escalation instead of directly addressing the request

**Example:** A customer reported that one of two purchased items had been cancelled even though they had not cancelled it.

**Observed behavior:** The agent responded with a generic support contact instead of addressing the unexpected cancellation.

**Hypothesis:** The retriever finds broadly relevant order-support conversations, but the generator may prefer a safe escalation response when the historical evidence does not contain a precise resolution.

### Failure Mode 2: Delivery complaints receive generic support responses

**Example:** A customer said their order was still not delivered despite the expected delivery date.

**Observed behavior:** The reply apologized and directed the customer to support without providing a more specific next step.

**Hypothesis:** Delivery-related conversations contain many similar historical replies, making retrieval useful for escalation but not always specific enough to resolve the exact situation.

### Failure Mode 3: Product troubleshooting can fall back to escalation

**Example:** A customer reported that they had already followed troubleshooting instructions but the device remained stuck in a reboot cycle.

**Observed behavior:** The agent directed the customer to phone support instead of continuing with a troubleshooting step.

**Hypothesis:** The current retrieval and generation pipeline is conservative when historical evidence does not clearly support another troubleshooting action.

### Failure Mode 4: Specific delivery problems can be collapsed into generic assistance

**Example:** A customer reported that a package was being returned while still in transit because the carrier claimed the customer had refused delivery.

**Observed behavior:** The agent provided a generic support contact rather than directly addressing the incorrect refusal/return situation.

**Hypothesis:** The historical knowledge base contains relevant delivery conversations, but TF-IDF similarity may not distinguish important details such as carrier error, refusal status, and package location.

### Failure Mode 5: Short, multilingual, or ambiguous messages are harder to handle

**Example:** A customer message in Japanese stated that the package had arrived normally and thanked Amazon.

**Observed behavior:** The response was polite but was rated only partially good by human review.

**Hypothesis:** The first-version system was primarily designed around English-language support patterns and a small intent taxonomy. Short or multilingual messages can therefore be harder to interpret and respond to precisely.

These failures suggest that improving retrieval specificity, multilingual robustness, and response-generation rules would likely improve practical support quality more than simply increasing the classifier accuracy.

## 8. What Is Misleading About My Headline Number?

The 93.3% Gemini classification accuracy is based on only 30 held-out examples.

Therefore, it should not be interpreted as proof that the system will achieve 93.3% accuracy on real-world Amazon customer messages.

The evaluation set is also imbalanced, with the `other` category representing a large portion of the examples.

A larger and more balanced evaluation set would provide stronger evidence.

---

## 9. Next Steps With One More Week

With another week, I would:

1. Expand the golden evaluation set from 150 to 250+ examples.
2. Improve the intent taxonomy using additional annotation.
3. Add retrieval-quality evaluation.
4. Compare TF-IDF retrieval with embedding-based retrieval.
5. Improve escalation using calibrated confidence and evidence thresholds.
6. Evaluate reply quality on a larger sample.
7. Perform deeper error analysis.
8. Test robustness on multilingual and noisy customer messages.
