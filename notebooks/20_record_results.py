results = """
Hiver AmazonHelp Intent Classification — Baseline Results

Dataset:
- Golden evaluation examples: 150
- Training examples: 120
- Test examples: 30

Baseline 1 — Majority Class:
- Most common intent: other
- Accuracy: 43.3%

Baseline 2 — TF-IDF + Logistic Regression:
- Accuracy: 46.7%
- Macro F1: 0.21
- Weighted F1: 0.40

Observation:
The ML baseline improves over the trivial majority-class baseline,
but the improvement is modest. Performance is stronger for the
majority classes and weaker for minority intents.
"""

with open("reports/baseline_results.txt", "w") as f:
    f.write(results)

print("Saved baseline results to: reports/baseline_results.txt")