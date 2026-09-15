import sys
sys.path.append(".")

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load fixed training and test sets
# --------------------------------------------------

train_df = pd.read_csv(
    "data/intent_train.csv",
    low_memory=False
)

test_df = pd.read_csv(
    "data/intent_test.csv",
    low_memory=False
)

# Prepare messages
train_df["message"] = train_df["customer_message"].fillna(
    train_df["text"]
)

test_df["message"] = test_df["customer_message"].fillna(
    test_df["text"]
)

train_df = train_df.dropna(
    subset=["message", "intent"]
)

test_df = test_df.dropna(
    subset=["message", "intent"]
)

X_train = train_df["message"].astype(str)
y_train = train_df["intent"].astype(str)

X_test = test_df["message"].astype(str)
y_test = test_df["intent"].astype(str)


# --------------------------------------------------
# 2. Build classifier
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])


# --------------------------------------------------
# 3. Train
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# 4. Predict
# --------------------------------------------------

predictions = model.predict(
    X_test
)


# --------------------------------------------------
# 5. Evaluate
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nINTENT CLASSIFIER EVALUATION")
print("=" * 70)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("Accuracy:", round(accuracy, 3))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# 6. Show every prediction
# --------------------------------------------------

print("\nPREDICTIONS")
print("=" * 70)

for i, (actual, predicted, message) in enumerate(
    zip(
        y_test,
        predictions,
        X_test
    ),
    start=1
):

    status = "CORRECT" if actual == predicted else "WRONG"

    print(f"\n{i}. {status}")
    print("Actual:   ", actual)
    print("Predicted: ", predicted)
    print("Message:  ", message)


    # --------------------------------------------------
# 7. Save wrong predictions for failure analysis
# --------------------------------------------------

results_df = pd.DataFrame({
    "message": X_test.values,
    "actual_intent": y_test.values,
    "predicted_intent": predictions
})

wrong_predictions = results_df[
    results_df["actual_intent"] != results_df["predicted_intent"]
]

wrong_predictions.to_csv(
    "data/intent_failures.csv",
    index=False
)

print("\nWrong predictions saved:", len(wrong_predictions))
print("Saved to: data/intent_failures.csv")