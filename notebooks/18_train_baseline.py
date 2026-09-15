import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load our 150 labelled examples
df = pd.read_csv("data/amazon_golden_set.csv")

# 2. Create one common message column
df["message"] = df["customer_message"].fillna(df["text"])

# Remove rows with missing messages or labels
df = df.dropna(subset=["message", "intent"])

# 3. Separate messages (X) and labels (y)
X = df["message"]
y = df["intent"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Build the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions on unseen test messages
predictions = model.predict(X_test)

# 8. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("\nBaseline Model Results")
print("----------------------")
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("Accuracy:", round(accuracy, 3))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))