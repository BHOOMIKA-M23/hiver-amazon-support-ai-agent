import pandas as pd
import re
from collections import Counter

file_path = "data/amazon_support.csv"

df = pd.read_csv(file_path, low_memory=False)

customer_messages = df[df["inbound"] == True]["text"].astype(str)

words = []

for text in customer_messages:
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Keep only letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    words.extend(text.split())

stop_words = {
    "the", "and", "to", "a", "i", "of", "is", "it",
    "in", "for", "on", "my", "me", "you", "that",
    "this", "was", "with", "but", "have", "has",
    "be", "are", "can", "do", "not", "your", "we",
    "they", "am", "so", "please", "just", "got",
    "from", "or", "what", "why", "how", "amazon",
    "help", "helpme"
}

filtered_words = [
    word for word in words
    if word not in stop_words and len(word) > 2
]

counts = Counter(filtered_words)

print("\nTop 50 customer-support terms:\n")

for word, count in counts.most_common(50):
    print(f"{word}: {count}")