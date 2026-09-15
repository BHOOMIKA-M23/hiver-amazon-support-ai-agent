import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. Load the knowledge base
# --------------------------------------------------

input_file = "data/amazon_knowledge_base_clean.csv"

df = pd.read_csv(input_file)

print("Knowledge-base examples:", len(df))


# --------------------------------------------------
# 2. Prepare customer messages
# --------------------------------------------------

messages = df["customer_message"].fillna("").astype(str)


# --------------------------------------------------
# 3. Convert messages into TF-IDF vectors
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

matrix = vectorizer.fit_transform(messages)

print("TF-IDF matrix created.")
print("Number of features:", len(vectorizer.get_feature_names_out()))


# --------------------------------------------------
# 4. Function to retrieve similar conversations
# --------------------------------------------------

def retrieve_similar_cases(query, top_k=3):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    results["similarity"] = similarities[top_indices]

    return results


# --------------------------------------------------
# 5. Test the retrieval system
# --------------------------------------------------

test_message = (
    "My package says it was delivered but "
    "I never received it."
)

results = retrieve_similar_cases(
    test_message,
    top_k=3
)


# --------------------------------------------------
# 6. Display retrieved cases
# --------------------------------------------------

print("\nNEW CUSTOMER MESSAGE:")
print(test_message)

print("\nTOP 3 HISTORICAL CASES:")

for _, row in results.iterrows():

    print("\nCUSTOMER:")
    print(row["customer_message"])

    print("\nAMAZON RESPONSE:")
    print(row["amazon_reply"])

    print("\nSIMILARITY SCORE:")
    print(round(row["similarity"], 3))

    print("\n" + "=" * 70)