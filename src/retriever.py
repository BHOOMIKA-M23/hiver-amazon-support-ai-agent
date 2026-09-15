import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AmazonRetriever:

    def __init__(
        self,
        knowledge_base_path="data/amazon_knowledge_base_clean.csv"
    ):

        # Load historical Amazon conversations
        self.df = pd.read_csv(
            knowledge_base_path
        )

        # Prepare customer messages
        self.messages = (
            self.df["customer_message"]
            .fillna("")
            .astype(str)
        )

        # Create TF-IDF representation
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )

        self.matrix = self.vectorizer.fit_transform(
            self.messages
        )

    def retrieve(
        self,
        query,
        top_k=3
    ):

        # Convert new customer message to TF-IDF
        query_vector = self.vectorizer.transform(
            [query]
        )

        # Calculate similarity
        similarities = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Get highest scoring examples
        top_indices = similarities.argsort()[
            -top_k:
        ][::-1]

        results = self.df.iloc[
            top_indices
        ].copy()

        results["similarity"] = similarities[
            top_indices
        ]

        return results