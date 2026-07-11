"""Prepare the Kyrgyz_News_Corpus for continued pretraining.

Continued pretraining is unsupervised: we just need raw Kyrgyz text, chunked
into training-length blocks. No (input, output) pairs needed here — the goal
is to shift the model's language distribution toward Kyrgyz (it currently
defaults to Kazakh even when explicitly asked for Kyrgyz), not to teach a
specific task yet.
"""

from datasets import Dataset, load_dataset

CORPUS_NAME = "the-cramer-project/Kyrgyz_News_Corpus"


def load_kyrgyz_text(min_chars=200, max_chars=3000, limit=None):
    corpus = load_dataset(CORPUS_NAME)["train"]
    texts = [
        row["text"] for row in corpus if row["text"] and min_chars <= len(row["text"]) <= max_chars
    ]
    if limit:
        texts = texts[:limit]
    return Dataset.from_dict({"text": texts})
