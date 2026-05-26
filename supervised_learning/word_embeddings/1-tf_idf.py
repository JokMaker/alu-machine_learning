#!/usr/bin/env python3
"""TF-IDF embedding."""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words; if None, all words are used

    Returns:
        embeddings: numpy.ndarray of shape (s, f) with TF-IDF values
        features: list of feature words used for embeddings
    """
    if vocab is not None:
        vocabulary = {word: idx for idx, word in enumerate(vocab)}
    else:
        vocabulary = None

    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
