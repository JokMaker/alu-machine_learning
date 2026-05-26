#!/usr/bin/env python3
"""Bag of Words embedding."""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words; if None, all words are used

    Returns:
        embeddings: numpy.ndarray of shape (s, f) with word counts
        features: list of feature words used for embeddings
    """
    def tokenize(sentence):
        """Lowercase, remove possessives, extract alphabetic words."""
        sentence = re.sub(r"'s\b", "", sentence.lower())
        return re.findall(r"[a-z]+", sentence)

    tokenized = [tokenize(s) for s in sentences]

    if vocab is None:
        all_words = set()
        for tokens in tokenized:
            all_words.update(tokens)
        features = sorted(all_words)
    else:
        features = list(vocab)

    word_to_idx = {w: i for i, w in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, tokens in enumerate(tokenized):
        for word in tokens:
            if word in word_to_idx:
                embeddings[i, word_to_idx[word]] += 1

    return embeddings, features
