#!/usr/bin/env python3
"""Word2Vec model training using gensim."""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a gensim word2vec model.

    Args:
        sentences: list of tokenized sentences to train on
        size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word for training
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: True for CBOW training; False for Skip-gram
        iterations: number of training iterations
        seed: seed for the random number generator
        workers: number of worker threads

    Returns:
        the trained gensim Word2Vec model
    """
    sg = 0 if cbow else 1
    model = Word2Vec(sentences, size=size, min_count=min_count, window=window,
                     negative=negative, sg=sg, iter=iterations, seed=seed,
                     workers=workers)
    return model
