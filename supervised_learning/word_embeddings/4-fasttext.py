#!/usr/bin/env python3
"""FastText model training using gensim."""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Create and train a gensim FastText model.

    Args:
        sentences: list of tokenized sentences to train on
        size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word for training
        negative: size of negative sampling
        window: maximum distance between current and predicted word
        cbow: True for CBOW training; False for Skip-gram
        iterations: number of training iterations
        seed: seed for the random number generator
        workers: number of worker threads

    Returns:
        the trained gensim FastText model
    """
    sg = 0 if cbow else 1
    model = FastText(sentences, size=size, min_count=min_count, window=window,
                     negative=negative, sg=sg, iter=iterations, seed=seed,
                     workers=workers)
    return model
