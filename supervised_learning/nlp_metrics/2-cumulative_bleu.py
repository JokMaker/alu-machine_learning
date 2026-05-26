#!/usr/bin/env python3
"""Cumulative N-gram BLEU score calculation."""
import numpy as np
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score for a sentence.

    All n-gram scores from 1 to n are weighted evenly.

    Args:
        references: list of reference translations, each a list of words
        sentence: list of words in the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        the cumulative n-gram BLEU score
    """
    def get_ngrams(words, size):
        """Extract all n-grams of given size from a word list."""
        return [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]

    def ngram_precision(gram_size):
        """Compute clipped precision for n-grams of a given size."""
        sent_ngrams = get_ngrams(sentence, gram_size)
        sent_counts = Counter(sent_ngrams)
        total = max(len(sentence) - gram_size + 1, 0)
        if total == 0:
            return 0.0
        clipped = 0
        for ngram, count in sent_counts.items():
            max_ref = max(
                Counter(get_ngrams(ref, gram_size))[ngram]
                for ref in references
            )
            clipped += min(count, max_ref)
        return clipped / total

    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest = min(ref_lens, key=lambda r: (abs(r - c), r))

    if c >= closest:
        bp = 1.0
    else:
        bp = np.exp(1 - closest / c)

    weight = 1 / n
    log_avg = sum(weight * np.log(ngram_precision(i)) for i in range(1, n + 1))

    return bp * np.exp(log_avg)
