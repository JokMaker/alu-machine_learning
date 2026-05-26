#!/usr/bin/env python3
"""N-gram BLEU score calculation."""
import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """Calculate the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations, each a list of words
        sentence: list of words in the model proposed sentence
        n: size of the n-gram to use for evaluation

    Returns:
        the n-gram BLEU score
    """
    def get_ngrams(words, size):
        """Extract all n-grams of given size from a word list."""
        return [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]

    sent_ngrams = get_ngrams(sentence, n)
    sent_counts = Counter(sent_ngrams)

    clipped = 0
    for ngram, count in sent_counts.items():
        max_ref = max(Counter(get_ngrams(ref, n))[ngram] for ref in references)
        clipped += min(count, max_ref)

    total = max(len(sentence) - n + 1, 0)
    if total == 0:
        return 0.0

    precision = clipped / total

    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest = min(ref_lens, key=lambda r: (abs(r - c), r))

    if c >= closest:
        bp = 1.0
    else:
        bp = np.exp(1 - closest / c)

    return bp * precision
