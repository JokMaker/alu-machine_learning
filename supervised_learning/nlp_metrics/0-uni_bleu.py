#!/usr/bin/env python3
"""Unigram BLEU score calculation."""
import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations, each a list of words
        sentence: list of words in the model proposed sentence

    Returns:
        the unigram BLEU score
    """
    sent_counts = Counter(sentence)

    clipped = 0
    for word, count in sent_counts.items():
        max_ref = max(ref.count(word) for ref in references)
        clipped += min(count, max_ref)

    precision = clipped / len(sentence)

    c = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest = min(ref_lens, key=lambda r: (abs(r - c), r))

    if c >= closest:
        bp = 1.0
    else:
        bp = np.exp(1 - closest / c)

    return bp * precision
