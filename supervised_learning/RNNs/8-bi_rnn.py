#!/usr/bin/env python3
"""Bidirectional RNN forward propagation"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for a bidirectional RNN.

    Args:
        bi_cell: BidirectionalCell instance
        X: numpy.ndarray of shape (t, m, i) - input data
        h_0: numpy.ndarray of shape (m, h) - initial forward hidden state
        h_t: numpy.ndarray of shape (m, h) - initial backward hidden state

    Returns:
        H: numpy.ndarray of shape (t, m, 2 * h) - concatenated hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t, m, h))
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev

    Hb = np.zeros((t, m, h))
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next

    H = np.concatenate([Hf, Hb], axis=2)
    Y = bi_cell.output(H)

    return H, Y
