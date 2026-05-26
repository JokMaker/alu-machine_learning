#!/usr/bin/env python3
"""RNN forward propagation"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: RNNCell instance for forward propagation
        X: numpy.ndarray of shape (t, m, i) - input data
        h_0: numpy.ndarray of shape (m, h) - initial hidden state

    Returns:
        H: numpy.ndarray containing all hidden states
        Y: numpy.ndarray containing all outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = []

    h_prev = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_next
        Y.append(y)
        h_prev = h_next

    return H, np.array(Y)
