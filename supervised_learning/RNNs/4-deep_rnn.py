#!/usr/bin/env python3
"""Deep RNN forward propagation"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells: list of RNNCell instances (length = number of layers)
        X: numpy.ndarray of shape (t, m, i) - input data
        h_0: numpy.ndarray of shape (l, m, h) - initial hidden states

    Returns:
        H: numpy.ndarray of shape (t + 1, l, m, h) - all hidden states
        Y: numpy.ndarray of shape (t, m, o) - outputs from last layer
    """
    t, m, _ = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]

    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0
    Y = []

    for step in range(t):
        x_in = X[step]
        for layer in range(layers):
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_in)
            H[step + 1, layer] = h_next
            x_in = h_next
        Y.append(y)

    return H, np.array(Y)
