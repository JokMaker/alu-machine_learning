#!/usr/bin/env python3
"""RNN Cell"""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initialize RNN cell with weights and biases."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat = np.concatenate([h_prev, x_t], axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        logits = h_next @ self.Wy + self.by
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)
        return h_next, y
