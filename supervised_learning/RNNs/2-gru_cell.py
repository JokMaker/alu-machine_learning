#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initialize GRU cell with weights and biases."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        concat = np.concatenate([h_prev, x_t], axis=1)

        z = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))
        r = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))

        concat_r = np.concatenate([r * h_prev, x_t], axis=1)
        h_tilde = np.tanh(concat_r @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        logits = h_next @ self.Wy + self.by
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)

        return h_next, y
