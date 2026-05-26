#!/usr/bin/env python3
"""Bidirectional Cell - Output"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Initialize bidirectional cell with weights and biases."""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation in the forward direction."""
        concat = np.concatenate([h_prev, x_t], axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """Perform forward propagation in the backward direction."""
        concat = np.concatenate([h_next, x_t], axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """Calculates all outputs for the RNN.

        Args:
            H: numpy.ndarray of shape (t, m, 2 * h) - concatenated hidden
               states from both directions

        Returns:
            Y: the outputs
        """
        logits = H @ self.Wy + self.by
        e = np.exp(logits - np.max(logits, axis=2, keepdims=True))
        return e / np.sum(e, axis=2, keepdims=True)
