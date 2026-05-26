#!/usr/bin/env python3
"""Bidirectional Cell - Backward Direction"""
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
        """Perform forward propagation in the backward direction.

        Args:
            h_next: numpy.ndarray of shape (m, h) - next hidden state
            x_t: numpy.ndarray of shape (m, i) - input at time t

        Returns:
            h_prev: the previous hidden state
        """
        concat = np.concatenate([h_next, x_t], axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev
