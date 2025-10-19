#!/usr/bin/env python3
"""Module for calculating correlation matrix"""
import numpy as np


def correlation(C):
    """Calculate a correlation matrix from a covariance matrix

    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix

    Returns:
        numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]
    std_devs = np.sqrt(np.diag(C))
    correlation_matrix = C / np.outer(std_devs, std_devs)

    return correlation_matrix
