#!/usr/bin/env python3
"""
Contains a function np_cat that concatenates two matrices along a specific axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis

    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray
        axis: Axis along which to concatenate

    Returns:
        A new numpy.ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)
