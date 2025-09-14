#!/usr/bin/env python3
"""
Contains a function np_elementwise that performs element-wise operations
"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication, and
    division

    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray

    Returns:
        A tuple containing (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
