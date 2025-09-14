#!/usr/bin/env python3
"""
Contains a function add_matrices2D that adds two 2D matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise

    Args:
        mat1: First matrix (2D list)
        mat2: Second matrix (2D list)

    Returns:
        A new matrix with element-wise sum, or None if shapes don't match
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
