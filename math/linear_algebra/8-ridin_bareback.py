#!/usr/bin/env python3
"""
Contains a function mat_mul that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication

    Args:
        mat1: First matrix (2D list)
        mat2: Second matrix (2D list)

    Returns:
        A new matrix that is the product of mat1 and mat2, or None if cannot multiply
    """
    if len(mat1[0]) != len(mat2):
        return None
    return [[sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2))) for j in range(len(mat2[0]))] for i in range(len(mat1))]
