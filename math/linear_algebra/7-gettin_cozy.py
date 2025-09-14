#!/usr/bin/env python3
"""
Contains a function cat_matrices2D that concatenates two matrices along a
specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis

    Args:
        mat1: First matrix (2D list)
        mat2: Second matrix (2D list)
        axis: Axis along which to concatenate (0 for rows, 1 for columns)

    Returns:
        A new matrix, or None if matrices cannot be concatenated
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]
    return None
