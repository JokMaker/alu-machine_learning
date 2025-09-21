#!/usr/bin/env python3
"""Module for calculating minor matrix"""


def determinant(matrix):
    """Calculate determinant for minor calculation

    Args:
        matrix: list of lists

    Returns:
        The determinant of matrix
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        submatrix = []
        for i in range(1, len(matrix)):
            row = []
            for k in range(len(matrix)):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)
    return det


def minor(matrix):
    """Calculate the minor matrix of a matrix

    Args:
        matrix: list of lists whose minor matrix should be calculated

    Returns:
        The minor matrix of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(n):
            submatrix = []
            for k in range(n):
                if k != i:
                    row = []
                    for col in range(n):
                        if col != j:
                            row.append(matrix[k][col])
                    submatrix.append(row)
            minor_row.append(determinant(submatrix))
        minor_matrix.append(minor_row)

    return minor_matrix