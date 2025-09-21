#!/usr/bin/env python3
"""Module for calculating inverse matrix"""


def determinant(matrix):
    """Calculate determinant for inverse calculation

    Args:
        matrix: list of lists

    Returns:
        The determinant of matrix
    """
    if matrix == [[]]:
        return 1
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
    """Calculate minor matrix for inverse calculation

    Args:
        matrix: list of lists

    Returns:
        The minor matrix of matrix
    """
    n = len(matrix)
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


def cofactor(matrix):
    """Calculate cofactor matrix for inverse calculation

    Args:
        matrix: list of lists

    Returns:
        The cofactor matrix of matrix
    """
    minor_matrix = minor(matrix)
    cofactor_matrix = []
    n = len(matrix)

    for i in range(n):
        cofactor_row = []
        for j in range(n):
            cofactor_row.append(((-1) ** (i + j)) * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def adjugate(matrix):
    """Calculate adjugate matrix for inverse calculation

    Args:
        matrix: list of lists

    Returns:
        The adjugate matrix of matrix
    """
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    n = len(matrix)

    for i in range(n):
        adjugate_row = []
        for j in range(n):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix


def inverse(matrix):
    """Calculate the inverse of a matrix

    Args:
        matrix: list of lists whose inverse should be calculated

    Returns:
        The inverse of matrix, or None if matrix is singular
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

    det = determinant(matrix)
    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    adj_matrix = adjugate(matrix)
    inverse_matrix = []

    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj_matrix[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
