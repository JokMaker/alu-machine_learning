#!/usr/bin/env python3
"""Module for calculating determinant of a matrix"""


def determinant(matrix):
    """Calculate the determinant of a matrix
    
    Args:
        matrix: list of lists whose determinant should be calculated
        
    Returns:
        The determinant of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    
    if matrix == [[]]:
        return 1
    
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
    
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")
    
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for j in range(n):
        submatrix = []
        for i in range(1, n):
            row = []
            for k in range(n):
                if k != j:
                    row.append(matrix[i][k])
            submatrix.append(row)
        det += ((-1) ** j) * matrix[0][j] * determinant(submatrix)
    
    return det
