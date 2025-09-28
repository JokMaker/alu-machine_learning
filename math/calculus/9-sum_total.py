#!/usr/bin/env python3
"""Module for calculating sum of squares"""


def summation_i_squared(n):
    """Calculate sum of i^2 from 1 to n
    
    Args:
        n: stopping condition
        
    Returns:
        Integer value of sum or None if invalid
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6

