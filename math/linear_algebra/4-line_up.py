#!/usr/bin/env python3
"""
Contains a function add_arrays that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise

    Args:
        arr1: First array (list of ints/floats)
        arr2: Second array (list of ints/floats)

    Returns:
        A new list with element-wise sum, or None if shapes don't match
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
