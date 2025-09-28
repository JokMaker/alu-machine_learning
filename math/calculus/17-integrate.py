#!/usr/bin/env python3
"""Module for calculating polynomial integrals"""


def poly_integral(poly, C=0):
    """Calculate integral of polynomial

    Args:
        poly: list of coefficients
        C: integration constant

    Returns:
        List of integral coefficients or None if invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    if not all(isinstance(x, (int, float)) for x in poly):
        return None

    result = [C]
    for i, coeff in enumerate(poly):
        integral_coeff = coeff / (i + 1)
        if integral_coeff == int(integral_coeff):
            integral_coeff = int(integral_coeff)
        result.append(integral_coeff)

    return result