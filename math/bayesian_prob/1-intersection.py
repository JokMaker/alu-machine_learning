#!/usr/bin/env python3
"""Module for calculating intersection in Bayesian probability"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculate intersection of obtaining data with hypothetical probabilities
    
    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray containing hypothetical probabilities
        Pr: 1D numpy.ndarray containing prior beliefs of P
        
    Returns:
        1D numpy.ndarray containing intersection for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    
    # Calculate likelihood
    from math import factorial
    binomial_coeff = factorial(n) / (factorial(x) * factorial(n - x))
    likelihood_vals = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))
    
    # Intersection = likelihood * prior
    return likelihood_vals * Pr
