#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initialize Binomial distribution"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - variance / mean
            n = round(mean / p)
            self.n = int(n)
            self.p = float(mean / n)

    def pmf(self, k):
        """Calculate PMF for given k"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # Calculate binomial coefficient
        binomial_coeff = 1
        for i in range(k):
            binomial_coeff = binomial_coeff * (self.n - i) / (i + 1)
        return binomial_coeff * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculate CDF for given k"""
        k = int(k)
        if k < 0:
            return 0
        if k >= self.n:
            return 1

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value
