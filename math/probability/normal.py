#!/usr/bin/env python3
"""Normal distribution"""


class Normal:
    """Represents a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """Calculate z-score for given x"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate x-value for given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculate PDF for given x"""
        pi = 3.1415926536
        e = 2.7182818285
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coefficient * e ** exponent

    def cdf(self, x):
        """Calculate CDF for given x"""
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9) / 216
        return 0.5 * (1 + (2 / (pi ** 0.5)) * erf)