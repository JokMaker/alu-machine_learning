#!/usr/bin/env python3
"""Module for valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs valid convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            convolved[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return convolved