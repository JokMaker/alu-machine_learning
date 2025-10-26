#!/usr/bin/env python3
"""Module for convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs convolution on grayscale images with custom padding

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
        padding: tuple of (ph, pw) for padding

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    
    convolved = np.zeros((m, output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            convolved[:, i, j] = (padded[:, i:i+kh, j:j+kw] * kernel).sum(axis=(1, 2))
    
    return convolved