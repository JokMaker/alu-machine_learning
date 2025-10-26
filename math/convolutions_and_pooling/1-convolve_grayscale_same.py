#!/usr/bin/env python3
"""Module for same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs same convolution on grayscale images

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    
    pad_h = kh // 2
    pad_w = kw // 2
    
    padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    convolved = np.zeros((m, h, w))
    
    for i in range(h):
        for j in range(w):
            convolved[:, i, j] = (padded[:, i:i+kh, j:j+kw] * kernel).sum(axis=(1, 2))
    
    return convolved