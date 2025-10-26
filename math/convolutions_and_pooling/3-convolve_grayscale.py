#!/usr/bin/env python3
"""Module for strided convolution on grayscale images"""
import numpy as np
from math import ceil, floor


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs convolution on grayscale images with stride

    Args:
        images: numpy.ndarray with shape (m, h, w) containing multiple images
        kernel: numpy.ndarray with shape (kh, kw) containing the kernel
        padding: 'same', 'valid', or tuple of (ph, pw)
        stride: tuple of (sh, sw) for stride

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding
    
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1
    
    convolved = np.zeros((m, output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            convolved[:, i, j] = (padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel).sum(axis=(1, 2))
    
    return convolved