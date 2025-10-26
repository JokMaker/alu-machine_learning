#!/usr/bin/env python3
"""Module for pooling on images"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images

    Args:
        images: numpy.ndarray with shape (m, h, w, c) containing images
        kernel_shape: tuple of (kh, kw) containing the kernel shape
        stride: tuple of (sh, sw) for stride
        mode: 'max' or 'avg' for pooling type

    Returns:
        numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    
    pooled = np.zeros((m, output_h, output_w, c))
    
    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                pooled[:, i, j, :] = np.max(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2)
                )
    
    return pooled