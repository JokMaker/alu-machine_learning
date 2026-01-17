#!/usr/bin/env python3
"""Module for deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network"""

    def __init__(self, nx, layers):
        """Initialize deep neural network

        Args:
            nx: number of input features
            layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            layer_size = layers[i]
            prev_layer_size = nx if i == 0 else layers[i - 1]
            key_w = 'W{}'.format(i + 1)
            key_b = 'b{}'.format(i + 1)
            self.__weights[key_w] = (np.random.randn(layer_size, prev_layer_size) *
                                     np.sqrt(2 / prev_layer_size))
            self.__weights[key_b] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
