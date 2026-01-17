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
            weight_init = (np.random.randn(layer_size, prev_layer_size) *
                           np.sqrt(2 / prev_layer_size))
            self.__weights[key_w] = weight_init
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

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data

        Returns:
            The output of the neural network and the cache
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(i)]
            b = self.__weights['b{}'.format(i)]
            A_prev = self.__cache['A{}'.format(i - 1)]
            z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache['A{}'.format(i)] = A
        return self.__cache['A{}'.format(self.__L)], self.__cache
