#!/usr/bin/env python3
"""Module for deep neural network performing classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            weight_init = (np.random.randn(layer_size, prev_layer_size) *
                           np.sqrt(2 / prev_layer_size))
            self.__weights[W_key] = weight_init
            self.__weights[b_key] = np.zeros((layer_size, 1))

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
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]
            z = np.matmul(W, A_prev) + b
            if i == self.__L:
                t = np.exp(z)
                A = t / np.sum(t, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-z))
            self.__cache["A{}".format(i)] = A
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model

        Args:
            Y: numpy.ndarray with shape (classes, m) containing correct labels
            A: numpy.ndarray with shape (classes, m) containing activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (classes, m) containing correct labels

        Returns:
            The neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        eye_matrix = np.eye(Y.shape[0])
        pred = eye_matrix[np.argmax(A, axis=0)].T
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (classes, m) containing correct labels
            cache: dictionary containing all the intermediary values
            alpha: learning rate
        """
        m = Y.shape[1]
        dz = cache["A{}".format(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(i - 1)]
            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                W_curr = self.__weights["W{}".format(i)]
                dz = np.matmul(W_curr.T, dz) * A_prev * (1 - A_prev)
            self.__weights["W{}".format(i)] -= alpha * dw
            self.__weights["b{}".format(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing input data
            Y: numpy.ndarray with shape (classes, m) containing correct labels
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: boolean that defines whether or not to print info
            graph: boolean that defines whether or not to graph info
            step: step for verbose and graph

        Returns:
            The evaluation of the training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError(
                    "step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if verbose:
                    msg = "Cost after {} iterations: {}"
                    print(msg.format(i, cost))
                if graph:
                    costs.append(cost)
                    iters.append(i)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format

        Args:
            filename: file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object

        Args:
            filename: file from which the object should be loaded

        Returns:
            The loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
        