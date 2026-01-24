#!/usr/bin/env python3
"""Forward propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=None, activations=None):
    """Creates the forward propagation graph for the neural network.

    Args:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        prediction of the network in tensor form
    """
    if layer_sizes is None:
        layer_sizes = []
    if activations is None:
        activations = []

    if len(layer_sizes) != len(activations):
        raise ValueError('layer_sizes and activations must be the same length')

    output = x
    for n, activation in zip(layer_sizes, activations):
        output = create_layer(output, n, activation)
    return output
