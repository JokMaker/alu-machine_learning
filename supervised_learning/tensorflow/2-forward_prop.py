#!/usr/bin/env python3
"""Forward propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


# Support TF2 environments used by some checkers
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network.

    Args:
        x: placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer

    Returns:
        prediction of the network in tensor form
    """
    output = x
    for n, activation in zip(layer_sizes, activations):
        output = create_layer(output, n, activation)
    return output
