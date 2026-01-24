#!/usr/bin/env python3
"""Forward propagation"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates forward propagation graph for neural network"""
    layer = x
    for i in range(len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer