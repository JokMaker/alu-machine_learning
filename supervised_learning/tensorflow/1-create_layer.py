#!/usr/bin/env python3
"""Create layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network

    Args:
        prev: tensor output of the previous layer
        n: number of nodes in the layer to create
        activation: activation function that the layer should use

    Returns:
        tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init,
                            name='layer')(prev)
    return layer
