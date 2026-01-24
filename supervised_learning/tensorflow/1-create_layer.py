#!/usr/bin/env python3
"""Create layer"""
import tensorflow as tf


# Support TF2 environments used by some checkers
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass


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
    # He et al initialization (required by the project spec)
    try:
        init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    except Exception:
        init = tf.compat.v1.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_avg',
            distribution='uniform',
        )

    dense = getattr(tf, 'layers', None)
    if dense is None and hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        dense = tf.compat.v1.layers

    layer = dense.dense(
        prev,
        n,
        activation=activation,
        kernel_initializer=init,
        name='layer',
    )
    return layer
