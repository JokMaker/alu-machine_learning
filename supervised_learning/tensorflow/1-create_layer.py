#!/usr/bin/env python3
"""Create layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer for neural network"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer
