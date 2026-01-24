#!/usr/bin/env python3
"""Create train operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates training operation using gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
