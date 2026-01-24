#!/usr/bin/env python3
"""Create placeholders"""
import tensorflow as tf


# Support TF2 environments used by some checkers
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass
    tf = tf.compat.v1


def create_placeholders(nx, classes):
    """Creates placeholders for neural network"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
