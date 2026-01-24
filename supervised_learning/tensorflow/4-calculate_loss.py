#!/usr/bin/env python3
"""Calculate loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates softmax cross-entropy loss"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
