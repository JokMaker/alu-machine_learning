#!/usr/bin/env python3
"""Calculate accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculates accuracy of prediction"""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy