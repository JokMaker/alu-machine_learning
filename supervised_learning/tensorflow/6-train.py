#!/usr/bin/env python3
"""Train neural network"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def _tf1_compat():
    """Return a TensorFlow 1.x-compatible API surface.

    Some checkers run with TensorFlow 2.x installed. In that case, we need to
    use `tf.compat.v1` and disable eager execution for graph code.
    """
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        try:
            tf.compat.v1.disable_eager_execution()
        except Exception:
            pass
        return tf.compat.v1
    return tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier

    Args:
        X_train: numpy.ndarray containing the training input data
        Y_train: numpy.ndarray containing the training labels
        X_valid: numpy.ndarray containing the validation input data
        Y_valid: numpy.ndarray containing the validation labels
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        alpha: learning rate
        iterations: number of iterations to train over
        save_path: designates where to save the model

    Returns:
        path where the model was saved
    """
    tf1 = _tf1_compat()

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf1.train.Saver()

    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())

        for i in range(iterations + 1):
            if i % 100 == 0 or i == 0 or i == iterations:
                train_cost = sess.run(loss,
                                      feed_dict={x: X_train, y: Y_train})
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: X_train, y: Y_train})
                valid_cost = sess.run(loss,
                                      feed_dict={x: X_valid, y: Y_valid})
                valid_accuracy = sess.run(accuracy,
                                          feed_dict={x: X_valid, y: Y_valid})

                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuracy))

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        return saver.save(sess, save_path)
