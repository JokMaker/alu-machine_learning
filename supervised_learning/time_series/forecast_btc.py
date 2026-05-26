#!/usr/bin/env python3
"""Trains and validates an LSTM model to forecast BTC hourly close price."""
import numpy as np
import tensorflow as tf
import pandas as pd


WINDOW = 24       # past 24 hours as input
BATCH = 64
EPOCHS = 10
DATA_FILE = 'btc_hourly.npy'


def make_dataset(data, window_size=WINDOW, batch_size=BATCH, shuffle=False):
    """Create a tf.data.Dataset of (input_window, target) pairs.

    Uses a sliding window of length window_size over the data.
    Each sample is (past window_size values, next value).

    Args:
        data: numpy.ndarray of shape (N, 1) containing normalized prices
        window_size: number of past time steps to use as input
        batch_size: number of samples per batch
        shuffle: whether to shuffle the dataset

    Returns:
        tf.data.Dataset yielding (inputs, targets) batches
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(
        lambda w: w.batch(window_size + 1, drop_remainder=True))
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def build_model(window_size=WINDOW):
    """Build a two-layer LSTM model for time series regression.

    Args:
        window_size: number of input time steps

    Returns:
        compiled tf.keras.Sequential model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True,
                             input_shape=(window_size, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse')
    return model


def main():
    """Load data, build model, train, and report validation MSE."""
    data = np.load(DATA_FILE).astype(np.float32).reshape(-1, 1)
    print("Loaded {} hourly data points".format(len(data)))

    # 80% train, 20% validation (keep temporal order)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    val_data = data[split - WINDOW:]  # include overlap for first window

    train_ds = make_dataset(train_data, shuffle=True)
    val_ds = make_dataset(val_data, shuffle=False)

    model = build_model()
    model.summary()

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

    final_mse = history.history['val_loss'][-1]
    print("Final validation MSE: {:.6f}".format(final_mse))


if __name__ == '__main__':
    main()
