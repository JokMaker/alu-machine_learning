#!/usr/bin/env python3
"""1-sparse

Sparse (fully-connected) autoencoder implemented with `tensorflow.keras`.

Adds L1 activity regularization on the latent representation.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Creates a sparse autoencoder.

    Args:
        input_dims (int): Dimensionality of the model input.
        hidden_layers (list[int]): Number of nodes for each hidden layer in the
            encoder (in order).
        latent_dims (int): Dimensionality of the latent space representation.
        lambtha (float): L1 regularization parameter on the encoded output.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: encoder model
            - decoder: decoder model
            - auto: full sparse autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha),
    )(x)

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=latent,
        name='encoder',
    )

    # Decoder
    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = decoder_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)

    decoder = keras.Model(
        inputs=decoder_inputs,
        outputs=decoder_outputs,
        name='decoder',
    )

    # Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(
        inputs=encoder_inputs,
        outputs=auto_outputs,
        name='autoencoder',
    )

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
