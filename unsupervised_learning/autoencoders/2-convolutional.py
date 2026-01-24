#!/usr/bin/env python3
"""2-convolutional

Convolutional autoencoder implemented with `tensorflow.keras`.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    Args:
        input_dims (tuple[int, int, int]): Dimensions of the model input
            (height, width, channels).
        filters (list[int]): Number of filters for each convolutional layer in
            the encoder (in order).
        latent_dims (tuple[int, int, int]): Shape of the latent representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: encoder model
            - decoder: decoder model
            - auto: full autoencoder model
    """
    # Encoder
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same',
        )(x)

    encoder = keras.Model(inputs=encoder_inputs, outputs=x, name='encoder')

    # Decoder
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs

    rev_filters = list(reversed(filters))

    # All decoder convolutions except the last two:
    # same padding + relu + upsample
    for f in rev_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
        )(x)
        x = keras.layers.UpSampling2D(
            size=(2, 2),
        )(x)

    # Second to last convolution: valid padding + relu + upsample
    x = keras.layers.Conv2D(
        filters=rev_filters[-1],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu',
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Last convolution: channels == input channels, sigmoid, no upsample
    decoder_outputs = keras.layers.Conv2D(
        filters=input_dims[2],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid',
    )(x)

    decoder = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs,
                          name='decoder')

    # Autoencoder
    auto_outputs = decoder(encoder(encoder_inputs))
    auto = keras.Model(inputs=encoder_inputs, outputs=auto_outputs,
                       name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
