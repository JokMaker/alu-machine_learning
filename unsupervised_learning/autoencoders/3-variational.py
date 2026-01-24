#!/usr/bin/env python3
"""3-variational

Variational autoencoder implemented with `tensorflow.keras`.

Encoder outputs:
- z: sampled latent representation
- mu: mean of the latent distribution
- log_sigma: log variance (log(sigma^2)) of the latent distribution

The model uses the reparameterization trick and adds KL divergence via
`Model.add_loss`.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    Args:
        input_dims (int): Dimensionality of the model input.
        hidden_layers (list[int]): Number of nodes for each hidden layer in the
            encoder (in order).
        latent_dims (int): Dimensionality of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: outputs (z, mu, log_sigma)
            - decoder: decoder model
            - auto: full VAE model
    """

    # Sampling function using (mu, log_sigma)
    K = keras.backend

    def sampling(args):
        mu, log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(mu)[0], latent_dims),
            mean=0.0,
            stddev=1.0,
        )
        return mu + K.exp(log_sigma / 2) * epsilon

    # Encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder_inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mu = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sigma = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [mu, log_sigma]
    )

    encoder = keras.Model(
        inputs=encoder_inputs,
        outputs=[z, mu, log_sigma],
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

    # VAE = encoder + decoder
    z_out, mu_out, log_sigma_out = encoder(encoder_inputs)
    outputs = decoder(z_out)

    auto = keras.Model(inputs=encoder_inputs, outputs=outputs, name='vae')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * K.sum(
        1 + log_sigma_out - K.square(mu_out) - K.exp(log_sigma_out),
        axis=1,
    )
    auto.add_loss(K.mean(kl_loss))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
