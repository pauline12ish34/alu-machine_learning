#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """[summary]

    Args:
        input_dims ([type]): [description]
        hidden_layers ([type]): [description]
        latent_dims ([type]): [description]

    Returns:
        [type]: [description]
    """
    backend = keras.backend

    def s_a(args):
        z_mean, z_log_sigma = args
        batch = backend.shape(
            z_mean
        )[0]
        epsilon = backend.random_normal(
            shape=(batch, latent_dims),
            mean=0.0,
            stddev=0.1)
        return z_mean + backend.exp(
            z_log_sigma) * epsilon

    encoder_In = keras.Input(
        shape=(
            input_dims,))
    encoder = encoder_In
    for nodes in hidden_layers:
        encoder = keras.layers.Dense(
            nodes,
            activation='relu'
        )(encoder)

    z_mean = keras.layers.Dense(
        latent_dims)(
            encoder)
    z_log_sigma = keras.layers.Dense(
        latent_dims
    )(encoder)

    z = keras.layers.Lambda(
        s_a)([z_mean, z_log_sigma]
             )

    decoder_In = keras.Input(
        shape=(latent_dims,
               ))
    decoder = decoder_In
    for nodes in hidden_layers[::-1]:
        decoder = keras.layers.Dense(
            nodes,
            activation='relu'
        )(
            decoder)
    decoder = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(
        decoder)

    encoder = keras.Model(encoder_In,
                          [z, z_mean, z_log_sigma]
                          )
    decoder = keras.Model(
        decoder_In,
        decoder)

    out = decoder(
        encoder(
            encoder_In))
    auto = keras.Model(
        encoder_In,
        out)

    def cost_f(val1, val2):
        reconstruction_loss = keras.losses.binary_crossentropy(
            encoder_In,
            out
        )
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma
        kl_loss = kl_loss - backend.square(
            z_mean) - backend.exp(
                z_log_sigma
        )
        kl_loss = backend.sum(
            kl_loss,
            axis=-1)
        kl_loss *= -0.5
        cost_f = backend.mean(
            reconstruction_loss + kl_loss
        )
        return cost_f
    auto.compile(
        optimizer='adam',
        loss=cost_f
    )

    return encoder, decoder, auto
