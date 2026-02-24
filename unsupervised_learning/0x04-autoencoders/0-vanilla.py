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
    encoder_In = keras.Input(
        shape=(input_dims,)
    )

    encoder = encoder_In

    for nodes in hidden_layers:
        encoder = keras.layers.Dense(nodes,
                                     activation='relu'
                                     )(
                                         encoder
        )
    encoder = keras.layers.Dense(latent_dims,
                                 activation='relu'
                                 )(
                                     encoder
    )
    decoder_In = keras.Input(
        shape=(latent_dims,)
    )
    decoder = decoder_In
    for nodes in hidden_layers[::-1]:
        decoder = keras.layers.Dense(nodes,
                                     activation='relu'
                                     )(
                                         decoder
        )
    decoder = keras.layers.Dense(input_dims,
                                 activation='sigmoid'
                                 )(
                                     decoder
    )

    encoder = keras.Model(encoder_In,
                          encoder
                          )
    decoder = keras.Model(decoder_In,
                          decoder
                          )
    auto = keras.Model(encoder_In,
                       decoder(
                           encoder(encoder_In)
                       ))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
