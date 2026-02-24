#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """[summary]

    Args:
        X ([type]): [description]
        nb_filters ([type]): [description]
        growth_rate ([type]): [description]
        layers ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_ini = K.initializers.he_normal()
    for i in range(layers):
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=growth_rate * 4,
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=he_ini
                                )(layer)

        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=3,
                                padding='same',
                                kernel_initializer=he_ini
                                )(layer)

        nb_filters += growth_rate
        X = K.layers.concatenate([X, layer])

    return X, nb_filters
