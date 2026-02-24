#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """[summary]

    Args:
        X ([type]): [description]
        nb_filters ([type]): [description]
        compression ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_ini = K.initializers.he_normal()

    l1 = K.layers.BatchNormalization()(X)
    l1 = K.layers.Activation('relu')(l1)

    nb_filters = int(nb_filters * compression)

    l2 = K.layers.Conv2D(filters=nb_filters,
                         kernel_size=1,
                         padding='same',
                         kernel_initializer=he_ini
                         )(l1)
    l2 = K.layers.AveragePooling2D(pool_size=2,
                                   padding='same'
                                   )(l2)
    return l2, nb_filters
