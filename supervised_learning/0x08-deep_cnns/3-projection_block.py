#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """[summary]

    Args:
        A_prev ([type]): [description]
        filters ([type]): [description]
        s (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    F11, F3, F12 = filters
    he_init = K.initializers.he_normal()

    l1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                         strides=(s, s),
                         padding='same',
                         kernel_initializer=he_init
                         )(A_prev)
    l1 = K.layers.BatchNormalization(axis=3)(l1)
    l1 = K.layers.Activation('relu')(l1)
    l2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=he_init
                         )(l1)
    l2 = K.layers.BatchNormalization(axis=3)(l2)
    l2 = K.layers.Activation('relu')(l2)
    l3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=he_init
                         )(l2)
    l3 = K.layers.BatchNormalization(axis=3)(l3)
    l1_pair = K.layers.Conv2D(filters=F12,
                              kernel_size=(1, 1),
                              strides=(s, s),
                              padding='same',
                              kernel_initializer=he_init
                              )(A_prev)
    l1_pair = K.layers.BatchNormalization(axis=3)(l1_pair)
    l3 = K.layers.Add()([l3, l1_pair])
    return K.layers.Activation('relu')(l3)
