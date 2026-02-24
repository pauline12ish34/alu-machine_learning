#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """[summary]

    Args:
        A_prev ([type]): [description]
        filters ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_init = K.initializers.he_normal()
    F1, F3R, F3, F5R, F5, FPP = filters
    l_F1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                           padding='same',
                           activation='relu',
                           kernel_initializer=he_init
                           )(A_prev)
    l_F3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he_init
                            )(A_prev)
    l_F3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                           padding='same',
                           activation='relu',
                           kernel_initializer=he_init
                           )(l_F3R)
    l_F5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he_init
                            )(A_prev)
    l_F5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                           padding='same',
                           activation='relu',
                           kernel_initializer=he_init
                           )(l_F5R)
    l_pooling_FPP = K.layers.MaxPool2D(pool_size=(3, 3),
                                       padding='same',
                                       strides=(1, 1)
                                       )(A_prev)
    l_FPP = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=he_init
                            )(l_pooling_FPP)
    return K.layers.concatenate([l_F1, l_F3, l_F5, l_FPP])
