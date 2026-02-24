#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """[summary]

    Args:
        prev ([type]): [description]
        n ([type]): [description]
        activation ([type]): [description]
        keep_prob ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=he_init,
                            kernel_regularizer=dropout)
    return layer(prev)
