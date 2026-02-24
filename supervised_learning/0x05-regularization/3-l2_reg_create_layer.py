#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """[summary]

    Args:
        prev ([type]): [description]
        n ([type]): [description]
        activation ([type]): [description]
        lambtha ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=he_init,
                            kernel_regularizer=l2_reg)
    return layer(prev)
