#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """[summary]

    Args:
        prev ([type]): [description]
        n ([type]): [description]
        activation ([type]): [description]

    Returns:
        [type]: [description]
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)
    mt, vt = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    zt = tf.nn.batch_normalization(z, mt, vt, beta, gamma, 1e-8)
    y_pred = activation(zt)
    return y_pred
