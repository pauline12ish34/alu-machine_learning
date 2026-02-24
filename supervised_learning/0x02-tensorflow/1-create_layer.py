#!/usr/bin/env python3
"""tensorflow 1"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """tensorflow 1"""
    he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, kernel_initializer=he_init,
        activation=activation, name='layer')
    l_output = layer(prev)

    return l_output
