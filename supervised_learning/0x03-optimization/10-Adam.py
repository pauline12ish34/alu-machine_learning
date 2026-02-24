#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """[summary]

    Args:
        loss ([type]): [description]
        alpha ([type]): [description]
        beta1 ([type]): [description]
        beta2 ([type]): [description]
        epsilon ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1, beta2=beta2,
                                  epsilon=epsilon).minimize(loss)
