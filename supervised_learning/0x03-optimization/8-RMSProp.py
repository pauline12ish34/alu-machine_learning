#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """[summary]

    Args:
        loss ([type]): [description]
        alpha ([type]): [description]
        beta2 ([type]): [description]
        epsilon ([type]): [description]

    Returns:
        [type]: [description]
    """
    RMSProp = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                        decay=beta2,
                                        epsilon=epsilon).minimize(loss)
    return RMSProp
