#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """[summary]

    Args:
        loss ([type]): [description]
        alpha ([type]): [description]
        beta1 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
