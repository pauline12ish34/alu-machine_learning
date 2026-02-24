#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
    """
import tensorflow as tf


def create_train_op(loss, alpha):
    """[summary]

    Args:
        loss ([type]): [description]
        alpha ([type]): [description]

    Returns:
        [type]: [description]
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
