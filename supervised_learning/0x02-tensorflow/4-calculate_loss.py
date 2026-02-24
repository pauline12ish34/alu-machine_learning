#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
    """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """[summary]

    Args:
        y ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)
