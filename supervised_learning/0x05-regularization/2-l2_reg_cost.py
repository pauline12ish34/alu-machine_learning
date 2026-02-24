#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """[summary]

    Args:
        cost ([type]): [description]

    Returns:
        [type]: [description]
    """
    return cost + tf.losses.get_regularization_losses()
