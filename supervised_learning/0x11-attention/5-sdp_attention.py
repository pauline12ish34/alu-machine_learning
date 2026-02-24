#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """[summary]

    Args:
        Q ([type]): [description]
        K ([type]): [description]
        V ([type]): [description]
        mask ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(
        tf.cast(tf.shape(Q)[-1],
                tf.float32))
    if mask is not None:
        scaled += (mask * -1e9)
    w = tf.nn.softmax(
        scaled, axis=-1)
    return tf.matmul(w, V), w
