#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, units):
        """[summary]

        Args:
            units ([type]): [description]
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(
            units=units
        )
        self.U = tf.keras.layers.Dense(
            units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """[summary]

        Args:
            s_prev ([type]): [description]
            hidden_states ([type]): [description]

        Returns:
            [type]: [description]
        """
        score = self.V(tf.nn.tanh(self.W(
            tf.expand_dims(s_prev, axis=1)
        ) + self.U(hidden_states)))
        w = tf.nn.softmax(
            score,
            axis=1
        )
        return tf.reduce_sum(
            w * hidden_states,
            axis=1), w
