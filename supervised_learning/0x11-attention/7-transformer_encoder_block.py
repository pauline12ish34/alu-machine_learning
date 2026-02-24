#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """[summary]

        Args:
            dm ([type]): [description]
            h ([type]): [description]
            hidden ([type]): [description]
            drop_rate (float, optional): [description]. Defaults to 0.1.
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """[summary]

        Args:
            x ([type]): [description]
            training ([type]): [description]
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(
            attn_output,
            training=training)
        out1 = self.layernorm1(x + attn_output)
        tpn = self.dense_hidden(out1)
        tpn = self.dense_output(tpn)
        tpn = self.dropout2(tpn, training=training)
        return self.layernorm2(out1 + tpn)
