#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, vocab, embedding, units, batch):
        """[summary]

        Args:
            vocab ([type]): [description]
            embedding ([type]): [description]
            units ([type]): [description]
            batch ([type]): [description]
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )

    def call(self, x, initial):
        """[summary]

        Args:
            x ([type]): [description]
            initial ([type]): [description]

        Returns:
            [type]: [description]
        """
        embeddings = self.embedding(x)
        outputs, hidden = self.gru(
            embeddings,
            initial_state=initial
        )
        return outputs, hidden

    def initialize_hidden_state(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        init_t = tf.keras.initializers.Zeros()
        return init_t(
            shape=(self.batch, self.units)
        )
