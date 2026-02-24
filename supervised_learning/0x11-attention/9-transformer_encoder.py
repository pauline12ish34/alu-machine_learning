#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """[summary]

        Args:
            N ([type]): [description]
            dm ([type]): [description]
            h ([type]): [description]
            hidden ([type]): [description]
            input_vocab ([type]): [description]
            max_seq_len ([type]): [description]
            drop_rate (float, optional): [description]. Defaults to 0.1.
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_vocab, dm)
        self.positional_encoding = positional_encoding(
            max_seq_len,
            self.dm
        )
        self.blocks = [EncoderBlock(dm,
                                    h,
                                    hidden,
                                    drop_rate
                                    )
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """[summary]

        Args:
            x ([type]): [description]
            training ([type]): [description]
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x
