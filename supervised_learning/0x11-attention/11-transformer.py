#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """[summary]

        Args:
            N ([type]): [description]
            dm ([type]): [description]
            h ([type]): [description]
            hidden ([type]): [description]
            input_vocab ([type]): [description]
            target_vocab ([type]): [description]
            max_seq_input ([type]): [description]
            max_seq_target ([type]): [description]
            drop_rate (float, optional): [description]. Defaults to 0.1.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N,
                               dm, h, hidden,
                               input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm,
                               h, hidden, target_vocab,
                               max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """[summary]

        Args:
            inputs ([type]): [description]
            target ([type]): [description]
            training ([type]): [description]
            encoder_mask ([type]): [description]
            look_ahead_mask ([type]): [description]
            decoder_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        return self.linear(self.decoder(
            target, self.encoder(inputs,
                                 training, encoder_mask),
            training,
            look_ahead_mask, decoder_mask))
