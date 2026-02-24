#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
import tensorflow.compat.v2 as tf


def positional_encoding(max_seq_len, dm):
    """[summary]

    Args:
        max_seq_len ([type]): [description]
        dm ([type]): [description]

    Returns:
        [type]: [description]
    """
    t = np.arange(max_seq_len)[:, np.newaxis]
    index = np.arange(dm)[np.newaxis, :]
    dm_float = np.float32(dm)
    W = 1 / (np.power(10000, (2 * (index // 2) / dm_float)))
    Wt = (W * t)
    positional_vect = np.zeros((max_seq_len, dm))
    positional_vect[:, 0::2] = np.sin(Wt[:, 0::2])
    positional_vect[:, 1::2] = np.cos(Wt[:, 1::2])
    return positional_vect


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
    dk = tf.shape(Q)[-1]
    dk_float = tf.cast(dk, tf.float32)
    scaled = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk_float)
    if mask is not None:
        scaled += (mask * -1e9)
    weights = tf.nn.softmax(scaled, axis=-1)
    output = tf.matmul(weights, V)
    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, dm, h):
        """[summary]

        Args:
            dm ([type]): [description]
            h ([type]): [description]
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)

        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """[summary]

        Args:
            x ([type]): [description]
            batch_size ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """[summary]

        Args:
            Q ([type]): [description]
            K ([type]): [description]
            V ([type]): [description]
            mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        batch_size = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)
        q = self.split_heads(Q, batch_size)
        k = self.split_heads(K, batch_size)
        v = self.split_heads(V, batch_size)
        scaled_attention, weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))
        output = self.linear(concat_attention)
        return output, weights


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
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class for machine translation"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """[summary]

        Args:
            dm ([type]): [description]
            h ([type]): [description]
            hidden ([type]): [description]
            drop_rate (float, optional): [description]. Defaults to 0.1.
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """[summary]

        Args:
            x ([type]): [description]
            encoder_output ([type]): [description]
            training ([type]): [description]
            look_ahead_mask ([type]): [description]
            padding_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(out1,
                                               encoder_output,
                                               encoder_output,
                                               padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Encoder(tf.keras.layers.Layer):
    """Encoder class for machine translation"""

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
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len,
                                                       self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
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


class Decoder(tf.keras.layers.Layer):
    """Decoder class for machine translation"""

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """[summary]

        Args:
            N ([type]): [description]
            dm ([type]): [description]
            h ([type]): [description]
            hidden ([type]): [description]
            target_vocab ([type]): [description]
            max_seq_len ([type]): [description]
            drop_rate (float, optional): [description]. Defaults to 0.1.
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """[summary]

        Args:
            x ([type]): [description]
            encoder_output ([type]): [description]
            training ([type]): [description]
            look_ahead_mask ([type]): [description]
            padding_mask ([type]): [description]

        Returns:
            [type]: [description]
        """
        seq_len = x.shape[1]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
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
        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
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
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output