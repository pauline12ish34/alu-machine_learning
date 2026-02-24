#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.compat.v2 as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, d_model, warmup_steps=4000):
        """[summary]

        Args:
            d_model ([type]): [description]
            warmup_steps (int, optional): [description]. Defaults to 4000.
        """
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """[summary]

        Args:
            step ([type]): [description]

        Returns:
            [type]: [description]
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """[summary]

    Args:
        N ([type]): [description]
        dm ([type]): [description]
        h ([type]): [description]
        hidden ([type]): [description]
        max_len ([type]): [description]
        batch_size ([type]): [description]
        epochs ([type]): [description]

    Returns:
        [type]: [description]
    """
    data = Dataset(batch_size, max_len)
    learning_rate = CustomSchedule(dm)
    optimizer = \
        tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                 epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    input_vocab_size = data.tokenizer_pt.vocab_size + 2
    target_vocab_size = data.tokenizer_en.vocab_size + 2
    transformer = \
        Transformer(N=N, dm=dm, h=h,
                    hidden=hidden,
                    input_vocab=input_vocab_size,
                    target_vocab=target_vocab_size,
                    max_seq_input=max_len,
                    max_seq_target=max_len)

    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = \
            create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions = transformer(inp, tar_inp,
                                      True,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(data.data_train):
            train_step(inp, tar)
            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch,
                    train_loss.result(), train_accuracy.result()))
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.
              format(epoch + 1,
                     train_loss.result(),
                     train_accuracy.result()))
