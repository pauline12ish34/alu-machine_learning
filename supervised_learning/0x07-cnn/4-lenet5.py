#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def lenet5(x, y):
    """[summary]

    Args:
        x ([type]): [description]
        y ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_ini = tf.contrib.layers.variance_scaling_initializer()

    l1_conv = tf.layers.Conv2D(filters=6,
                               kernel_size=(5, 5),
                               padding='same',
                               activation=tf.nn.relu,
                               kernel_initializer=he_ini)(x)
    l1_pool = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                     strides=(2, 2))(l1_conv)

    l2_conv = tf.layers.Conv2D(filters=16,
                               kernel_size=(5, 5),
                               padding='valid',
                               activation=tf.nn.relu,
                               kernel_initializer=he_ini)(l1_pool)
    l2_pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(l2_conv)

    flat_layer = tf.layers.Flatten()(l2_pool)

    l3_fully_c = tf.layers.Dense(units=120,
                                 activation=tf.nn.relu,
                                 kernel_initializer=he_ini)(flat_layer)

    l4_fullyc = tf.layers.Dense(units=84,
                                activation=tf.nn.relu,
                                kernel_initializer=he_ini)(l3_fully_c)

    l5_fully = tf.layers.Dense(units=10,
                               kernel_initializer=he_ini)(l4_fullyc)

    y_hat = tf.nn.softmax(l5_fully)
    y_hat_tag = tf.argmax(l5_fully, 1)
    pred = tf.argmax(y, 1)
    accu = tf.equal(y_hat_tag, pred)
    acc = tf.reduce_mean(tf.cast(accu, tf.float32))
    loss = tf.losses.softmax_cross_entropy(y, l5_fully)
    trainer = tf.train.AdamOptimizer().minimize(loss)
    return y_hat, trainer, loss, acc
