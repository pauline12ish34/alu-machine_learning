#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """[summary]

    Args:
        X ([type]): [description]
        Y ([type]): [description]
        save_path ([type]): [description]
    prediction, accuracy, and loss
    Returns:
        [type]: [description]
    """
    saved = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        saved.restore(sess, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        y_pred, accuracy, loss = sess.run(
            [y_pred, accuracy, loss],
            feed_dict={x: X, y: Y})
        return y_pred, accuracy, loss
