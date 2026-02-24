#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
    """
import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """[summary]
    Args:
        X ([type]): [description]
        Y ([type]): [description]
    Returns:
        [type]: [description]
    """
    shuff = np.random.permutation(X.shape[0])
    return X[shuff], Y[shuff]


def create_placeholders(nx, classes):
    """tensorflow"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return (x, y)


def create_layer(prev, n, activation):
    """tensorflow 1"""
    he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n, kernel_initializer=he_init,
        activation=activation, name='layer')
    l_output = layer(prev)

    return l_output


def forward_prop(x, layer_sizes, activations, epsilon=1e-8):
    """[summary]

    Args:
        x ([type]): [description]
        layer_sizes ([type]): [description]
        activations ([type]): [description]
        epsilon ([type], optional): [description]. Defaults to 1e-8.

    Returns:---
        [type]: [description]
    """
    for i in range(len(layer_sizes)):
        if i < len(layer_sizes) - 1:
            layer = create_batch_norm_layer(x, layer_sizes[i], activations[i],
                                            epsilon=1e-8)
        else:
            layer = create_layer(x, layer_sizes[i], activations[i])
        x = layer
    return layer


def calculate_accuracy(y, y_pred):
    """[summary]
    Args:
        y ([type]): [description]
        y_pred ([type]): [description]
    Returns:
        [type]: [description]---
    """
    y = tf.argmax(y, axis=1)
    y_hat = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y, y_hat), dtype=tf.float32))


def calculate_loss(y, y_pred):
    """[summary]
    Args:
        y ([type]): [description]
        y_pred ([type]): [description]
    Returns:
        [type]: [description] ---
    """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """[summary]
    Args:
        loss ([type]): [description]
        alpha ([type]): [description]
        beta1 ([type]): [description]
        beta2 ([type]): [description]
        epsilon ([type]): [description]
    Returns:
        [type]: [description] ---
    """
    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1, beta2=beta2,
                                  epsilon=epsilon).minimize(loss)


def create_batch_norm_layer(prev, n, activation, epsilon=1e-8):
    """[summary]

    Args:
        prev ([type]): [description]
        n ([type]): [description]
        activation ([type]): [description]
        epsilon ([type], optional): [description]. Defaults to 1e-8.

    Returns:
        [type]: [description]
    """

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)
    mt, vt = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    zt = tf.nn.batch_normalization(z, mt, vt, beta, gamma, epsilon)
    y_pred = activation(zt)
    return y_pred


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """[summary]

    Args:
        alpha ([type]): [description]
        decay_rate ([type]): [description]
        global_step ([type]): [description]
        decay_step ([type]): [description]

    Returns:
        [type]: [description]
    """

    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)


def get_batch(t, batch_size):
    """[summary]
    Args:
        t ([type]): [description]
        batch_size ([type]): [description]
    Returns:
        [type]: [description]
    """

    batch_list = []
    i = 0
    m = t.shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    for b in range(batches):
        if b != batches - 1:
            batch_list.append(t[i:(i + batch_size)])
        else:
            batch_list.append(t[i:])
        i += batch_size

    return batch_list


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Function that builds, trains, and saves a NN model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization:
    Arguments:
     - Data_train is a tuple containing the training inputs and training
        labels, respectively
     - Data_valid is a tuple containing the validation inputs and
        validation labels, respectively
     - layers is a list containing the number of nodes in each layer of
        the network
     - activation is a list containing the activation functions used
        for each layer of the network
     - alpha is the learning rate
     - beta1 is the weight for the first moment of Adam Optimization
     - beta2 is the weight for the second moment of Adam Optimization
     - epsilon is a small number used to avoid division by zero
     - decay_rate is the decay rate for inverse time decay of the
        learning rate (the corresponding decay step should be 1)
     - batch_size is the number of data points that should be in a mini-batch
     - epochs is the number of times the training should pass through
        the whole dataset
     - save_path is the path where the model should be saved to
    Returns:
     The path where the model was saved
    """

    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layers, activations, epsilon)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    m = Data_train[0].shape[0]
    batches = int(m / batch_size) + (m % batch_size > 0)

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign_add(global_step, 1,
                                          name='increment_global_step')
    alpha = learning_rate_decay(alpha, decay_rate, global_step, batches)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(epochs + 1):
            x_t, y_t = shuffle_data(X_train, Y_train)
            loss_t, acc_t = sess.run((loss, accuracy),
                                     feed_dict={x: X_train, y: Y_train})
            loss_v, acc_v = sess.run((loss, accuracy),
                                     feed_dict={x: X_valid, y: Y_valid})
            print('After {} epochs:'.format(e))
            print('\tTraining Cost: {}'.format(loss_t))
            print('\tTraining Accuracy: {}'.format(acc_t))
            print('\tValidation Cost: {}'.format(loss_v))
            print('\tValidation Accuracy: {}'.format(acc_v))

            if e < epochs:
                X_batch_t = get_batch(x_t, batch_size)
                Y_batch_t = get_batch(y_t, batch_size)
                for b in range(1, len(X_batch_t) + 1):
                    sess.run((increment_global_step, train_op),
                             feed_dict={x: X_batch_t[b - 1],
                                        y: Y_batch_t[b - 1]})
                    loss_t, acc_t = sess.run((loss, accuracy),
                                             feed_dict={x: X_batch_t[b - 1],
                                                        y: Y_batch_t[b - 1]})
                    if not b % 100:
                        print('\tStep {}:'.format(b))
                        print('\t\tCost: {}'.format(loss_t))
                        print('\t\tAccuracy: {}'.format(acc_t))
        save_path = saver.save(sess, save_path)
    return save_path
