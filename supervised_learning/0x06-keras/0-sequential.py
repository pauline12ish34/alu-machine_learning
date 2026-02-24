#!/usr/bin/env python3
"""
Building a model with Keras
"""
import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """[summary]

    Args:
        nx ([type]): [description]
        layers ([type]): [description]
        activations ([type]): [description]
        lambtha ([type]): [description]
        keep_prob ([type]): [description]

    Returns:
        [type]: [description]
    """
    nn = keras.Sequential()
    l2 = keras.regularizers.l2(lambtha)
    nn.add(keras.layers.Dense(layers[0], input_shape=(nx,),
                              activation=activations[0],
                              kernel_regularizer=l2))
    for x in range(1, len(layers)):
        nn.add(keras.layers.Dropout(1 - keep_prob))
        nn.add(keras.layers.Dense(layers[x],
                                  activation=activations[x],
                                  kernel_regularizer=l2))

    return nn
