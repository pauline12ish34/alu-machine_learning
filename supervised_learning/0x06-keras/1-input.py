#!/usr/bin/env python3
"""
Buiding a model with Keras using Input
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
    inputs = keras.Input(shape=(nx,))
    regularizer = keras.regularizers.l2(lambtha)
    outputs = keras.layers.Dense(layers[0], input_shape=(nx,),
                                 activation=activations[0],
                                 kernel_regularizer=regularizer)(inputs)
    for layer in range(1, len(layers)):
        dname = 'dense_' + str(layer)
        dropout = keras.layers.Dropout(rate=(1 - keep_prob))(outputs)
        outputs = keras.layers.Dense(layers[layer],
                                     activation=activations[layer],
                                     kernel_regularizer=regularizer,
                                     name=dname)(dropout)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    return model
