#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as keras


def optimize_model(network, alpha, beta1, beta2):
    """[summary]

    Args:
        network ([type]): [description]
        alpha ([type]): [description]
        beta1 ([type]): [description]
        beta2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    network.compile(optimizer=keras.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
