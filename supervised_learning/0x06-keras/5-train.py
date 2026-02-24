#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """[summary]

    Args:
        network ([type]): [description]
        data ([type]): [description]
        labels ([type]): [description]
        batch_size ([type]): [description]
        epochs ([type]): [description]
        validation_data ([type], optional): [description]. Defaults to None.
        verbose (bool, optional): [description]. Defaults to True.
        shuffle (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       verbose=verbose, shuffle=shuffle
                       )
