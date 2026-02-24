#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """[summary]

    Args:
        network ([type]): [description]
        data ([type]): [description]
        labels ([type]): [description]
        batch_size ([type]): [description]
        epochs ([type]): [description]
        validation_data ([type], optional): [description]. Defaults to None.
        early_stopping (bool, optional): [description]. Defaults to False.
        patience (int, optional): [description]. Defaults to 0.
        verbose (bool, optional): [description]. Defaults to True.
        shuffle (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    callback = []
    if validation_data and early_stopping:
        c = K.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience)
        callback.append(c)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callback)
