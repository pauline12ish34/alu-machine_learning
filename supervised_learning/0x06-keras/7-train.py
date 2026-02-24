#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
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
        learning_rate_decay (bool, optional): [description]. Defaults to False.
        alpha (float, optional): [description]. Defaults to 0.1.
        decay_rate (int, optional): [description]. Defaults to 1.
        verbose (bool, optional): [description]. Defaults to True.
        shuffle (bool, optional): [description]. Defaults to False.
    """
    def decay(epoch):
        """[summary]

        Args:
            epoch ([type]): [description]

        Returns:
            [type]: [description]
        """
        return alpha / (1 + decay_rate * epoch)

    callback = []

    if validation_data:

        if early_stopping:
            es = K.callbacks.EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=patience)
            callback.append(es)
        if learning_rate_decay:
            lrd = K.callbacks.LearningRateScheduler(decay, verbose=1)
            callback.append(lrd)
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callback)
