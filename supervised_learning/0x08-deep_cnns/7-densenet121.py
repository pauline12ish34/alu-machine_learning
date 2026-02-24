#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """[summary]

    Args:
        growth_rate (int, optional): [description]. Defaults to 32.
        compression (float, optional): [description]. Defaults to 1.0.

    Returns:
        [type]: [description]
    """
    # input_ = K.Input(shape=(224, 224, 3))
    # he_ini = K.initializers.he_normal()
    # layers = [12, 24, 16]

    # l = K.layers.BatchNormalization(axis=3)(input_)
    # l = K.layers.Activation('relu')(l)
    # l = K.layers.Conv2D(filters=2 * growth_rate,
    #                     kernel_size=(7, 7),
    #                     strides=(2, 2),
    #                     padding='same',
    #                     kernel_initializer=he_ini
    #                     )(l)
    # l = K.layers.MaxPool2D(pool_size=(3, 3),
    #                        padding='same',
    #                        strides=(2, 2)
    #                        )(l)

    # nb_filters = 2 * growth_rate
    # l, nb_filters = dense_block(l, nb_filters, growth_rate, 6)

    # for la in layers:
    #     l, nb_filters = transition_layer(l, nb_filters, compression)
    #     l, nb_filters = dense_block(l, nb_filters, growth_rate, la)
    # l = K.layers.AveragePooling2D(pool_size=(7, 7),
    #                               padding='same'
    #                               )(l)
    # l = K.layers.Dense(units=1000,
    #                    activation='softmax',
    #                    kernel_initializer=he_ini
    #                    )(l)
    # m = K.models.Model(inputs=input_, outputs=l)
    # return m
    he_normal = K.initializers.he_normal()
    layers = [12, 24, 16]
    nb_filters = 2 * growth_rate

    x = K.Input(shape=(224, 224, 3))

    conv = K.layers.BatchNormalization(axis=-1)(x)
    conv = K.layers.Activation('relu')(conv)
    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           padding='same',
                           kernel_initializer=he_normal)(conv)

    max_pool = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(conv)

    layer, nb_filters = dense_block(max_pool,
                                    nb_filters,
                                    growth_rate,
                                    6)
    for num_layers in layers:
        layer, nb_filters = transition_layer(layer,
                                             nb_filters,
                                             compression)
        layer, nb_filters = dense_block(layer,
                                        nb_filters,
                                        growth_rate,
                                        num_layers)

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(layer)

    out = K.layers.Dense(units=1000,
                         activation='softmax',
                         kernel_initializer=he_normal)(avg_pool)

    model = K.Model(inputs=x, outputs=out)

    return model
