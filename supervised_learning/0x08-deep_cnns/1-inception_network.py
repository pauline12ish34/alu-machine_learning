#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """[summary]

    Returns:
        [type]: [description]
    """
    he_init = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    l1_conv = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                              strides=(2, 2),
                              padding='same',
                              activation='relu',
                              kernel_initializer=he_init
                              )(X)
    l1_pooling = K.layers.MaxPool2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2)
                                    )(l1_conv)
    l3_conv = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                              strides=(1, 1),
                              padding='same',
                              activation='relu',
                              kernel_initializer=he_init
                              )(l1_pooling)
    l4_conv = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                              strides=(1, 1),
                              padding='same',
                              activation='relu',
                              kernel_initializer=he_init
                              )(l3_conv)
    l4_pooling = K.layers.MaxPool2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2)
                                    )(l4_conv)
    l5_inception = inception_block(l4_pooling,
                                   [64, 96, 128, 16, 32, 32])
    l6_inception = inception_block(l5_inception,
                                   [128, 128, 192, 32, 96, 64])
    l7_padding = K.layers.MaxPool2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(2, 2)
                                    )(l6_inception)
    l8_inception = inception_block(l7_padding,
                                   [192, 96, 208, 16, 48, 64])
    l9_inception = inception_block(l8_inception,
                                   [160, 112, 224, 24, 64, 64])
    l10_inception = inception_block(l9_inception,
                                    [128, 128, 256, 24, 64, 64])
    l11_inception = inception_block(l10_inception,
                                    [112, 144, 288, 32, 64, 64])
    l12_inception = inception_block(
        l11_inception, [256, 160, 320, 32, 128, 128])
    l13_maxp = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2)
                                  )(l12_inception)
    l14_inception = inception_block(l13_maxp,
                                    [256, 160, 320, 32, 128, 128])
    l15_inception = inception_block(
        l14_inception,
        [384, 192, 384, 48, 128, 128])
    l16_maxp = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same'
                                         )(l15_inception)
    l17_drop = K.layers.Dropout(rate=0.4
                                )(l16_maxp)
    l18_fully = K.layers.Dense(units=1000,
                               activation='softmax',
                               kernel_initializer=he_init)(l17_drop)
    return K.models.Model(inputs=X, outputs=l18_fully)
