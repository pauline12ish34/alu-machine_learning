#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """[summary]

    Returns:
        [type]: [description]
    """
    X = K.Input(shape=(224, 224, 3))
    he_init = K.initializers.he_normal()

    l1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=he_init
                         )(X)
    l1 = K.layers.BatchNormalization(axis=3)(l1)
    l1 = K.layers.Activation('relu')(l1)

    l2 = K.layers.MaxPool2D(pool_size=(3, 3),
                            padding='same',
                            strides=(2, 2)
                            )(l1)

    combo1 = projection_block(l2, [64, 64, 256], 1)
    for i in range(2):
        combo1 = identity_block(combo1, [64, 64, 256])

    comb2 = projection_block(combo1, [128, 128, 512])
    for i in range(3):
        comb2 = identity_block(comb2, [128, 128, 512])

    combo3 = projection_block(comb2, [256, 256, 1024])
    for i in range(5):
        combo3 = identity_block(combo3, [256, 256, 1024])

    combo4 = projection_block(combo3, [512, 512, 2048])
    for i in range(2):
        combo4 = identity_block(combo4, [512, 512, 2048])

    pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                     padding='same'
                                     )(combo4)

    soft = K.layers.Dense(units=1000,
                          activation='softmax',
                          kernel_initializer=he_init
                          )(pool)

    return K.models.Model(inputs=X, outputs=soft)
