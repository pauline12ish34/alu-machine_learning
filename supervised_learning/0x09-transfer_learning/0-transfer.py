#!/usr/bin/env python3
import tensorflow.keras as K


def process_image(X):
    return K.backend.resize_images(X, 7, 7,
                                   data_format="channels_last",
                                   interpolation='bilinear')


def scale_data(X, Y):
    X = K.applications.resnet50.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y


if __name__ == "__main__":
    (X_train, Y_train), (
        X_test, Y_test) = K.datasets.cifar10.load_data()
    Adam_op = K.optimizers.Adam(lr=0.00001)
    ResNet50_model = K.applications.ResNet50(weights='imagenet',
                                             include_top=False,
                                             input_shape=(224, 224, 3))
    ResNet50_model.trainable = False
    input_m = K.Input(shape=(32, 32, 3))

    layer = ResNet50_model(K.layers.Lambda(
        process_image)(input_m), training=False)

    layer = K.layers.Flatten()(layer)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Dense(130, activation='relu'
                           )(layer)
    layer = K.layers.Dropout(0.5)(layer)
    layer = K.layers.BatchNormalization()(layer)

    layer = K.layers.Dense(60, activation='relu'
                           )(layer)
    layer = K.layers.Dropout(0.5)(layer)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Dense(10, activation='softmax'
                           )(layer)

    model = K.Model(input_m, layer)
    model.compile(optimizer=Adam_op,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x_train, t_train = scale_data(X_train, Y_train)
    x_test, y_test = scale_data(X_test, Y_test)

    call_back = [K.callbacks.ModelCheckpoint(
        filepath='cifar10.h5', save_best_only=True)]
    model.fit(x_train,
              t_train,
              batch_size=256,
              validation_data=(x_test, y_test),
              epochs=20,
              callbacks=call_back)
