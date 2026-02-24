#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
import tensorflow as tf


class WindowGenerator:
    """[summary]
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        """[summary]

        Args:
            input_width ([type]): [description]
            label_width ([type]): [description]
            shift ([type]): [description]
            train_df ([type]): [description]
            val_df ([type]): [description]
            test_df ([type]): [description]
            label_columns ([type], optional): [description]. Defaults to None.
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in
                enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = \
            np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = \
            np.arange(self.total_window_size)[self.labels_slice]

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        """[summary]

        Args:
            model ([type], optional): [description]. Defaults to None.
            plot_col (str, optional): [description]. Defaults to 'Close'.
            max_subplots (int, optional): [description]. Defaults to 3.
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel('{} [normed]'.format(plot_col))
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col,
                    None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue
            plt.scatter(self.label_indices,
                        labels[n, :, label_col_index],
                        edgecolors='k',
                        label='Labels',
                        c='#2ca02c',
                        s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices,
                            predictions[n, :, label_col_index],
                            marker='X', edgecolors='k',
                            label='Predictions',
                            c='#ff7f0e', s=64)
            if n == 0:
                plt.legend()
        plt.xlabel('Time [h]')

    def split_window(self, features):
        """[summary]

        Args:
            features ([type]): [description]

        Returns:
            [type]: [description]
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )
        ds = ds.map(self.split_window)
        return ds


def build_model():
    """[summary]

    Returns:
        [type]: [description]
    """
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(24, input_shape=[24, 7], return_sequences=True),
        tf.keras.layers.Dense(units=1)
    ])
    lstm_model.summary()
    return lstm_model


def compile_and_fit(model, window, patience=2, epochs=500):
    """[summary]

    Args:
        model ([type]): [description]
        window ([type]): [description]
        patience (int, optional): [description]. Defaults to 2.
        epochs (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[
                      tf.metrics.MeanAbsoluteError()])
    history = model.fit(
        window.train, epochs=epochs,
        validation_data=window.val,
        callbacks=[early_stopping])
    print(model.summary())
    return history


class Baseline(tf.keras.Model):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, label_index=None):
        """[summary]

        Args:
            label_index ([type], optional): [description]. Defaults to None.
        """
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        """[summary]

        Args:
            inputs ([type]): [description]

        Returns:
            [type]: [description]
        """
        if self.label_index is None:
            return inputs
        result = inputs[
            :, :, self.label_index]
        return result[
            :, :, tf.newaxis]
