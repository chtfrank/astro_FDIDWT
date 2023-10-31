
import os
import numpy as np
import tensorflow as tf


def MLP_MODEL(n_input,
              output_classes,
              hidden_units,
              dropout_list,
              hidden_act_fn='relu',
              output_act_fn='softmax'):

    assert len(hidden_units) == len(dropout_list)

    input = tf.keras.layers.Input(shape=(n_input))
    net = input

    for i in range(len(hidden_units)):
        unit = hidden_units[i]
        dropout = dropout_list[i]

        net = tf.keras.layers.Dense(units=unit)(net)
        net = tf.keras.layers.BatchNormalization(axis=-1)(net)
        net = tf.keras.layers.Activation(hidden_act_fn)(net)
        net = tf.keras.layers.Dropout(rate=dropout)(net)
    
    net = tf.keras.layers.Dense(units=output_classes)(net)
    if output_act_fn:
        net = tf.keras.layers.Activation(output_act_fn)(net)

    return tf.keras.Model(input, net)



def MINI_MATCHBOX_NET(
                    n_input,
                    output_classes=2,

                    B=2,
                    R=1,
                    C=64,

                    prologue_output_channels=128,
                    prologue_block_kernel=3,
                    prologue_block_stride=1,

                    epilogue_output_channels=128,
                    epilogue_block_kernel=3,
                    epilogue_block_stride=1,

                    kernel_sizes=[3, 3],
                    dropout_list=[0.9, 0.9],

                    hidden_act_fn='tanh',
                    output_act_fn='softmax',
                    dropout = 0.9
                    ):
    assert len(kernel_sizes) == len(dropout_list)
    assert len(dropout_list) == B

    input = tf.keras.layers.Input(shape=(n_input))
    net = tf.keras.layers.Reshape((n_input, 1), input_shape=(n_input,))(input)


    # prologue block
    net = tf.keras.layers.Conv1D(
                                filters=prologue_output_channels,
                                kernel_size=prologue_block_kernel,
                                strides=prologue_block_stride,
                                padding='same',
                                data_format='channels_last',
                                activation=None,
                                use_bias=True,
                                )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if hidden_act_fn == 'prelu':
        net = tf.keras.layers.PReLU()(net)
    else:
        net = tf.keras.layers.Activation(hidden_act_fn)(net)



    # intermediate blocks
    for i in range(B):
        residual = tf.keras.layers.Conv1D(
                                    filters=C,
                                    kernel_size=1,
                                    strides=1,
                                    padding='same',
                                    data_format='channels_last',
                                    activation=None,
                                    use_bias=True,
                                    )(net)
        residual = tf.keras.layers.BatchNormalization()(residual)
        for j in range(R):
            net = tf.keras.layers.Conv1D(
                                        filters=C,
                                        kernel_size=kernel_sizes[i],
                                        strides=1,
                                        padding='same',
                                        data_format='channels_last',
                                        activation=None,
                                        use_bias=True,
                                        )(net)
            net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Add()([net, residual])
        if hidden_act_fn == 'prelu':
            net = tf.keras.layers.PReLU()(net)
        else:
            net = tf.keras.layers.Activation(hidden_act_fn)(net)
        net = tf.keras.layers.Dropout(rate=dropout_list[i])(net)



    # the epilogue layers
    net = tf.keras.layers.Conv1D(
                                filters=epilogue_output_channels,
                                kernel_size=epilogue_block_kernel,
                                strides=epilogue_block_stride,
                                dilation_rate=1,
                                padding='same',
                                data_format='channels_last',
                                activation=None,
                                use_bias=True,
                                )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if hidden_act_fn == 'prelu':
        net = tf.keras.layers.PReLU()(net)
    else:
        net = tf.keras.layers.Activation(hidden_act_fn)(net)

    net = tf.keras.layers.Conv1D(
                                filters=128,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format='channels_last',
                                activation=None,
                                use_bias=True,
                                )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if hidden_act_fn == 'prelu':
        net = tf.keras.layers.PReLU()(net)
    else:
        net = tf.keras.layers.Activation(hidden_act_fn)(net)

    net = tf.keras.layers.Conv1D(
                                filters=output_classes,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format='channels_last',
                                activation=None,
                                use_bias=True,
                                )(net)
    net = tf.keras.layers.BatchNormalization()(net)
    if hidden_act_fn == 'prelu':
        net = tf.keras.layers.PReLU()(net)
    else:
        net = tf.keras.layers.Activation(hidden_act_fn)(net)

    net = tf.keras.layers.Flatten()(net)

    net = tf.keras.layers.Dense(units=output_classes)(net)
    if output_act_fn:
        net = tf.keras.layers.Activation(output_act_fn)(net)


    return tf.keras.Model(input, net)


