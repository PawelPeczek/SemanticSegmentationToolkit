from typing import Optional, Tuple

import tensorflow as tf


def bottleneck_conv2d(x: tf.Tensor,
                      num_filters: int,
                      activation: Optional[str] = 'relu',
                      padding: str = 'SAME',
                      name: Optional[str] = None,
                      **kwargs) -> tf.Tensor:
    """
    Convolution layer that change (usually reduce) filter numbers
    while preserving spatial dimensions.
    """
    return tf.layers.conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=(1, 1),
        padding=padding,
        activation=activation,
        name=name,
        *kwargs)


def downsample_conv2d(x: tf.Tensor,
                      num_filters: int,
                      kernel_size: Tuple[int, int],
                      strides: Tuple[int, int] = (2, 2),
                      activation: Optional[str] = 'relu',
                      name: Optional[str] = None,
                      padding: str = 'SAME',
                      **kwargs) -> tf.Tensor:
    """
    Convolution layer that reduce the spatial dimensions.
    """
    return tf.layers.conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=name,
        *kwargs)


def dim_hold_conv2d(x: tf.Tensor,
                    num_filters: int,
                    kernel_size: Tuple[int, int],
                    activation: Optional[str] = 'relu',
                    padding: str = 'SAME',
                    name: Optional[str] = None,
                    **kwargs) -> tf.Tensor:
    """
    Convolution layer that holds the spatial dimensions while
    performing convolution operation with given kernel.
    """
    return tf.layers.conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        name=name,
        *kwargs)


def atrous_conv2d(x: tf.Tensor,
                  num_filters: int,
                  kernel_size: Tuple[int, int],
                  dilation_rate: Tuple[int, int] = (2, 2),
                  activation: Optional[str] = 'relu',
                  padding: str = 'SAME',
                  name: Optional[str] = None,
                  **kwargs) -> tf.Tensor:
    return tf.layers.conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        activation=activation,
        name=name,
        *kwargs)


def separable_bottleneck_conv2d(x: tf.Tensor,
                                num_filters: int,
                                activation: Optional[str] = 'relu',
                                padding: str = 'SAME',
                                name: Optional[str] = None,
                                **kwargs) -> tf.Tensor:
    return separable_conv2d(x=x,
                            num_filters=num_filters,
                            kernel_size=(1, 1),
                            activation=activation,
                            padding=padding,
                            name=name,
                            *kwargs)


def separable_conv2d(x: tf.Tensor,
                     num_filters: int,
                     kernel_size: Tuple[int, int],
                     strides: Tuple[int, int] = (1, 1),
                     activation: Optional[str] = 'relu',
                     name: Optional[str] = None,
                     padding: str = 'SAME',
                     **kwargs) -> tf.Tensor:
    return tf.layers.separable_conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=name,
        *kwargs)


def atrous_separable_conv2d(x: tf.Tensor,
                            num_filters: int,
                            kernel_size: Tuple[int, int],
                            dilation_rate: Tuple[int, int] = (2, 2),
                            activation: Optional[str] = 'relu',
                            padding: str = 'SAME',
                            name: Optional[str] = None,
                            **kwargs) -> tf.Tensor:
    return tf.layers.separable_conv2d(
        inputs=x,
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        activation=activation,
        name=name,
        *kwargs)
