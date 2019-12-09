from enum import Enum
from typing import Tuple, Optional

import tensorflow as tf


class FusionMethod(Enum):
    CONCAT = 0
    SUM = 1


def max_pool2d(x: tf.Tensor,
               window_shape: Tuple[int, int],
               padding: str = 'SAME',
               strides: Tuple[int, int] = (2, 2),
               name: Optional[str] = None,
               **kwargs) -> tf.Tensor:
    return pool2d(
        x=x,
        window_shape=window_shape,
        pooling_type='MAX',
        padding=padding,
        strides=strides,
        name=name,
        *kwargs)


def avg_pool2d(x: tf.Tensor,
               window_shape: Tuple[int, int],
               padding: str = 'SAME',
               strides: Tuple[int, int] = (2, 2),
               name: Optional[str] = None,
               **kwargs) -> tf.Tensor:
    return pool2d(
        x=x,
        window_shape=window_shape,
        pooling_type='AVG',
        padding=padding,
        strides=strides,
        name=name,
        *kwargs)


def atrous_max_pool2d(x: tf.Tensor,
                      window_shape: Tuple[int, int],
                      padding: str = 'SAME',
                      dilation_rate: Tuple[int, int] = (2, 2),
                      name: Optional[str] = None,
                      **kwargs) -> tf.Tensor:
    return pool2d(
        x=x,
        window_shape=window_shape,
        pooling_type='MAX',
        padding=padding,
        dilation_rate=dilation_rate,
        name=name,
        *kwargs)


def atrous_avg_pool2d(x: tf.Tensor,
                      window_shape: Tuple[int, int],
                      padding: str = 'SAME',
                      dilation_rate: Tuple[int, int] = (2, 2),
                      name: Optional[str] = None,
                      **kwargs) -> tf.Tensor:
    return pool2d(
        x=x,
        window_shape=window_shape,
        pooling_type='AVG',
        padding=padding,
        dilation_rate=dilation_rate,
        name=name,
        *kwargs)


def pool2d(x: tf.Tensor,
           window_shape: Tuple[int, int],
           pooling_type: str,
           padding: str = 'SAME',
           strides: Optional[Tuple[int, int]] = None,
           dilation_rate: Optional[Tuple[int, int]] = None,
           name: Optional[str] = None,
           **kwargs) -> tf.Tensor:
    return tf.nn.pool(
        input=x,
        window_shape=window_shape,
        pooling_type=pooling_type,
        padding=padding,
        strides=strides,
        dilation_rate=dilation_rate,
        name=name,
        *kwargs
    )
