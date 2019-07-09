from enum import Enum
from functools import reduce
from typing import Tuple, Optional, List

import tensorflow as tf

from src.model.layers.conv import dim_hold_conv2d


class PoolingFusion(Enum):
    CONCAT = 0
    SUM = 1


def pyramid_pool2d(x: tf.Tensor,
                   windows_shapes: List[int],
                   fuse_filters: int,
                   fuse_kernel: Tuple[int, int] = (3, 3),
                   pooling_type: str = 'AVG',
                   strides: Tuple[int, int] = (2, 2),
                   fusion_method: PoolingFusion = PoolingFusion.CONCAT,
                   name: Optional[str] = None) -> tf.Tensor:

    def __prepare_pooling_layer(layer_parameters: Tuple[int, int]) -> tf.Tensor:
        pool_layer_id, window_shape = layer_parameters
        return pool2d(
            x,
            window_shape=(window_shape, window_shape),
            pooling_type=pooling_type,
            strides=strides,
            name=__pooling_name_assigner(pool_layer_id))

    def __pooling_name_assigner(pool_layer_id: int) -> Optional[str]:
        if name is None:
            return None
        else:
            return f'{name}/pooling_{pool_layer_id}'

    enumerated_shapes = list(enumerate(windows_shapes))
    pooling_layers = list(map(__prepare_pooling_layer, enumerated_shapes))
    if fusion_method is PoolingFusion.CONCAT:
        fusion = tf.concat(pooling_layers, axis=-1)
    else:
        fusion = reduce(lambda acc, elem: acc + elem, pooling_layers)
    fuse_conv_name = f'{name}/fuse_conv' if name is not None else None
    return dim_hold_conv2d(
        x=fusion,
        num_filters=fuse_filters,
        kernel_size=fuse_kernel,
        name=fuse_conv_name)


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
