from functools import reduce
from typing import Tuple, Optional, List

import tensorflow as tf

from src.model.layers.conv import dim_hold_conv2d
from src.model.layers.pooling import FusionMethod, pool2d


def pyramid_pool_fusion(x: tf.Tensor,
                        windows_shapes: List[int],
                        fuse_filters: int,
                        fuse_kernel: Tuple[int, int] = (3, 3),
                        pooling_type: str = 'AVG',
                        pooling_strides: Tuple[int, int] = (2, 2),
                        fusion_method: FusionMethod = FusionMethod.CONCAT,
                        name: Optional[str] = None) -> tf.Tensor:
    """
    Layer performs pooling operation with rectangular kernels according to
    :windows_shapes parameter. After that - pooling results are being
    concatenated along channel dimension and fusion operation (conv2d with
    strides=(1,1) and kernel size given by :fuse_kernel) is being applied.
    """

    def __prepare_pooling_layer(layer_parameters: Tuple[int, int]) -> tf.Tensor:
        pool_layer_id, window_shape = layer_parameters
        return pool2d(
            x,
            window_shape=(window_shape, window_shape),
            pooling_type=pooling_type,
            strides=pooling_strides,
            name=__pooling_name_assigner(pool_layer_id))

    def __pooling_name_assigner(pool_layer_id: int) -> Optional[str]:
        if name is None:
            return None
        else:
            return f'{name}/pooling_{pool_layer_id}'

    enumerated_shapes = list(enumerate(windows_shapes))
    pooling_layers = list(map(__prepare_pooling_layer, enumerated_shapes))
    if fusion_method is FusionMethod.CONCAT:
        fusion = tf.concat(pooling_layers, axis=-1)
    else:
        fusion = reduce(lambda acc, elem: acc + elem, pooling_layers)
    fuse_conv_name = f'{name}/fuse_conv' if name is not None else None
    return dim_hold_conv2d(
        x=fusion,
        num_filters=fuse_filters,
        kernel_size=fuse_kernel,
        name=fuse_conv_name)
