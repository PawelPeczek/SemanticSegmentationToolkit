from functools import reduce, partial
from typing import Tuple, Optional, List, Union

import tensorflow as tf

from src.model.building_blocks.utils import prepare_block_operation_name
from src.model.layers.conv import dim_hold_conv2d
from src.model.layers.interpolation import resize_bilinear
from src.model.layers.pooling import FusionMethod, pool2d, avg_pool2d

PyramidPoolingConfig = List[Tuple[Tuple[int, int], Tuple[int, int]]]
SizeElement = Union[int, tf.Dimension]
Size = Tuple[SizeElement, SizeElement]


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


def pyramid_pooling(x: tf.Tensor,
                    pooling_config: PyramidPoolingConfig,
                    output_size: Size,
                    name: Optional[str] = None) -> tf.Tensor:
    pooling_heads = []
    for head_id, (window_shape, strides) in enumerate(pooling_config):
        pooling_head = _pyramid_pooling_head(
            x=x,
            window_shape=window_shape,
            strides=strides,
            output_size=output_size,
            name=name,
            head_id=head_id)
        pooling_heads.append(pooling_head)
    return _pyramid_pooling_fusion(inputs=pooling_heads, name=name)


def _pyramid_pooling_head(x: tf.Tensor,
                          window_shape: Tuple[int, int],
                          strides: Tuple[int, int],
                          output_size: Size,
                          name: Optional[str],
                          head_id: int) -> tf.Tensor:
    pool_op_name = None
    if name is not None:
        pool_op_name = prepare_block_operation_name(name, f'pool_{head_id}')
    pool_op = avg_pool2d(
        x=x,
        window_shape=window_shape,
        strides=strides,
        name=pool_op_name)
    interp_op_name = None
    if name is not None:
        interp_op_name = prepare_block_operation_name(
            name,
            f'pool_{head_id}',
            'interp'
        )
    height, width = output_size
    return resize_bilinear(
        x=pool_op,
        height=height,
        width=width,
        name=interp_op_name)


def _pyramid_pooling_fusion(inputs: List[tf.Tensor],
                            name: Optional[str]) -> tf.Tensor:
    fusion_op_name = None
    if name is not None:
        fusion_op_name = prepare_block_operation_name(name, 'sum')
    return tf.add_n(inputs=inputs, name=fusion_op_name)
