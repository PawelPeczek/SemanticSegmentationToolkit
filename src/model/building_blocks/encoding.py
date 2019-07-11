from functools import reduce, partial
from typing import Tuple, Optional, List, Union

import tensorflow as tf

from src.model.building_blocks.errors import AtrousPyramidEncoderError, \
    ATROUS_ENCODER_PYRAMID_CFG_ERROR_MSG, ATROUS_ENCODER_RESIDUAL_ERROR_MSG
from src.model.building_blocks.utils import prepare_block_operation_name
from src.model.layers.conv import dim_hold_conv2d, bottleneck_conv2d, \
    separable_bottleneck_conv2d
from src.model.layers.pooling import FusionMethod, pool2d


def atrous_pyramid_encoder(x: tf.Tensor,
                           output_filters: int,
                           pyramid_heads_dilation_rate: List[int],
                           pyramid_heads_kernels: Union[int, List[int]] = 3,
                           use_separable_conv_in_pyramid: bool = False,
                           pyramid_heads_activations: Optional[str] = 'relu',
                           input_filters_after_reduction: Optional[int] = 64,
                           separate_reduction_head: bool = True,
                           use_separable_conv_on_input: bool = False,
                           reduction_activation: Optional[str] = 'relu',
                           use_residual_connection: bool = True,
                           fusion_method: FusionMethod = FusionMethod.SUM,
                           fusion_blending_kernel: Optional[int] = None,
                           use_separable_conv_while_fusion: bool = False,
                           name: Optional[str] = None) -> tf.Tensor:
    _validate_atrous_encoder_input(
        x,
        pyramid_heads_dilation_rate,
        pyramid_heads_kernels,
        use_residual_connection,
        input_filters_after_reduction,
        output_filters)
    heads_number = len(pyramid_heads_dilation_rate)
    inputs = _atrous_encoder_input(
        x,
        heads_number,
        input_filters_after_reduction,
        separate_reduction_head,
        reduction_activation,
        use_separable_conv_on_input,
        name)
    out_1 = tf.layers.conv2d(feed_1, output_filters, (3, 3), padding='SAME',
                             activation='relu')
    out_2 = tf.layers.conv2d(feed_2, output_filters, (3, 3), padding='SAME',
                             activation='relu',
                             dilation_rate=(2, 2))
    out_3 = tf.layers.conv2d(feed_3, output_filters, (3, 3), padding='SAME',
                             activation='relu',
                             dilation_rate=(4, 4))
    out_4 = tf.layers.conv2d(feed_4, output_filters, (3, 3), padding='SAME',
                             activation='relu',
                             dilation_rate=(8, 8))
    out = out_1 + out_2 + out_3 + out_4
    if use_residual_connection:
        out = tf.math.add(out, X)

    return out


def _validate_atrous_encoder_input(x: tf.Tensor,
                                   pyramid_heads_dilation_rate: List[int],
                                   pyramid_heads_kernels: Union[int, List[int]],
                                   use_residual_connection: bool,
                                   input_filters_after_reduction: Optional[int],
                                   output_filters: int) -> None:
    _check_pyramid_input(pyramid_heads_dilation_rate, pyramid_heads_kernels)
    _check_residual_connection(
        x,
        use_residual_connection,
        input_filters_after_reduction,
        output_filters)


def _check_pyramid_input(pyramid_heads_dilation_rate: List[int],
                         pyramid_heads_kernels: Union[int, List[int]]) -> None:
    if isinstance(pyramid_heads_kernels, list):
        if len(pyramid_heads_dilation_rate) != len(pyramid_heads_kernels):
            raise AtrousPyramidEncoderError(
                ATROUS_ENCODER_PYRAMID_CFG_ERROR_MSG
            )
    return None


def _check_residual_connection(x: tf.Tensor,
                               use_residual_connection: bool,
                               input_filters_after_reduction: Optional[int],
                               output_filters: int) -> None:
    if not use_residual_connection:
        return None
    if input_filters_after_reduction is None:
        if x.shape[-1] != output_filters:
            raise AtrousPyramidEncoderError(ATROUS_ENCODER_RESIDUAL_ERROR_MSG)
        else:
            return None
    elif input_filters_after_reduction != output_filters:
        raise AtrousPyramidEncoderError(ATROUS_ENCODER_RESIDUAL_ERROR_MSG)
    return None


def _atrous_encoder_input(x: tf.Tensor,
                          heads_number: int,
                          input_filters_after_reduction: Optional[int] = 64,
                          separate_reduction_head: bool = True,
                          reduction_activation: Optional[str] = 'relu',
                          use_separable_conv_on_input: bool = False,
                          name: Optional[str] = None) -> List[tf.Tensor]:
    if input_filters_after_reduction is None:
        return [x for _ in range(heads_number)]
    create_input_head = partial(
        _atrous_encoder_input_head,
        x=x,
        filters=input_filters_after_reduction,
        activation=reduction_activation,
        use_separable_conv=use_separable_conv_on_input,
        name=name)
    if separate_reduction_head is True:
        return [create_input_head(head_id=i) for i in range(heads_number)]
    else:
        input_head = create_input_head(0)
        return [input_head for _ in range(heads_number)]


def _atrous_encoder_input_head(x: tf.Tensor,
                               filters: int,
                               activation: Optional[str],
                               use_separable_conv: bool,
                               name: Optional[str],
                               head_id: Optional[str]) -> tf.Tensor:
    if name is not None:
        name = prepare_block_operation_name(
            name,
            'input_head',
            f'reduction_1x1_conv_{head_id}')
    if use_separable_conv:
        return separable_bottleneck_conv2d(
            x,
            filters,
            activation=activation,
            name=name)
    else:
        return bottleneck_conv2d(
            x,
            filters,
            activation=activation,
            name=name)

