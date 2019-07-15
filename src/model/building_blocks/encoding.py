from functools import partial
from typing import Optional, List, Union, Tuple

import tensorflow as tf

from src.model.building_blocks.errors import AtrousPyramidEncoderError, \
    ATROUS_ENCODER_PYRAMID_CFG_ERROR_MSG, ATROUS_ENCODER_RESIDUAL_ERROR_MSG
from src.model.building_blocks.utils import prepare_block_operation_name
from src.model.layers.conv import dim_hold_conv2d, bottleneck_conv2d, \
    separable_bottleneck_conv2d, atrous_separable_conv2d, atrous_conv2d, \
    separable_conv2d
from src.model.layers.pooling import FusionMethod


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
                           fusion_blending_kernel: Optional[int] = 3,
                           use_separable_conv_while_fusion: bool = False,
                           fusion_activation: Optional[str] = 'relu',
                           name: Optional[str] = None) -> tf.Tensor:
    _validate_atrous_encoder_input(
        x=x,
        pyramid_heads_dilation_rate=pyramid_heads_dilation_rate,
        pyramid_heads_kernels=pyramid_heads_kernels,
        use_residual_connection=use_residual_connection,
        input_filters_after_reduction=input_filters_after_reduction,
        output_filters=output_filters)
    heads_number = len(pyramid_heads_dilation_rate)
    inputs = _atrous_encoder_input(
        x=x,
        heads_number=heads_number,
        input_filters_after_reduction=input_filters_after_reduction,
        separate_reduction_head=separate_reduction_head,
        reduction_activation=reduction_activation,
        use_separable_conv_on_input=use_separable_conv_on_input,
        name=name)
    pyramid_output = _atrous_pyramid(
        inputs=inputs,
        output_filters=output_filters,
        heads_dilation_rate=pyramid_heads_dilation_rate,
        heads_kernels=pyramid_heads_kernels,
        use_separable_conv_in_pyramid=use_separable_conv_in_pyramid,
        pyramid_heads_activations=pyramid_heads_activations,
        name=name)
    output_is_last_op_in_block = not use_residual_connection
    out = _output_fusion(
        pyramid_output=pyramid_output,
        output_filters=output_filters,
        fusion_method=fusion_method,
        fusion_blending_kernel=fusion_blending_kernel,
        use_separable_conv_while_fusion=use_separable_conv_while_fusion,
        fusion_activation=fusion_activation,
        name=name,
        output_is_last_op_in_block=output_is_last_op_in_block
    )
    if use_residual_connection:
        if name is not None:
            name = prepare_block_operation_name(name, 'out')
        out = tf.math.add(out, x, name=name)
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


def _atrous_pyramid(inputs: List[tf.Tensor],
                    output_filters: int,
                    heads_dilation_rate: List[int],
                    heads_kernels: Union[int, List[int]],
                    use_separable_conv_in_pyramid: bool,
                    pyramid_heads_activations: Optional[str],
                    name: Optional[str]) -> List[tf.Tensor]:
    if isinstance(heads_kernels, int):
        heads_no = len(heads_dilation_rate)
        heads_kernels = [heads_kernels for _ in range(heads_no)]
    pyramid_output = []
    builing_it = enumerate(zip(inputs, heads_kernels, heads_dilation_rate))
    for head_id, (x, kernel, dilation) in builing_it:
        pyramid_head = _atrous_pyramid_head(
            x=x,
            output_filters=output_filters,
            kernel=kernel,
            dilation_rate=dilation,
            use_separable_conv=use_separable_conv_in_pyramid,
            activation=pyramid_heads_activations,
            name=name,
            head_id=head_id
        )
        pyramid_output.append(pyramid_head)
    return pyramid_output


def _atrous_pyramid_head(x: tf.Tensor,
                         output_filters: int,
                         kernel: int,
                         dilation_rate: int,
                         use_separable_conv: bool,
                         activation: Optional[str],
                         name: Optional[str],
                         head_id: int) -> tf.Tensor:
    if name is not None:
        name = prepare_block_operation_name(
            name,
            'pyramid',
            f'{kernel}x{kernel}_conv_{head_id}')
    kernel = kernel, kernel
    dilation_rate = dilation_rate, dilation_rate
    if use_separable_conv:
        return atrous_separable_conv2d(
            x=x,
            num_filters=output_filters,
            kernel_size=kernel,
            dilation_rate=dilation_rate,
            activation=activation,
            name=name
        )
    else:
        return atrous_conv2d(
            x=x,
            num_filters=output_filters,
            kernel_size=kernel,
            dilation_rate=dilation_rate,
            activation=activation,
            name=name
        )


def _output_fusion(pyramid_output: List[tf.Tensor],
                   output_filters: int,
                   fusion_method: FusionMethod,
                   fusion_blending_kernel: Optional[int],
                   use_separable_conv_while_fusion: bool,
                   fusion_activation: Optional[str],
                   name: Optional[str],
                   output_is_last_op_in_block: bool) -> tf.Tensor:
    fused = _fusion_op(
        pyramid_output=pyramid_output,
        fusion_method=fusion_method,
        fusion_blending_kernel=fusion_blending_kernel,
        name=name,
        output_is_last_op_in_block=output_is_last_op_in_block)
    if fusion_blending_kernel is None:
        return fused
    return _blending_layer(
        x=fused,
        output_filters=output_filters,
        kernel=fusion_blending_kernel,
        use_separable_conv=use_separable_conv_while_fusion,
        activation=fusion_activation,
        name=name,
        output_is_last_op_in_block=output_is_last_op_in_block)


def _fusion_op(pyramid_output: List[tf.Tensor],
               fusion_method: FusionMethod,
               fusion_blending_kernel: Optional[int],
               name: Optional[str],
               output_is_last_op_in_block: bool) -> tf.Tensor:
    fusion_name = None
    if name is not None:
        concat_is_final_op = output_is_last_op_in_block and \
                             fusion_blending_kernel is None
        fusion_name_postfix = 'out' if concat_is_final_op else 'fusion'
        fusion_name = prepare_block_operation_name(name, fusion_name_postfix)
    if fusion_method is FusionMethod.SUM:
        fused = tf.math.add_n(pyramid_output, name=fusion_name)
    else:
        fused = tf.concat(pyramid_output, axis=-1, name=fusion_name)
    return fused


def _blending_layer(x: tf.Tensor,
                    output_filters: int,
                    kernel: int,
                    use_separable_conv: bool,
                    activation: Optional[str],
                    name: Optional[str],
                    output_is_last_op_in_block: bool) -> tf.Tensor:
    blending_name = None
    if name is not None:
        blending_postfix = 'out' if output_is_last_op_in_block else 'blend'
        blending_name = prepare_block_operation_name(name, blending_postfix)
    kernel = kernel, kernel
    if use_separable_conv is True:
        return separable_conv2d(
            x=x,
            num_filters=output_filters,
            kernel_size=kernel,
            activation=activation,
            name=blending_name)
    else:
        return dim_hold_conv2d(
            x=x,
            num_filters=output_filters,
            kernel_size=kernel,
            activation=activation,
            name=blending_name)


def residual_conv_encoder(x: tf.Tensor,
                          output_filters: Tuple[int, int, int],
                          dilation_rate: int = 1,
                          use_relu_at_output: bool = True,
                          name: Optional[str] = None) -> tf.Tensor:
    projection_conv = _projection_conv(x, output_filters[-1], name)
    increased = _reduce_conv_increase_block(
        x=x,
        output_filters=output_filters,
        dilation_rate=dilation_rate,
        name=name
    )
    sum = tf.add(projection_conv, increased, name=name)
    if not use_relu_at_output:
        return sum
    if name is not None:
        name = prepare_block_operation_name(name, 'relu')
    return tf.nn.relu(sum, name=name)


def _projection_conv(x: tf.Tensor,
                     filters: int,
                     name: Optional[str]) -> tf.Tensor:
    if name is not None:
        name = prepare_block_operation_name(name, '1x1_proj')
    return bottleneck_conv2d(
        x=x,
        num_filters=filters,
        activation=None,
        name=name)


def _reduce_conv_increase_block(x: tf.Tensor,
                                output_filters: Tuple[int, int, int],
                                dilation_rate: int,
                                name: Optional[str]) -> tf.Tensor:
    reduced = _reduce_conv(
        x=x,
        filters=output_filters[0],
        name=name)
    internal_conv = _internal_conv3x3(
        x=reduced,
        filters=output_filters[1],
        dilation_rate=dilation_rate,
        name=name)
    return _increase_conv(
        x=internal_conv,
        filters=output_filters[-1],
        name=name)


def _reduce_conv(x: tf.Tensor,
                 filters: int,
                 name: Optional[str]) -> tf.Tensor:
    if name is not None:
        name = prepare_block_operation_name(name, '1x1_reduce', 'relu')
    return bottleneck_conv2d(
        x=x,
        num_filters=filters,
        name=name
    )


def _internal_conv3x3(x: tf.Tensor,
                      filters: int,
                      dilation_rate: int,
                      name: Optional[str]) -> tf.Tensor:
    if name is not None:
        name = prepare_block_operation_name(name, '3x3', 'relu')
    dilation_rate = dilation_rate, dilation_rate
    return atrous_conv2d(
        x=x,
        num_filters=filters,
        kernel_size=(3, 3),
        dilation_rate=dilation_rate,
        name=name
    )


def _increase_conv(x: tf.Tensor,
                   filters: int,
                   name: Optional[str]) -> tf.Tensor:
    if name is not None:
        name = prepare_block_operation_name(name, '1x1_increase')
    return bottleneck_conv2d(
        x=x,
        num_filters=filters,
        activation=None,
        name=name
    )

