from abc import ABC
from typing import Optional, List, Tuple

import tensorflow as tf

from src.model.building_blocks.encoding import residual_conv_encoder
from src.model.building_blocks.pyramid_pooling import pyramid_pooling, Size
from src.model.layers.conv import downsample_conv2d, bottleneck_conv2d, \
    dim_hold_conv2d, atrous_conv2d
from src.model.layers.interpolation import downsample_bilinear, \
    upsample_bilinear
from src.model.layers.pooling import max_pool2d
from src.model.network import Network, NetworkOutput, RequiredNodes


class ICNetBackbone(Network, ABC):

    EncoderConfigs = List[Tuple[dict, List[str]]]
    FlatEncoderConfigs = List[dict]

    __MEDIUM_BRANCH_ENCODERS_CONFIGS = [
        ({'output_filters': (32, 32, 128)}, ['conv2_1']),
        (
            {
                'output_filters': (32, 32, 128),
                'project_input': False,
                'name': 'conv2_2'
            },
            ['conv2_2', 'conv2_3']
        ),
        ({'output_filters': (64, 64, 256), 'input_stride': 2}, ['conv_3_1'])
    ]

    __SMALL_BRANCH_ENCODERS_CONFIGS = [
        (
            {'output_filters': (64, 64, 256), 'project_input': False},
            ['conv_3_2', 'conv_3_3', 'conv_3_4']
        ),
        (
            {'output_filters': (128, 128, 512), 'dilation_rate': 2},
            ['conv_4_1']
        ),
        (
            {
                'output_filters': (128, 128, 512),
                'project_input': False,
                'dilation_rate': 2,
            },
            ['conv_4_2', 'conv_4_3', 'conv_4_4', 'conv_4_5', 'conv_4_6']
        ),
        (
            {'output_filters': (256, 256, 1024), 'dilation_rate': 4},
            ['conv_5_1']
        ),
        (
            {
                'output_filters': (256, 256, 1024),
                'project_input': False,
                'dilation_rate': 4,
            },
            ['conv_5_2', 'conv_5_3']
        )
    ]

    @staticmethod
    def __get_default_config() -> dict:
        return {
            'lambda_1': 0.16,
            'lambda_2': 0.4,
            'lambda_3': 1.0
        }

    def __init__(self,
                 output_classes: int,
                 image_mean: Optional[List[float]] = None,
                 ignore_labels: Optional[List[int]] = None,
                 config: Optional[dict] = None):
        super().__init__(
            output_classes=output_classes,
            image_mean=image_mean,
            ignore_labels=ignore_labels,
            config=config)
        if config is None:
            config = ICNetBackbone.__get_default_config()
            self._config = config
        self._labmda_1 = config['lambda_1']
        self._lambda_2 = config['lambda_2']
        self._lambda_3 = config['lambda_3']

    def feed_forward(self,
                     x: tf.Tensor,
                     is_training: bool = True,
                     nodes_to_return: RequiredNodes = None) -> NetworkOutput:
        if self._image_mean is not None:
            x -= self._image_mean
        x = tf.math.divide(x, 255.0)
        input_size = x.shape[1], x.shape[2]
        conv3_sub1_proj = self.__big_images_branch(x=x, is_training=is_training)
        conv3_1 = self.__medium_images_branch(x)
        conv3_1_sub2_proj = bottleneck_conv2d(
            x=conv3_1,
            num_filters=128,
            activation=None,
            name='conv3_1_sub2_proj')
        conv_sub4 = self.__small_images_branch(
            conv3_1=conv3_1,
            input_size=input_size)
        conv_sub2 = self.__medium_small_branch_fusion(
            conv_sub4=conv_sub4,
            conv3_1_sub2_proj=conv3_1_sub2_proj,
            is_training=is_training)
        conv6_cls = self.__big_medium_branch_fusion(
            conv_sub2=conv_sub2,
            conv3_sub1_proj=conv3_sub1_proj)
        conv6_interp = upsample_bilinear(
            x=conv6_cls,
            zoom_factor=4)
        return self._construct_output(
            feedforward_output=conv6_interp,
            nodes_to_return=nodes_to_return)

    def infer(self, x: tf.Tensor) -> NetworkOutput:
        return self.feed_forward(
            x=x,
            is_training=False)

    def __big_images_branch(self,
                            x: tf.Tensor,
                            is_training: bool = True) -> tf.Tensor:
        conv1_sub1 = downsample_conv2d(
            x=x,
            num_filters=32,
            kernel_size=(3, 3),
            name='conv1_sub1')
        conv1_sub1_bn = tf.layers.batch_normalization(
            inputs=conv1_sub1,
            training=is_training,
            name='conv1_sub1_bn')
        conv2_sub1 = downsample_conv2d(
            x=conv1_sub1_bn,
            num_filters=32,
            kernel_size=(3, 3),
            name='conv2_sub1')
        conv2_sub1_bn = tf.layers.batch_normalization(
            inputs=conv2_sub1,
            training=is_training,
            name='conv2_sub1_bn')
        conv3_sub1 = downsample_conv2d(
            x=conv2_sub1_bn,
            num_filters=64,
            kernel_size=(3, 3),
            name='conv3_sub1')
        conv3_sub1_bn = tf.layers.batch_normalization(
            inputs=conv3_sub1,
            training=is_training,
            name='conv3_sub1_bn')
        conv3_sub1_proj = bottleneck_conv2d(
            x=conv3_sub1_bn,
            num_filters=128,
            activation=None,
            name='conv3_sub1_proj')
        return tf.layers.batch_normalization(
            inputs=conv3_sub1_proj,
            training=is_training,
            name='conv3_sub1_proj_bn')

    def __medium_images_branch(self,
                               x: tf.Tensor,
                               is_training: bool = True) -> tf.Tensor:
        data_sub2 = downsample_bilinear(
            x=x,
            shrink_factor=2)
        conv1_1_3x3_s2 = downsample_conv2d(
            x=data_sub2,
            num_filters=32,
            kernel_size=(3, 3),
            name='conv1_1_3x3_s2')
        conv1_1_3x3_s2_bn = tf.layers.batch_normalization(
            inputs=conv1_1_3x3_s2,
            training=is_training,
            name='conv1_1_3x3_s2_bn')
        conv1_2_3x3 = dim_hold_conv2d(
            x=conv1_1_3x3_s2_bn,
            num_filters=32,
            kernel_size=(3, 3),
            name='conv1_2_3x3')
        conv1_2_3x3_bn = tf.layers.batch_normalization(
            inputs=conv1_2_3x3,
            training=is_training,
            name='conv1_2_3x3_bn')
        conv1_3_3x3 = dim_hold_conv2d(
            x=conv1_2_3x3_bn,
            num_filters=64,
            kernel_size=(3, 3),
            name='conv1_3_3x3')
        conv1_3_3x3_bn = tf.layers.batch_normalization(
            inputs=conv1_3_3x3,
            training=is_training,
            name='conv1_3_3x3_bn')
        pool1_3x3_s2 = max_pool2d(
            x=conv1_3_3x3_bn,
            window_shape=(3, 3),
            name='pool1_3x3_s2')
        return self.__residual_encoder_chain(
            x=pool1_3x3_s2,
            encoders_configs=ICNetBackbone.__MEDIUM_BRANCH_ENCODERS_CONFIGS,
            is_training=is_training)

    @Network.Block.output_registered('conv_sub4')
    def __small_images_branch(self,
                              conv3_1: tf.Tensor,
                              input_size: Size,
                              is_training: bool = True) -> tf.Tensor:
        conv3_1_sub4 = downsample_bilinear(
            x=conv3_1,
            shrink_factor=2)
        conv_5_3 = self.__residual_encoder_chain(
            x=conv3_1_sub4,
            encoders_configs=ICNetBackbone.__SMALL_BRANCH_ENCODERS_CONFIGS,
            is_training=is_training)
        pyramid_pooling_config = [
            ((32, 64), (32, 64)), ((16, 32), (16, 32)),
            ((13, 25), (10, 20)), ((8, 16), (5, 10))
        ]
        pooling_output_size = input_size[0] // 32, input_size[1] // 32
        conv_5_3_sum = pyramid_pooling(
            x=conv_5_3,
            pooling_config=pyramid_pooling_config,
            output_size=pooling_output_size)
        conv5_4_k1 = bottleneck_conv2d(
            x=conv_5_3_sum,
            num_filters=256,
            name='conv5_4_k1')
        conv5_4_k1_bn = tf.layers.batch_normalization(
            inputs=conv5_4_k1,
            training=is_training,
            name='conv5_4_k1_bn')
        conv5_4_interp = upsample_bilinear(
            x=conv5_4_k1_bn,
            zoom_factor=2)
        conv_sub4 = atrous_conv2d(
            x=conv5_4_interp,
            num_filters=128,
            kernel_size=(3, 3),
            name='conv_sub4')
        return tf.layers.batch_normalization(
            inputs=conv_sub4,
            training=is_training,
            name='conv_sub4_bn')

    def __residual_encoder_chain(self,
                                 x: tf.Tensor,
                                 encoders_configs: List[Tuple[dict, List[str]]],
                                 is_training: bool = True) -> tf.Tensor:
        current_op = x
        encoders_configs = self.__unpack_encoder_configs(encoders_configs)
        for encoder_config in encoders_configs:
            current_op = residual_conv_encoder(
                x=current_op,
                is_training=is_training,
                **encoder_config
            )
        return current_op

    def __unpack_encoder_configs(self,
                                 configs: EncoderConfigs) -> FlatEncoderConfigs:
        def __update_config(config: dict, name: str) -> dict:
            config['name'] = name
            return config

        return [
            __update_config(config, name)
            for config, names in configs for name in names
        ]

    @Network.Block.output_registered('conv_sub2')
    def __medium_small_branch_fusion(self,
                                     conv_sub4: tf.Tensor,
                                     conv3_1_sub2_proj: tf.Tensor,
                                     is_training: bool = True) -> tf.Tensor:
        conv_sub2 = self.__branch_fusion(
            first_branch=conv_sub4,
            second_branch=conv3_1_sub2_proj,
            output_filters=128,
            fusion_name='sub24_sum',
            output_name='conv_sub2')
        return tf.layers.batch_normalization(
            inputs=conv_sub2,
            training=is_training,
            name='conv_sub2_bn')

    @Network.Block.output_registered('conv6_cls')
    def __big_medium_branch_fusion(self,
                                   conv_sub2: tf.Tensor,
                                   conv3_sub1_proj: tf.Tensor) -> tf.Tensor:
        return self.__branch_fusion(
            first_branch=conv_sub2,
            second_branch=conv3_sub1_proj,
            output_filters=self._output_classes,
            fusion_name='sub12_sum',
            output_name='conv6_cls',
            output_activation=None,
            dilation_rate=(1, 1))

    def __branch_fusion(self,
                        first_branch: tf.Tensor,
                        second_branch: tf.Tensor,
                        output_filters: int,
                        fusion_name: str,
                        output_name: str,
                        output_activation: Optional[str] = 'relu',
                        dilation_rate: Tuple[int, int] = (2, 2)) -> tf.Tensor:
        sum = tf.add(first_branch, second_branch, name=fusion_name)
        sum_relu = tf.nn.relu(sum, name=f'{fusion_name}/relu')
        sum_interp = upsample_bilinear(
            x=sum_relu,
            zoom_factor=2)
        return atrous_conv2d(
            x=sum_interp,
            num_filters=output_filters,
            kernel_size=(3, 3),
            dilation_rate=dilation_rate,
            activation=output_activation,
            name=output_name)
