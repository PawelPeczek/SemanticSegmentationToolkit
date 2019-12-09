from typing import Optional, List

import tensorflow as tf

from src.model.building_blocks.encoding import atrous_pyramid_encoder
from src.model.building_blocks.pyramid_pooling import pyramid_pool_fusion
from src.model.layers.conv import downsample_conv2d, dim_hold_conv2d, \
    bottleneck_conv2d
from src.model.layers.interpolation import downsample_bilinear, \
    upsample_bilinear
from src.model.losses.cascade import cascade_loss
from src.model.network import Network, BlockOutput, NetworkOutput, RequiredNodes


class ICNetV12(Network):

    @staticmethod
    def __get_default_config() -> dict:
        return {
            'lambda_1': 1.0,
            'lambda_2': 1.0,
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
            config = ICNetV12.__get_default_config()
            self._config = config
        self._labmda_1 = config['lambda_1']
        self._lambda_2 = config['lambda_2']
        self._lambda_3 = config['lambda_3']

    def feed_forward(self,
                     x: tf.Tensor,
                     is_training: bool = True,
                     nodes_to_return: RequiredNodes = None
                     ) -> NetworkOutput:
        if self._image_mean is not None:
            x -= self._image_mean
        x = tf.math.divide(x, 255.0)
        big_branch_output = self.__big_images_branch(
            x=x,
            is_training=is_training
        )
        half_size_input = downsample_bilinear(x=x, shrink_factor=2)
        medium_branch_common = self.__medium_branch_head(
            x=half_size_input,
            is_training=is_training
        )
        medium_branch_tail = self.__medium_branch_tail(
            x=medium_branch_common,
            is_training=is_training
        )
        small_branch_output = self.__small_branch(
            x=medium_branch_common,
            is_training=is_training
        )
        medium_small_fusion = self.__medium_small_branch_fusion(
            small_branch_output=small_branch_output,
            medium_branch_output=medium_branch_tail,
            is_training=is_training
        )
        big_medium_fusion = self.__big_medium_branch_fusion(
            fused_medium_branch=medium_small_fusion,
            big_branch_outtput=big_branch_output,
            is_training=is_training
        )
        cls = self.__prediction_branch(big_medium_fusion=big_medium_fusion)
        cls_up = upsample_bilinear(
            x=cls,
            zoom_factor=4
        )
        out = tf.math.argmax(cls_up, axis=3, output_type=tf.dtypes.int32)
        return self._construct_output(
            feedforward_output=out,
            nodes_to_return=nodes_to_return
        )

    def training_pass(self,
                      x: tf.Tensor,
                      y: tf.Tensor
                      ) -> tf.Operation:
        nodes_to_return = ['medium_small_fusion', 'big_medium_fusion', 'cls']
        model_output = self.feed_forward(
            x=x,
            is_training=True,
            nodes_to_return=nodes_to_return
        )
        medium_small_fusion = model_output['medium_small_fusion']
        big_medium_fusion = model_output['big_medium_fusion']
        cls = model_output['cls']
        medium_small_fusion = bottleneck_conv2d(
            x=medium_small_fusion,
            num_filters=self._output_classes,
            activation=None,
            name='medium_small_fusion_cls'
        )
        big_medium_fusion = bottleneck_conv2d(
            x=big_medium_fusion,
            num_filters=self._output_classes,
            activation=None,
            name='big_medium_fusion_cls'
        )
        cls_outputs = [medium_small_fusion, big_medium_fusion, cls]
        cls_weights = [self._labmda_1, self._lambda_2, self._lambda_3]
        return cascade_loss(
            cascade_output_nodes=cls_outputs,
            y=y,
            loss_weights=cls_weights,
            labels_to_ignore=self._ignore_labels
        )

    def infer(self, x: tf.Tensor) -> NetworkOutput:
        return self.feed_forward(
            x=x,
            is_training=False
        )

    @Network.Block.output_registered('h1_sub3_bn')
    def __big_images_branch(self,
                            x: tf.Tensor,
                            is_training: bool = True) -> tf.Tensor:
        h1_sub1 = downsample_conv2d(
            x=x,
            num_filters=8,
            kernel_size=(3, 3),
            name='h1_sub1'
        )
        h1_sub2 = downsample_conv2d(
            x=h1_sub1,
            num_filters=16,
            kernel_size=(3, 3),
            name='h1_sub2'
        )
        h1_sub3 = downsample_conv2d(
            x=h1_sub2,
            num_filters=64,
            kernel_size=(3, 3),
            name='h1_sub3'
        )
        return tf.layers.batch_normalization(
            inputs=h1_sub3,
            training=is_training,
            name='h1_sub3_bn'
        )

    @Network.Block.output_registered('h2_sub3_bn')
    def __medium_branch_head(self,
                             x: tf.Tensor,
                             is_training: bool
                             ) -> tf.Tensor:
        h2_sub1 = downsample_conv2d(
            x=x,
            num_filters=16,
            kernel_size=(3, 3),
            name='h2_sub1'
        )
        h2_conv1 = dim_hold_conv2d(
            x=h2_sub1,
            num_filters=16,
            kernel_size=(3, 3),
            name='h2_conv1'
        )
        h2_conv1_bn = tf.layers.batch_normalization(
            inputs=h2_conv1,
            training=is_training,
            name='h2_conv1_bn'
        )
        h2_sub2 = downsample_conv2d(
            x=h2_conv1_bn,
            num_filters=32,
            kernel_size=(3, 3),
            name='h2_sub2'
        )
        h2_conv2 = dim_hold_conv2d(
            x=h2_sub2,
            num_filters=64,
            kernel_size=(3, 3),
            name='h2_conv2'
        )
        return tf.layers.batch_normalization(
            inputs=h2_conv2,
            training=is_training,
            name='h2_conv2_bn'
        )

    @Network.Block.output_registered('h2_add')
    def __medium_branch_tail(self,
                             x: tf.Tensor,
                             is_training: bool
                             ) -> tf.Tensor:
        h2_fs1 = bottleneck_conv2d(
            x=x,
            num_filters=64,
            name='h2_fs1'
        )
        h2_fs_bn = tf.layers.batch_normalization(
            inputs=h2_fs1,
            training=is_training,
            name='h2_fs_bn'
        )
        h2_conv3 = dim_hold_conv2d(
            x=h2_fs_bn,
            num_filters=128,
            kernel_size=(3, 3),
            name='h2_conv3'
        )
        h2_fs2 = bottleneck_conv2d(
            x=h2_conv3,
            num_filters=64,
            name='h2_fs2'
        )
        h2_conv4 = dim_hold_conv2d(
            x=h2_fs2,
            num_filters=128,
            kernel_size=(3, 3),
            name='h2_conv4'
        )
        h2_fs3 = bottleneck_conv2d(
            x=h2_conv4,
            num_filters=64,
            name='h2_fs3'
        )
        fuse = h2_fs1 + h2_fs3
        fuse_bn = tf.layers.batch_normalization(
            inputs=fuse,
            training=is_training,
            name='fuse_bn'
        )
        pp1 = pyramid_pool_fusion(
            x=fuse_bn,
            windows_shapes=[2, 3, 5],
            fuse_filters=128,
            name='h2_pp1'
        )
        dilated_block1 = atrous_pyramid_encoder(
            x=pp1,
            output_filters=128,
            pyramid_heads_dilation_rate=[1, 2, 4, 8],
            use_residual_connection=False,
            name='h2_dilation_block'
        )
        h2_dilated_block1_bn = tf.layers.batch_normalization(
            inputs=dilated_block1,
            training=is_training,
            name='h2_dilated_block1_bn'
        )
        h2_fs4 = bottleneck_conv2d(
            x=h2_dilated_block1_bn,
            num_filters=64,
            name='h2_fs4'
        )
        h2_conv5 = dim_hold_conv2d(
            x=h2_fs4,
            num_filters=128,
            kernel_size=(3, 3),
            name='h2_conv5'
        )
        h2_fs5 = bottleneck_conv2d(
            x=h2_conv5,
            num_filters=64,
            name='h2_fs5'
        )
        h2_conv6 = dim_hold_conv2d(
            x=h2_fs5,
            num_filters=256,
            kernel_size=(3, 3),
            name='h2_conv6'
        )
        h2_fs6 = bottleneck_conv2d(
            x=h2_conv6,
            num_filters=128,
            name='h2_fs6'
        )
        return tf.math.add(h2_fs6, pp1, name='h2_add')

    @Network.Block.output_registered('h3_add6_bn')
    def __small_branch(self, x: tf.Tensor, is_training: bool) -> tf.Tensor:
        h3_fs1 = bottleneck_conv2d(x=x, num_filters=64, name='h3_fs1')
        h3_fs1_bn = tf.layers.batch_normalization(
            inputs=h3_fs1,
            training=is_training,
            name='h3_fs1_bn'
        )
        h3_pp1 = pyramid_pool_fusion(
            x=h3_fs1_bn,
            windows_shapes=[2, 3, 5],
            fuse_filters=128,
            name='h2_pp1'
        )
        h3_fs2 = bottleneck_conv2d(x=h3_pp1, num_filters=64, name='h3_fs2')
        h3_conv1 = dim_hold_conv2d(
            x=h3_fs2,
            num_filters=128,
            kernel_size=(3, 3),
            name='h3_conv1'
        )
        h3_fs3 = bottleneck_conv2d(x=h3_conv1, num_filters=64, name='h3_fs3')
        h3_conv2 = dim_hold_conv2d(
            x=h3_fs3,
            num_filters=256,
            kernel_size=(3, 3),
            name='h3_conv2'
        )
        h3_fs4 = bottleneck_conv2d(x=h3_conv2, num_filters=128, name='h3_fs4')
        h3_add1 = tf.math.add(h3_pp1, h3_fs4, name='h3_add1')
        h3_conv3 = dim_hold_conv2d(
            x=h3_add1,
            num_filters=256,
            kernel_size=(3, 3),
            name='h3_conv3'
        )
        h3_fs5 = bottleneck_conv2d(x=h3_conv3, num_filters=128, name='h3_fs5')
        h3_fs5_bn = tf.layers.batch_normalization(
            inputs=h3_fs5,
            training=is_training,
            name='h3_fs5_bn'
        )

        h3_pp2 = pyramid_pool_fusion(
            x=h3_fs5_bn,
            windows_shapes=[2, 3, 5],
            fuse_filters=256,
            name='h3_pp2'
        )
        h3_dilated_block_1 = atrous_pyramid_encoder(
            x=h3_pp2,
            output_filters=256,
            pyramid_heads_dilation_rate=[1, 2, 4, 8],
            use_residual_connection=False,
            name='h3_dilation_block_1'
        )
        h3_dilated_block_1_bn = tf.layers.batch_normalization(
            inputs=h3_dilated_block_1,
            training=is_training,
            name='h3_dilated_block_1_bn'
        )
        h3_fs6 = bottleneck_conv2d(
            x=h3_dilated_block_1_bn,
            num_filters=64,
            name='h3_fs6'
        )
        h3_conv4 = dim_hold_conv2d(
            x=h3_fs6,
            num_filters=256,
            kernel_size=(3, 3),
            name='h3_conv4'
        )
        h3_fs7 = bottleneck_conv2d(
            x=h3_conv4,
            num_filters=64,
            name='h3_fs7'
        )
        h3_conv5 = dim_hold_conv2d(
            x=h3_fs7,
            num_filters=512,
            kernel_size=(3, 3),
            name='h3_conv5'
        )
        h3_fs8 = bottleneck_conv2d(
            x=h3_conv5,
            num_filters=256,
            name='h3_fs8'
        )
        h3_add2 = tf.math.add(h3_pp2, h3_fs8, name='h3_add2')
        h3_add2_bn = tf.layers.batch_normalization(
            inputs=h3_add2,
            training=is_training,
            name='h3_add2_bn'
        )

        h3_fs9 = bottleneck_conv2d(
            x=h3_add2_bn,
            num_filters=128,
            name='h3_fs9'
        )
        h3_conv6 = dim_hold_conv2d(
            x=h3_fs9,
            num_filters=512,
            kernel_size=(3, 3),
            name='h3_conv6'
        )
        h3_fs10 = bottleneck_conv2d(
            x=h3_conv6,
            num_filters=128,
            name='h3_fs10'
        )
        h3_add3 = tf.math.add(h3_fs9, h3_fs10, name='h3_add3')
        h3_add3_bn = tf.layers.batch_normalization(
            inputs=h3_add3,
            training=is_training,
            name='h3_add3_bn'
        )
        h3_conv7 = dim_hold_conv2d(
            x=h3_add3_bn,
            num_filters=512,
            kernel_size=(3, 3),
            name='h3_conv7'
        )
        h3_fs11 = bottleneck_conv2d(
            x=h3_conv7,
            num_filters=128,
            name='h3_fs11'
        )
        h3_conv8 = dim_hold_conv2d(
            x=h3_fs11,
            num_filters=512,
            kernel_size=(3, 3),
            name='h3_conv8'
        )
        h3_fs12 = bottleneck_conv2d(
            x=h3_conv8,
            num_filters=128,
            name='h3_fs12'
        )
        h3_add4 = tf.math.add(h3_fs12, h3_add3_bn, name='h3_add4')
        h3_add4_bn = tf.layers.batch_normalization(
            inputs=h3_add4,
            training=is_training,
            name='h3_add4_bn'
        )
        h3_conv9 = dim_hold_conv2d(
            x=h3_add4_bn,
            num_filters=768,
            kernel_size=(3, 3),
            name='h3_conv9'
        )
        h3_fs13 = bottleneck_conv2d(
            x=h3_conv9,
            num_filters=128,
            name='h3_fs13'
        )
        h3_conv10 = dim_hold_conv2d(
            x=h3_fs13,
            num_filters=768,
            kernel_size=(3, 3),
            name='h3_conv10'
        )
        h3_fs14 = bottleneck_conv2d(
            x=h3_conv10,
            num_filters=128,
            name='h3_fs14'
        )
        h3_add5 = tf.math.add(h3_fs14, h3_add4_bn, name='h3_add5')
        h3_add5_bn = tf.layers.batch_normalization(
            inputs=h3_add5,
            training=is_training,
            name='h3_add5_bn'
        )
        h3_conv11 = dim_hold_conv2d(
            x=h3_add5_bn,
            num_filters=1024,
            kernel_size=(3, 3),
            name='h3_conv11'
        )
        h3_fs15 = bottleneck_conv2d(
            x=h3_conv11,
            num_filters=256,
            name='h3_fs15'
        )
        h3_conv12 = dim_hold_conv2d(
            x=h3_fs15,
            num_filters=1024,
            kernel_size=(3, 3),
            name='h3_conv12'
        )
        h3_fs16 = bottleneck_conv2d(
            x=h3_conv12,
            num_filters=256,
            name='h3_fs16'
        )
        h3_add6 = tf.math.add(h3_fs15, h3_fs16, name='h3_add6')
        return tf.layers.batch_normalization(
            inputs=h3_add6,
            training=is_training,
            name='h3_add6_bn'
        )

    @Network.Block.output_registered('medium_small_fusion')
    def __medium_small_branch_fusion(self,
                                     small_branch_output: tf.Tensor,
                                     medium_branch_output: tf.Tensor,
                                     is_training: bool) -> tf.Tensor:
        return self.__cascade_fusion_block(
            smaller_input=small_branch_output,
            bigger_input=medium_branch_output,
            is_training=is_training,
            output_filters=64,
            base_name='medium_small_fusion'
        )

    @Network.Block.output_registered('big_medium_fusion')
    def __big_medium_branch_fusion(self,
                                   fused_medium_branch: tf.Tensor,
                                   big_branch_outtput: tf.Tensor,
                                   is_training: bool) -> tf.Tensor:
        return self.__cascade_fusion_block(
            smaller_input=fused_medium_branch,
            bigger_input=big_branch_outtput,
            is_training=is_training,
            output_filters=64,
            base_name='big_medium_fusion'
        )

    @Network.Block.output_registered('cls')
    def __prediction_branch(self,
                            big_medium_fusion: tf.Tensor
                            ) -> tf.Tensor:
        quater_size_output = upsample_bilinear(
            x=big_medium_fusion,
            zoom_factor=2
        )
        return dim_hold_conv2d(
            x=quater_size_output,
            num_filters=self._output_classes,
            kernel_size=(3, 3),
            activation=None,
            name='cls'
        )

    def __cascade_fusion_block(self,
                               smaller_input: tf.Tensor,
                               bigger_input: tf.Tensor,
                               is_training: bool,
                               output_filters: int,
                               base_name: str,
                               ) -> tf.Tensor:
        upsampled = upsample_bilinear(
            x=smaller_input,
            zoom_factor=2
        )
        upsampled = dim_hold_conv2d(
            x=upsampled,
            num_filters=output_filters,
            kernel_size=(3, 3),
            name=f'{base_name}/fusion_conv'
        )
        upsampled_bn = tf.layers.batch_normalization(
            inputs=upsampled,
            training=is_training,
            name=f'{base_name}/fusion_conv_bn'
        )
        bigger_input = bottleneck_conv2d(
            x=bigger_input,
            num_filters=output_filters,
            name=f'{base_name}/bigger_input_fs'
        )
        out = tf.math.add(upsampled_bn, bigger_input, name=f'{base_name}/add')
        return tf.nn.relu(out, name=f'{base_name}/relu')
