from typing import Tuple, Optional, List

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class MPPNet(SemanticSegmentationModel):

    def run(self, X: tf.Tensor, num_classes: int, is_training: bool = True, y: Optional[tf.Tensor] = None) -> Tuple[
        tf.Tensor, Optional[tf.Tensor]]:
        if is_training:
            y = tf.expand_dims(y, axis=-1)
        downsampled_x2 = tf.layers.conv2d(X, 32, (3, 3), strides=(2, 2), padding='SAME', activation='relu')
        print(f'downsampled_x2.shape: {downsampled_x2.shape}')
        downsampled_x2_bn = tf.layers.batch_normalization(downsampled_x2, training=is_training)
        downsampled_x4 = tf.layers.conv2d(downsampled_x2_bn, 32, (3, 3), strides=(2, 2), padding='SAME',
                                          activation='relu')
        print(f'downsampled_x4.shape: {downsampled_x4.shape}')

        downsampled_x4_bn = tf.layers.batch_normalization(downsampled_x4, training=is_training)
        downsampled_x8 = tf.layers.conv2d(downsampled_x4_bn, 32, (3, 3), strides=(2, 2), padding='SAME',
                                          activation='relu')
        print(f'downsampled_x8.shape: {downsampled_x8.shape}')
        downsampled_x8_bn = tf.layers.batch_normalization(downsampled_x8, training=is_training)

        down_x8_pp_1 = self.__pyramid_pool_layer(downsampled_x8_bn, [(32, 64), (16, 32), (8, 16)], 64,
                                                 downsample_input=False)
        print(f'down_x8_pp_1.shape: {down_x8_pp_1.shape}')
        down_x8_conv_1 = tf.layers.conv2d(down_x8_pp_1, 128, (3, 3), strides=(2, 2), padding='SAME', activation='relu')
        print(f'down_x8_conv_1.shape: {down_x8_conv_1.shape}')
        down_x8_conv_2 = tf.layers.conv2d(down_x8_conv_1, 64, (1, 1), strides=(1, 1), padding='SAME', activation='relu')
        print(f'down_x8_conv_2.shape: {down_x8_conv_2.shape}')
        down_x8_bn_2 = tf.layers.batch_normalization(down_x8_conv_2, training=is_training)
        down_x8_conv_3 = tf.layers.conv2d(down_x8_bn_2, 128, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        print(f'down_x8_conv_3.shape: {down_x8_conv_3.shape}')
        down_x8_conv_4 = tf.layers.conv2d(down_x8_conv_3, 64, (1, 1), strides=(1, 1), padding='SAME', activation='relu')
        print(f'down_x8_conv_4.shape: {down_x8_conv_4.shape}')
        down_x8_elemem_sum = down_x8_bn_2 + down_x8_conv_4
        down_x8_bn_3 = tf.layers.batch_normalization(down_x8_elemem_sum, training=is_training)
        down_x8_pp_2 = self.__pyramid_pool_layer(down_x8_bn_3, [(16, 32), (8, 16), (4, 8)], 256)
        print(f'down_x8_pp_2.shape: {down_x8_pp_2.shape}')
        down_x32_loss, down_x16_loss, down_x8_loss, down_x_4_loss, overall_loss = None, None, None, None, None
        if is_training:
            prediction = tf.layers.conv2d(down_x8_pp_2, num_classes, (1, 1), strides=(1, 1), padding='SAME',
                                          activation=None, name='down_x32_cls')
            down_x32_label = self.__resize_layer_by_factor(y, 1 / 32, ResizeMethod.NEAREST_NEIGHBOR)
            down_x32_label = tf.squeeze(down_x32_label, axis=-1)
            down_x32_label = tf.cast(down_x32_label, dtype=tf.int32)
            down_x32_loss = self.__compute_loss(prediction, down_x32_label)
        down_x8_conv_5 = tf.layers.conv2d(down_x8_pp_2, 128, (1, 1), strides=(1, 1), padding='SAME', activation='relu')
        print(f'down_x8_conv_5.shape: {down_x8_conv_5.shape}')
        down_x8_conv_6 = tf.layers.conv2d(down_x8_conv_5, 256, (3, 3), strides=(1, 1), padding='SAME',
                                          activation='relu')
        print(f'down_x8_conv_6.shape: {down_x8_conv_6.shape}')
        down_x8_conv_7 = tf.layers.conv2d(down_x8_conv_6, 128, (1, 1), strides=(1, 1), padding='SAME',
                                          activation='relu')
        print(f'down_x8_conv_7.shape: {down_x8_conv_7.shape}')
        down_x8_conv_8 = tf.layers.conv2d(down_x8_conv_7, 256, (3, 3), strides=(1, 1), padding='SAME',
                                          activation='relu')
        print(f'down_x8_conv_8.shape: {down_x8_conv_8.shape}')
        down_x8_bn_4 = tf.layers.batch_normalization(down_x8_conv_8, training=is_training)
        down_x8_conv_9 = tf.layers.conv2d(down_x8_bn_4, 128, (1, 1), strides=(1, 1), padding='SAME',
                                          activation='relu')
        print(f'down_x8_conv_9.shape: {down_x8_conv_9.shape}')
        down_x8_conv_10 = tf.layers.conv2d(down_x8_conv_9, 512, (3, 3), strides=(1, 1), padding='SAME',
                                           activation='relu')
        print(f'down_x8_conv_10.shape: {down_x8_conv_10.shape}')
        down_x8_conv_11 = tf.layers.conv2d(down_x8_conv_10, 128, (1, 1), strides=(1, 1), padding='SAME',
                                           activation='relu')
        print(f'down_x8_conv_11.shape: {down_x8_conv_11.shape}')
        down_x8_conv_12 = tf.layers.conv2d(down_x8_conv_11, 512, (3, 3), strides=(1, 1), padding='SAME',
                                           activation='relu')
        print(f'down_x8_conv_12.shape: {down_x8_conv_12.shape}')
        down_x8_conv_13 = tf.layers.conv2d(down_x8_conv_12, 128, (1, 1), strides=(1, 1), padding='SAME',
                                           activation='relu')
        print(f'down_x8_conv_13.shape: {down_x8_conv_13.shape}')
        down_x8_bn_5 = tf.layers.batch_normalization(down_x8_conv_13, training=is_training)
        down_x8_pp_2 = self.__pyramid_pool_layer(down_x8_bn_5, [(16, 32), (8, 16), (4, 8)], 256, downsample_input=False)
        print(f'down_x8_pp_2.shape: {down_x8_pp_2.shape}')
        down_x8_up = self.__resize_layer_by_factor(down_x8_pp_2, 2)
        print(f'down_x8_up.shape: {down_x8_up.shape}')

        down_x4_pp_1 = self.__pyramid_pool_layer(downsampled_x4_bn, [(16, 32), (8, 16), (4, 8)], 64)
        print(f'down_x4_pp_1.shape: {down_x4_pp_1.shape}')
        down_x4_conv_1_1 = tf.layers.conv2d(down_x4_pp_1, 128, (3, 3), strides=(2, 2), padding='SAME',
                                            activation='relu')
        print(f'down_x4_conv_1_1.shape: {down_x4_conv_1_1.shape}')
        down_x4_conv_2_1 = tf.layers.conv2d(down_x4_conv_1_1, 256, (1, 1), strides=(1, 1), padding='SAME',
                                            activation='relu')
        print(f'down_x4_conv_2_1.shape: {down_x4_conv_2_1.shape}')
        down_x4_conv_1_2 = tf.layers.conv2d(down_x4_pp_1, 128, (3, 3), strides=(1, 1), padding='SAME',
                                            activation='relu')
        print(f'down_x4_conv_1_2.shape: {down_x4_conv_1_2.shape}')
        down_x4_conv_2_2 = tf.layers.conv2d(down_x4_conv_1_2, 256, (1, 1), strides=(1, 1), padding='SAME',
                                            activation='relu')
        print(f'down_x4_conv_2_2.shape: {down_x4_conv_2_2.shape}')

        down_x4_fused_dwon_x8 = down_x8_up + down_x4_conv_2_1
        down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_1 = \
            tf.layers.conv2d(down_x4_fused_dwon_x8, 128, (3, 3), strides=(1, 1), padding='SAME', activation='relu')
        print(
            f'down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_1.shape: {down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_1.shape}')

        if is_training:
            prediction = tf.layers.conv2d(down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_1, num_classes, (1, 1),
                                          strides=(1, 1), padding='SAME',
                                          activation=None, name='down_x16_cls')
            down_x16_label = self.__resize_layer_by_factor(y, 1 / 16, ResizeMethod.NEAREST_NEIGHBOR)
            down_x16_label = tf.squeeze(down_x16_label, axis=-1)
            down_x16_label = tf.cast(down_x16_label, dtype=tf.int32)
            down_x16_loss = self.__compute_loss(prediction, down_x16_label)
        down_x4_fused_dwon_x8down_x4_fused_dwon_x8_bn_1 = tf.layers.batch_normalization(
            down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_1, training=is_training)
        down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_2 = \
            tf.layers.conv2d(down_x4_fused_dwon_x8down_x4_fused_dwon_x8_bn_1, 256, (1, 1), strides=(1, 1),
                             padding='SAME', activation='relu')
        print(
            f'down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_2.shape: {down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_2.shape}')
        down_x4_fused_dwon_x8down_x4_fused_dwon_x8_up = self.__resize_layer_by_factor(
            down_x4_fused_dwon_x8down_x4_fused_dwon_x8_conv_2, 2)
        print(
            f'down_x4_fused_dwon_x8down_x4_fused_dwon_x8_up.shape: {down_x4_fused_dwon_x8down_x4_fused_dwon_x8_up.shape}')

        down_x8_fusion = down_x4_fused_dwon_x8down_x4_fused_dwon_x8_up + down_x4_conv_2_2
        down_x8_fusion_improved = tf.layers.conv2d(down_x8_fusion, 64, (3, 3), strides=(1, 1), padding='SAME',
                                                   activation='relu')
        if is_training:
            prediction = tf.layers.conv2d(down_x8_fusion_improved, num_classes, (1, 1),
                                          strides=(1, 1), padding='SAME',
                                          activation=None, name='down_x8_cls')
            down_x8_label = self.__resize_layer_by_factor(y, 1 / 8, ResizeMethod.NEAREST_NEIGHBOR)
            down_x8_label = tf.squeeze(down_x8_label, axis=-1)
            down_x8_label = tf.cast(down_x8_label, dtype=tf.int32)
            down_x8_loss = self.__compute_loss(prediction, down_x8_label)

        down_x8_fusion_improved_up = self.__resize_layer_by_factor(down_x8_fusion_improved, 2)
        down_x2_conv_1 = tf.layers.conv2d(downsampled_x2_bn, 64, (3, 3), strides=(2, 2), padding='SAME',
                                          activation='relu')
        print(f'down_x2_conv_1.shape: {down_x2_conv_1.shape}')
        final_fusion = down_x8_fusion_improved_up + down_x2_conv_1
        final_fusion_improved = tf.layers.conv2d(final_fusion, 64, (3, 3), strides=(1, 1), padding='SAME',
                                                 activation='relu')
        print(f'final_fusion_improved.shape: {final_fusion_improved.shape}')
        final_cls = tf.layers.conv2d(final_fusion_improved, num_classes, (1, 1), strides=(1, 1), padding='SAME',
                                     activation=None)
        print(f'final_cls.shape: {final_cls.shape}')
        if is_training:
            down_x4_label = self.__resize_layer_by_factor(y, 1 / 4, ResizeMethod.NEAREST_NEIGHBOR)
            down_x4_label = tf.squeeze(down_x4_label, axis=-1)
            down_x4_label = tf.cast(down_x4_label, dtype=tf.int32)
            down_x4_loss = self.__compute_loss(final_cls, down_x4_label)
            down_x32_loss = tf.reduce_mean(down_x32_loss)
            down_x16_loss = tf.reduce_mean(down_x16_loss)
            down_x8_loss = tf.reduce_mean(down_x8_loss)
            down_x4_loss = tf.reduce_mean(down_x4_loss)
            overall_loss = tf.stack([down_x32_loss, down_x16_loss, down_x8_loss, down_x4_loss], axis=0)
            overall_loss = tf.reduce_mean(overall_loss, axis=0)
        final_cls_up = self.__resize_layer_by_factor(final_cls, 4)
        print(f'final_cls_up.shape: {final_cls_up.shape}')
        return final_cls_up, overall_loss

    def __pyramid_pool_layer(self, input: tf.Tensor, kernels: List[Tuple[int, int]],
                             out_slices: int, downsample_input: bool = True) -> tf.Tensor:
        avg_1 = tf.nn.avg_pool(input, [1, kernels[0][0], kernels[0][1], 1], [1, kernels[0][0], kernels[0][1], 1],
                               padding='VALID')
        avg_2 = tf.nn.avg_pool(input, [1, kernels[1][0], kernels[1][1], 1], [1, kernels[1][0], kernels[0][1], 1],
                               padding='VALID')
        avg_3 = tf.nn.avg_pool(input, [1, kernels[2][0], kernels[2][1], 1], [1, kernels[2][0], kernels[0][1], 1],
                               padding='VALID')
        if downsample_input:
            output_size = self.__calculate_resized_input_dim(input.shape, 0.5)
        else:
            output_size = input.shape
        avg_1 = self.__resize_layer_to_target(avg_1, output_size)
        avg_2 = self.__resize_layer_to_target(avg_2, output_size)
        avg_3 = self.__resize_layer_to_target(avg_3, output_size)
        out = avg_1 + avg_2 + avg_3
        out = tf.layers.conv2d(out, out_slices, (3, 3), padding='SAME', activation='relu')
        return out

    def __resize_layer_to_target(self, input: tf.Tensor,
                                 target_size: Tuple[tf.Dimension, tf.Dimension, tf.Dimension, tf.Dimension],
                                 method: ResizeMethod = ResizeMethod.BILINEAR) -> tf.Tensor:
        return tf.image.resize(input, (target_size[1], target_size[2]), method=method)

    def __resize_layer_by_factor(self, input: tf.Tensor, resize_factor: float,
                                 method: ResizeMethod = ResizeMethod.BILINEAR) -> tf.Tensor:
        new_dim = self.__calculate_resized_input_dim(input.shape, resize_factor)
        new_dim = new_dim[1], new_dim[2]
        return tf.image.resize(input, new_dim, method=method)

    def __calculate_resized_input_dim(self, input_size: Tuple[tf.Dimension, tf.Dimension, tf.Dimension, tf.Dimension],
                                      scaling_factor: float) -> \
            Tuple[tf.Dimension, tf.Dimension, tf.Dimension, tf.Dimension]:

        def get_new_dimension_size(dim_size: tf.Dimension) -> tf.Dimension:
            if scaling_factor < 1:
                return dim_size // (1 / scaling_factor)
            else:
                return dim_size * scaling_factor

        dim_0, dim_1, dim_2, dim_3 = input_size
        dim_1 = get_new_dimension_size(dim_1)
        dim_2 = get_new_dimension_size(dim_2)
        return dim_0, dim_1, dim_2, dim_3

    def __compute_loss(self, prediction: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        to_ignore = tf.cast(tf.not_equal(gt, 0), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                              labels=gt)
        return tf.multiply(loss, to_ignore)
