from typing import Tuple, Optional

import tensorflow as tf

from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class ICNetV14(SemanticSegmentationModel):

    def run(self, X: tf.Tensor, num_classes: int, is_training: bool = True, y: Optional[tf.Tensor] = None) -> Tuple[
        tf.Tensor, Optional[tf.Tensor]]:
        if is_training:
            y = tf.expand_dims(y, axis=-1)
        input_image_size = X.shape[1], X.shape[2]
        half_input_size = self.__downsample_input_size(input_image_size, 2)
        one_fourth_input_size = self.__downsample_input_size(input_image_size, 4)
        one_eight_input_size = self.__downsample_input_size(input_image_size, 8)
        one_sixteenth_input_size = self.__downsample_input_size(input_image_size, 16)
        half_input = tf.image.resize_images(X, half_input_size)
        big_branch_output = self.__top_branch(X, is_training)
        print('big_branch_output {}'.format(big_branch_output))
        medium_branch_common_output = self.__medium_branch_head(half_input, is_training)
        print('medium_branch_common_output {}'.format(medium_branch_common_output.shape))
        medium_branch_output = self.__medium_branch_tail(medium_branch_common_output, is_training)
        print('medium_branch_output {}'.format(medium_branch_output.shape))
        small_branch_output = self.__small_branch_head(medium_branch_common_output, is_training)
        print('small_branch_output {}'.format(small_branch_output.shape))
        one_sixteenth_label = None
        if is_training:
            one_sixteenth_label = tf.image.resize_nearest_neighbor(y, one_sixteenth_input_size)
            one_sixteenth_label = tf.squeeze(one_sixteenth_label, axis=-1)
            one_sixteenth_label = tf.cast(one_sixteenth_label, dtype=tf.int32)
            print('one_sixteenth_label {}'.format(one_sixteenth_label.shape))
        small_medium_fused, cascade_loss_1 = self.__cascade_fusion_block(small_branch_output, medium_branch_output,
                                                                         is_training, 1, label=one_sixteenth_label,
                                                                         num_classes=num_classes)
        print('small_medium_fused {}'.format(small_medium_fused.shape))
        one_eight_label = None
        if is_training:
            one_eight_label = tf.image.resize_nearest_neighbor(y, one_eight_input_size)
            one_eight_label = tf.squeeze(one_eight_label, axis=-1)
            one_eight_label = tf.cast(one_eight_label, dtype=tf.int32)
            print('one_eight_label {}'.format(one_eight_label.shape))
        medium_big_fused, cascade_loss_2 = self.__cascade_fusion_block(small_medium_fused, big_branch_output,
                                                                       is_training, 2, label=one_eight_label,
                                                                       num_classes=num_classes)
        print('medium_big_fused {}'.format(medium_big_fused.shape))
        upsample_2_size = medium_big_fused.shape[1] * 2, medium_big_fused.shape[2] * 2
        upsampled_2 = tf.image.resize_bilinear(medium_big_fused, upsample_2_size, align_corners=True)
        upsampled_2 = tf.layers.conv2d(upsampled_2, num_classes, (1, 1), padding='SAME', activation=None)
        print('upsampled_2 {}'.format(upsampled_2.shape))
        one_fourth_label = None
        cascade_loss_3 = None
        if is_training:
            one_fourth_label = tf.image.resize_nearest_neighbor(y, one_fourth_input_size)
            one_fourth_label = tf.squeeze(one_fourth_label, axis=-1)
            one_fourth_label = tf.cast(one_fourth_label, dtype=tf.int32)
            print('one_fourth_label {}'.format(one_fourth_label.shape))
            cascade_loss_3 = self.__compute_loss(upsampled_2, one_fourth_label)
        upsample_4_size = upsampled_2.shape[1] * 4, upsampled_2.shape[2] * 4
        classifier = tf.image.resize_bilinear(upsampled_2, upsample_4_size, align_corners=True)
        overall_loss = None
        if is_training:
            cascade_loss_1 = tf.reduce_mean(cascade_loss_1)
            cascade_loss_2 = tf.reduce_mean(cascade_loss_2)
            cascade_loss_3 = tf.reduce_mean(cascade_loss_3)
            overall_loss = tf.stack([cascade_loss_1, cascade_loss_2, cascade_loss_3], axis=0)
            overall_loss = tf.reduce_mean(overall_loss, axis=0)
        return classifier, overall_loss

    def __compute_loss(self, prediction: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        to_ignore = tf.cast(tf.not_equal(gt, 0), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                              labels=gt)
        return tf.multiply(loss, to_ignore)

    def __top_branch(self, input: tf.Tensor, is_training: bool) -> tf.Tensor:
        halved_conv = self.__downsample_conv_block(input, 8, (3, 3))
        quater_conv = self.__downsample_conv_block(halved_conv, 16, (3, 3))
        one_eight_conv = self.__downsample_conv_block(quater_conv, 64, (3, 3))
        bn_1 = tf.layers.batch_normalization(one_eight_conv, training=is_training)
        return bn_1

    def __medium_branch_head(self, half_input: tf.Tensor, is_training: bool) -> tf.Tensor:
        quater_conv_1 = self.__downsample_conv_block(half_input, 16, (3, 3))
        quater_conv_2 = tf.layers.conv2d(quater_conv_1, 16, (3, 3), padding='SAME')
        bn_1 = tf.layers.batch_normalization(quater_conv_2, training=is_training)
        one_eiqght_conv_1 = self.__downsample_conv_block(bn_1, 32, (3, 3))
        one_eiqght_conv_2 = tf.layers.conv2d(one_eiqght_conv_1, 64, (3, 3), padding='SAME')
        bn_2 = tf.layers.batch_normalization(one_eiqght_conv_2, training=is_training)
        return bn_2

    def __medium_branch_tail(self, medium_branch_common_output, is_training):
        conv_1 = self.__filters_scaling(medium_branch_common_output, 64, name='medium_branch_tail_conv_1')
        conv_2 = tf.layers.conv2d(conv_1, 128, (3, 3), padding='SAME', activation='relu', name='medium_branch_tail_conv_2')
        conv_3 = self.__filters_scaling(conv_2, 64, name='medium_branch_tail_conv_3')
        conv_4 = tf.layers.conv2d(conv_3, 128, (3, 3), padding='SAME', activation='relu',
                                  name='medium_branch_tail_conv_4')
        conv_5 = self.__filters_scaling(conv_4, 64, name='medium_branch_tail_conv_5')
        fuse = conv_1 + conv_5
        bn_1 = tf.layers.batch_normalization(fuse, training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1, 128)
        dil_2_1 = self.__dilated_block(pool_1, 128)
        bn_2 = tf.layers.batch_normalization(dil_2_1, training=is_training)
        conv_6 = self.__filters_scaling(bn_2, 64, name='medium_branch_tail_conv_6')
        conv_7 = tf.layers.conv2d(conv_6, 128, (3, 3), padding='SAME', activation='relu',
                                  name='medium_branch_tail_conv_7')
        conv_8 = self.__filters_scaling(conv_7, 64, name='medium_branch_tail_conv_8')
        conv_9 = tf.layers.conv2d(conv_8, 256, (3, 3), padding='SAME', activation='relu',
                                  name='medium_branch_tail_conv_9')
        conv_10 = self.__filters_scaling(conv_9, 128, name='medium_branch_tail_conv_10')
        add_1 = tf.math.add(conv_10, pool_1)

        return add_1

    def __downsample_conv_block(self, input: tf.Tensor, filters: int, kernel_size: Tuple[int, int]) -> tf.Tensor:
        return tf.layers.conv2d(input, filters, kernel_size, strides=[2, 2], padding='SAME')

    def __small_branch_head(self, medium_branch_common_output: tf.Tensor, is_training: bool) -> tf.Tensor:
        input_scaled = self.__filters_scaling(medium_branch_common_output, 64)
        bn_1 = tf.layers.batch_normalization(input_scaled, training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1, 128)
        conv_1 = self.__filters_scaling(pool_1, 64, name='small_branch_conv_1')
        conv_2 = tf.layers.conv2d(conv_1, 128, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_2')
        conv_3 = self.__filters_scaling(conv_2, 64, name='small_branch_conv_3')
        conv_4 = tf.layers.conv2d(conv_3, 256, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_4')
        conv_5 = self.__filters_scaling(conv_4, 128, name='small_branch_conv_5')
        add_1 = tf.math.add(conv_5, pool_1)
        conv_5b = tf.layers.conv2d(add_1, 256, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_5b')
        conv_5c = self.__filters_scaling(conv_5b, 128, name='small_branch_conv_5c')

        bn_3 = tf.layers.batch_normalization(conv_5c, training=is_training)
        pool_2 = self.__pyramid_pooling(bn_3, 256)
        dil_3_1 = self.__dilated_block(pool_2, 256)
        conv_6 = self.__filters_scaling(dil_3_1, 64, name='small_branch_conv_6')
        conv_7 = tf.layers.conv2d(conv_6, 256, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_7')
        conv_8 = self.__filters_scaling(conv_7, 64, name='small_branch_conv_8')
        conv_9 = tf.layers.conv2d(conv_8, 512, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_9')
        conv_10 = self.__filters_scaling(conv_9, 256, name='small_branch_conv_10')
        add_2 = tf.math.add(conv_10, pool_2)

        bn_7 = tf.layers.batch_normalization(add_2, training=is_training)
        bn_8 = tf.layers.batch_normalization(bn_7, training=is_training)
        conv_11 = self.__filters_scaling(bn_8, 128, name='small_branch_conv_11')
        conv_12 = tf.layers.conv2d(conv_11, 512, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_12')
        conv_13 = self.__filters_scaling(conv_12, 128, name='small_branch_conv_13')
        conv_13 = conv_13 + conv_11
        conv_14 = tf.layers.conv2d(conv_13, 512, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_14')
        conv_15 = self.__filters_scaling(conv_14, 128, name='small_branch_conv_15')
        conv_15_a = tf.layers.conv2d(conv_15, 512, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_15_a')
        conv_15_b = self.__filters_scaling(conv_15_a, 128, name='small_branch_conv_15_b')
        conv_15_b = conv_15_b + conv_13
        conv_16 = tf.layers.conv2d(conv_15_b, 768, (3, 3), padding='SAME', activation='relu', name='small_branch_conv_16')
        conv_17 = self.__filters_scaling(conv_16, 128, name='small_branch_conv_17')
        conv_17a = tf.layers.conv2d(conv_17, 768, (3, 3), padding='SAME', activation='relu',
                                    name='small_branch_conv_16a')
        conv_17b = self.__filters_scaling(conv_17a, 128, name='small_branch_conv_17b')
        conv_17b = conv_17b + conv_15_b
        conv_18 = tf.layers.conv2d(conv_17b, 1024, (3, 3), padding='SAME', activation='relu',
                                   name='small_branch_conv_18')
        conv_19 = self.__filters_scaling(conv_18, 256, name='small_branch_conv_19')
        conv_19a = tf.layers.conv2d(conv_19, 1024, (3, 3), padding='SAME', activation='relu',
                                    name='small_branch_conv_19a')
        conv_19b = self.__filters_scaling(conv_19a, 256, name='small_branch_conv_19b')
        conv_19b = conv_19b + conv_19
        add_3 = tf.math.add(conv_19b, add_2)

        conv_20 = tf.layers.conv2d(add_3, 2048, (3, 3), padding='SAME', activation='relu',
                                   name='small_branch_conv_20')
        conv_21 = self.__filters_scaling(conv_20, 512, name='small_branch_conv_21')
        conv_22 = tf.layers.conv2d(conv_21, 2048, (3, 3), padding='SAME',
                                   activation='relu',
                                   name='small_branch_conv_22')
        conv_23 = self.__filters_scaling(conv_22, 256,
                                         name='small_branch_conv_23')
        return add_3 + conv_23

    def __dilated_block(self, X: tf.Tensor, output_filters: int, dim_red: bool = True,
                        residual: bool = True) -> tf.Tensor:
        if dim_red:
            feed_1 = self.__filters_scaling(X, 64)
            feed_2 = self.__filters_scaling(X, 64)
            feed_3 = self.__filters_scaling(X, 64)
            feed_4 = self.__filters_scaling(X, 64)
        else:
            feed_1 = feed_2 = feed_3 = feed_4 = X
        out_1 = tf.layers.conv2d(feed_1, output_filters, (3, 3), padding='SAME', activation='relu')
        out_2 = tf.layers.conv2d(feed_2, output_filters, (3, 3), padding='SAME', activation='relu',
                                 dilation_rate=(2, 2))
        out_3 = tf.layers.conv2d(feed_3, output_filters, (3, 3), padding='SAME', activation='relu',
                                 dilation_rate=(4, 4))
        out_4 = tf.layers.conv2d(feed_4, output_filters, (3, 3), padding='SAME', activation='relu',
                                 dilation_rate=(8, 8))
        out = out_1 + out_2 + out_3 + out_4
        if residual:
            out = tf.math.add(out, X)

        return out

    def __filters_scaling(self, X: tf.Tensor, num_filters: int, name: Optional[str] = None) -> tf.Tensor:
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation='relu', name=name)

    def __pyramid_pooling(self, X: tf.Tensor, num_filters: int) -> tf.Tensor:
        pool_1 = tf.nn.pool(X, [2, 2], 'AVG', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        pool_2 = tf.nn.pool(X, [5, 5], 'AVG', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        pool_3 = tf.nn.pool(X, [8, 8], 'AVG', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        concat = tf.concat([pool_1, pool_2, pool_3], axis=-1)
        out = tf.layers.conv2d(concat, num_filters, (3, 3), padding='SAME', activation='relu')
        return out

    def __cascade_fusion_block(self, smaller_input: tf.Tensor, bigger_input: tf.Tensor, is_training: bool,
                               casc_op_number: int,
                               label: Optional[tf.Tensor] = None, num_classes: Optional[int] = None) -> Tuple[
        tf.Tensor, tf.Tensor]:
        upsample_size = smaller_input.shape[1] * 2, smaller_input.shape[2] * 2
        upsampled = tf.image.resize_bilinear(smaller_input, upsample_size, align_corners=True)
        cascade_loss = None
        if is_training:
            intermediate_classifier = self.__filters_scaling(upsampled, num_classes,
                                                             'cascade_classifier_{}'.format(casc_op_number))
            cascade_loss = self.__compute_loss(intermediate_classifier, label)
        upsampled = tf.layers.conv2d(upsampled, 64, (3, 3), padding='SAME', activation='relu')
        upsampled_bn = tf.layers.batch_normalization(upsampled, training=is_training)
        bigger_input = self.__filters_scaling(bigger_input, 64)
        print('{} {}'.format(upsampled_bn.shape, bigger_input.shape))
        out = tf.math.add(upsampled_bn, bigger_input)
        return tf.nn.relu(out), cascade_loss

    def __downsample_input_size(self, input_size: Tuple[tf.Dimension, tf.Dimension], downsampling_factor: int) -> Tuple[
        tf.Dimension, tf.Dimension]:

        def get_new_dimenstion_size(dim_size: tf.Dimension) -> tf.Dimension:
            return dim_size // downsampling_factor

        dim_1, dim_2 = input_size
        dim_1 = get_new_dimenstion_size(dim_1)
        dim_2 = get_new_dimenstion_size(dim_2)
        return dim_1, dim_2
