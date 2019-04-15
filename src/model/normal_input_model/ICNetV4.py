from typing import Tuple, Optional

import tensorflow as tf

from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class ICNetV4(SemanticSegmentationModel):

    def run(self, X: tf.Tensor, num_classes: int, is_training: bool = True, y: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
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
            one_sixteenth_label = tf.image.resize_images(y, one_sixteenth_input_size)
            one_sixteenth_label = tf.squeeze(one_sixteenth_label, axis=-1)
            one_sixteenth_label = tf.cast(one_sixteenth_label, dtype=tf.int32)
            print('one_sixteenth_label {}'.format(one_sixteenth_label.shape))
        small_medium_fused, cascade_loss_1 = self.__cascade_fusion_block(small_branch_output, medium_branch_output,
                                                                         is_training, 1, label=one_sixteenth_label,
                                                                         num_classes=num_classes)
        print('small_medium_fused {}'.format(small_medium_fused.shape))
        one_eight_label = None
        if is_training:
            one_eight_label = tf.image.resize_images(y, one_eight_input_size)
            one_eight_label = tf.squeeze(one_eight_label, axis=-1)
            one_eight_label = tf.cast(one_eight_label, dtype=tf.int32)
            print('one_eight_label {}'.format(one_eight_label.shape))
        medium_big_fused, cascade_loss_2 = self.__cascade_fusion_block(small_medium_fused, big_branch_output,
                                                                       is_training, 2, label=one_eight_label,
                                                                       num_classes=num_classes)
        print('medium_big_fused {}'.format(medium_big_fused.shape))
        upsample_2_size = medium_big_fused.shape[1] * 2, medium_big_fused.shape[2] * 2
        upsampled_2 = tf.image.resize_bilinear(medium_big_fused, upsample_2_size)
        upsampled_2 = tf.layers.conv2d(upsampled_2, num_classes, (1, 1), padding='SAME', activation=None)
        print('upsampled_2 {}'.format(upsampled_2.shape))
        one_fourth_label = None
        cascade_loss_3 = None
        if is_training:
            one_fourth_label = tf.image.resize_images(y, one_fourth_input_size)
            one_fourth_label = tf.squeeze(one_fourth_label, axis=-1)
            one_fourth_label = tf.cast(one_fourth_label, dtype=tf.int32)
            print('one_fourth_label {}'.format(one_fourth_label.shape))
            intermediate_classifier = self.__filters_scaling(upsampled_2, num_classes, 'cascade_classfier_one_fourth')
            cascade_loss_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=intermediate_classifier,
                                                                            labels=one_fourth_label)
        upsample_4_size = upsampled_2.shape[1] * 4, upsampled_2.shape[2] * 4
        classifier = tf.image.resize_bilinear(upsampled_2, upsample_4_size)
        main_loss = None
        if is_training:
            y = tf.squeeze(y, axis=-1)
            main_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classifier, labels=y)
        overall_loss = None
        if is_training:
            cascade_loss_1 = tf.reduce_mean(cascade_loss_1)
            cascade_loss_2 = tf.reduce_mean(cascade_loss_2)
            cascade_loss_3 = tf.reduce_mean(cascade_loss_3)
            main_loss = tf.reduce_mean(main_loss)
            overall_loss = tf.stack([cascade_loss_1, cascade_loss_2, cascade_loss_3, main_loss], axis=0)
            overall_loss = tf.reduce_mean(overall_loss, axis=0)
        return classifier, overall_loss

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
        dil_1 = self.__dilated_block(medium_branch_common_output, 128, residual=False)
        bn_1 = tf.layers.batch_normalization(dil_1, training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1, 128)
        dil_2_1 = self.__dilated_block(pool_1, 128)
        bn_2 = tf.layers.batch_normalization(dil_2_1, training=is_training)
        dil_2_2 = self.__dilated_block(bn_2, 128)
        add_1 = tf.math.add(dil_2_2, pool_1)

        return add_1

    def __downsample_conv_block(self, input: tf.Tensor, filters: int, kernel_size: Tuple[int, int]) -> tf.Tensor:
        return tf.layers.conv2d(input, filters, kernel_size, strides=[2, 2], padding='SAME')

    def __small_branch_head(self, medium_branch_common_output: tf.Tensor, is_training: bool) -> tf.Tensor:
        input_scaled = self.__filters_scaling(medium_branch_common_output, 64)

        dil_1 = self.__dilated_block(input_scaled, 128, dim_red=False, residual=False)
        bn_1 = tf.layers.batch_normalization(dil_1, training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1, 128)
        dil_2_1 = self.__dilated_block(pool_1, 128)
        bn_2 = tf.layers.batch_normalization(dil_2_1, training=is_training)
        dil_2_2 = self.__dilated_block(bn_2, 128)
        add_1 = tf.math.add(dil_2_2, pool_1)

        bn_3 = tf.layers.batch_normalization(add_1, training=is_training)
        pool_2 = self.__pyramid_pooling(bn_3, 256)
        dil_3_1 = self.__dilated_block(pool_2, 256)
        bn_4 = tf.layers.batch_normalization(dil_3_1, training=is_training)
        dil_3_2 = self.__dilated_block(bn_4, 256)
        add_2 = tf.math.add(dil_3_2, pool_2)

        bn_7 = tf.layers.batch_normalization(add_2, training=is_training)
        dil_3_5 = self.__dilated_block(bn_7, 512, residual=False)
        bn_8 = tf.layers.batch_normalization(dil_3_5, training=is_training)
        dil_3_6 = self.__dilated_block(bn_8, 512)
        dil_3_6_down = self.__filters_scaling(dil_3_6, 256)
        add_3 = tf.math.add(dil_3_6_down, add_2)

        return add_3

    def __dilated_block(self, X: tf.Tensor, output_filters: int, dim_red: bool = True, residual: bool = True) -> tf.Tensor:
        if dim_red:
            feed_1 = self.__filters_scaling(X, 64)
            feed_2 = self.__filters_scaling(X, 64)
            feed_3 = self.__filters_scaling(X, 64)
            feed_4 = self.__filters_scaling(X, 64)
        else:
            feed_1 = feed_2 = feed_3 = feed_4 = X
        out_1 = tf.layers.conv2d(feed_1, output_filters, (3, 3), padding='SAME', activation='relu')
        out_2 = tf.layers.conv2d(feed_2, output_filters, (3, 3), padding='SAME', activation='relu', dilation_rate=(2, 2))
        out_3 = tf.layers.conv2d(feed_3, output_filters, (3, 3), padding='SAME', activation='relu', dilation_rate=(4, 4))
        out_4 = tf.layers.conv2d(feed_4, output_filters, (3, 3), padding='SAME', activation='relu', dilation_rate=(8, 8))
        out = out_1 + out_2 + out_3 + out_4
        if residual:
            out = tf.math.add(out, X)

        return out

    def __filters_scaling(self, X: tf.Tensor, num_filters: int, name: Optional[str] = None) -> tf.Tensor:
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation='relu', name=name)

    def __pyramid_pooling(self, X: tf.Tensor, num_filters: int) -> tf.Tensor:
        feed_1 = self.__filters_scaling(X, 64)
        pool_1 = tf.nn.pool(feed_1, [2, 2], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        feed_2 = self.__filters_scaling(X, 64)
        pool_2 = tf.nn.pool(feed_2, [3, 3], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        feed_3 = self.__filters_scaling(X, 64)
        pool_3 = tf.nn.pool(feed_3, [5, 5], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        out = pool_1 + pool_2 + pool_3
        out = self.__filters_scaling(out, num_filters)
        return out

    def __cascade_fusion_block(self, smaller_input: tf.Tensor, bigger_input: tf.Tensor, is_training: bool, casc_op_number: int,
                               label: Optional[tf.Tensor] = None, num_classes: Optional[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        upsample_size = smaller_input.shape[1] * 2, smaller_input.shape[2] * 2
        upsampled = tf.image.resize_bilinear(smaller_input, upsample_size)
        cascade_loss = None
        if is_training:
            intermediate_classifier = self.__filters_scaling(upsampled, num_classes, 'cascade_classifier_{}'.format(casc_op_number))
            cascade_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=intermediate_classifier, labels=label)
        upsampled = tf.layers.conv2d(upsampled, 64, (3, 3), padding='SAME', activation=None, dilation_rate=(2, 2))
        upsampled_bn = tf.layers.batch_normalization(upsampled, training=is_training)
        bigger_input = self.__filters_scaling(bigger_input, 64)
        print('{} {}'.format(upsampled_bn.shape, bigger_input.shape))
        out = tf.math.add(upsampled_bn, bigger_input)
        return tf.nn.relu(out), cascade_loss

    def __downsample_input_size(self, input_size: Tuple[tf.Dimension, tf.Dimension], downsampling_factor: int) -> Tuple[tf.Dimension, tf.Dimension]:


        def get_new_dimenstion_size(dim_size: tf.Dimension) -> tf.Dimension:
            return dim_size // downsampling_factor

        dim_1, dim_2 = input_size
        dim_1 = get_new_dimenstion_size(dim_1)
        dim_2 = get_new_dimenstion_size(dim_2)
        return dim_1, dim_2

    

