from typing import Tuple, Optional

import tensorflow as tf
from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class OriginalICNet(SemanticSegmentationModel):

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
        small_branch_output = self.__small_branch_head(medium_branch_common_output, is_training)
        medium_branch_common_output = self.__filters_scaling(medium_branch_common_output, 128, activation=None)
        fused_1 = tf.nn.relu(small_branch_output + medium_branch_common_output)
        cascade_loss_1, cascade_loss_2, cascade_loss_3, overall_loss = None, None, None, None
        if is_training:
            intermediate_classifier = self.__filters_scaling(fused_1, num_classes, 'cascade_classifier_1')
            label = tf.cast(tf.image.resize_images(y, one_sixteenth_input_size), dtype=tf.int32)
            label = tf.squeeze(label, axis=-1)
            print(intermediate_classifier.shape)
            print(label.shape)
            cascade_loss_1 = tf.reduce_mean(self.__compute_loss(intermediate_classifier, label))
        upsample = tf.image.resize(fused_1, one_eight_input_size)
        upsample_improved = tf.layers.conv2d(upsample, 128, kernel_size=(3, 3), dilation_rate=(2, 2), activation=None, padding='SAME')
        fused_2 = tf.nn.relu(upsample_improved + big_branch_output)
        if is_training:
            intermediate_classifier = self.__filters_scaling(fused_2, num_classes, 'cascade_classifier_2')
            label = tf.cast(tf.image.resize_images(y, one_eight_input_size), dtype=tf.int32)
            label = tf.squeeze(label, axis=-1)
            cascade_loss_2 = tf.reduce_mean(self.__compute_loss(intermediate_classifier, label))
            print(overall_loss)
        upsample_2 = tf.image.resize(fused_2, one_fourth_input_size)
        fused_3 = self.__filters_scaling(upsample_2, num_classes)
        if is_training:
            intermediate_classifier = self.__filters_scaling(fused_3, num_classes, 'cascade_classifier_3')
            label = tf.cast(tf.image.resize_images(y, one_fourth_input_size), dtype=tf.int32)
            label = tf.squeeze(label, axis=-1)
            cascade_loss_3 = tf.reduce_mean(self.__compute_loss(intermediate_classifier, label))
            print(overall_loss)
        classifier = tf.image.resize(fused_3, input_image_size)
        print(classifier.shape)
        if is_training:
            y = tf.squeeze(y, axis=-1)
            overall_loss = self.__compute_loss(classifier, y)
            overall_loss = tf.reduce_mean(overall_loss)
            overall_loss = tf.stack([0.064 * cascade_loss_1, 0.16 * cascade_loss_2, 0.4 * cascade_loss_3, overall_loss], axis=0)
            overall_loss = tf.reduce_mean(overall_loss, axis=0)
            print(overall_loss)
        print('RETURN: {}'.format(overall_loss))
        return classifier, overall_loss

    def __compute_loss(self, prediction: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        to_ignore = tf.cast(tf.not_equal(gt, 0), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                              labels=gt)
        return tf.multiply(loss, to_ignore)

    def __top_branch(self, input: tf.Tensor, is_training: bool) -> tf.Tensor:
        halved_conv = self.__downsample_conv_block(input, 32, (3, 3))
        quater_conv = self.__downsample_conv_block(halved_conv, 32, (3, 3))
        one_eight_conv = self.__downsample_conv_block(quater_conv, 64, (3, 3))
        top_out = self.__filters_scaling(one_eight_conv, 128, activation=None)
        return top_out

    def __medium_branch_head(self, half_input: tf.Tensor, is_training: bool) -> tf.Tensor:
        quater_conv_1 = self.__downsample_conv_block(half_input, 32, (3, 3))
        quater_conv_2 = tf.layers.conv2d(quater_conv_1, 32, (3, 3), padding='SAME')
        quater_conv_3 = tf.layers.conv2d(quater_conv_2, 64, (3, 3), padding='SAME')
        pool_1 = tf.nn.pool(quater_conv_3, [3, 3], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        proj_conv = self.__filters_scaling(pool_1, 128, activation=None)
        conv_2_proj = self.__reduce_increase_block(pool_1)
        block_1 = tf.math.add(proj_conv, conv_2_proj)
        block_1 = tf.nn.relu(block_1)
        block_2 = self.__reduce_increase_block(block_1)
        block_2 = tf.math.add(block_1, block_2)
        block_2 = tf.nn.relu(block_2)
        block_3 = self.__reduce_increase_block(block_2)
        block_3 = tf.math.add(block_2, block_3)
        block_3 = tf.nn.relu(block_3)
        conv_out_1 = tf.layers.conv2d(block_3, 64, (1, 1), strides=(2, 2), activation='relu', padding='SAME')
        conv_out_2 = tf.layers.conv2d(conv_out_1, 64, (3, 3), strides=(1, 1), activation='relu', padding='SAME')
        conv_out_3 = self.__filters_scaling(conv_out_2, 256, activation=None)
        conv_3_proj = self.__downsample_conv_block(block_3, 256, (1, 1), acticvation=None)
        out = tf.math.add(conv_out_3, conv_3_proj)
        return tf.nn.relu(out)

    def __reduce_increase_block(self, input: tf.Tensor, factor: int = 1, dilation: Tuple[int, int] = (1, 1)) -> tf.Tensor:
        conv_1 = self.__filters_scaling(input, 32 * factor)
        conv_2 = tf.layers.conv2d(conv_1, 32 * factor, (3, 3), padding='SAME', activation='relu', dilation_rate=dilation)
        conv_2_proj = self.__filters_scaling(conv_2, 128 * factor, activation=None)
        return conv_2_proj

    def __downsample_conv_block(self, input: tf.Tensor, filters: int, kernel_size: Tuple[int, int], acticvation: Optional[str] = 'relu') -> tf.Tensor:
        return tf.layers.conv2d(input, filters, kernel_size, strides=[2, 2], padding='SAME', activation=acticvation)

    def __small_branch_head(self, medium_branch_common_output: tf.Tensor, is_training: bool) -> tf.Tensor:
        shape = medium_branch_common_output.shape[1], medium_branch_common_output.shape[2]
        half_input_size = self.__downsample_input_size(shape, 2)
        half_input = tf.image.resize_images(medium_branch_common_output, half_input_size)
        block_1 = self.__reduce_increase_block(half_input, factor=2)
        block_1 = tf.math.add(half_input, block_1)
        block_1 = tf.nn.relu(block_1)
        block_2 = self.__reduce_increase_block(block_1, factor=2)
        block_2 = tf.math.add(block_1, block_2)
        block_2 = tf.nn.relu(block_2)
        block_3 = self.__reduce_increase_block(block_2, factor=2)
        block_3 = tf.math.add(block_2, block_3)
        block_3 = tf.nn.relu(block_3)
        conv_reduce_1 = self.__filters_scaling(block_3, 128)
        conv_2 = tf.layers.conv2d(conv_reduce_1, 128, kernel_size=(3, 3), padding='SAME', dilation_rate=(2, 2))
        conv_increase = self.__filters_scaling(conv_2, 512, activation=None)
        block_projection = self.__filters_scaling(block_3, 512, activation=None)
        add_1 = tf.math.add(conv_increase, block_projection)
        add_1_relu = tf.nn.relu(add_1)
        block_4 = self.__reduce_increase_block(add_1_relu, factor=4, dilation=(2, 2))
        block_4 = tf.math.add(block_4, add_1_relu)
        block_4 = tf.nn.relu(block_4)
        block_5 = self.__reduce_increase_block(block_4, factor=4, dilation=(2, 2))
        block_5 = tf.math.add(block_5, block_4)
        block_5 = tf.nn.relu(block_5)
        block_6 = self.__reduce_increase_block(block_5, factor=4, dilation=(2, 2))
        block_6 = tf.math.add(block_5, block_6)
        block_6 = tf.nn.relu(block_6)
        block_7 = self.__reduce_increase_block(block_6, factor=4, dilation=(2, 2))
        block_7 = tf.math.add(block_7, block_6)
        block_7 = tf.nn.relu(block_7)
        block_8 = self.__reduce_increase_block(block_7, factor=4, dilation=(2, 2))
        block_8 = tf.math.add(block_7, block_8)
        block_8 = tf.nn.relu(block_8)
        block_9 = self.__reduce_increase_block(block_8, factor=8, dilation=(4, 4))
        block_8_proj = self.__filters_scaling(block_8, 1024)
        block_9 = tf.math.add(block_9, block_8_proj)
        block_9 = tf.nn.relu(block_9)
        block_10 = self.__reduce_increase_block(block_9, factor=8, dilation=(4, 4))
        block_10 = tf.math.add(block_9, block_10)
        block_10 = tf.nn.relu(block_10)
        block_11 = self.__reduce_increase_block(block_10, factor=8, dilation=(4, 4))
        block_11 = tf.math.add(block_11, block_10)
        block_11 = tf.nn.relu(block_11)
        pool_1 = tf.layers.average_pooling2d(block_11, (32, 64), (32, 64), 'SAME')
        pool_1_interp = tf.image.resize(pool_1, (32, 64))
        pool_2 = tf.layers.average_pooling2d(block_11, (16, 32), (16, 32), 'SAME')
        pool_2_interp = tf.image.resize(pool_2, (32, 64))
        pool_3 = tf.layers.average_pooling2d(block_11, (13, 25), (10, 20), 'SAME')
        pool_3_interp = tf.image.resize(pool_3, (32, 64))
        pool_4 = tf.layers.average_pooling2d(block_11, (8, 16), (5, 10), 'SAME')
        pool_4_interp = tf.image.resize(pool_4, (32, 64))
        pyramid_pooling = pool_1_interp + pool_2_interp + pool_3_interp + pool_4_interp
        pyramid_pooling = self.__filters_scaling(pyramid_pooling, 256)
        zoom = tf.image.resize(pyramid_pooling, shape)
        zoom_improved = tf.layers.conv2d(zoom, 128, kernel_size=(3, 3), dilation_rate=(2, 2), activation=None, padding='SAME')
        print(zoom_improved.shape)
        return zoom_improved

    def __filters_scaling(self, X: tf.Tensor, num_filters: int, name: Optional[str] = None, activation: Optional[str] = 'relu') -> tf.Tensor:
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation=activation, name=name)

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
