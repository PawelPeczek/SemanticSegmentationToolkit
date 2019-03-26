import tensorflow as tf
from typing import Union
from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class ThinModelV4(SemanticSegmentationModel):

    def run(self, X: tf.Tensor, num_classes: int, is_training: bool = True) -> tf.Tensor:
        #encoder
        input_scaled = self.__filters_scaling(X, 32)
        dil_1 = self.__dilated_block(input_scaled, 32)
        bn_1 = tf.layers.batch_normalization(dil_1, training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1, 64)
        dil_2_1 = self.__dilated_block(pool_1, 64)
        bn_2 = tf.layers.batch_normalization(dil_2_1, training=is_training)
        dil_2_2 = self.__dilated_block(bn_2, 64)
        bn_3 = tf.layers.batch_normalization(dil_2_2, training=is_training)
        pool_2 = self.__pyramid_pooling(bn_3, 128)
        dil_3_1 = self.__dilated_block(pool_2, 128)
        bn_4 = tf.layers.batch_normalization(dil_3_1, training=is_training)
        dil_3_2 = self.__dilated_block(bn_4, 128)
        bn_5 = tf.layers.batch_normalization(dil_3_2, training=is_training)
        dil_3_3 = self.__dilated_block(bn_5, 128)
        bn_5_a = tf.layers.batch_normalization(dil_3_3, training=is_training)
        dil_3_4 = self.__dilated_block(bn_5_a, 128)
        bn_6 = tf.layers.batch_normalization(dil_3_4, training=is_training)
        pool_3 = self.__pyramid_pooling(bn_6, 256)
        dil_4_1 = self.__dilated_block(pool_3, 256)
        bn_7 = tf.layers.batch_normalization(dil_4_1, training=is_training)
        dil_4_2 = self.__dilated_block(bn_7, 256)
        bn_8 = tf.layers.batch_normalization(dil_4_2, training=is_training)
        dil_4_3 = self.__dilated_block(bn_8, 256)
        bn_8_a = tf.layers.batch_normalization(dil_4_3, training=is_training)
        dil_4_4 = self.__dilated_block(bn_8_a, 256)
        bn_8_b = tf.layers.batch_normalization(dil_4_4, training=is_training)
        dil_4_5 = self.__dilated_block(bn_8_b, 256)

        #decoder
        deconv_1 = self.__deconv_block(dil_4_5, 128)
        bn_9 = tf.layers.batch_normalization(deconv_1, training=is_training)
        dec_res_1 = tf.math.add(bn_9, pool_2)
        dec_res_1_conv = self.__filters_scaling(dec_res_1, dec_res_1.shape[-1])
        deconv_2 = self.__deconv_block(dec_res_1_conv, 64)
        bn_10 = tf.layers.batch_normalization(deconv_2, training=is_training)
        dec_res_2 = tf.math.add(bn_10, pool_1)
        dec_res_2_conv = self.__filters_scaling(dec_res_2, dec_res_2.shape[-1])
        deconv_3 = self.__deconv_block(dec_res_2_conv, 32)
        bn_11 = tf.layers.batch_normalization(deconv_3, training=is_training)
        dec_res_3 = tf.math.add(bn_11, input_scaled)
        dec_res_3_conv = self.__filters_scaling(dec_res_3, dec_res_3.shape[-1])

        return self.__filters_scaling(dec_res_3_conv, num_classes, activation=None)

    def __dilated_block(self, X: tf.Tensor, output_slices: int) -> tf.Tensor:
        slice_depth = int(output_slices / 4)
        out_1 = tf.layers.conv2d(X, slice_depth, (3, 3), padding='SAME', activation='relu')
        out_2 = tf.layers.conv2d(X, slice_depth, (3, 3), padding='SAME', activation='relu', dilation_rate=(2, 2))
        out_3 = tf.layers.conv2d(X, slice_depth, (3, 3), padding='SAME', activation='relu', dilation_rate=(4, 4))
        out_4 = tf.layers.conv2d(X, slice_depth, (3, 3), padding='SAME', activation='relu', dilation_rate=(6, 6))
        out = tf.concat([out_1, out_2, out_3, out_4], axis=3)
        return out

    def __filters_scaling(self, X: tf.Tensor, num_filters: int, activation: Union[None, str] = 'relu') -> tf.Tensor:
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation=activation)

    def __pyramid_pooling(self, X, output_slices):
        slice_depth = int(output_slices / 4)
        feed_1 = self.__filters_scaling(X, slice_depth)
        pool_1 = tf.nn.pool(feed_1, [2, 2], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        feed_2 = self.__filters_scaling(X, slice_depth)
        pool_2 = tf.nn.pool(feed_2, [3, 3], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        feed_3 = self.__filters_scaling(X, slice_depth)
        pool_3 = tf.nn.pool(feed_3, [4, 4], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        feed_4 = self.__filters_scaling(X, slice_depth)
        pool_4 = tf.nn.pool(feed_4, [6, 6], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        out = tf.concat([pool_1, pool_2, pool_3, pool_4], axis=3)
        return out

    def __deconv_block(self, X: tf.Tensor, output_slices: int) -> tf.Tensor:
        slice_depth = int(output_slices / 4)
        out_1 = tf.layers.conv2d_transpose(X, slice_depth, (2, 2), strides=(2, 2), padding='SAME', activation='relu')
        out_2 = tf.layers.conv2d_transpose(X, slice_depth, (3, 3), strides=(2, 2), padding='SAME', activation='relu')
        out_3 = tf.layers.conv2d_transpose(X, slice_depth, (5, 5), strides=(2, 2), padding='SAME', activation='relu')
        out_4 = tf.layers.conv2d_transpose(X, slice_depth, (6, 6), strides=(2, 2), padding='SAME', activation='relu')
        out = tf.concat([out_1, out_2, out_3, out_4], axis=3)
        return out
