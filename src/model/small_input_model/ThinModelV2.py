import tensorflow as tf
from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class ThinModelV2(SemanticSegmentationModel):

    def run(self, X, num_classes, training=True):
        #encoder
        print(X.shape)
        input_scaled = self.__filters_scaling(X, 64)
        print(input_scaled.shape)
        dil_1 = self.__dilated_block(input_scaled)
        bn_1 = tf.contrib.layers.batch_norm(dil_1, is_training=training)
        pool_1 = self.__pyramid_pooling(bn_1)
        dil_2_1 = self.__dilated_block(pool_1)
        bn_2 = tf.contrib.layers.batch_norm(dil_2_1, is_training=training)
        dil_2_2 = self.__dilated_block(bn_2)
        res_2 = tf.math.add(dil_2_2, pool_1)
        print(res_2.shape)
        bn_3 = tf.contrib.layers.batch_norm(res_2, is_training=training)
        pool_2 = self.__pyramid_pooling(bn_3)
        dil_3_1 = self.__dilated_block(pool_2)
        bn_4 = tf.contrib.layers.batch_norm(dil_3_1, is_training=training)
        dil_3_2 = self.__dilated_block(bn_4)
        bn_5 = tf.contrib.layers.batch_norm(dil_3_2, is_training=training)
        dil_3_3 = self.__dilated_block(bn_5)
        bn_5_a = tf.contrib.layers.batch_norm(dil_3_3, is_training=training)
        dil_3_4 = self.__dilated_block(bn_5_a)
        res_3 = tf.math.add(dil_3_4, pool_2)
        print(res_3.shape)
        bn_6 = tf.contrib.layers.batch_norm(res_3, is_training=training)
        pool_3 = self.__pyramid_pooling(bn_6)
        dil_4_1 = self.__dilated_block(pool_3)
        bn_7 = tf.contrib.layers.batch_norm(dil_4_1, is_training=training)
        dil_4_2 = self.__dilated_block(bn_7)
        bn_8 = tf.contrib.layers.batch_norm(dil_4_2, is_training=training)
        dil_4_3 = self.__dilated_block(bn_8)
        bn_8_a = tf.contrib.layers.batch_norm(dil_4_3, is_training=training)
        dil_4_4 = self.__dilated_block(bn_8_a)
        bn_8_b = tf.contrib.layers.batch_norm(dil_4_4, is_training=training)
        dil_4_5 = self.__dilated_block(bn_8_b)
        res_4 = tf.math.add(dil_4_5, pool_3)
        print(res_4.shape)
        #decoder
        deconv_1 = self.__deconv_block(res_4)
        bn_9 = tf.contrib.layers.batch_norm(deconv_1, is_training=training)
        dec_res_1 = tf.math.add(bn_9, pool_2)
        deconv_2 = self.__deconv_block(dec_res_1)
        bn_10 = tf.contrib.layers.batch_norm(deconv_2, is_training=training)
        dec_res_2 = tf.math.add(bn_10, pool_1)
        deconv_3 = self.__deconv_block(dec_res_2)
        bn_11 = tf.contrib.layers.batch_norm(deconv_3, is_training=training)
        dec_res_3 = tf.math.add(bn_11, input_scaled)
        print(dec_res_3.shape)
        return self.__filters_scaling(dec_res_3, num_classes)

    def __dilated_block(self, X, dim_red=True, residual=True):
        if dim_red:
            feed_1 = self.__filters_scaling(X, 16)
            feed_2 = self.__filters_scaling(X, 16)
            feed_3 = self.__filters_scaling(X, 16)
            feed_4 = self.__filters_scaling(X, 16)
        else:
            feed_1 = feed_2 = feed_3 = feed_4 = X
        out_1 = tf.layers.conv2d(feed_1, 16, (5, 5), padding='SAME', activation='relu')
        print(out_1.shape)
        out_2 = tf.layers.conv2d(feed_2, 16, (3, 3), padding='SAME', activation='relu', dilation_rate=(2, 2))
        print(out_2.shape)
        out_3 = tf.layers.conv2d(feed_3, 16, (3, 3), padding='SAME', activation='relu', dilation_rate=(4, 4))
        print(out_3.shape)
        out_4 = tf.layers.conv2d(feed_4, 16, (3, 3), padding='SAME', activation='relu', dilation_rate=(8, 8))
        print(out_4.shape)
        out = out_1 + out_2 + out_3 + out_4
        out = self.__filters_scaling(out, 64)
        if residual:
            out = tf.math.add(out, X)
        print(out.shape)
        return out

    def __filters_scaling(self, X, num_filters):
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation='relu')

    def __pyramid_pooling(self, X):
        feed_1 = self.__filters_scaling(X, 16)
        pool_1 = tf.nn.pool(feed_1, [2, 2], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_1.shape)
        feed_2 = self.__filters_scaling(X, 16)
        pool_2 = tf.nn.pool(feed_2, [3, 3], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_2.shape)
        feed_3 = self.__filters_scaling(X, 16)
        pool_3 = tf.nn.pool(feed_3, [5, 5], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_3.shape)
        feed_4 = self.__filters_scaling(X, 16)
        pool_4 = tf.nn.pool(feed_4, [8, 8], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_4.shape)
        out = pool_1 + pool_2 + pool_3 + pool_4
        out = self.__filters_scaling(out, 64)
        print(out.shape)
        return out

    def __deconv_block(self, X):
        feed_1 = self.__filters_scaling(X, 16)
        out_1 = tf.layers.conv2d_transpose(feed_1, 16, (2, 2), strides=(2, 2), padding='SAME', activation='relu')
        print(out_1.shape)
        feed_2 = self.__filters_scaling(X, 16)
        out_2 = tf.layers.conv2d_transpose(feed_2, 16, (3, 3), strides=(2, 2), padding='SAME', activation='relu')
        print(out_2.shape)
        feed_3 = self.__filters_scaling(X, 16)
        out_3 = tf.layers.conv2d_transpose(feed_3, 16, (5, 5), strides=(2, 2), padding='SAME', activation='relu')
        print(out_3.shape)
        feed_4 = self.__filters_scaling(X, 16)
        out_4 = tf.layers.conv2d_transpose(feed_4, 16, (6, 6), strides=(2, 2), padding='SAME', activation='relu')
        print(out_4.shape)
        out = out_1 + out_2 + out_3 + out_4
        out = self.__filters_scaling(out, 64)
        print(out.shape)
        return out
