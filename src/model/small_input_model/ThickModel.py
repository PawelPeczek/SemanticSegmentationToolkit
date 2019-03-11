import tensorflow as tf
from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class ThickModel(SemanticSegmentationModel):

    def run(self, X, num_classes, is_training=True):
        #encoder
        print(X.shape)
        input_scaled = self.__filters_scaling(X, 64)
        print(input_scaled.shape)
        dil_1 = self.__dilated_block(input_scaled)
        bn_1 = tf.contrib.layers.batch_norm(dil_1, is_training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1)
        dil_2_1 = self.__dilated_block(pool_1)
        bn_2 = tf.contrib.layers.batch_norm(dil_2_1, is_training=is_training)
        dil_2_2 = self.__dilated_block(bn_2)
        res_2 = tf.math.add(dil_2_2, pool_1)
        print(res_2.shape)
        bn_3 = tf.contrib.layers.batch_norm(res_2, is_training=is_training)
        pool_2 = self.__pyramid_pooling(bn_3)
        dil_3_1 = self.__dilated_block(pool_2)
        bn_4 = tf.contrib.layers.batch_norm(dil_3_1, is_training=is_training)
        dil_3_2 = self.__dilated_block(bn_4)
        bn_5 = tf.contrib.layers.batch_norm(dil_3_2, is_training=is_training)
        dil_3_3 = self.__dilated_block(bn_5)
        res_3 = tf.math.add(dil_3_3, pool_2)
        print(res_3.shape)
        bn_6 = tf.contrib.layers.batch_norm(res_3, is_training=is_training)
        pool_3 = self.__pyramid_pooling(bn_6)
        dil_4_1 = self.__dilated_block(pool_3)
        bn_7 = tf.contrib.layers.batch_norm(dil_4_1, is_training=is_training)
        dil_4_2 = self.__dilated_block(bn_7)
        bn_8 = tf.contrib.layers.batch_norm(dil_4_2, is_training=is_training)
        dil_4_3 = self.__dilated_block(bn_8)
        res_4 = tf.math.add(dil_4_3, pool_3)
        print(res_4.shape)
        #decoder
        deconv_1 = self.__deconv_block(res_4)
        bn_9 = tf.contrib.layers.batch_norm(deconv_1, is_training=is_training)
        dec_res_1 = tf.math.add(bn_9, pool_2)
        deconv_2 = self.__deconv_block(dec_res_1)
        bn_10 = tf.contrib.layers.batch_norm(deconv_2, is_training=is_training)
        dec_res_2 = tf.math.add(bn_10, pool_1)
        deconv_3 = self.__deconv_block(dec_res_2)
        bn_11 = tf.contrib.layers.batch_norm(deconv_3, is_training=is_training)
        dec_res_3 = tf.math.add(bn_11, input_scaled)
        print(dec_res_3.shape)
        flatten = tf.reshape(dec_res_3, [-1, 64])
        print(flatten.shape)
        out = tf.layers.dense(flatten, num_classes, activation='relu')
        print(out.shape)
        return tf.reshape(out, [-1, 128, 256, num_classes])

    def __dilated_block(self, X, residual=True):
        out_1 = tf.layers.conv2d(X, 64, (5, 5), padding='SAME', activation='relu')
        print(out_1.shape)
        out_2 = tf.layers.conv2d(X, 64, (3, 3), padding='SAME', activation='relu', dilation_rate=(2, 2))
        print(out_2.shape)
        out_3 = tf.layers.conv2d(X, 64, (3, 3), padding='SAME', activation='relu', dilation_rate=(4, 4))
        print(out_3.shape)
        out_4 = tf.layers.conv2d(X, 64, (3, 3), padding='SAME', activation='relu', dilation_rate=(8, 8))
        print(out_4.shape)
        out = out_1 + out_2 + out_3 + out_4
        if residual:
            out = tf.math.add(out, X)
        print(out.shape)
        return out

    def __filters_scaling(self, X, num_filters, activation='relu'):
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation=activation)

    def __pyramid_pooling(self, X):
        pool_1 = tf.nn.pool(X, [2, 2], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_1.shape)
        pool_2 = tf.nn.pool(X, [3, 3], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_2.shape)
        pool_3 = tf.nn.pool(X, [5, 5], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_3.shape)
        pool_4 = tf.nn.pool(X, [8, 8], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        print(pool_4.shape)
        out = pool_1 + pool_2 + pool_3 + pool_4
        print(out.shape)
        return out

    def __deconv_block(self, X):
        out_1 = tf.layers.conv2d_transpose(X, 64, (2, 2), strides=(2, 2), padding='SAME', activation='relu')
        print(out_1.shape)
        out_2 = tf.layers.conv2d_transpose(X, 64, (3, 3), strides=(2, 2), padding='SAME', activation='relu')
        print(out_2.shape)
        out_3 = tf.layers.conv2d_transpose(X, 64, (5, 5), strides=(2, 2), padding='SAME', activation='relu')
        print(out_3.shape)
        out_4 = tf.layers.conv2d_transpose(X, 64, (6, 6), strides=(2, 2), padding='SAME', activation='relu')
        print(out_4.shape)
        out = out_1 + out_2 + out_3 + out_4
        print(out.shape)
        return out
