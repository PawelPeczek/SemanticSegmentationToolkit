import tensorflow as tf
from src.model.SemanticSegmentationModel import SemanticSegmentationModel


class ThickModel(SemanticSegmentationModel):

    def run(self, X, num_classes, is_training=True):
        #encoder
        input_scaled = self.__filters_scaling(X, 64)
        
        dil_1 = self.__dilated_block(input_scaled)
        bn_1 = tf.layers.batch_normalization(dil_1, training=is_training)
        pool_1 = self.__pyramid_pooling(bn_1)
        dil_2_1 = self.__dilated_block(pool_1)
        bn_2 = tf.layers.batch_normalization(dil_2_1, training=is_training)
        dil_2_2 = self.__dilated_block(bn_2)
        res_2 = tf.math.add(dil_2_2, pool_1)
        
        bn_3 = tf.layers.batch_normalization(res_2, training=is_training)
        pool_2 = self.__pyramid_pooling(bn_3)
        dil_3_1 = self.__dilated_block(pool_2)
        bn_4 = tf.layers.batch_normalization(dil_3_1, training=is_training)
        dil_3_2 = self.__dilated_block(bn_4)
        bn_5 = tf.layers.batch_normalization(dil_3_2, training=is_training)
        dil_3_3 = self.__dilated_block(bn_5)
        res_3 = tf.math.add(dil_3_3, pool_2)

        bn_6 = tf.layers.batch_normalization(res_3, training=is_training)
        pool_3 = self.__pyramid_pooling(bn_6)
        dil_4_1 = self.__dilated_block(pool_3)
        bn_7 = tf.layers.batch_normalization(dil_4_1, training=is_training)
        dil_4_2 = self.__dilated_block(bn_7)
        bn_8 = tf.layers.batch_normalization(dil_4_2, training=is_training)
        dil_4_3 = self.__dilated_block(bn_8)
        res_4 = tf.math.add(dil_4_3, pool_3)

        #decoder
        deconv_1 = self.__deconv_block(res_4)
        bn_9 = tf.layers.batch_normalization(deconv_1, training=is_training)
        dec_res_1 = tf.math.add(bn_9, pool_2)
        deconv_2 = self.__deconv_block(dec_res_1)
        bn_10 = tf.layers.batch_normalization(deconv_2, training=is_training)
        dec_res_2 = tf.math.add(bn_10, pool_1)
        deconv_3 = self.__deconv_block(dec_res_2)
        bn_11 = tf.layers.batch_normalization(deconv_3, training=is_training)
        dec_res_3 = tf.math.add(bn_11, input_scaled)
        flatten = tf.reshape(dec_res_3, [-1, 64])
        out = tf.layers.dense(flatten, num_classes, activation='relu')
        
        return tf.reshape(out, [-1, 128, 256, num_classes])

    def __dilated_block(self, X, residual=True):
        out_1 = tf.layers.conv2d(X, 64, (5, 5), padding='SAME', activation='relu')
        out_2 = tf.layers.conv2d(X, 64, (3, 3), padding='SAME', activation='relu', dilation_rate=(2, 2))
        out_3 = tf.layers.conv2d(X, 64, (3, 3), padding='SAME', activation='relu', dilation_rate=(4, 4))
        out_4 = tf.layers.conv2d(X, 64, (3, 3), padding='SAME', activation='relu', dilation_rate=(8, 8))
        out = out_1 + out_2 + out_3 + out_4
        if residual:
            out = tf.math.add(out, X)
        
        return out

    def __filters_scaling(self, X, num_filters, activation='relu'):
        return tf.layers.conv2d(X, num_filters, (1, 1), padding='SAME', activation=activation)

    def __pyramid_pooling(self, X):
        pool_1 = tf.nn.pool(X, [2, 2], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        pool_2 = tf.nn.pool(X, [3, 3], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        pool_3 = tf.nn.pool(X, [5, 5], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        pool_4 = tf.nn.pool(X, [8, 8], 'MAX', 'SAME', dilation_rate=[1, 1], strides=[2, 2])
        out = pool_1 + pool_2 + pool_3 + pool_4
        
        return out

    def __deconv_block(self, X):
        out_1 = tf.layers.conv2d_transpose(X, 64, (2, 2), strides=(2, 2), padding='SAME', activation='relu')
        out_2 = tf.layers.conv2d_transpose(X, 64, (3, 3), strides=(2, 2), padding='SAME', activation='relu')
        out_3 = tf.layers.conv2d_transpose(X, 64, (5, 5), strides=(2, 2), padding='SAME', activation='relu')
        out_4 = tf.layers.conv2d_transpose(X, 64, (6, 6), strides=(2, 2), padding='SAME', activation='relu')
        out = out_1 + out_2 + out_3 + out_4
        
        return out
