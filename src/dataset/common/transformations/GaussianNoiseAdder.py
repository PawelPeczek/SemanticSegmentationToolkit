from typing import Optional, Tuple

import tensorflow as tf

from src.dataset.common.transformations.DatasetTransformation import DatasetTransformation


class GaussianNoiseAdder(DatasetTransformation):

    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        stddev = 15.0
        if parameters is not None and 'stddev' in parameters:
            stddev = parameters['stddev']

        def proceed_application():
            noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=stddev, dtype=tf.float32)
            result_image = tf.clip_by_value(image + noise, 0.0, 256.0)
            return result_image, label

        application_probab = tf.constant(application_probab)
        bayesian_coin = tf.random.uniform([])
        return tf.cond(bayesian_coin < application_probab, proceed_application, lambda: (image, label))
