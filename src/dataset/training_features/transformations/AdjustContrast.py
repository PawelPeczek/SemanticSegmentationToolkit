from typing import Optional, Tuple

import tensorflow as tf

from src.dataset.training_features.transformations.DatasetTransformation import DatasetTransformation


class AdjustContrast(DatasetTransformation):

    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        lower = 0.2
        if parameters is not None and 'lower' in parameters:
            lower = parameters['lower']
        upper = 0.5
        if parameters is not None and 'upper' in parameters:
            upper = parameters['upper']

        def proceed_application():
            normalized_image = tf.div(image, 256.0)
            result_image = tf.image.random_contrast(normalized_image, lower=lower, upper=upper)
            result_image = tf.multiply(result_image, 256.0)
            return result_image, label

        application_probab = tf.constant(application_probab)
        bayesian_coin = tf.random.uniform([])
        return tf.cond(tf.less(bayesian_coin, application_probab), proceed_application, lambda: (image, label))
