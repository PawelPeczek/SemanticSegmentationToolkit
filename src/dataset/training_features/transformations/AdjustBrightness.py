from typing import Optional, Tuple

import tensorflow as tf

from src.dataset.training_features.transformations.DatasetTransformation import DatasetTransformation


class AdjustBrightness(DatasetTransformation):

    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        max_delta = 0.2
        if parameters is not None and 'max_delta' in parameters:
            max_delta = parameters['max_delta']

        def proceed_application():
            result_image = tf.image.random_brightness(image, max_delta)
            return result_image, label

        application_probab = tf.constant(application_probab)
        bayesian_coin = tf.random.uniform([])
        return tf.cond(bayesian_coin < application_probab, proceed_application, lambda: (image, label))
