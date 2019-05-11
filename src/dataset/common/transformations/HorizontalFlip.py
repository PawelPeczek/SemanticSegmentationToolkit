from typing import Optional, Tuple

import tensorflow as tf

from src.dataset.common.transformations.DatasetTransformation import DatasetTransformation


class HorizontalFlip(DatasetTransformation):

    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict]) -> Tuple[tf.Tensor, tf.Tensor]:

        def proceed_application():
            result_image = tf.image.flip_left_right(image)
            label_exp_dim = tf.expand_dims(label, -1)
            result_label = tf.image.flip_left_right(label_exp_dim)
            result_label = tf.squeeze(result_label)
            return result_image, result_label

        application_probab = tf.constant(application_probab)
        bayesian_coin = tf.random.uniform([])
        return tf.cond(bayesian_coin < application_probab, proceed_application, lambda: (image, label))
