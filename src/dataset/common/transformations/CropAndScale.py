import random
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.dataset.common.transformations.DatasetTransformation import DatasetTransformation


class CropAndScale(DatasetTransformation):

    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict] = None) -> Tuple[tf.Tensor, tf.Tensor]:

        def proceed_application():
            image_height, image_width, _ = image.get_shape().as_list()
            bounding_box = self.__calculate_crop_bounding_box(image_height, image_width)
            box_ind = np.zeros((1,), dtype=np.int32)
            image_exp_dim = tf.expand_dims(image, 0)
            label_exp_dim = tf.expand_dims(label, 0)
            label_exp_dim = tf.expand_dims(label_exp_dim, -1)
            result_image = tf.image.crop_and_resize(image_exp_dim, bounding_box, box_ind, [image_height, image_width])
            result_label = tf.image.crop_and_resize(label_exp_dim, bounding_box, box_ind, [image_height, image_width])
            result_image = tf.squeeze(result_image)
            result_label = tf.squeeze(result_label)
            result_label = tf.cast(result_label, dtype=tf.int32)
            return result_image, result_label

        application_probab = tf.constant(application_probab)
        bayesian_coin = tf.random.uniform([])
        return tf.cond(bayesian_coin < application_probab, proceed_application, lambda: (image, label))

    def __calculate_crop_bounding_box(self, image_height: int, image_width: int) -> tf.Tensor:
        crop_fraction = tf.random.uniform([], 0.55, 0.75)
        crop_height, crop_width = tf.cast(image_height * crop_fraction, dtype=tf.int32), \
                                  tf.cast(image_width * crop_fraction, dtype=tf.int32)
        y_max_value = image_height - crop_height
        x_max_value = image_width - crop_width
        left_top_y = tf.random.uniform([], 0, y_max_value, dtype=tf.int32)
        left_top_x = tf.random.uniform([], 0, x_max_value, dtype=tf.int32)
        right_bottom_y = left_top_y + crop_height
        right_bottom_x = left_top_x + crop_width
        left_top_y_normalized = left_top_y / image_height
        left_top_x_normalized = left_top_x / image_width
        right_bottom_y_normalized = right_bottom_y / image_height
        right_bottom_x_normalized = right_bottom_x / image_width
        return tf.convert_to_tensor([[left_top_y_normalized, left_top_x_normalized,
                                     right_bottom_y_normalized, right_bottom_x_normalized]], dtype=np.float32)
