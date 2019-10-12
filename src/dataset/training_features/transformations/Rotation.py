import random
from typing import Optional, Tuple
from math import pi
import numpy as np

import tensorflow as tf

from src.dataset.training_features.transformations.DatasetTransformation import DatasetTransformation


class Rotation(DatasetTransformation):

    SAFE_CROP_RATIO = 0.73

    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict]) -> Tuple[tf.Tensor, tf.Tensor]:

        def proceed_application():
            rotation_angle = tf.random.uniform([], -10, 10, dtype=tf.float32)
            pi_const = tf.constant(pi)
            normalizer = tf.constant(180.0)
            rotation_angle_radians = tf.div(tf.multiply(rotation_angle, pi_const), normalizer)
            image_height, image_width, _ = image.shape
            image_rotated = tf.contrib.image.rotate(image, rotation_angle_radians)
            label_rotated = tf.contrib.image.rotate(label, rotation_angle_radians)
            bounding_box = self.__calculate_crop_bounding_box()
            box_ind = np.zeros((1,), dtype=np.int32)
            image_rotated = tf.expand_dims(image_rotated, 0)
            label_rotated = tf.expand_dims(label_rotated, 0)
            label_rotated = tf.expand_dims(label_rotated, -1)
            result_image = tf.image.crop_and_resize(image_rotated, bounding_box, box_ind, [image_height, image_width])
            result_label = tf.image.crop_and_resize(label_rotated, bounding_box, box_ind, [image_height, image_width])
            result_image = tf.squeeze(result_image)
            result_label = tf.squeeze(result_label)
            result_label = tf.cast(result_label, tf.int32)
            return result_image, result_label

        application_probab = tf.constant(application_probab)
        bayesian_coin = tf.random.uniform([])
        return tf.cond(bayesian_coin < application_probab, proceed_application, lambda: (image, label))

    def __calculate_crop_bounding_box(self) -> tf.Tensor:
        x1 = y1 = 0.5 - 0.5 * Rotation.SAFE_CROP_RATIO
        x2 = y2 = 0.5 + 0.5 * Rotation.SAFE_CROP_RATIO
        return tf.convert_to_tensor([[y1, x1, y2, x2]], dtype=np.float32)
