from abc import ABC, abstractmethod
from typing import Optional, Tuple

import tensorflow as tf

from src.dataset.common.transformations.TransformationType import TransformationType


class DatasetTransformation(ABC):

    def __init__(self, transformation_type: TransformationType):
        self._transformation_type = transformation_type

    @abstractmethod
    def apply(self, image: tf.Tensor, label: tf.Tensor, application_probab: float,
              parameters: Optional[dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError('This method must be implemented')

    def get_transformation_type(self) -> TransformationType:
        return self._transformation_type
