from abc import ABC, abstractmethod
from typing import Optional, Tuple

import tensorflow as tf


class SemanticSegmentationModel(ABC):

    @abstractmethod
    def run(self, X: tf.Tensor, num_classes: int, is_training: bool = True, y: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        raise NotImplementedError('This method must be implemented.')
