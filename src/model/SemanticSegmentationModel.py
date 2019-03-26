from abc import ABC, abstractmethod
import tensorflow as tf

class SemanticSegmentationModel(ABC):

    @abstractmethod
    def run(self, X: tf.Tensor, num_classes: int, is_training: bool = True) -> tf.Tensor:
        raise NotImplementedError('This method must be implemented.')
