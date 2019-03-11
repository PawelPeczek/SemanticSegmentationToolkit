from abc import ABC, abstractmethod


class SemanticSegmentationModel(ABC):

    @abstractmethod
    def run(self, X, num_classes, is_training=True):
        raise NotImplementedError('This method must be implemented.')
