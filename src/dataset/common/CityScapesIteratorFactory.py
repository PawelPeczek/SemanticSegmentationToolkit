from enum import Enum
from src.dataset.common.CityScapesDataset import CityScapesDataset


class IteratorType(Enum):
    TRAINING_ITERATOR = 1
    DUMMY_ITERATOR = 2
    VALIDATION_ITERATOR = 3


class CityScapesIteratorFactory:

    def __init__(self, config):
        self.__config = config
        self.__cityscapes_dataset = CityScapesDataset(config)

    def get_iterator(self, iterator_type):
        """
        :param iterator_type: IteratorType
        :return: tf.iterator
        """
        batch_size = self.__config.batch_size
        if iterator_type == IteratorType.TRAINING_ITERATOR:
            return self.__cityscapes_dataset.get_training_iterator(batch_size)
        elif iterator_type == IteratorType.VALIDATION_ITERATOR:
            return self.__cityscapes_dataset.get_evaluation_iterator(batch_size)
        else:
            tfrecords_files_to_use = self.__config.dummy_iterator_tfrecords_files
            return self.__cityscapes_dataset.get_dummy_val_iterator(tfrecords_files_to_use, batch_size)

