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
            return self.__prepare_dummy_iterator()

    def __prepare_dummy_iterator(self):
        batch_size = self.__config.batch_size
        tfrecords_files_to_use = self.__config.dummy_iterator_tfrecords_files
        if self.__config.mode == 'train':
            return self.__cityscapes_dataset.get_initializable_dummy_iterator(tfrecords_files_to_use, batch_size)
        else:
            return self.__cityscapes_dataset.get_one_shot_dummy_iterator(tfrecords_files_to_use, batch_size)

