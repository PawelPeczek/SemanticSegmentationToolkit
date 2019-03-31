from enum import Enum
import tensorflow as tf
from src.dataset.common.CityScapesDataset import CityScapesDataset
from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader


class IteratorType(Enum):
    INITIALIZABLE_TRAIN_SET_ITERATOR = 1
    DUMMY_ITERATOR = 2
    OS_VALIDATION_ITERATOR = 3
    OS_TRAIN_ITERATOR = 4
    INITIALIZABLE_VALIDATION_ITERATOR = 5

class CityScapesIteratorFactory:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config
        self.__cityscapes_dataset = CityScapesDataset(config)

    def get_iterator(self, iterator_type: IteratorType) -> tf.data.Iterator:
        batch_size = self.__config.batch_size
        if iterator_type == IteratorType.INITIALIZABLE_TRAIN_SET_ITERATOR:
            return self.__cityscapes_dataset.get_initializable_train_iterator(batch_size)
        elif iterator_type == IteratorType.OS_VALIDATION_ITERATOR:
            return self.__cityscapes_dataset.get_one_shot_validation_iterator(batch_size)
        elif iterator_type == IteratorType.OS_TRAIN_ITERATOR:
            return self.__cityscapes_dataset.get_one_shot_train_iterator(batch_size)
        elif iterator_type == IteratorType.INITIALIZABLE_VALIDATION_ITERATOR:
            return self.__cityscapes_dataset.get_initializable_validation_iterator(batch_size)
        else:
            return self.__prepare_dummy_iterator()

    def __prepare_dummy_iterator(self) -> tf.data.Iterator:
        batch_size = self.__config.batch_size
        tfrecords_files_to_use = self.__config.dummy_iterator_tfrecords_files
        if self.__config.mode == 'train':
            return self.__cityscapes_dataset.get_initializable_dummy_iterator(tfrecords_files_to_use, batch_size)
        else:
            return self.__cityscapes_dataset.get_one_shot_dummy_iterator(tfrecords_files_to_use, batch_size)

