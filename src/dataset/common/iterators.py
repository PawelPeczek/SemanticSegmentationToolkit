from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf
import numpy as np
import multiprocessing
from glob import glob
from typing import List, Tuple, Union
import os

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.DatasetTransformer import DatasetTransformer


class CityScapesIterator(ABC):

    def __init__(self, config: GraphExecutorConfigReader):
        self._config = config
        self._dataset_transformer = DatasetTransformer(config)

    @abstractmethod
    def build(self) -> tf.data.Iterator:
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')

    def _compose_dummy_iterator_source(self,
                                       tfrecords_to_include: Union[str, int],
                                       batch_size: int) -> tf.data.Dataset:
        tfrecords_filenames = self._get_tfrecords_list('val')
        if tfrecords_to_include != 'all':
            tfrecords_filenames = tfrecords_filenames[:tfrecords_to_include]
        return self.__compose_dataset(tfrecords_filenames, batch_size)

    def _get_tfrecords_list(self, subset_name: str) -> List[str]:
        tfrecords_base_dir = self._config.tfrecords_dir
        base_tfrecords_name = self._config.tfrecords_base_name
        tfrecords_file_name_template = f'*{base_tfrecords_name}*'
        return glob(os.path.join(
            tfrecords_base_dir,
            subset_name,
            tfrecords_file_name_template))

    def _get_one_shot_iterator(self,
                               tfrecords_filenames: List[str],
                               batch_size: int,
                               augmented: bool = False) -> tf.data.Iterator:
        dataset = self.__compose_dataset(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=batch_size,
            augmented=augmented)
        return dataset.make_one_shot_iterator()

    def _get_initializable_iterator(self,
                                    tfrecords_filenames: List[str],
                                    batch_size: int,
                                    augmented: bool = False) -> tf.data.Iterator:
        dataset = self.__compose_dataset(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=batch_size,
            augmented=augmented)
        return dataset.make_initializable_iterator()

    def __compose_dataset(self,
                          tfrecords_filenames: List[str],
                          batch_size: int,
                          augmented: bool = False) -> tf.data.Dataset:
        num_cpu = multiprocessing.cpu_count()
        dataset = tf.data.TFRecordDataset(
            tfrecords_filenames,
            num_parallel_reads=num_cpu)
        dataset = dataset.map(self.__parse, num_parallel_calls=num_cpu)
        if augmented is True:
            dataset_transformer = DatasetTransformer(self._config)
            dataset = dataset.map(
                dataset_transformer.augment_data,
                num_parallel_calls=num_cpu)
        dataset = dataset.shuffle(buffer_size=8 * batch_size)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(8 * batch_size)
        return dataset

    def __parse(self,
                serialized_example: tf.string) -> Tuple[tf.Tensor, tf.Tensor]:
        features = \
            {
                'example': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string)
            }
        parsed_example = tf.parse_single_example(
            serialized=serialized_example,
            features=features)
        image_raw = parsed_example['example']
        image = tf.decode_raw(image_raw, np.uint8)
        destination_size = self._config.destination_size
        destination_size = destination_size[1], destination_size[0]
        image_shape = (destination_size[0], destination_size[1], 3)
        image = tf.reshape(image, shape=image_shape)
        image = tf.cast(image, tf.float32)
        label_raw = parsed_example['gt']
        label = tf.decode_raw(label_raw, tf.uint8)
        label = tf.cast(label, tf.int32)
        label = tf.reshape(label, destination_size)
        return image, label


class InitializableDummyIterator(CityScapesIterator):

    def build(self) -> tf.data.Iterator:
        dataset = self._compose_dummy_iterator_source(
            tfrecords_to_include=self._config.dummy_iterator_tfrecords_files,
            batch_size=self._config.batch_size)
        return dataset.make_initializable_iterator()


class OneShotDummyIterator(CityScapesIterator):

    def build(self) -> tf.data.Iterator:
        dataset = self._compose_dummy_iterator_source(
            tfrecords_to_include=self._config.dummy_iterator_tfrecords_files,
            batch_size=self._config.batch_size)
        return dataset.make_one_shot_iterator()


class OneShotValidationIterator(CityScapesIterator):

    def build(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('val')
        return self._get_one_shot_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size)


class OneShotTrainIterator(CityScapesIterator):

    def build(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('train')
        return self._get_one_shot_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size,
            augmented=self._config.radnom_data_transformation)


class InitializableValidationIterator(CityScapesIterator):

    def build(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('val')
        return self._get_initializable_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size)


class InitializableTrainIterator(CityScapesIterator):

    def build(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('train')
        return self._get_initializable_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size,
            augmented=self._config.radnom_data_transformation)


class IteratorType(Enum):
    INITIALIZABLE_TRAIN_SET_ITERATOR = InitializableTrainIterator
    OS_VALIDATION_ITERATOR = OneShotValidationIterator
    OS_TRAIN_ITERATOR = OneShotTrainIterator
    INITIALIZABLE_VALIDATION_ITERATOR = InitializableValidationIterator


class CityScapesIteratorFactory:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config

    def get_iterator(self, iterator_type: IteratorType) -> tf.data.Iterator:
        factorized_iterator = None
        for iterator in IteratorType:
            if iterator_type == iterator:
                factorized_iterator = iterator(self.__config)
                break
        if factorized_iterator is None:
            factorized_iterator = self.__prepare_dummy_iterator()
        return factorized_iterator.build()

    def __prepare_dummy_iterator(self) -> CityScapesIterator:
        if self.__config.mode == 'train':
            return InitializableDummyIterator(self.__config)
        else:
            return OneShotDummyIterator(self.__config)
