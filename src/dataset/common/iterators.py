from abc import ABC, abstractmethod
from enum import Enum
import os
from typing import List, Tuple, Union, Optional

import tensorflow as tf
import numpy as np
import multiprocessing
from glob import glob
import random

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.DatasetTransformer import DatasetTransformer
from src.model.layers.interpolation import resize_bilinear


class CityScapesIterator(ABC):

    def __init__(self, config: GraphExecutorConfigReader):
        self._config = config
        self._dataset_transformer = DatasetTransformer(config)

    def build(self) -> tf.data.Iterator:
        task = self._config.get_or_else('task', 'segmentation')
        if task.lower() == 'segmentation':
            return self.build_segmentation_iterator()
        else:
            return self.build_auto_encoding_iterator()

    @abstractmethod
    def build_segmentation_iterator(self) -> tf.data.Iterator:
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')

    @abstractmethod
    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        raise NotImplementedError('This method must be implemented in '
                                  'derived class.')

    def _get_dummy_segmentation_iterator(self,
                                         tfrecords_to_include: Union[str, int],
                                         batch_size: int) -> tf.data.Dataset:
        tfrecords_filenames = self._get_tfrecords_list('val')
        if tfrecords_to_include != 'all':
            tfrecords_filenames = tfrecords_filenames[:tfrecords_to_include]
        return self.__get_segmentation_dataset(tfrecords_filenames, batch_size)

    def _get_tfrecords_list(self, subset_name: str) -> List[str]:
        tfrecords_base_dir = self._config.tfrecords_dir
        base_tfrecords_name = self._config.tfrecords_base_name
        tfrecords_file_name_template = f'*{base_tfrecords_name}*'
        return glob(os.path.join(
            tfrecords_base_dir,
            subset_name,
            tfrecords_file_name_template))

    def _get_one_shot_segmentation_iterator(self,
                                            tfrecords_filenames: List[str],
                                            batch_size: int,
                                            augmented: bool = False) -> tf.data.Iterator:
        dataset = self.__get_segmentation_dataset(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=batch_size,
            augmented=augmented)
        return dataset.make_one_shot_iterator()

    def _get_initializable_segmentation_iterator(self,
                                                 tfrecords_filenames: List[str],
                                                 batch_size: int,
                                                 augmented: bool = False) -> tf.data.Iterator:
        dataset = self.__get_segmentation_dataset(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=batch_size,
            augmented=augmented)
        return dataset.make_initializable_iterator()

    def __get_segmentation_dataset(self,
                                   tfrecords_filenames: List[str],
                                   batch_size: int,
                                   augmented: bool = False) -> tf.data.Dataset:
        num_cpu = multiprocessing.cpu_count()
        dataset = tf.data.TFRecordDataset(
            tfrecords_filenames,
            num_parallel_reads=num_cpu)
        dataset = dataset.map(self.__parse_tfrecords, num_parallel_calls=num_cpu)
        if augmented is True:
            dataset_transformer = DatasetTransformer(self._config)
            dataset = dataset.map(
                dataset_transformer.augment_data,
                num_parallel_calls=num_cpu)
        dataset = dataset.shuffle(buffer_size=8 * batch_size)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(8 * batch_size)
        return dataset

    def __parse_tfrecords(self,
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

    def _get_training_auto_encoding_dataset(self,
                                            batch_size: int,
                                            max_examples: Optional[int] = None) -> tf.data.Dataset:
        examples_list_path = self._config.training_examples_list
        return self._get_auto_encoding_dataset(
            examples_list_path=examples_list_path,
            batch_size=batch_size,
            max_examples=max_examples)

    def _get_validation_auto_encoding_dataset(self,
                                              batch_size: int,
                                              max_examples: Optional[int] = None) -> tf.data.Dataset:
        examples_list_path = self._config.validation_examples_list
        return self._get_auto_encoding_dataset(
            examples_list_path=examples_list_path,
            batch_size=batch_size,
            max_examples=max_examples)

    def _get_auto_encoding_dataset(self,
                                   examples_list_path: str,
                                   batch_size: int,
                                   max_examples: Optional[int] = None) -> tf.data.Dataset:
        num_cpu = multiprocessing.cpu_count()
        dataset = tf.contrib.data.CsvDataset(
            filenames=[examples_list_path],
            record_defaults=[tf.string])
        dataset = dataset.map(self.__parse_csv, num_parallel_calls=num_cpu)
        dataset = dataset.shuffle(buffer_size=8 * batch_size)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(8 * batch_size)
        return dataset

    def __parse_csv(self, example: tf.string) -> Tuple[tf.Tensor, tf.Tensor]:
        destination_size = self._config.destination_size
        height, width = destination_size[1], destination_size[0]
        example_bytes = tf.io.read_file(example)
        x = tf.io.decode_image(example_bytes, channels=3, dtype=tf.uint8)
        x = tf.image.resize_image_with_crop_or_pad(
            x,
            target_height=height,
            target_width=width)
        x = tf.reshape(x, [height, width, 3])
        y = tf.identity(x)
        x = self.__mask_example(x)
        x = tf.reshape(x, [height, width, 3])
        return tf.cast(x, dtype=tf.float32), tf.cast(y, dtype=tf.float32)

    def __mask_example(self, x: tf.Tensor) -> tf.Tensor:
        mask = tf.py_func(
            func=self.__prepare_mask,
            inp=[x],
            Tout=tf.uint8)
        return tf.math.multiply(x, mask)

    def __prepare_mask(self, x: np.ndarray) -> np.ndarray:
        mask_shape = x.shape[0], x.shape[1], 3
        mask = np.ones(mask_shape, dtype=np.uint8)
        img_height, img_width, _ = mask_shape
        half_height, half_width = int(round(img_height / 2)), int(round(img_width / 2))
        max_box_height = int(round(0.15 * img_height))
        max_box_width = int(round(0.15 * img_width))
        min_box_height = int(round(0.4 * max_box_height))
        min_box_width = int(round(0.4 * max_box_width))
        black_boxes = random.randint(10, 20)
        for _ in range(black_boxes):
            box_height = random.randint(min_box_height, max_box_height)
            box_width = random.randint(min_box_width, max_box_width)
            box_y = int(round(random.gauss(half_height, 2/3 * half_height)))
            box_x = int(round(random.gauss(half_width, 2/3 * half_width)))
            mask[box_y:box_y+box_height, box_x:box_x+box_width, :] = 0
        return mask


class InitializableDummyIterator(CityScapesIterator):

    def build_segmentation_iterator(self) -> tf.data.Iterator:
        dataset = self._get_dummy_segmentation_iterator(
            tfrecords_to_include=self._config.dummy_iterator_tfrecords_files,
            batch_size=self._config.batch_size)
        return dataset.make_initializable_iterator()

    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        max_examples = None
        dataset = self._get_training_auto_encoding_dataset(
            batch_size=self._config.batch_size,
            max_examples=max_examples)
        return dataset.make_initializable_iterator()


class OneShotDummyIterator(CityScapesIterator):

    def build_segmentation_iterator(self) -> tf.data.Iterator:
        dataset = self._get_dummy_segmentation_iterator(
            tfrecords_to_include=self._config.dummy_iterator_tfrecords_files,
            batch_size=self._config.batch_size)
        return dataset.make_one_shot_iterator()

    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        max_examples = None
        dataset = self._get_training_auto_encoding_dataset(
            batch_size=self._config.batch_size,
            max_examples=max_examples)
        return dataset.make_one_shot_iterator()


class OneShotValidationIterator(CityScapesIterator):

    def build_segmentation_iterator(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('val')
        return self._get_one_shot_segmentation_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size)

    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        batch_size = self._config.batch_size
        dataset = self._get_validation_auto_encoding_dataset(batch_size)
        return dataset.make_one_shot_iterator()


class OneShotTrainIterator(CityScapesIterator):

    def build_segmentation_iterator(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('train')
        return self._get_one_shot_segmentation_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size,
            augmented=self._config.radnom_data_transformation)

    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        batch_size = self._config.batch_size
        dataset = self._get_training_auto_encoding_dataset(batch_size)
        return dataset.make_one_shot_iterator()


class InitializableValidationIterator(CityScapesIterator):

    def build_segmentation_iterator(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('val')
        return self._get_initializable_segmentation_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size)

    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        batch_size = self._config.batch_size
        dataset = self._get_validation_auto_encoding_dataset(batch_size)
        return dataset.make_initializable_iterator()


class InitializableTrainIterator(CityScapesIterator):

    def build_segmentation_iterator(self) -> tf.data.Iterator:
        tfrecords_filenames = self._get_tfrecords_list('train')
        return self._get_initializable_segmentation_iterator(
            tfrecords_filenames=tfrecords_filenames,
            batch_size=self._config.batch_size,
            augmented=self._config.radnom_data_transformation)

    def build_auto_encoding_iterator(self) -> tf.data.Iterator:
        batch_size = self._config.batch_size
        dataset = self._get_training_auto_encoding_dataset(batch_size)
        return dataset.make_initializable_iterator()


class IteratorType(Enum):
    INITIALIZABLE_TRAIN_SET_ITERATOR = InitializableTrainIterator
    OS_VALIDATION_ITERATOR = OneShotValidationIterator
    OS_TRAIN_ITERATOR = OneShotTrainIterator
    INITIALIZABLE_VALIDATION_ITERATOR = InitializableValidationIterator
    DUMMY_ITERATOR = None


class CityScapesIteratorFactory:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config

    def get_iterator(self, iterator_type: IteratorType) -> tf.data.Iterator:
        factorized_iterator = None
        for iterator in IteratorType:
            if iterator_type == iterator:
                factorized_iterator = iterator.value
                break
        if factorized_iterator is None:
            factorized_iterator = self.__prepare_dummy_iterator()
        else:
            factorized_iterator = factorized_iterator(self.__config)
        return factorized_iterator.build()

    def __prepare_dummy_iterator(self) -> CityScapesIterator:
        if self.__config.mode == 'train':
            return InitializableDummyIterator(self.__config)
        else:
            return OneShotDummyIterator(self.__config)
