import tensorflow as tf
import numpy as np
import multiprocessing
from glob import glob
from typing import List, Tuple, Union
import os

from src.dataset.common.DatasetTransformer import DatasetTransformer
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader


class CityScapesDataset:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config
        self.__dataset_transformer = DatasetTransformer(config)

    def get_initializable_dummy_iterator(self, tfrecords_files_included: Union[str, int], batch_size: int) -> tf.data.Iterator:
        dataset = self.__compose_data_source_for_dummy_iterator(tfrecords_files_included, batch_size)
        return dataset.make_initializable_iterator()

    def get_one_shot_dummy_iterator(self, tfrecords_files_included: Union[str, int], batch_size: int) -> tf.data.Iterator:
        dataset = self.__compose_data_source_for_dummy_iterator(tfrecords_files_included, batch_size)
        return dataset.make_one_shot_iterator()

    def get_one_shot_validation_iterator(self, batch_size: int) -> tf.data.Iterator:
        tfrecords_filenames = self.__get_tfrecords_list('val')
        return self.__compose_one_shot_iterator_from_tfrecords(tfrecords_filenames, batch_size)

    def get_one_shot_train_iterator(self, batch_size: int) -> tf.data.Iterator:
        tfrecords_filenames = self.__get_tfrecords_list('train')
        return self.__compose_one_shot_iterator_from_tfrecords(tfrecords_filenames, batch_size,
                                                               self.__config.radnom_data_transformation)

    def get_initializable_validation_iterator(self, batch_size: int) -> tf.data.Iterator:
        tfrecords_filenames = self.__get_tfrecords_list('val')
        return self.__compose_initializable_iterator_from_tfrecords(tfrecords_filenames, batch_size)

    def get_initializable_train_iterator(self, batch_size: int) -> tf.data.Iterator:
        tfrecords_filenames = self.__get_tfrecords_list('train')
        return self.__compose_initializable_iterator_from_tfrecords(tfrecords_filenames, batch_size,
                                                                    self.__config.radnom_data_transformation)

    def __compose_data_source_for_dummy_iterator(self, tfrecords_files_included: Union[str, int], batch_size: int) -> tf.data.Dataset:
        if tfrecords_files_included == 'all':
            tfrecords_filenames = self.__get_tfrecords_list('val')
        else:
            tfrecords_filenames = self.__get_tfrecords_list('val')[:tfrecords_files_included]
        return self.__compose_dataset(tfrecords_filenames, batch_size)

    def __get_tfrecords_list(self, subset_name: str) -> List[str]:
        tfrecords_base_dir = self.__config.tfrecords_dir
        tfrecords_file_name_template = '*{}*'.format(self.__config.tfrecords_base_name)
        return glob(os.path.join(tfrecords_base_dir, subset_name, tfrecords_file_name_template))

    def __compose_one_shot_iterator_from_tfrecords(self, tfrecords_filenames: List[str], batch_size: int, use_augmentation: bool = False) -> tf.data.Iterator:
        dataset = self.__compose_dataset(tfrecords_filenames, batch_size, use_augmentation)
        return dataset.make_one_shot_iterator()

    def __compose_dataset(self, tfrecords_filenames: List[str], batch_size: int, use_augmentation: bool = False) -> tf.data.Dataset:
        num_cpu = multiprocessing.cpu_count()
        dataset = tf.data.TFRecordDataset(tfrecords_filenames, num_parallel_reads=num_cpu)
        dataset = dataset.map(self.__parse, num_parallel_calls=num_cpu)
        if use_augmentation is True:
            dataset_transformer = DatasetTransformer(self.__config)
            dataset = dataset.map(dataset_transformer.augment_data, num_parallel_calls=num_cpu)
        dataset = dataset.shuffle(buffer_size=8 * batch_size)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(8 * batch_size)
        return dataset

    def __compose_initializable_iterator_from_tfrecords(self, tfrecords_filenames: List[str], batch_size: int,  use_augmentation: bool = False) -> tf.data.Iterator:
        dataset = self.__compose_dataset(tfrecords_filenames, batch_size, use_augmentation)
        return dataset.make_initializable_iterator()

    def __parse(self, serialized_example: tf.string) -> Tuple[tf.Tensor, tf.Tensor]:
        features = \
            {
                'example': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string)
            }
        parsed_example = tf.parse_single_example(serialized=serialized_example,
                                                 features=features)
        image_raw = parsed_example['example']
        image = tf.decode_raw(image_raw, np.uint8)
        image = tf.reshape(image, [self.__config.destination_size[1], self.__config.destination_size[0], 3])
        image = tf.cast(image, tf.float32)
        label_raw = parsed_example['gt']
        label = tf.decode_raw(label_raw, tf.uint8)
        label = tf.cast(label, tf.int32)
        label = tf.reshape(label, [self.__config.destination_size[1], self.__config.destination_size[0]])
        return image, label

