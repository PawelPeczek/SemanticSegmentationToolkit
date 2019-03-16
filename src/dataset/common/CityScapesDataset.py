import tensorflow as tf
import numpy as np
import multiprocessing
from glob import glob
import os


class CityScapesDataset:

    def __init__(self, config):
        """
        :param config: TrainingConfigReader or TestingConfigReader
        """
        self.__config = config

    def get_dummy_val_iterator(self, tfrecords_files_included, batch_size):
        if tfrecords_files_included == 'all':
            tfrecords_filenames = self.__get_tfrecords_list('val')
        else:
            tfrecords_filenames = self.__get_tfrecords_list('val')[:tfrecords_files_included]
        dataset = self.__get_dataset(tfrecords_filenames, batch_size)
        return dataset.make_one_shot_iterator()

    def get_evaluation_iterator(self, batch_size):
        tfrecords_filenames = self.__get_tfrecords_list('val')
        dataset = self.__get_dataset(tfrecords_filenames, batch_size)
        return dataset.make_one_shot_iterator()

    def get_training_iterator(self, batch_size):
        tfrecords_filenames = self.__get_tfrecords_list('train')
        dataset = self.__get_dataset(tfrecords_filenames, batch_size)
        return dataset.make_initializable_iterator()

    def __get_tfrecords_list(self, subset_name):
        tfrecords_base_dir = self.__config.tfrecords_dir
        tfrecords_file_name_template = '*{}*'.format(self.__config.tfrecords_base_name)
        return glob(os.path.join(tfrecords_base_dir, subset_name, tfrecords_file_name_template))

    def __get_dataset(self, tfrecords_filenames, batch_size):
        num_cpu = multiprocessing.cpu_count()
        dataset = tf.data.TFRecordDataset(tfrecords_filenames, num_parallel_reads=num_cpu)
        dataset = dataset.map(self.__parse, num_parallel_calls=num_cpu)
        dataset = dataset.shuffle(buffer_size=num_cpu * batch_size)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(num_cpu * batch_size)
        return dataset

    def __parse(self, serialized_example):
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
