import os
from glob import glob
import re
from functools import reduce
import yaml
import math
import numpy as np
from threading import Thread
import tensorflow as tf
import cv2 as cv
from src.utils.filesystem_utils import create_directory


class DatasetPreprocessor:

    def __init__(self, config):
        """
        :param config: ConfigReader objects that wraps config file
        """
        self.__config = config

    def transform_dataset(self):
        create_directory(self.__config.output_tfrecords_dir)
        self.__transform_dataset_subset('train')
        self.__transform_dataset_subset('val')

    def __transform_dataset_subset(self, subset_name):
        files_list = self.__prepare_dataset_list(subset_name)
        self.__prepare_tfrecords(subset_name, files_list)

    def __prepare_dataset_list(self, subset_name):
        gt_path = self.__construct_dataset_subset_path(self.__config.gt_dir, subset_name)
        examples_path = self.__construct_dataset_subset_path(self.__config.examples_dir, subset_name)
        gt_dict = self.__prepare_dataset_subset_list(gt_path)
        examples_dict = self.__prepare_dataset_subset_list(examples_path)
        return self.__connect_dicts_in_tuples_list(gt_dict, examples_dict)

    def __construct_dataset_subset_path(self, base_bath, subset_name):
        return os.path.join(base_bath, subset_name)

    def __prepare_dataset_subset_list(self, path):
        result = {}
        for folder in [x[0] for x in os.walk(path) if x[0] != path]:
            folder_content = glob(os.path.join(folder, '*_color.png'))
            if len(folder_content) is 0:
                folder_content = glob(os.path.join(folder, '*_leftImg8bit.png'))
            folder_content = list(map(self.__map_paths_to_tuples_with_extracted_name, folder_content))
            folder_dict = reduce(self.__reduce_content_to_dict, folder_content, {})
            result = {**result, **folder_dict}
        return result

    def __map_paths_to_tuples_with_extracted_name(self, path):
        file_name = path.split('/')[-1]
        pattern = "^([a-zA-Z]+_[0-9]{6}_[0-9]{6})_.*$"
        match = re.match(pattern, file_name).group(1)
        return match, path

    def __reduce_content_to_dict(self, dictionary, elem):
        dictionary[elem[0]] = elem[1]
        return dictionary

    def __connect_dicts_in_tuples_list(self, gt_dict, examples_dict):
        result = []
        for key, gt_path in gt_dict.items():
            example_path = examples_dict[key]
            result.append((example_path, gt_path))
        return result

    def __prepare_tfrecords(self, subset_name, files_list):
        print('========================================================================')
        print('Start creating *.tfrecords for {} subset'.format(subset_name))
        create_directory(os.path.join(self.__config.output_tfrecords_dir, subset_name))
        paths_list_len = len(files_list)
        colour_to_id = self.__get_colour_to_id_mapping()
        batch_size = self.__config.binary_batch_size
        threads = []
        for i in range(0, int(math.ceil(paths_list_len / batch_size))):
            if (i + 1) * batch_size < paths_list_len:
                chunk = files_list[i * batch_size:(i + 1) * batch_size]
            else:
                chunk = files_list[i * batch_size:]
            thread = Thread(target=self.__prepare_tfrecords_batch, args=[subset_name, i, chunk, colour_to_id])
            thread.start()
            threads.append((i, thread))
            print('Task for creating {}-th {} batch just started.'.format(i, subset_name))
        for task_id, thread in threads:
            thread.join()
            print('Task for creating {}-th {} batch just finished.')
        print('Finished creating *.tfrecords for {} subset'.format(subset_name))
        print('========================================================================')

    def __get_colour_to_id_mapping(self):
        mapping = self.__get_mapping_file_content()
        result = {}
        for idx, colour in mapping.values():
            result[tuple(colour)] = idx
        return result

    def __get_mapping_file_content(self):
        with open(self.__config.mapping_file, 'r') as stream:
            return yaml.load(stream)

    def __map_classes_id(self, gt, colour_to_id):

        def class_mapper(colour):
            colour = tuple(colour)
            if colour in colour_to_id:
                return colour_to_id[colour]
            else:
                return 0

        return np.apply_along_axis(class_mapper, 2, gt)

    def __prepare_tfrecords_batch(self, subset_name, batch_id, data, colour_to_id):
        output_dir = self.__config.output_tfrecords_dir
        tfrecords_file_name = '{}-{}'.format(self.__config.tfrecords_base_name, batch_id)
        output_file_name = os.path.join(output_dir, subset_name, tfrecords_file_name)
        with tf.python_io.TFRecordWriter(output_file_name) as writer:
            for example_path, gt_path in data:
                example, gt = self.__prepare_images_to_save(example_path, gt_path, colour_to_id)
                data = \
                    {
                        'example': self.__wrap_bytes(example),
                        'gt': self.__wrap_bytes(gt)
                    }
                feature = tf.train.Features(feature=data)
                example = tf.train.Example(features=feature)
                serialized = example.SerializeToString()
                writer.write(serialized)

    def __prepare_images_to_save(self, example_path, gt_path, colour_to_id):
        example, gt = cv.imread(example_path), cv.imread(gt_path)
        example, gt = cv.resize(example, self.__config.destination_size), cv.resize(gt, self.__config.destination_size)
        gt = gt[:, :, (2, 1, 0)]
        gt = self.__map_classes_id(gt, colour_to_id)
        gt = gt.astype(np.uint8)
        return example.tostring(), gt.tostring()

    def __wrap_bytes(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


