from typing import Dict, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import numpy as np

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.dataset.utils.mapping_utils import get_id_to_colour_mapping, map_colour
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager
from src.train_eval.core.persistence.TrainingPersistenceManager import TrainingPersistenceManager


class DatasetTransformationTestExecutor(GraphExecutor):

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)
        self._config.transoformation_options['application_probability'] = 1.0

    def execute(self) -> None:
        _, _, image, label = self._build_computation_graph()
        image = tf.cast(image, tf.uint8)
        config = self._get_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                self.__proceed_inference(sess, image, label)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.OS_TRAIN_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return TrainingPersistenceManager(self._descriptive_name, self._config)

    def __proceed_inference(self, sess: tf.Session, image: tf.Tensor, label: tf.Tensor) -> None:
        mappings = get_id_to_colour_mapping(self._config.mapping_file)
        try:
            while True:
                self.__proceed_inference_on_batch(sess, image, label, mappings)
        except tf.errors.OutOfRangeError:
            print('Inference [DONE]')

    def __proceed_inference_on_batch(self, sess: tf.Session, image: tf.Tensor, label: tf.Tensor,
                                     mappings: Dict[int, Tuple[int, int, int]]) -> None:
        print('DUPA')
        params = self.__prepare_prediction_to_color_mapping()
        label = tf.expand_dims(label, axis=-1)
        label = tf.image.resize_nearest_neighbor(label, (256, 512))
        label = tf.squeeze(label, axis=-1)
        label = tf.gather(params, label)
        print(label.shape)
        label = tf.cast(label, dtype=tf.uint8)
        # label = tf.expand_dims(label, axis=-1)
        # label = tf.image.resize_nearest_neighbor(label, (512, 1024))
        # label = tf.squeeze(label, axis=-1)
        base, gt = sess.run([image, label])

        # fig = plt.figure(figsize=(20, 40))
        for i in range(0, base.shape[0]):
        #     to_show = base[i][..., ::-1]
        #     fig.add_subplot(base.shape[0], 2, 2 * i + 1)
        #     plt.imshow(to_show)
        #     fig.add_subplot(base.shape[0], 2, 2 * i + 2)
            ground_truth = gt[i]
            print(ground_truth)

            cv.imshow('dupa', ground_truth.copy())
            cv.waitKey(1)
        #     plt.imshow(ground_truth)
        # image_path = self._persistence_manager.generate_random_transformation_test_image_path()
        # plt.savefig(image_path)
        # plt.close()

    def __prepare_prediction_to_color_mapping(self) -> np.ndarray:
        mappings = get_id_to_colour_mapping(self._config.mapping_file)
        return np.array([(0, 0, 0)] + list(mappings.values()))
