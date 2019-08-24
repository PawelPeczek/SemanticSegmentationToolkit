import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.iterators import IteratorType
from src.dataset.utils.mapping_utils import get_id_to_color_mapping, \
    map_colour, Id2ColorMapping
from src.train_eval.core.graph_executors.graph_executor import GraphExecutor
from src.train_eval.core.persistence.managers import PersistenceManager, \
    TrainingPersistenceManager


class DatasetTransformationTestExecutor(GraphExecutor):

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)
        self._config.transoformation_options['application_probability'] = 1.0

    def execute(self) -> None:
        iterator = self._get_iterator()
        image, label = iterator.get_next()
        image = tf.cast(image, tf.uint8)
        session_config = self._get_session_config()
        with tf.Session(config=session_config) as session:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                self.__proceed_inference(session, image, label)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.OS_TRAIN_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return TrainingPersistenceManager(self._descriptive_name, self._config)

    def __proceed_inference(self,
                            session: tf.Session,
                            image: tf.Tensor,
                            label: tf.Tensor) -> None:
        try:
            self.__proceed_inference_loop(
                session=session,
                image=image,
                label=label)
        except tf.errors.OutOfRangeError:
            print('Inference [DONE]')

    def __proceed_inference_loop(self,
                                 session: tf.Session,
                                 image: tf.Tensor,
                                 label: tf.Tensor):
        while True:
            self.__proceed_inference_on_batch(
                session=session,
                image=image,
                label=label)

    def __proceed_inference_on_batch(self,
                                     session: tf.Session,
                                     image: tf.Tensor,
                                     label: tf.Tensor) -> None:
        x, gt = session.run([image, label])
        task = self._config.get_or_else('task', 'segmentation')
        if task.lower() == 'segmentation':
            self.__visualise_segmentation_input(x=x, gt=gt)
        else:
            self.__visualise_auto_encoding_input(x=x, gt=gt)

    def __visualise_segmentation_input(self,
                                       x: np.ndarray,
                                       gt: np.ndarray) -> None:
        mappings = get_id_to_color_mapping(self._config.mapping_file)
        fig = plt.figure(figsize=(20, 40))
        for i in range(0, x.shape[0]):
            to_show = x[i][..., ::-1]
            fig.add_subplot(x.shape[0], 2, 2 * i + 1)
            plt.imshow(to_show)
            fig.add_subplot(x.shape[0], 2, 2 * i + 2)
            ground_truth = gt[i]
            ground_truth = map_colour(ground_truth, mappings)
            plt.imshow(ground_truth)
        path = self._persistence_manager.generate_transformation_image_path()
        plt.savefig(path)
        plt.close()

    def __visualise_auto_encoding_input(self,
                                        x: np.ndarray,
                                        gt: np.ndarray) -> None:
        for i in range(0, x.shape[0]):
            single_x = x[i][..., ::-1]
            single_gt = gt[i][..., ::-1]
            self.__persist_single_auto_encoding_input(
                x=single_x,
                gt=single_gt)

    def __persist_single_auto_encoding_input(self,
                                             x: np.ndarray,
                                             gt: np.ndarray) -> None:
        stacked_image = np.column_stack((x, gt))
        path = self._persistence_manager.generate_transformation_image_path()
        cv.imwrite(path, stacked_image)

    def __prepare_prediction_to_color_mapping(self) -> np.ndarray:
        mappings = get_id_to_color_mapping(self._config.mapping_file)
        return np.array([(0, 0, 0)] + list(mappings.values()))
