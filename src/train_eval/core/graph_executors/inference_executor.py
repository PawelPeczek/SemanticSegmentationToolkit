import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.iterators import IteratorType
from src.dataset.utils.mapping_utils import get_id_to_color_mapping, map_colour
from src.train_eval.core.graph_executors.graph_executor import GraphExecutor
from src.train_eval.core.persistence.managers import PersistenceManager, \
    EvaluationPersistenceManager


class InferenceExecutor(GraphExecutor):

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        iterator = self._get_iterator()
        x, y = iterator.get_next()
        variable_scope = tf.get_variable_scope()
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            prediction = self._model.infer(x)
        x_casted = tf.cast(x, tf.uint8)
        with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
            self.__proceed_inference(x_casted, prediction, y)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.OS_VALIDATION_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(
            descriptive_name=self._descriptive_name,
            config=self._config)

    def __proceed_inference(self,
                            x_casted: tf.Tensor,
                            prediction: tf.Tensor,
                            y: tf.Tensor) -> None:
        saver = tf.train.Saver()
        session_config = self._get_session_config()
        with tf.Session(config=session_config) as session:
            saver.restore(session, self._config.checkpoint_name)
            try:
                self.__proceed_inference_loop(
                    session=session,
                    x_casted=x_casted,
                    prediction=prediction,
                    y=y)
            except tf.errors.OutOfRangeError:
                print('Inference [DONE]')

    def __proceed_inference_loop(self,
                                 session: tf.Session,
                                 x_casted: tf.Tensor,
                                 prediction: tf.Tensor,
                                 y: tf.Tensor) -> None:
        while True:
            self.__proceed_inference_on_batch(
                session=session,
                x_casted=x_casted,
                prediction=prediction,
                y=y)

    def __proceed_inference_on_batch(self,
                                     session: tf.Session,
                                     x_casted: tf.Tensor,
                                     prediction: tf.Tensor,
                                     y: tf.Tensor) -> None:
        input_image, inferred_image, gt = session.run([x_casted, prediction, y])
        task = self._config.get_or_else('task', 'segmentation')
        if task.lower() == 'segmentation':
            self.__proceed_segmentation_inference(
                input_image=input_image,
                inferred_image=inferred_image,
                gt=gt)
        else:
            self.__proceed_auto_encoding_inference(
                input_image=input_image,
                inferred_image=inferred_image,
                gt=gt)

    def __proceed_segmentation_inference(self,
                                         input_image: np.ndarray,
                                         inferred_image: np.ndarray,
                                         gt: np.ndarray) -> None:
        mappings = get_id_to_color_mapping(self._config.mapping_file)
        fig = plt.figure(figsize=(20, 40))
        for i in range(0, input_image.shape[0]):
            to_show = input_image[i]
            fig.add_subplot(input_image.shape[0], 3, 3 * i + 1)
            plt.imshow(to_show)
            fig.add_subplot(input_image.shape[0], 3, 3 * i + 2)
            result = inferred_image[i]
            result = map_colour(result, mappings)
            plt.imshow(result)
            fig.add_subplot(input_image.shape[0], 3, 3 * i + 3)
            ground_truth = gt[i]
            ground_truth = map_colour(ground_truth, mappings)
            plt.imshow(ground_truth)
        image_path = self._persistence_manager.generate_inference_image_path()
        plt.savefig(image_path)
        plt.close()

    def __proceed_auto_encoding_inference(self,
                                          input_image: np.ndarray,
                                          inferred_image: np.ndarray,
                                          gt: np.ndarray) -> None:
        for i in range(0, input_image.shape[0]):
            single_x = input_image[i][..., ::-1]
            single_inference = inferred_image[i][..., ::-1]
            single_gt = gt[i][..., ::-1]
            self.__persist_single_auto_encoding_inference_result(
                x=single_x,
                y_dash=single_inference,
                gt=single_gt)

    def __persist_single_auto_encoding_inference_result(self,
                                                        x: np.ndarray,
                                                        y_dash: np.ndarray,
                                                        gt: np.ndarray) -> None:
        stacked_image = np.concatenate([x, y_dash, gt], axis=-2)
        print(stacked_image.shape)
        path = self._persistence_manager.generate_inference_image_path()
        print(path)
        cv.imwrite(path, stacked_image)
