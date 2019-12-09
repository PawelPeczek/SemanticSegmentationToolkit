from functools import partial
from typing import Optional

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.training_features.iterators import IteratorType
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
        iterator_element = iterator.get_next()
        idx = None
        if len(iterator_element) == 3:
            idx, x, y = iterator_element
        else:
            x, y = iterator_element
        variable_scope = tf.get_variable_scope()
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            prediction = self._model.infer(x)
        x_casted = tf.cast(x, tf.uint8)
        with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
            self.__proceed_inference(x_casted, prediction, y, idx)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.OS_VALIDATION_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(
            descriptive_name=self._descriptive_name,
            config=self._config)

    def __proceed_inference(self,
                            x_casted: tf.Tensor,
                            prediction: tf.Tensor,
                            y: tf.Tensor,
                            idx: Optional[tf.Tensor]) -> None:
        saver = tf.train.Saver()
        session_config = self._get_session_config()
        with tf.Session(config=session_config) as session:
            saver.restore(session, self._config.checkpoint_name)
            try:
                self.__proceed_inference_loop(
                    session=session,
                    x_casted=x_casted,
                    prediction=prediction,
                    y=y,
                    idx=idx)
            except tf.errors.OutOfRangeError:
                print('Inference [DONE]')

    def __proceed_inference_loop(self,
                                 session: tf.Session,
                                 x_casted: tf.Tensor,
                                 prediction: tf.Tensor,
                                 y: tf.Tensor,
                                 idx: Optional[tf.Tensor]) -> None:
        while True:
            self.__proceed_inference_on_batch(
                session=session,
                x_casted=x_casted,
                prediction=prediction,
                y=y,
                idx=idx)

    def __proceed_inference_on_batch(self,
                                     session: tf.Session,
                                     x_casted: tf.Tensor,
                                     prediction: tf.Tensor,
                                     y: tf.Tensor,
                                     idx: Optional[tf.Tensor]) -> None:

        task = self._config.get_or_else('task', 'segmentation')
        if task.lower() == 'segmentation':
            images_idx, input_image, inferred_image, gt = session.run(
                [idx, x_casted, prediction, y]
            )
            self.__proceed_segmentation_inference(
                images_idx=images_idx,
                input_image=input_image,
                inferred_image=inferred_image,
                gt=gt
            )
        else:
            input_image, inferred_image, gt = session.run(
                [x_casted, prediction, y]
            )
            self.__proceed_auto_encoding_inference(
                input_image=input_image,
                inferred_image=inferred_image,
                gt=gt)

    def __proceed_segmentation_inference(self,
                                         images_idx: np.ndarray,
                                         input_image: np.ndarray,
                                         inferred_image: np.ndarray,
                                         gt: np.ndarray) -> None:
        mappings = get_id_to_color_mapping(self._config.mapping_file)
        for i in range(0, input_image.shape[0]):
            idx = images_idx[i]
            x = input_image[i][..., ::-1]
            y_dash = inferred_image[i]
            y_dash = map_colour(y_dash, mappings)[..., ::-1]
            y = gt[i]
            y = map_colour(y, mappings)[..., ::-1]
            diff_map = self.__prepare_difference_map(
                first_image=y_dash,
                second_image=y
            )
            self.__persist_single_semantic_inference_result(
                idx=idx,
                x=x,
                y_dash=y_dash,
                gt=y,
                diff_map=diff_map
            )

    def __persist_single_semantic_inference_result(self,
                                                   idx: int,
                                                   x: np.ndarray,
                                                   y_dash: np.ndarray,
                                                   gt: np.ndarray,
                                                   diff_map: np.ndarray
                                                   ) -> None:
        full_results = self.__stack_inference_results(
            x=x,
            y_dash=y_dash,
            gt=gt,
            diff_map=diff_map
        )
        full_results_name = f'{idx}_full_results'
        full_results_path = \
            self._persistence_manager.generate_inference_image_path(
                name=full_results_name
            )
        cv.imwrite(full_results_path, full_results)
        inference_only_name = f'{idx}_inference_only'
        inference_only_path = \
            self._persistence_manager.generate_inference_image_path(
                name=inference_only_name
            )
        cv.imwrite(inference_only_path, y_dash)

    def __proceed_auto_encoding_inference(self,
                                          input_image: np.ndarray,
                                          inferred_image: np.ndarray,
                                          gt: np.ndarray) -> None:
        for i in range(0, input_image.shape[0]):
            single_x = input_image[i][..., ::-1]
            single_inference = inferred_image[i][..., ::-1]
            single_gt = gt[i][..., ::-1]
            diff_map = self.__prepare_difference_map(
                first_image=single_inference,
                second_image=single_gt
            )
            self.__persist_single_auto_encoding_inference_result(
                x=single_x,
                y_dash=single_inference,
                gt=single_gt,
                diff_map=diff_map
            )

    def __persist_single_auto_encoding_inference_result(self,
                                                        x: np.ndarray,
                                                        y_dash: np.ndarray,
                                                        gt: np.ndarray,
                                                        diff_map: np.ndarray
                                                        ) -> None:
        stacked_image = self.__stack_inference_results(
            x=x,
            y_dash=y_dash,
            gt=gt,
            diff_map=diff_map
        )
        path = self._persistence_manager.generate_inference_image_path()
        cv.imwrite(path, stacked_image)

    def __prepare_difference_map(self,
                                 first_image: np.ndarray,
                                 second_image: np.ndarray
                                 ) -> np.ndarray:
        first_image = first_image.astype(np.uint8)
        second_image = second_image.astype(np.uint8)
        images = [first_image, second_image]
        color_conversion = partial(cv.cvtColor, code=cv.COLOR_RGB2GRAY)
        images = list(map(color_conversion, images))
        diff_picture = np.absolute(images[0] - images[1])
        return cv.cvtColor(diff_picture, cv.COLOR_GRAY2RGB)

    def __stack_inference_results(self,
                                  x: np.ndarray,
                                  y_dash: np.ndarray,
                                  gt: np.ndarray,
                                  diff_map: np.ndarray
                                  ) -> np.ndarray:
        top_row = np.concatenate([x, y_dash], axis=-2)
        bottom_row = np.concatenate([gt, diff_map], axis=-2)
        stacked_image = np.concatenate([top_row, bottom_row], axis=0)
        return stacked_image
