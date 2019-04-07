import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.dataset.utils.mapping_utils import get_id_to_colour_mapping, map_colour
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.persistence.EvaluationPersistenceManager import EvaluationPersistenceManager
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class InferenceExecutor(GraphExecutor):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        iterator, (model_out, _), X, y = self._build_computation_graph()
        X_casted = tf.cast(X, tf.uint8)
        prediction = tf.math.argmax(model_out, axis=3, output_type=tf.dtypes.int32)
        saver = tf.train.Saver()
        config = self._get_tf_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                saver.restore(sess, self._config.checkpoint_name)
                self.__proceed_inference(sess, X_casted, prediction, y)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.DUMMY_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(self._descriptive_name, self._config)

    def __proceed_inference(self, sess: tf.Session, X_casted: tf.Tensor, prediction: tf.Tensor, y: tf.Tensor) -> None:
        mappings = get_id_to_colour_mapping(self._config.mapping_file)
        try:
            while True:
                self.__proceed_inference_on_batch(sess, X_casted, prediction, y, mappings)
        except tf.errors.OutOfRangeError:
            print('Inference [DONE]')

    def __proceed_inference_on_batch(self, sess: tf.Session, X_casted: tf.Tensor, prediction: tf.Tensor, y: tf.Tensor,
                                     mappings: Dict[int, Tuple[int, int, int]]) -> None:
        base, inf_res, gt = sess.run([X_casted, prediction, y])
        fig = plt.figure(figsize=(20, 40))
        for i in range(0, inf_res.shape[0]):
            to_show = base[i]
            fig.add_subplot(inf_res.shape[0], 3, 3 * i + 1)
            plt.imshow(to_show)
            fig.add_subplot(inf_res.shape[0], 3, 3 * i + 2)
            result = inf_res[i]
            result = map_colour(result, mappings)
            plt.imshow(result)
            fig.add_subplot(inf_res.shape[0], 3, 3 * i + 3)
            ground_truth = gt[i]
            ground_truth = map_colour(ground_truth, mappings)
            plt.imshow(ground_truth)
        image_path = self._persistence_manager.generate_random_inference_image_path()
        plt.savefig(image_path)
        plt.close()
