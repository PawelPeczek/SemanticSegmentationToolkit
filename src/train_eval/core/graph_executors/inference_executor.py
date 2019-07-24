import tensorflow as tf
import matplotlib.pyplot as plt

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.iterators import IteratorType
from src.dataset.utils.mapping_utils import get_id_to_color_mapping, map_colour, \
    Id2ColorMapping
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
        prediction = self._model.infer(x)
        x_casted = tf.cast(x, tf.uint8)
        with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                self.__proceed_inference(x_casted, prediction, y)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.DUMMY_ITERATOR

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
        mappings = get_id_to_color_mapping(self._config.mapping_file)
        while True:
            self.__proceed_inference_on_batch(
                session=session,
                x_casted=x_casted,
                prediction=prediction,
                y=y,
                mappings=mappings)

    def __proceed_inference_on_batch(self,
                                     session: tf.Session,
                                     x_casted: tf.Tensor,
                                     prediction: tf.Tensor,
                                     y: tf.Tensor,
                                     mappings: Id2ColorMapping) -> None:
        base, inf_res, gt = session.run([x_casted, prediction, y])
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
        image_path = self._persistence_manager.generate_inference_image_path()
        plt.savefig(image_path)
        plt.close()
