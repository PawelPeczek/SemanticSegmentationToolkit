import tensorflow as tf

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.persistence.EvaluationPersistenceManager import EvaluationPersistenceManager
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class EvaluationExecutor(GraphExecutor):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        if self._config.batch_size is not 1:
            self._config.batch_size = 1
        _, model_out, _, y = self._build_computation_graph()
        prediction = tf.math.argmax(model_out, axis=3, output_type=tf.dtypes.int32)
        weights = tf.to_float(tf.not_equal(y, 0))
        mean_iou, mean_iou_update = tf.metrics.mean_iou(prediction, y, self._config.num_classes, weights=weights)
        saver = tf.train.Saver()
        config = self._get_tf_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                sess.run(tf.initializers.variables(tf.local_variables()))
                saver.restore(sess, self._config.checkpoint_name)
                self.__proceed_evaluation(sess, mean_iou, mean_iou_update)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.VALIDATION_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(self._descriptive_name, self._config)

    def __proceed_evaluation(self, sess: tf.Session, mean_iou: tf.Operation, mean_iou_update: tf.Operation) -> None:
        try:
            while True:
                sess.run(mean_iou_update)
        except tf.errors.OutOfRangeError:
            miou_metrics = sess.run(mean_iou)
            print("Mean IOU: {}".format(miou_metrics))
