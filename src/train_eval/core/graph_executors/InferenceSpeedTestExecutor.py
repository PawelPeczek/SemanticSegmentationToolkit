import tensorflow as tf
from tensorflow.python.client import timeline
import time

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.persistence.EvaluationPersistenceManager import EvaluationPersistenceManager
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class InferenceSpeedTestExecutor(GraphExecutor):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        if self._config.batch_size is not 1:
            self._config.batch_size = 1
        _, model_out, _, _ = self._build_computation_graph()
        prediction = tf.math.argmax(model_out, axis=3, output_type=tf.dtypes.int32)
        saver = tf.train.Saver()
        config = self._get_tf_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                saver.restore(sess, self._config.checkpoint_name)
                self.__proceed_time_inference_test(sess, prediction)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.DUMMY_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(self._descriptive_name, self._config)

    def __proceed_time_inference_test(self, sess: tf.Session, prediction: tf.Tensor) -> None:
        times = []
        try:
            for i in range(0, 5):
                # dummy warm-up predictions
                sess.run(prediction)
            self.__profile_inference(sess, prediction)
            while True:
                start_t = time.time()
                sess.run(prediction)
                diff_t = (time.time() - start_t)
                times.append(diff_t)
        except tf.errors.OutOfRangeError:
            avg_inf_time = sum(times) / len(times) if len(times) > 0 else None
            print("Avg inference time: {}s per frame".format(avg_inf_time))
            print('Inference speed test [DONE]')

    def __profile_inference(self, sess: tf.Session, prediction: tf.Tensor) -> None:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(prediction, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        trace = fetched_timeline.generate_chrome_trace_format()
        self._persistence_manager.save_profiling_trace(trace)
