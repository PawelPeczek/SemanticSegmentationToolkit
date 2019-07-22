import tensorflow as tf

from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.persistence.EvaluationPersistenceManager import EvaluationPersistenceManager
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class VisualisationExecutor(GraphExecutor):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        _, (model_out, _), _, _ = self._build_computation_graph()
        config = self._get_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                sess.run(tf.global_variables_initializer())
                self._persistence_manager.save_graph_summary(sess.graph)
        self.__print_tensorboard_start_command()

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.DUMMY_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(self._descriptive_name, self._config)

    def __print_tensorboard_start_command(self):
        logdir = self._persistence_manager.get_graph_summary_dir_path()
        print('To run tensorboard use the following command:')
        print('tensorboard --logdir={}'.format(logdir))
