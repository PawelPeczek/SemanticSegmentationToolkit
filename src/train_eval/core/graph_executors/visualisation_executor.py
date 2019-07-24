import tensorflow as tf

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.common.iterators import IteratorType
from src.train_eval.core.graph_executors.graph_executor import GraphExecutor
from src.train_eval.core.persistence.managers import PersistenceManager, \
    EvaluationPersistenceManager


class VisualisationExecutor(GraphExecutor):

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        iterator = self._get_iterator()
        x, _ = iterator.get_next()
        self._model.infer(x)
        config = self._get_session_config()
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
                sess.run(tf.global_variables_initializer())
                self._persistence_manager.save_graph_summary(sess.graph)
        self.__print_tensorboard_start_command()

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.DUMMY_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(
            descriptive_name=self._descriptive_name,
            config=self._config)

    def __print_tensorboard_start_command(self):
        logdir = self._persistence_manager.get_graph_summary_dir_path()
        print('To run tensorboard use the following command:')
        print('tensorboard --logdir={}'.format(logdir))
