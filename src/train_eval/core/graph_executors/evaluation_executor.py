import tensorflow as tf

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.training_features.iterators import IteratorType
from src.train_eval.core.graph_executors.graph_executor import GraphExecutor
from src.train_eval.core.graph_executors.utils import evaluate_miou, \
    get_validation_operation, ValidationOperation
from src.train_eval.core.persistence.managers import PersistenceManager, \
    EvaluationPersistenceManager


class EvaluationExecutor(GraphExecutor):

    def __init__(self,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def execute(self) -> None:
        iterator = self._get_iterator()
        num_classes = self._config.num_classes
        labels_to_ignore = self._config.get_or_else('ignore_labels', None)
        validation_operation = get_validation_operation(
            iterator=iterator,
            model=self._model,
            num_classes=num_classes,
            labels_to_ignore=labels_to_ignore)
        with tf.device("/gpu:{}".format(self._config.gpu_to_use)):
            self.__proceed_evaluation(validation_operation)

    def _get_iterator_type(self) -> IteratorType:
        return IteratorType.OS_VALIDATION_ITERATOR

    def _get_persistence_manager(self) -> PersistenceManager:
        return EvaluationPersistenceManager(
            descriptive_name=self._descriptive_name,
            config=self._config)

    def __proceed_evaluation(self,
                             validation_operation: ValidationOperation) -> None:
        saver = tf.train.Saver()
        config = self._get_session_config()
        with tf.Session(config=config) as session:
            saver.restore(session, self._config.checkpoint_name)
            miou_metrics = evaluate_miou(
                session=session,
                validation_operation=validation_operation)
            print("Mean IOU: {}".format(miou_metrics))
