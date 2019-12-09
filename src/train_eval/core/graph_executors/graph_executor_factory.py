from src.common.config_utils import GraphExecutorConfigReader
from enum import Enum

from src.dataset.training_features.iterators import IteratorType
from src.train_eval.core.graph_executors.dataset_transformation_test_executor import DatasetTransformationTestExecutor
from src.train_eval.core.graph_executors.evaluation_executor import EvaluationExecutor
from src.train_eval.core.graph_executors.graph_executor import GraphExecutor
from src.train_eval.core.graph_executors.inference_executor import InferenceExecutor
from src.train_eval.core.graph_executors.inference_speed_test_executor import InferenceSpeedTestExecutor
from src.train_eval.core.graph_executors.training_executor import TrainingExecutor
from src.train_eval.core.graph_executors.tensor_board_executor import TensorBoardExecutor


class GraphExecutorFactoryError(Exception):
    pass


class GraphExecutorType(Enum):
    EVALUATION = (EvaluationExecutor, {})
    INFERENCE = (InferenceExecutor, {})
    INFERENCE_SPEED_TEST = (InferenceSpeedTestExecutor, {})
    FULL_TRAIN = (
        TrainingExecutor,
        {
            'iterator_type': IteratorType.INITIALIZABLE_TRAIN_SET_ITERATOR
        }
    )
    OVERFIT_TRAIN = (
        TrainingExecutor,
        {
            'iterator_type': IteratorType.DUMMY_ITERATOR
        }
    )
    GRAPH_VISUALISATION = (TensorBoardExecutor, {})
    TEST_DATA_TRANSFORMATION = (DatasetTransformationTestExecutor, {})


class GraphExecutorFactory:

    __ERROR_MSG = 'There is no such graph executor as {}'

    @staticmethod
    def assembly(executor_type: GraphExecutorType,
                 descriptive_name: str,
                 config: GraphExecutorConfigReader) -> GraphExecutor:
        for executor in GraphExecutorType:
            if executor_type == executor:
                executor_class, kwargs = executor.value
                return executor_class(descriptive_name, config, **kwargs)
        error_msg = GraphExecutorFactory.__ERROR_MSG.format(executor_type)
        raise GraphExecutorFactoryError(error_msg)
