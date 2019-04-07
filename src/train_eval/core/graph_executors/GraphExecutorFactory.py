from src.dataset.common.CityScapesIteratorFactory import IteratorType
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from enum import Enum

from src.train_eval.core.graph_executors.EvaluationExecutor import EvaluationExecutor
from src.train_eval.core.graph_executors.GraphExecutor import GraphExecutor
from src.train_eval.core.graph_executors.InferenceExecutor import InferenceExecutor
from src.train_eval.core.graph_executors.InferenceSpeedTestExecutor import InferenceSpeedTestExecutor
from src.train_eval.core.graph_executors.TrainingExecutor import TrainingExecutor
from src.train_eval.core.graph_executors.VisualisationExecutor import VisualisationExecutor


class GraphExecutorType(Enum):
    EVALUATION = 1
    INFERENCE = 2
    INFERENCE_SPEED_TEST = 3
    FULL_TRAIN = 4
    OVERFIT_TRAIN = 5
    GRAPH_VISUALISATION = 6


class GraphExecutorFactory:

    def assembly(self, executor_type: GraphExecutorType, descriptive_name: str, config: GraphExecutorConfigReader) -> GraphExecutor:
        if executor_type == GraphExecutorType.EVALUATION:
            return EvaluationExecutor(descriptive_name, config)
        if executor_type == GraphExecutorType.INFERENCE:
            return InferenceExecutor(descriptive_name, config)
        if executor_type == GraphExecutorType.INFERENCE_SPEED_TEST:
            return InferenceSpeedTestExecutor(descriptive_name, config)
        if executor_type == GraphExecutorType.FULL_TRAIN:
            return TrainingExecutor(descriptive_name, config, IteratorType.INITIALIZABLE_TRAIN_SET_ITERATOR)
        if executor_type == GraphExecutorType.OVERFIT_TRAIN:
            return TrainingExecutor(descriptive_name, config, IteratorType.DUMMY_ITERATOR)
        else:
            return VisualisationExecutor(descriptive_name, config)

