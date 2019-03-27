from fire import Fire
from typing import Union

from src.train_eval.ExecutionSupervisor import ExecutionSupervisor
from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutorFactory import GraphExecutorType


class EvaluationSupervisor(ExecutionSupervisor):

    def evaluate_model(self, descriptive_name: Union[str, None] = None, config_path: Union[str, None] = None) -> None:
        try:
            config = GraphExecutorConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'model_evaluation'
            self._execute_graph_operation_pipeline(GraphExecutorType.EVALUATION, descriptive_name, config)
        except Exception as ex:
            print('Failed to proceed model evaluation. {}'.format(ex))

    def get_predictions_from_model(self, descriptive_name: Union[str, None] = None, config_path: Union[str, None] = None) -> None:
        try:
            config = GraphExecutorConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'model_predictions'
            self._execute_graph_operation_pipeline(GraphExecutorType.INFERENCE, descriptive_name, config)
        except Exception as ex:
            print('Failed to proceed taking inference from model. {}'.format(ex))

    def measure_inference_speed(self, descriptive_name: Union[str, None] = None, config_path: Union[str, None] = None) -> None:
        try:
            config = GraphExecutorConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'model_inference_speed_measurement'
            self._execute_graph_operation_pipeline(GraphExecutorType.INFERENCE_SPEED_TEST, descriptive_name, config)
        except Exception as ex:
            print('Failed to proceed speed evaluation. {}'.format(ex))

    def visualise_graph(self, descriptive_name: Union[str, None] = None, config_path: Union[str, None] = None) -> None:
        try:
            config = GraphExecutorConfigReader(config_path, reader_type='val')
            if descriptive_name is None:
                descriptive_name = 'graph_visualisation'
            self._execute_graph_operation_pipeline(GraphExecutorType.GRAPH_VISUALISATION, descriptive_name, config)
        except Exception as ex:
            print('Failed to visualise graph. {}'.format(ex))


if __name__ == '__main__':
    Fire(EvaluationSupervisor)
