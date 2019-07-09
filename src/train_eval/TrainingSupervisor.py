from fire import Fire
from typing import Union

from src.train_eval.ExecutionSupervisor import ExecutionSupervisor
from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutorFactory import GraphExecutorType


class TrainingSupervisor(ExecutionSupervisor):

    def full_training(self, descriptive_name: str, config_path: Union[str, None] = None) -> None:
        # try:
        config = GraphExecutorConfigReader(config_path)
        self._execute_graph_operation_pipeline(GraphExecutorType.FULL_TRAIN, descriptive_name, config)
        # except Exception as ex:
        #     print('Failed to proceed full training. {}'.format(ex))

    def overfit_training(self, descriptive_name: str, config_path: Union[str, None] = None) -> None:
        try:
            config = GraphExecutorConfigReader(config_path)
            self._execute_graph_operation_pipeline(GraphExecutorType.OVERFIT_TRAIN, descriptive_name, config)
        except Exception as ex:
            print('Failed to proceed overfitting training. {}'.format(ex))

    def visualise_data_augmentation(self, descriptive_name: str, config_path: Union[str, None] = None) -> None:
        try:
            config = GraphExecutorConfigReader(config_path)
            self._execute_graph_operation_pipeline(GraphExecutorType.TEST_DATA_TRANSFORMATION, descriptive_name, config)
        except Exception as ex:
            print('Failed to proceed data augmentation visualisation. {}'.format(ex))


if __name__ == '__main__':
    Fire(TrainingSupervisor)
