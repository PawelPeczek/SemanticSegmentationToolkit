from fire import Fire
from typing import Optional

from src.common.config_utils import GraphExecutorConfigReader
from src.train_eval.ExecutionSupervisor import ExecutionSupervisor
from src.train_eval.core.graph_executors.graph_executor_factory \
    import GraphExecutorType


class TrainingSupervisor(ExecutionSupervisor):

    def full_training(self,
                      descriptive_name: str,
                      config_path: Optional[str] = None
                      ) -> None:
        try:
            config = GraphExecutorConfigReader(config_path)
            self._execute_graph_operation_pipeline(
                executor_type=GraphExecutorType.FULL_TRAIN,
                descriptive_name=descriptive_name,
                config=config
            )
        except Exception as ex:
            print(f'Failed to proceed full training. {ex}')

    def overfit_training(self,
                         descriptive_name: str,
                         config_path: Optional[str] = None
                         ) -> None:
        try:
            config = GraphExecutorConfigReader(config_path)
            self._execute_graph_operation_pipeline(
                executor_type=GraphExecutorType.OVERFIT_TRAIN,
                descriptive_name=descriptive_name,
                config=config
            )
        except Exception as ex:
            print(f'Failed to proceed overfitting training. {ex}')

    def visualise_data_augmentation(self,
                                    descriptive_name: str,
                                    config_path: Optional[str] = None
                                    ) -> None:
        try:
            config = GraphExecutorConfigReader(config_path)
            self._execute_graph_operation_pipeline(
                executor_type=GraphExecutorType.TEST_DATA_TRANSFORMATION,
                descriptive_name=descriptive_name,
                config=config
            )
        except Exception as ex:
            print(f'Failed to proceed data augmentation visualisation. {ex}')


if __name__ == '__main__':
    Fire(TrainingSupervisor)
