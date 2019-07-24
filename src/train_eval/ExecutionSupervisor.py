from abc import ABC

from src.common.config_utils import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.graph_executor_factory import GraphExecutorType, GraphExecutorFactory


class ExecutionSupervisor(ABC):

    def _execute_graph_operation_pipeline(self, executor_type: GraphExecutorType, descriptive_name: str,
                                          config: GraphExecutorConfigReader) -> None:
        factory = GraphExecutorFactory()
        executor = factory.assembly(executor_type, descriptive_name, config)
        executor.execute()
