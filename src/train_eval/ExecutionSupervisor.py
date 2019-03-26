from abc import ABC

from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.graph_executors.GraphExecutorFactory import GraphExecutorType, GraphExecutorFactory


class ExecutionSupervisor(ABC):

    def _execute_graph_operation_pipeline(self, executor_type: GraphExecutorType, descriptive_name: str,
                                          config: GraphExecutorConfigReader) -> None:
        factory = GraphExecutorFactory()
        executor = factory.assembly(executor_type, descriptive_name, config)
        executor.execute()
