from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class EvaluationPersistenceManager(PersistenceManager):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def _generate_model_dir_path(self) -> str:
        return self._config.model_dir
