import os
import datetime

from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class TrainingPersistenceManager(PersistenceManager):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def _generate_model_dir_path(self) -> str:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M")
        training_dir_name = '{}_{}'.format(self.__descriptive_name, timestamp)
        return os.path.join(self._config.model_storage_directory, training_dir_name)
