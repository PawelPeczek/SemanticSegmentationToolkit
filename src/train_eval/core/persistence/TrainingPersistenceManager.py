import os
from shutil import copyfile

from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.train_eval.core.persistence.PersistenceManager import PersistenceManager


class TrainingPersistenceManager(PersistenceManager):

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        super().__init__(descriptive_name, config)

    def _generate_model_dir_path(self) -> str:
        timestamp = self._get_current_timestamp()
        training_dir_name = '{}_{}'.format(self._descriptive_name, timestamp)
        return os.path.join(self._config.model_storage_directory, training_dir_name)

    def _prepare_storage(self) -> None:
        super()._prepare_storage()
        config_copy_path = os.path.join(self._model_directory_path, 'train-config.yml')
        copyfile(self._config.get_config_path(), config_copy_path)
