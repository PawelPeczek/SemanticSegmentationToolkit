import os
from shutil import copyfile
import tensorflow as tf
import uuid
from abc import ABC, abstractmethod

from src.train_eval.core.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.utils.filesystem_utils import create_directory


class PersistenceManager(ABC):

    ERROR_LOG_FILE_NAME = 'loss.csv'

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        self.__descriptive_name = descriptive_name
        self._config = config
        self.__model_directory_path = self._generate_model_dir_path()
        self.__loss_log_file_path = self.__generate_loss_log_file_name()
        self.__save_path = self.__generate_checkpoint_path()
        self.__profiling_path = self.__generate_model_profiling_dir_path()
        self.__model_inference_results_path = self.__generate_model_inference_results_dir_path()

    def prepare_storage(self) -> None:
        create_directory(self.__config.model_storage_directory)
        create_directory(self.__model_directory_path)
        create_directory(self.__profiling_path)
        create_directory(self.__model_inference_results_path)
        config_copy_path = os.path.join(self.__model_directory_path, 'train-config.yaml')
        copyfile(self.__config.get_config_path(), config_copy_path)

    def log_loss(self, epoch: int, loss_value: float) -> None:
        with open(self.__loss_log_file_path, 'a') as log_file:
            log_file.write('{},{}{}'.format(epoch, loss_value, os.linesep))

    def persist_model(self, session: tf.Session, epoch: int) -> None:
        print('===========================================================')
        print('Checkpoint saving [IN PROGRESS]')
        saver = tf.train.Saver()
        saver.save(session, self.__save_path, global_step=epoch)
        print('Checkpoint saving [DONE]')
        print('===========================================================')

    def generate_random_inference_image_path(self):
        rand_name = '{}-{}.png'.format(self.__descriptive_name, uuid.uuid4())
        return os.path.join(self.__config.model_dir, rand_name)

    @abstractmethod
    def _generate_model_dir_path(self) -> str:
        raise NotImplementedError('This method must be implemented.')

    def __generate_loss_log_file_name(self) -> str:
        return os.path.join(self.__model_directory_path, self.ERROR_LOG_FILE_NAME)

    def __generate_checkpoint_path(self) -> str:
        return os.path.join(self.__model_directory_path, '{}.ckpt'.format(self.__config.checkpoint_name))

    def __generate_model_profiling_dir_path(self) -> str:
        return os.path.join(self.__model_directory_path, 'profiler_output')

    def __generate_model_inference_results_dir_path(self) -> str:
        return os.path.join(self.__model_directory_path, 'model_inference')

