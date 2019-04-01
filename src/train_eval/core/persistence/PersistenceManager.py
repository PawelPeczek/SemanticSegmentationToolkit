import os
import tensorflow as tf
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from src.train_eval.core.config_readers.GraphExecutorConfigReader import GraphExecutorConfigReader
from src.utils.filesystem_utils import create_directory


class PersistenceManager(ABC):

    ERROR_LOG_FILE_NAME = 'loss.csv'

    def __init__(self, descriptive_name: str, config: GraphExecutorConfigReader):
        self._descriptive_name = descriptive_name
        self._config = config
        self._model_directory_path = self._generate_model_dir_path()
        self._loss_log_file_path = self.__generate_loss_log_file_name()
        self._save_path = self.__generate_checkpoint_path()
        self._profiling_path = self.__generate_model_profiling_dir_path()
        self._model_inference_results_path = self.__generate_model_inference_results_dir_path()
        self._graph_summary_path = self.__generate_graph_summary_path()
        self._prepare_storage()

    def log_loss(self, epoch: int, loss_value: float, train_acc: float = -1.0, val_acc: float = -1.0) -> None:
        log_content = '{},{},'.format(epoch, loss_value)
        if train_acc >= 0:
            log_content = '{}{},'.format(log_content, train_acc)
        else:
            log_content = '{},'.format(log_content)
        if val_acc >= 0:
            log_content = '{}{}'.format(log_content, val_acc)
        if not os.path.exists(self._loss_log_file_path):
            header = 'Epoch no.,Loss,Train acc.,Val acc.{}'.format(os.linesep)
            log_content = '{}{}'.format(header, log_content)
        with open(self._loss_log_file_path, 'a') as log_file:
            log_file.write('{}{}'.format(log_content, os.linesep))

    def persist_model(self, session: tf.Session, epoch: int) -> None:
        print('===========================================================')
        print('Checkpoint saving [IN PROGRESS]')
        saver = tf.train.Saver()
        saver.save(session, self._save_path, global_step=epoch)
        print('Checkpoint saving [DONE]')
        print('===========================================================')

    def generate_random_inference_image_path(self) -> str:
        rand_name = '{}-{}.png'.format(self._descriptive_name, uuid.uuid4())
        return os.path.join(self._config.model_dir, rand_name)

    def save_profiling_trace(self, trace: str) -> None:
        timestamp = self._get_current_timestamp()
        trace_file_path = os.path.join(self._profiling_path, 'profiling_trace_{}.json'.format(timestamp))
        with open(trace_file_path, 'w') as file:
            file.write(trace)

    def save_graph_summary(self, graph: tf.Graph) -> None:
        tf.summary.FileWriter(self._graph_summary_path, graph)

    def get_graph_summary_dir_path(self) -> str:
        return self._graph_summary_path

    @abstractmethod
    def _generate_model_dir_path(self) -> str:
        raise NotImplementedError('This method must be implemented.')

    def _prepare_storage(self) -> None:
        create_directory(self._model_directory_path)
        create_directory(self._profiling_path)
        create_directory(self._model_inference_results_path)
        create_directory(self._graph_summary_path)

    def _get_current_timestamp(self) -> str:
        return datetime.now().strftime("%Y_%m_%d_%H:%M")

    def __generate_loss_log_file_name(self) -> str:
        return os.path.join(self._model_directory_path, self.ERROR_LOG_FILE_NAME)

    def __generate_checkpoint_path(self) -> str:
        return os.path.join(self._model_directory_path, '{}.ckpt'.format(self._config.checkpoint_name))

    def __generate_model_profiling_dir_path(self) -> str:
        return os.path.join(self._model_directory_path, 'profiler_output')

    def __generate_model_inference_results_dir_path(self) -> str:
        return os.path.join(self._model_directory_path, 'model_inference')

    def __generate_graph_summary_path(self):
        return os.path.join(self._model_directory_path, 'graph_summary')

