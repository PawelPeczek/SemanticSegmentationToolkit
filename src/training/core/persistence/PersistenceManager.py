import datetime
import os
from shutil import copyfile
import tensorflow as tf
from src.utils.filesystem_utils import create_directory


class PersistenceManager:

    ERROR_LOG_FILE_NAME = 'loss.csv'

    def __init__(self, descriptive_name, config):
        """
        :param descriptive_name: training process descriptive identifier
        :param config: TrainingConfigReader object
        """
        self.__descriptive_name = descriptive_name
        self.__config = config
        self.__training_directory_path = self.__generate_training_dir_path()
        self.__loss_log_file_path = self.__generate_loss_log_file_name()
        self.__save_path = self.__generate_checkpoint_path()

    def prepare_storage(self):
        create_directory(self.__config.model_storage_directory)
        create_directory(self.__training_directory_path)
        config_copy_path = os.path.join(self.__training_directory_path, 'config.yaml')
        copyfile(self.__config.get_config_path(), config_copy_path)

    def log_loss(self, epoch, loss_value):
        with open(self.__loss_log_file_path, 'a') as log_file:
            log_file.write('{},{}{}'.format(epoch, loss_value, os.linesep))

    def persist_model(self, session, epoch):
        print('===========================================================')
        print('Checkpoint saving [IN PROGRESS]')
        saver = tf.train.Saver()
        saver.save(session, self.__save_path, global_step=epoch)
        print('Checkpoint saving [DONE]')
        print('===========================================================')

    def __generate_training_dir_path(self):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M")
        training_dir_name = '{}_{}'.format(self.__descriptive_name, timestamp)
        return os.path.join(self.__config.model_storage_directory, training_dir_name)

    def __generate_loss_log_file_name(self):
        return os.path.join(self.__training_directory_path, self.ERROR_LOG_FILE_NAME)

    def __generate_checkpoint_path(self):
        return os.path.join(self.__training_directory_path, '{}.ckpt'.format(self.__config.checkpoint_name))

