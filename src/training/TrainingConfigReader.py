import yaml
import os
from src.common.ConfigReader import ConfigReader


class TrainingConfigReader(ConfigReader):

    def __init__(self, config_path=None):
        super().__init__(config_path)

    def _get_default_config_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

    def _read_config(self, config_path):
        with open(config_path, 'r') as config_file:
            return yaml.load(config_file)

    def _adjust_config_dict(self):
        self.__adjust_dataset_directories_paths()
        self.__adjust_mappping_file_path()
        self.__adjust_destination_size()

    def __adjust_dataset_directories_paths(self):
        base_dataset_dir = self.__conf_dict['dataset_dir']
        self.__conf_dict['tfrecords_dir'] = os.path.join(base_dataset_dir, self.__conf_dict['tfrecords_dir'])

    def __adjust_mappping_file_path(self):
        base_dataset_dir = self.__conf_dict['dataset_dir']
        self.__conf_dict['mapping_file'] = os.path.join(base_dataset_dir, self.__conf_dict['mapping_file'])

    def __adjust_destination_size(self):
        dest_size = self.__conf_dict['destination_size']
        self.__conf_dict['destination_size'] = (dest_size[0], dest_size[1])

    def __getattr__(self, name):
        return self.__conf_dict[name]
