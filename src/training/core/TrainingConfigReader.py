import yaml
import os
from src.common.ConfigReader import ConfigReader


class TrainingConfigReader(ConfigReader):

    def __init__(self, config_path=None):
        super().__init__(config_path)
        if config_path is None:
            self.__config_path = self._get_default_config_path()
        else:
            self.__config_path = config_path

    def get_config_path(self):
        return self.__config_path

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
        base_dataset_dir = self._conf_dict['dataset_dir']
        self._conf_dict['tfrecords_dir'] = os.path.join(base_dataset_dir, self._conf_dict['tfrecords_dir'])

    def __adjust_mappping_file_path(self):
        base_dataset_dir = self._conf_dict['dataset_dir']
        self._conf_dict['mapping_file'] = os.path.join(base_dataset_dir, self._conf_dict['mapping_file'])

    def __adjust_destination_size(self):
        dest_size = self._conf_dict['destination_size']
        self._conf_dict['destination_size'] = (dest_size[0], dest_size[1])

