import yaml
import os
from typing import Dict
from src.common.ConfigReader import ConfigReader


class DataPreProcessingConfigReader(ConfigReader):

    def __init__(self, config_path=None):
        super().__init__(config_path)

    def _get_default_config_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'train-config.yaml')

    def _read_config(self, config_path) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.load(config_file)

    def _adjust_config_dict(self) -> None:
        self.__adjust_dataset_directories_paths()
        self.__adjust_mappping_file_path()
        self.__adjust_destination_size()

    def __adjust_dataset_directories_paths(self) -> None:
        base_dataset_dir = self._conf_dict['dataset_dir']
        keys_to_adjust = ['gt_dir', 'examples_dir', 'output_tfrecords_dir']
        for key in keys_to_adjust:
            self._conf_dict[key] = os.path.join(base_dataset_dir, self._conf_dict[key])

    def __adjust_mappping_file_path(self) -> None:
        base_dataset_dir = self._conf_dict['dataset_dir']
        self._conf_dict['mapping_file'] = os.path.join(base_dataset_dir, self._conf_dict['mapping_file'])

    def __adjust_destination_size(self) -> None:
        dest_size = self._conf_dict['destination_size']
        self._conf_dict['destination_size'] = (dest_size[0], dest_size[1])
