from typing import Union

import os
from src.common.ConfigReader import ConfigReader


class GraphExecutorConfigReader(ConfigReader):

    def __init__(self, config_path: Union[str, None], reader_type: str = 'train'):
        self.__reader_type = reader_type
        super().__init__(config_path)
        if config_path is None:
            self.__config_path = self._get_default_config_path()
        else:
            self.__config_path = config_path

    def get_config_path(self) -> str:
        return self.__config_path

    def _get_default_config_path(self) -> str:
        default_config_file_name = '{}-config.yaml'.format(self.__reader_type)
        return os.path.join(os.path.dirname(__file__), '..', '..', 'config', default_config_file_name)

    def _adjust_config_dict(self) -> None:
        self.__adjust_dataset_directories_paths()
        self.__adjust_mappping_file_path()
        self.__adjust_destination_size()
        if self.__reader_type == 'val':
            self.__adjust_checkpoint_path()

    def __adjust_dataset_directories_paths(self) -> None:
        base_dataset_dir = self._conf_dict['dataset_dir']
        self._conf_dict['tfrecords_dir'] = os.path.join(base_dataset_dir, self._conf_dict['tfrecords_dir'])

    def __adjust_mappping_file_path(self) -> None:
        base_dataset_dir = self._conf_dict['dataset_dir']
        self._conf_dict['mapping_file'] = os.path.join(base_dataset_dir, self._conf_dict['mapping_file'])

    def __adjust_destination_size(self) -> None:
        dest_size = self._conf_dict['destination_size']
        self._conf_dict['destination_size'] = (dest_size[0], dest_size[1])

    def __adjust_checkpoint_path(self) -> None:
        model_dir_path = self._conf_dict['model_dir']
        self._conf_dict['checkpoint_name'] = os.path.join(model_dir_path, self._conf_dict['checkpoint_name'])
