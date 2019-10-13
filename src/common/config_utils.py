from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
import os

from src.utils.filesystem_utils import read_yaml_file


class ConfigReader(ABC):

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self._get_default_config_path()
        self._conf_dict = self._read_config(config_path)
        self._adjust_config_dict()

    def is_option_set(self, option_name: str) -> bool:
        return option_name in self._conf_dict

    def get_or_else(self,
                    option_name: str,
                    else_value: Optional[Any] = None) -> Optional[Any]:
        if self.is_option_set(option_name):
            return self._conf_dict[option_name]
        else:
            return else_value

    def update_config_value(self, option_name: str, new_value: Any) -> None:
        if option_name not in self._conf_dict:
            raise RuntimeError('Trying to set unknown config value')
        self._conf_dict[option_name] = new_value

    @abstractmethod
    def _get_default_config_path(self) -> str:
        raise NotImplementedError('this method must be implemented')

    @abstractmethod
    def _adjust_config_dict(self) -> None:
        raise NotImplementedError('this method must be implemented')

    def _read_config(self, config_path) -> dict:
        return read_yaml_file(config_path)

    def _adjust_config_path(self,
                            config_path_key: str,
                            base_dir_key: str) -> None:
        fixed_path = os.path.join(
            self._conf_dict[base_dir_key],
            self._conf_dict[config_path_key])
        self._conf_dict[config_path_key] = fixed_path

    def _adjust_destination_size(self) -> None:
        dest_size = self._conf_dict['destination_size']
        self._conf_dict['destination_size'] = (dest_size[0], dest_size[1])

    def __getattr__(self, name: str) -> Any:
        return self._conf_dict[name]


class GraphExecutorConfigReader(ConfigReader):

    __DEFAULT_CONFIG_PATHS = {
        'train': os.path.join(
            os.path.dirname(__file__),
            '..',
            'train_eval',
            'config',
            'train-config.yml'
        ),
        'val': os.path.join(
            os.path.dirname(__file__),
            '..',
            'train_eval',
            'config',
            'val-config.yml'
        )
    }

    def __init__(self, config_path: Optional[str],
                 reader_type: str = 'train'):
        self.__reader_type = reader_type
        super().__init__(config_path)
        if config_path is None:
            self.__config_path = self._get_default_config_path()
        else:
            self.__config_path = config_path

    def get_config_path(self) -> str:
        return self.__config_path

    def _get_default_config_path(self) -> str:
        reader_type = self.__reader_type
        return GraphExecutorConfigReader.__DEFAULT_CONFIG_PATHS[reader_type]

    def _adjust_config_dict(self) -> None:
        self._adjust_config_path(
            config_path_key='tfrecords_dir',
            base_dir_key='dataset_dir')
        self._adjust_config_path(
            config_path_key='mapping_file',
            base_dir_key='dataset_dir')
        self._adjust_destination_size()
        if self.__reader_type == 'val':
            self._adjust_config_path(
                config_path_key='checkpoint_name',
                base_dir_key='model_dir')


class InferenceConfigReader(ConfigReader):

    __DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'train_eval',
        'config',
        'inference-config.yml'
    )

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)

    def _get_default_config_path(self) -> str:
        return InferenceConfigReader.__DEFAULT_CONFIG_PATH

    def _adjust_config_dict(self) -> None:
        self._adjust_config_path(
            config_path_key='checkpoint_name',
            base_dir_key='model_dir')


class DataPreProcessingConfigReader(ConfigReader):

    __DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'dataset',
        'config',
        'config.yml'
    )

    def __init__(self, config_path=None):
        super().__init__(config_path)

    def _get_default_config_path(self) -> str:
        return DataPreProcessingConfigReader.__DEFAULT_CONFIG_PATH

    def _adjust_config_dict(self) -> None:
        self.__adjust_dataset_directories_paths()
        self._adjust_destination_size()

    def __adjust_dataset_directories_paths(self) -> None:
        keys_to_adjust = ['gt_dir', 'examples_dir',
                          'output_tfrecords_dir', 'mapping_file']
        for key in keys_to_adjust:
            self._adjust_config_path(
                config_path_key=key,
                base_dir_key='dataset_dir')


class DataAnalysisConfigReader(ConfigReader):

    __DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__),
        '..',
        'dataset',
        'config',
        'analysis_config.yml'
    )

    def _get_default_config_path(self) -> str:
        return DataAnalysisConfigReader.__DEFAULT_CONFIG_PATH

    def _adjust_config_dict(self) -> None:
        pass
