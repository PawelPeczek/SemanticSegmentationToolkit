import os
from typing import Dict, Union

from src.common.ConfigReader import ConfigReader


class InferenceConfigReader(ConfigReader):

    def __init__(self, config_path: Union[str, None] = None):
        super().__init__(config_path)

    def _get_default_config_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'inference-config.yaml')

    def _adjust_config_dict(self) -> None:
        model_dir_path = self._conf_dict['model_dir']
        self._conf_dict['checkpoint_name'] = os.path.join(model_dir_path, self._conf_dict['checkpoint_name'])

