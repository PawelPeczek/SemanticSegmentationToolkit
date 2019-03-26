from abc import ABC, abstractmethod
from typing import Dict, Any
import tensorflow as tf


class OptimizerWrapper(ABC):

    def __init__(self, config_dict: Dict):
        self._config_dict = config_dict

    @abstractmethod
    def get_optimizer(self) -> tf.train.Optimizer:
        raise NotImplementedError('This method must be implemented.')

    def _get_parameter_value(self, parameter_name: str, default_value: Any = None) -> Any:
        if parameter_name in self._config_dict:
            return self._config_dict[parameter_name]
        else:
            return default_value
