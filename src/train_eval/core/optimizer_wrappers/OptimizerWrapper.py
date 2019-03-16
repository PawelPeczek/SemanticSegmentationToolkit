from abc import ABC, abstractmethod


class OptimizerWrapper(ABC):

    def __init__(self, config_dict):
        self._config_dict = config_dict

    @abstractmethod
    def get_optimizer(self):
        raise NotImplementedError('This method must be implemented.')

    def _get_parameter_value(self, parameter_name, default_value=None):
        if parameter_name in self._config_dict:
            return self._config_dict[parameter_name]
        else:
            return default_value
