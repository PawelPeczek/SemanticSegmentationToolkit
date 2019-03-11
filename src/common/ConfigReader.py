from abc import ABC, abstractmethod


class ConfigReader(ABC):

    def __init__(self, config_path=None):
        """
        :param config_path: Path to config *.yaml file
        """
        if config_path is None:
            config_path = self._get_default_config_path()

        self.__conf_dict = self._read_config(config_path)
        self._adjust_config_dict()

    @abstractmethod
    def _get_default_config_path(self):
        raise NotImplementedError('this method must be implemented')

    @abstractmethod
    def _read_config(self, config_path):
        raise NotImplementedError('this method must be implemented')

    @abstractmethod
    def _adjust_config_dict(self):
        raise NotImplementedError('this method must be implemented')

    def __getattr__(self, name):
        return self.__conf_dict[name]

