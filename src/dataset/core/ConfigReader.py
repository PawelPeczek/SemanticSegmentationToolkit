import yaml
import os


class ConfigReader:

    def __init__(self, config_path=None):
        """
        :param config_path: Path to config *.yaml file
        """
        if config_path is None:
            config_path = self.__get_default_config_path()

        self.__conf_dict = self.__read_config(config_path)
        self.__adjust_config_dict()

    def __get_default_config_path(self):
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

    def __read_config(self, config_path):
        with open(config_path, 'r') as config_file:
            return yaml.load(config_file)

    def __adjust_config_dict(self):
        self.__adjust_dataset_directories_paths()
        self.__adjust_mappping_file_path()
        self.__adjust_destination_size()

    def __adjust_dataset_directories_paths(self):
        base_dataset_dir = self.__conf_dict['dataset_dir']
        keys_to_adjust = ['gt_dir', 'examples_dir', 'output_tfrecords_dir']
        for key in keys_to_adjust:
            self.__conf_dict[key] = os.path.join(base_dataset_dir, self.__conf_dict[key])

    def __adjust_mappping_file_path(self):
        base_dataset_dir = self.__conf_dict['dataset_dir']
        self.__conf_dict['mapping_file'] = os.path.join(base_dataset_dir, self.__conf_dict['mapping_file'])

    def __adjust_destination_size(self):
        dest_size = self.__conf_dict['destination_size']
        self.__conf_dict['destination_size'] = (dest_size[0], dest_size[1])

    def __getattr__(self, name):
        return self.__conf_dict[name]
