import os
from typing import Dict

import yaml


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, 0o755)


def read_yaml_file(file_path: str) -> Dict:
    with open(file_path, 'r') as config_file:
        return yaml.load(config_file)
