import os
import csv
from typing import Dict, List

import yaml


def dump_content_to_csv(file_path: str, content: List[list]) -> None:
    target_dir = os.path.dirname(file_path)
    create_directory(target_dir)
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(content)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, 0o755)


def read_yaml_file(file_path: str) -> Dict:
    with open(file_path, 'r') as config_file:
        return yaml.load(config_file)
