import os
import csv
from typing import Dict, List

import yaml


def read_csv_file(file_path: str, delimiter: str = ',') -> List[list]:
    file_content = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            file_content.append(row)
    return file_content


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


def read_text_file_lines(file_path: str) -> List[str]:
    file_content = []
    with open(file_path, 'r') as f:
        line = f.readline()
        line = line.strip()
        file_content.append(line)
    return file_content


def dump_text_file(file_path: str, content: str) -> None:
    dir_path = os.path.dirname(file_path)
    create_directory(directory_path=dir_path)
    with open(file_path, 'w') as f:
        f.write(content)
