import os


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.mkdir(directory_path, 0o755)
