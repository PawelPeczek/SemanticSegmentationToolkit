import numpy as np
import yaml


def map_colour(X, mappings):

    def map_colour_in_row(row):
        return list(map(lambda e: mappings[e] if e in mappings else (0, 0, 0), row))

    return np.array(list(map(map_colour_in_row, X)))


def get_colour_to_id_mapping(mapping_path):
    mapping = get_mapping_file_content(mapping_path)
    result = {}
    for idx, colour in mapping.values():
        result[tuple(colour)] = idx
    return result


def get_id_to_colour_mapping(mapping_path):
    mapping = get_mapping_file_content(mapping_path)
    result = {}
    for idx, colour in mapping.values():
        result[idx] = colour
    return result


def get_mapping_file_content(mapping_path):
    with open(mapping_path, 'r') as stream:
        return yaml.load(stream)

