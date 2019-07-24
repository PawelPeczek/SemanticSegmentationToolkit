import numpy as np
import yaml
from typing import Dict, Tuple, List

Color = Tuple[int, int, int]
Id2ColorMapping = Dict[int, Color]
Color2IdMapping = Dict[Color, int]


def map_colour(x: np.ndarray,
               mappings: Id2ColorMapping) -> np.ndarray:

    def __map_color_for_pixel(pixel: int) -> Color:
        return mappings[pixel] if pixel in mappings else (0, 0, 0)

    def __map_color_in_row(row: np.ndarray) -> List[Color]:
        return list(map(__map_color_for_pixel, row))

    return np.array(list(map(__map_color_in_row, x)))


def get_color_to_id_mapping(mapping_path: str) -> Color2IdMapping:
    mapping = get_mapping_file_content(mapping_path)
    result = {}
    for idx, colour in mapping.values():
        result[tuple(colour)] = idx
    return result


def get_id_to_color_mapping(mapping_path: str) -> Id2ColorMapping:
    mapping = get_mapping_file_content(mapping_path)
    result = {}
    for idx, colour in mapping.values():
        result[idx] = colour
    return result


def get_mapping_file_content(mapping_path: str) -> Dict:
    with open(mapping_path, 'r') as stream:
        return yaml.load(stream)

