from typing import Tuple, List, Union

import numpy as np

Number = Union[int, float]
Point = Tuple[int, int]
ArrayContentSpec = List[Tuple[Point, Point, int, Number]]
FullArraySpec = Tuple[int, int, int, ArrayContentSpec, np.dtype]


def assembly_array(height: int,
                   width: int,
                   depth: int,
                   content_spec: ArrayContentSpec,
                   dtype: np.dtype
                   ) -> np.ndarray:
    array = np.zeros((height, width, depth), dtype=dtype)
    for (min_x, min_y), (max_x, max_y), channel, value in content_spec:
        array[min_y:max_y, min_x:max_x, channel] = value
    return array
