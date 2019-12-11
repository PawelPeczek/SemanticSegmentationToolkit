import unittest
from typing import List, Any

from parameterized import parameterized
import numpy as np

from test.utils import assembly_array, ArrayContentSpec


class UtilsTest(unittest.TestCase):

    @parameterized.expand([
        [1, 1, 1, np.uint8],
        [10, 10, 10, np.float32],
        [100, 80, 3, np.int16]
    ])
    def test_empty_array(self,
                         height: int,
                         width: int,
                         depth: int,
                         dtype: np.dtype
                         ) -> None:
        # given
        expected_result = np.zeros((height, width, depth), dtype=dtype)

        # when
        result = assembly_array(
            height=height,
            width=width,
            depth=depth,
            content_spec=[],
            dtype=dtype
        )

        # then
        self.assertTrue(np.array_equal(result, expected_result))
        self.assertTrue(result.dtype, expected_result.dtype)

    @parameterized.expand([
        [1, 1, 1, np.uint8, [((0, 0), (1, 1), 0, 10)], [[10]]],
        [
            2, 2, 1, np.uint8,
            [
                ((0, 0), (1, 2), 0, 10),
                ((1, 0), (2, 2), 0, 20),
            ], [[10], [20], [10], [20]]
        ],
        [
            2, 2, 1, np.uint8,
            [
                ((0, 0), (1, 2), 0, 10),
                ((0, 0), (2, 2), 0, 20),
            ], [[20], [20], [20], [20]]
        ],
        [
            2, 2, 3, np.uint8,
            [
                ((0, 0), (1, 2), 0, 10),
                ((1, 0), (2, 2), 0, 20),
                ((0, 0), (1, 2), 1, 11),
                ((1, 0), (2, 2), 1, 21),
                ((0, 0), (1, 2), 2, 12),
                ((1, 0), (2, 2), 2, 22),
            ],
            [[10, 11, 12], [20, 21, 22], [10, 11, 12], [20, 21, 22]]
        ],
        [
            2, 2, 3, np.uint8,
            [
                ((0, 0), (1, 2), 0, 10),
                ((1, 0), (2, 2), 0, 20),
                ((0, 0), (1, 2), 1, 11),
                ((1, 0), (2, 2), 1, 21),
            ],
            [[10, 11, 0], [20, 21, 0], [10, 11, 0], [20, 21, 0]]
        ],
        [
            2, 2, 3, np.uint8,
            [
                ((0, 0), (1, 2), 0, 10),
                ((1, 0), (2, 2), 0, 20),
                ((0, 0), (1, 2), 1, 11),
                ((1, 0), (2, 2), 1, 21),
                ((0, 0), (1, 2), 2, 12),
                ((1, 0), (2, 2), 2, 22),
                ((0, 0), (2, 2), 2, 122),
            ],
            [[10, 11, 122], [20, 21, 122], [10, 11, 122], [20, 21, 122]]
        ],
    ])
    def test_array_with_values(self,
                               height: int,
                               width: int,
                               depth: int,
                               dtype: np.dtype,
                               content_specs: ArrayContentSpec,
                               expected_depthwise_values: List[List[Any]]
                               ) -> None:
        # given
        result = assembly_array(
            height=height,
            width=width,
            depth=depth,
            content_spec=content_specs,
            dtype=dtype
        )

        # then
        self.assertTrue(result.dtype, dtype)
        idx = 0
        for row in range(height):
            for col in range(width):
                depth_slice = result[row, col].tolist()
                self.assertListEqual(
                    depth_slice, expected_depthwise_values[idx]
                )
                idx += 1

