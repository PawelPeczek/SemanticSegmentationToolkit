import unittest

from parameterized import parameterized
import numpy as np

from src.dataset.preprocessing.images import _GroundTruthClassMapper
from src.dataset.utils.mapping_utils import Color2IdMapping
from test.utils import ArrayContentSpec, assembly_array


class GroundTruthClassMapper(unittest.TestCase):

    @parameterized.expand([
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3
            },
            (
                10, 10, 3,
                [],
                np.uint8
            ),
            (
                10, 10, 1,
                [],
                np.uint8
            )
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 101,
                (0, 1, 0): 2,
                (0, 0, 1): 3
            },
            (
                    10, 10, 3,
                    [((0, 0), (5, 5), 0, 1)],
                    np.uint8
            ),
            (
                    10, 10, 1,
                    [((0, 0), (5, 5), 0, 101)],
                    np.uint8
            )
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 101,
                (0, 1, 0): 2,
                (0, 0, 1): 3
            },
            (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 0, 1),
                        ((9, 9), (10, 10), 2, 1),
                    ],
                    np.uint8
            ),
            (
                    10, 10, 1,
                    [
                        ((0, 0), (5, 5), 0, 101),
                        ((9, 9), (10, 10), 0, 3),
                    ],
                    np.uint8
            )
        ]
    ])
    def test_mapping(self,
                     color2id: Color2IdMapping,
                     input_image_spec: ArrayContentSpec,
                     expected_output_spec: ArrayContentSpec
                     ) -> None:
        # given
        mapper = _GroundTruthClassMapper(
            color2id=color2id
        )
        input_image = assembly_array(*input_image_spec)
        expected_output = np.squeeze(assembly_array(*expected_output_spec))

        # when
        result = mapper.map_classes_id(input_image)
        # then
        self.assertTrue(np.array_equal(result, expected_output))
