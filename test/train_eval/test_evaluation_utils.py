import unittest
from typing import Optional, Set, List

from parameterized import parameterized
import numpy as np

from src.dataset.utils.mapping_utils import Color2IdMapping
from src.train_eval.core.evaluation_utils import EvaluationAccumulatorEntry, \
    _Evaluator
from test.utils import FullArraySpec, assembly_array


class EvaluationAccumulatorEntryTest(unittest.TestCase):

    @parameterized.expand([
        [0, 0, None],
        [1, 0, None],
        [1, 1, 1.0],
        [1, 2, 1 / 2],
        [1, 100, 1 / 100],
        [1, 3, 1 / 3]
    ])
    def test_iou_value(self,
                       intersection: int,
                       union: int,
                       expected_result: Optional[float]
                       ) -> None:
        # given
        entry = EvaluationAccumulatorEntry(
            intersection=intersection,
            union=union
        )

        # when
        result = entry.get_iou()

        # then
        if expected_result is None:
            self.assertEqual(result, expected_result)
        else:
            self.assertAlmostEqual(result, expected_result)


class _EvaluatorTest(unittest.TestCase):

    @parameterized.expand([
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {},
            (10, 10, 3, [], np.uint8),
            (10, 10, 3, [], np.uint8),
            [1.0, None, None, None],
            1.0
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {},
            (10, 10, 3, [((0, 0), (10, 10), 0, 1)], np.uint8),
            (10, 10, 3, [], np.uint8),
            [0.0, 0.0, None, None],
            0.0
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [], np.uint8),
            [0.0, None, None, 0.0],
            0.0
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [], np.uint8),
            [None, None, None, 0.0],
            0.0
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [], np.uint8),
            [None, None, None, 0.0],
            0.0
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 3},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [], np.uint8),
            [None, None, None, None],
            None
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            [None, None, None, 1.0],
            1.0
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [((0, 0), (5, 5), 2, 1)], np.uint8),
            [None, None, None, 0.25],
            0.25
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (10, 10, 3, [((0, 0), (10, 10), 2, 1)], np.uint8),
            (10, 10, 3, [((0, 0), (9, 9), 2, 1)], np.uint8),
            [None, None, None, 81 / 100],
            81 / 100
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
            ),
            (10, 10, 3, [((0, 0), (2, 2), 2, 1)], np.uint8),
            [None, 0.0, None, 4 / 25],
            2 / 25
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
            ),
            (
                    10, 10, 3,
                    [
                        ((0, 0), (2, 2), 2, 1),
                        ((5, 5), (10, 10), 0, 1),
                    ],
                    np.uint8
            ),
            [None, 1.0, None, 4 / 25],
            29 / 50
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
            ),
            (
                    10, 10, 3,
                    [
                        ((0, 0), (2, 2), 2, 1),
                        ((5, 5), (7, 7), 0, 1),
                    ],
                    np.uint8
            ),
            [None, 4 / 25, None, 4 / 25],
            4 / 25
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {2},
            (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
            ),
            (
                    10, 10, 3,
                    [
                        ((0, 0), (2, 2), 2, 1),
                        ((5, 5), (7, 7), 0, 1),
                    ],
                    np.uint8
            ),
            [50 / 92, 4 / 25, None, 4 / 25],
            (50 / 92 + 8 / 25) / 3
        ],
    ])
    def test_single_evaluation(self,
                               mapping: Color2IdMapping,
                               ignored_classes: Set[int],
                               inference_result_spec: FullArraySpec,
                               gt_spec: FullArraySpec,
                               expected_ious: List[Optional[float]],
                               expected_miou: Optional[float]
                               ) -> None:
        # given
        evaluator = _Evaluator(
            mapping=mapping,
            ignored_classes=ignored_classes
        )
        inference_result = assembly_array(*inference_result_spec)
        gt = assembly_array(*gt_spec)
        # when
        evaluator.evaluate_example(inference_result=inference_result, gt=gt)

        # then
        for class_idx, expected_iou in enumerate(expected_ious):
            result_iou = evaluator.get_class_iou(class_id=class_idx)
            if expected_iou is None:
                self.assertEqual(result_iou, expected_iou)
            else:
                self.assertAlmostEqual(result_iou, expected_iou)
        result_miou = evaluator.get_miou()
        if expected_miou is None:
            self.assertEqual(result_miou, expected_miou)
        else:
            self.assertAlmostEqual(result_miou, expected_miou)

    @parameterized.expand([
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            [
                (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
                ),
            ],
            [
                (10, 10, 3, [((0, 0), (2, 2), 2, 1)], np.uint8),
            ],
            [None, 0.0, None, 4 / 25],
            2 / 25
        ],
        [
            {
                (0, 0, 0): 0,
                (1, 0, 0): 1,
                (0, 1, 0): 2,
                (0, 0, 1): 3,
            },
            {0, 2},
            [
                (
                        10, 10, 3,
                        [
                            ((0, 0), (5, 5), 2, 1),
                            ((5, 5), (10, 10), 0, 1)
                        ],
                        np.uint8
                ),
                (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
                )
            ],
            [
                (10, 10, 3, [((0, 0), (2, 2), 2, 1)], np.uint8),
                (
                    10, 10, 3,
                    [
                        ((0, 0), (5, 5), 2, 1),
                        ((5, 5), (10, 10), 0, 1)
                    ],
                    np.uint8
                 )
            ],
            [None, 0.5, None, 29 / 50],
            (25 + 29) / 100
        ],
    ])
    def test_multiple_evaluation(self,
                                 mapping: Color2IdMapping,
                                 ignored_classes: Set[int],
                                 inference_result_specs: List[FullArraySpec],
                                 gt_specs: List[FullArraySpec],
                                 expected_ious: List[Optional[float]],
                                 expected_miou: Optional[float]
                                 ) -> None:
        # given
        evaluator = _Evaluator(
            mapping=mapping,
            ignored_classes=ignored_classes
        )
        inference_results = []
        for inference_result_spec in inference_result_specs:
            inference_result = assembly_array(*inference_result_spec)
            inference_results.append(inference_result)
        gts = []
        for gt_spec in gt_specs:
            gt = assembly_array(*gt_spec)
            gts.append(gt)
        # when
        for inference_result, gt in zip(inference_results, gts):
            evaluator.evaluate_example(inference_result=inference_result, gt=gt)

        # then
        for class_idx, expected_iou in enumerate(expected_ious):
            result_iou = evaluator.get_class_iou(class_id=class_idx)
            if expected_iou is None:
                self.assertEqual(result_iou, expected_iou)
            else:
                self.assertAlmostEqual(result_iou, expected_iou)
        result_miou = evaluator.get_miou()
        if expected_miou is None:
            self.assertEqual(result_miou, expected_miou)
        else:
            self.assertAlmostEqual(result_miou, expected_miou)
