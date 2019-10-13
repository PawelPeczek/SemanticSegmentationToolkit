import os
import statistics
from typing import Dict, List, Tuple, Set

import numpy as np
import cv2 as cv
from tqdm import tqdm

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.utils.mapping_utils import get_color_to_id_mapping, \
    Color2IdMapping
from src.utils.filesystem_utils import read_csv_file


class EvaluationAccumulatorEntry:

    def __init__(self, intersection: int = 0, union: int = 0):
        self.intersection = intersection
        self.union = union

    def get_iou(self) -> float:
        return self.intersection / self.union if self.union != 0 else 0.0


class PostInferenceEvaluator:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config

    def evaluate(self) -> None:
        id2gt = self.__get_id2gt_mapping()
        samples_to_evaluate = self.__prepare_evaluation_samples(id2gt)
        mapping = get_color_to_id_mapping(self.__config.mapping_file)
        ignored_classes = set(self.__config.ignore_labels)
        self.__proceed_evaluation(
            samples=samples_to_evaluate,
            mapping=mapping,
            ignored_classes=ignored_classes
        )

    def __get_id2gt_mapping(self) -> Dict[str, str]:
        mapping_path = self.__config.validation_samples_mapping
        mapping_file_raw = read_csv_file(mapping_path)
        id2gt = {}
        for idx, file_path in mapping_file_raw:
            id2gt[idx] = file_path
        return id2gt

    def __prepare_evaluation_samples(self,
                                     id2gt: Dict[str, str]
                                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
        result = []
        for idx, _, gt_path in id2gt.items():
            inference_path = self.__prepare_inference_sample_path(idx)
            inference_image = cv.imread(inference_path)
            if inference_image is None:
                print(f'Lack of inference results for image id {idx}')
                continue
            inference_image = inference_image[..., ::-1]
            gt = cv.imread(gt_path)[..., ::-1]
            result.append((inference_image, gt))
        return result

    def __prepare_inference_sample_path(self, idx: str) -> str:
        return os.path.join(
            self.__config.model_dir,
            'model_inference',
            f'{idx}_inference_only.png'
        )

    def __proceed_evaluation(self,
                             samples: List[Tuple[np.ndarray, np.ndarray]],
                             mapping: Color2IdMapping,
                             ignored_classes: Set[int]
                             ) -> None:
        accumulator = {}
        for inference_result, gt in tqdm(samples):
            accumulator = self.__evaluate_example(
                inference_result=inference_result,
                gt=gt,
                mapping=mapping,
                ignored_classes=ignored_classes,
                accumulator=accumulator
            )
        self.__summarize(accumulator)

    def __evaluate_example(self,
                           inference_result: np.ndarray,
                           gt: np.ndarray,
                           mapping: Color2IdMapping,
                           ignored_classes: Set[int],
                           accumulator: Dict[int, EvaluationAccumulatorEntry]
                           ) -> Dict[int, EvaluationAccumulatorEntry]:
        for color, color_id in mapping.items():
            if color_id in ignored_classes:
                continue
            inference_color_area = \
                cv.inRange(inference_result, color, color).astype(np.bool)
            gt_color_area = cv.inRange(gt, color, color).astype(np.bool)
            intersection = np.logical_and(inference_color_area, gt_color_area)
            union = np.logical_or(inference_color_area, gt_color_area)
            intersection_area = np.count_nonzero(intersection)
            union_area = np.count_nonzero(union)
            accumulator = self.__update_accumulator(
                accumulator=accumulator,
                color_id=color_id,
                intersection_area=intersection_area,
                union_area=union_area
            )
        return accumulator

    def __update_accumulator(self,
                             accumulator: Dict[int, EvaluationAccumulatorEntry],
                             color_id: int,
                             intersection_area: int,
                             union_area: int
                             ) -> Dict[int, EvaluationAccumulatorEntry]:
        if color_id in accumulator:
            accumulator[color_id].intersection += intersection_area
            accumulator[color_id].union = union_area
        else:
            accumulator_entry = EvaluationAccumulatorEntry(
                intersection=intersection_area,
                union=union_area
            )
            accumulator[color_id] = accumulator_entry
        return accumulator

    def __summarize(self,
                    accumulator: Dict[int, EvaluationAccumulatorEntry]
                    ) -> None:
        ious = []
        for class_id, evaluation_entry in accumulator.items():
            class_iou = evaluation_entry.get_iou()
            print(f'Class {class_id}: {class_iou}')
            ious.append(class_iou)
        mean_iou = statistics.mean(ious)
        print(f'mIou: {mean_iou}')



