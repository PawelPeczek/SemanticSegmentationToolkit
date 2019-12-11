import os
import statistics
from typing import Dict, List, Tuple, Set, Optional
import logging

import numpy as np
import cv2 as cv
from tqdm import tqdm

from src.common.config_utils import GraphExecutorConfigReader
from src.dataset.utils.mapping_utils import get_color_to_id_mapping, \
    Color2IdMapping
from src.utils.filesystem_utils import read_csv_file


logging.getLogger().setLevel(logging.INFO)


class EvaluationAccumulatorEntry:

    def __init__(self, intersection: int = 0, union: int = 0):
        self.intersection = intersection
        self.union = union

    def get_iou(self) -> Optional[float]:
        return self.intersection / self.union if self.union != 0 else None


class PostInferenceEvaluator:

    def __init__(self, config: GraphExecutorConfigReader):
        self.__config = config

    def evaluate(self) -> None:
        id2gt = self.__get_id2gt_mapping()
        samples_to_evaluate = self.__prepare_evaluation_samples(id2gt)
        mapping = get_color_to_id_mapping(self.__config.mapping_file)
        evaluator = _Evaluator(
            mapping=mapping,
            ignored_classes=set(self.__config.ignore_labels)
        )
        self.__proceed_evaluation(
            samples=samples_to_evaluate,
            evaluator=evaluator
        )

    def __get_id2gt_mapping(self) -> Dict[str, str]:
        mapping_path = self.__config.validation_samples_mapping
        mapping_file_raw = read_csv_file(mapping_path)
        id2gt = {}
        for idx, _, file_path in mapping_file_raw:
            id2gt[idx] = file_path
        return id2gt

    def __prepare_evaluation_samples(self,
                                     id2gt: Dict[str, str]
                                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
        result = []
        for idx, gt_path in id2gt.items():
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
                             evaluator: '_Evaluator'
                             ) -> None:
        for inference_result, gt in tqdm(samples):
            evaluator.evaluate_example(
                inference_result=inference_result,
                gt=gt
            )
        self.__summarize(evaluator)

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
            accumulator[color_id].union += union_area
        else:
            accumulator_entry = EvaluationAccumulatorEntry(
                intersection=intersection_area,
                union=union_area
            )
            accumulator[color_id] = accumulator_entry
        return accumulator

    def __summarize(self,
                    evaluator: '_Evaluator'
                    ) -> None:
        ious = evaluator.get_classes_ious()
        for class_id, class_iou in ious:
            print(f'Class {class_id}: {class_iou}')
        print(f'mIou: {evaluator.get_miou()}')


class _Evaluator:

    def __init__(self,
                 mapping: Color2IdMapping,
                 ignored_classes: Set[int]):
        self.__mapping = mapping
        self.__ignored_classes = ignored_classes
        self.__accumulator = self.__initialize_accumulator()

    def get_classes_ious(self) -> List[Tuple[int, Optional[float]]]:
        result = []
        for class_id in self.__accumulator:
            class_iou = self.get_class_iou(class_id)
            result.append((class_id, class_iou))
        return result

    def get_class_iou(self, class_id: int) -> Optional[float]:
        if class_id not in self.__accumulator:
            return None
        class_evaluation = self.__accumulator[class_id]
        return class_evaluation.get_iou()

    def get_miou(self) -> Optional[float]:
        ious = []
        for class_id in self.__accumulator:
            class_iou = self.__accumulator[class_id].get_iou()
            if class_iou is None and class_iou not in self.__ignored_classes:
                logging.warning(f'Unable to calculate IoU for class {class_id}')
            else:
                ious.append(class_iou)
        if len(ious) == 0:
            return None
        else:
            return statistics.mean(ious)

    def evaluate_example(self,
                         inference_result: np.ndarray,
                         gt: np.ndarray) -> None:
        for color, color_id in self.__mapping.items():
            if color_id in self.__ignored_classes:
                continue
            inference_color_area = \
                cv.inRange(inference_result, color, color).astype(np.bool)
            gt_color_area = cv.inRange(gt, color, color).astype(np.bool)
            intersection = np.logical_and(inference_color_area, gt_color_area)
            union = np.logical_or(inference_color_area, gt_color_area)
            intersection_area = np.count_nonzero(intersection)
            union_area = np.count_nonzero(union)
            self.__update_accumulator(
                color_id=color_id,
                intersection_area=intersection_area,
                union_area=union_area
            )

    def __update_accumulator(self,
                             color_id: int,
                             intersection_area: int,
                             union_area: int
                             ) -> None:
        self.__accumulator[color_id].intersection += intersection_area
        self.__accumulator[color_id].union += union_area

    def __initialize_accumulator(self) -> Dict[int, EvaluationAccumulatorEntry]:
        result = dict()
        for _, class_id in self.__mapping.items():
            result[class_id] = EvaluationAccumulatorEntry()
        return result

