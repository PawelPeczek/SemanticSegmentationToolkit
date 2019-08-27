import math
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Any, Tuple

import numpy as np

from src.dataset.analysis.primitives import PreprocessedGroundTruth, \
    AnalysisResult, ContourDetail, Number
from src.dataset.utils.mapping_utils import get_color_to_id_mapping


def calculate_weighted_sum(elems: List[Tuple[Number, Number]]) -> Number:

    def summator(acc: Tuple[Number, Number],
                 element: Tuple[Number, Number]) -> Tuple[Number, Number]:
        acc_weight, acc_value = acc
        element_weight, element_value = element
        nominator = (acc_weight * acc_value + element_weight * element_value)
        denominator = acc_weight + element_weight
        return denominator, nominator / denominator

    return reduce(summator, elems)[1]


class GroundTruthAnalyzer(ABC):

    @abstractmethod
    def analyze(self, ground_truth: PreprocessedGroundTruth) -> AnalysisResult:
        raise RuntimeError('This method must be implemented by derived class.')


class GroundTruthAnalysisConsolidator(ABC):

    @abstractmethod
    def consolidate(self,
                    already_consolidated: AnalysisResult,
                    to_consolidate: AnalysisResult) -> AnalysisResult:
        raise RuntimeError('This method must be implemented by derived class.')


class ClassOccupationAnalyzer(GroundTruthAnalyzer):

    def analyze(self, ground_truth: PreprocessedGroundTruth) -> AnalysisResult:
        image_height, image_width = ground_truth.image.shape[:2]
        image_area = image_height * image_width
        classes_percentages = {}
        for class_id in ground_truth.classes:
            class_contours = ground_truth.classes[class_id].contours
            class_area = self.__calculate_class_area(contours=class_contours)
            class_percentage = class_area / image_area
            classes_percentages[class_id] = class_percentage
        return AnalysisResult(
            name='classes_percentage_occupation',
            value=classes_percentages
        )

    def __calculate_class_area(self, contours: List[ContourDetail]) -> float:

        def area_adder(acc: float, contour: ContourDetail) -> float:
            return acc + contour.area

        return reduce(area_adder, contours, 0.0)


class ClassAverageConsolidator(GroundTruthAnalysisConsolidator):

    def consolidate(self,
                    accumulator: AnalysisResult,
                    to_consolidate: AnalysisResult) -> AnalysisResult:
        new_value = self.__connect_values(
            accumulator=accumulator,
            next_element=to_consolidate
        )
        total_samples = \
            accumulator.analyzed_samples + to_consolidate.analyzed_samples
        return AnalysisResult(
            name=accumulator.name,
            value=new_value,
            analyzed_samples=total_samples
        )

    def __connect_values(self,
                         accumulator: AnalysisResult,
                         next_element: AnalysisResult) -> Any:
        acc_weight = accumulator.analyzed_samples
        sample_weight = next_element.analyzed_samples
        new_value = {}
        for class_id in accumulator.value:
            acc_percentage = accumulator.value[class_id]
            sample_percentage = next_element.value[class_id]
            new_class_percentage = calculate_weighted_sum(
                [
                    (acc_weight, acc_percentage),
                    (sample_weight, sample_percentage)
                ]
            )
            new_value[class_id] = new_class_percentage
        return new_value


class PolyLineComplexityAnalyzer(GroundTruthAnalyzer):

    def analyze(self, ground_truth: PreprocessedGroundTruth) -> AnalysisResult:
        classes_complexity = {}
        for class_id in ground_truth.classes:
            preprocessed_ground_truth = ground_truth.classes[class_id]
            contours = preprocessed_ground_truth.contours
            avg_complexity = self.__calculate_avg_contours_complexity(
                contours=contours
            )
            classes_complexity[class_id] = avg_complexity
        return AnalysisResult(
            name='classes_polygons_complexity',
            value=classes_complexity
        )

    def __calculate_avg_contours_complexity(self,
                                            contours: List[ContourDetail]
                                            ) -> float:
        complexity = map(lambda c: c.contour.shape[0], contours)
        return sum(complexity) / len(contours) if len(contours) > 0 else 0.0


class InstancesAnalyzer(GroundTruthAnalyzer):

    def analyze(self, ground_truth: PreprocessedGroundTruth) -> AnalysisResult:
        classes_instances = {}
        for class_id in ground_truth.classes:
            preprocessed_ground_truth = ground_truth.classes[class_id]
            contours = preprocessed_ground_truth.contours
            classes_instances[class_id] = len(contours)
        return AnalysisResult(
            name='instances_number',
            value=classes_instances
        )


class ReceptiveFieldAnalyzer(GroundTruthAnalyzer):

    def __init__(self,
                 mapping_path: str,
                 kernel_size: Tuple[int, int],
                 stride: Tuple[int, int]):
        self.__mapping = get_color_to_id_mapping(mapping_path=mapping_path)
        self.__kernel_size = kernel_size
        self.__stride = stride

    def analyze(self, ground_truth: PreprocessedGroundTruth) -> AnalysisResult:
        average_classes = self.__proceed_conv(image=ground_truth.image)
        return AnalysisResult(
            name='receptive_field_variety',
            value=average_classes
        )

    def __proceed_conv(self, image: np.ndarray) -> float:
        steps_w, steps_h = self.__calculate_convolutional_steps(
            image=image
        )
        accumulator = 0
        for i in range(steps_h):
            for j in range(steps_w):
                receptive_field = self.__get_receptive_field(
                    image=image,
                    y_pos_id=i,
                    x_pos_id=j,
                )
                classes_number = self.__get_classes_number_in_receptive_field(
                    receptive_field=receptive_field
                )
                accumulator += classes_number
        steps = steps_h * steps_w
        return accumulator / steps if steps > 0 else 0

    def __calculate_convolutional_steps(self,
                                        image: np.ndarray) -> Tuple[int, int]:
        image_h, image_w = image.shape[:2]
        kernel_w, kernel_h = self.__kernel_size
        stride_w, stride_h = self.__stride
        steps_w = int(math.floor((image_w - kernel_w) / stride_w)) + 1
        steps_h = int(math.floor((image_h - kernel_h) / stride_h)) + 1
        return steps_w, steps_h

    def __get_receptive_field(self,
                              image: np.ndarray,
                              y_pos_id: int,
                              x_pos_id: int) -> np.ndarray:
        start_x = x_pos_id * self.__stride[0]
        end_x = start_x + self.__kernel_size[0]
        start_y = y_pos_id * self.__stride[1]
        end_y = start_y + self.__kernel_size[1]
        return image[start_y:end_y, start_x:end_x]

    def __get_classes_number_in_receptive_field(self,
                                                receptive_field: np.ndarray
                                                ) -> int:
        classes_encountered = set()
        for i in range(receptive_field.shape[0]):
            for j in range(receptive_field.shape[1]):
                current_colour = receptive_field[i, j]
                current_class = self.__mapping.get(current_colour, -1)
                classes_encountered.add(current_class)
        return len(classes_encountered)


class AverageConsolidator(GroundTruthAnalysisConsolidator):

    def consolidate(self,
                    already_consolidated: AnalysisResult,
                    to_consolidate: AnalysisResult) -> AnalysisResult:
        acc_weight = already_consolidated.analyzed_samples
        sample_weight = to_consolidate.analyzed_samples
        new_value = calculate_weighted_sum(
            [
                (acc_weight * already_consolidated.value),
                (sample_weight * to_consolidate.value)
            ]
        )
        total_samples = acc_weight + sample_weight
        return AnalysisResult(
            name=already_consolidated.name,
            value=new_value,
            analyzed_samples=total_samples
        )
