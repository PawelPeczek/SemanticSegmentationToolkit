from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Any, Tuple

import numpy as np

from src.dataset.analysis.primitives import PreprocessedGroundTruth, \
    AnalysisResult, ContourDetail, Number


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


class ClassOccupationConsolidator(GroundTruthAnalysisConsolidator):

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

    def __calculate_new_class_percentage(self,
                                         acc_percentage: float,
                                         acc_weight: int,
                                         sample_percentage: float,
                                         sample_weight: int) -> float:
        new_value = \
            acc_percentage * acc_weight + sample_percentage * sample_weight
        return new_value / (acc_weight + sample_weight)


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
