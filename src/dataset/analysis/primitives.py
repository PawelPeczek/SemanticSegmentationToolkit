from typing import List, Dict, Union, Any

import numpy as np
import cv2 as cv


Number = Union[int, float]


class ContourDetail:

    def __init__(self,
                 contour: np.ndarray):
        self.__contour = contour
        self.__area = cv.contourArea(contour)

    @property
    def contour(self) -> np.ndarray:
        return self.__contour

    @property
    def area(self) -> float:
        return self.__area


class GroundTruthClass:

    def __init__(self,
                 segmentation_mask: np.ndarray,
                 contours: List[ContourDetail]):
        self.__segmentation_mask = segmentation_mask
        self.__contours = contours

    @property
    def segmentation_mask(self) -> np.ndarray:
        return self.__segmentation_mask

    @property
    def contours(self) -> List[ContourDetail]:
        return self.__contours


class PreprocessedGroundTruth:

    def __init__(self,
                 image: np.ndarray,
                 classes: Dict[int, GroundTruthClass]):
        self.__image = image
        self.__classes = classes

    @property
    def image(self) -> np.ndarray:
        return self.__image

    @property
    def classes(self) -> Dict[int, GroundTruthClass]:
        return self.__classes


class AnalysisResult:

    def __init__(self,
                 name: str,
                 value: Any,
                 analyzed_samples: int = 1):
        self.__name = name
        self.__value = value
        self.__analyzed_samples = analyzed_samples

    @property
    def name(self) -> str:
        return self.__name

    @property
    def value(self) -> Any:
        return self.__value

    @property
    def analyzed_samples(self) -> int:
        return self.__analyzed_samples
